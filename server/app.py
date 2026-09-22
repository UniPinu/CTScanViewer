"""The FastAPI service the viewer talks to (CONTEXT.md §4 Stage 6, §7).

Endpoint surface is §7's, plus two diagnostics the UI genuinely needs:

    GET  /cohort?split=test&limit=50     list patients
    GET  /patient/{id}                   metadata, rounds, what is cached
    POST /patient/{id}/fetch             stream + preprocess  -> job_id
    POST /infer                          run inference        -> job_id
    GET  /jobs/{job_id}                  status and progress
    GET  /jobs/{job_id}/stream           the same, as server-sent events
    GET  /results/{patient_id}           the results object (§5.3)
    GET  /random-patient                 backs the existing button
    GET  /system                         whether a risk model is installed
    GET  /health                         liveness

Run it with:
    .venv/Scripts/uvicorn server.app:app --reload --port 8000

The viewer's Vite dev server proxies /api to this (see viewer/vite.config.ts),
so the browser sees one origin and there is no CORS in development.
"""

from __future__ import annotations

import asyncio
import json
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from core import jsonio, paths
from core.provenance import DISCLAIMER, stamp
from ingest.stream import SeriesCache
from ml.risk import load_backend
from server import pipeline
from server.jobs import Job, progress_reporter, registry

app = FastAPI(
    title="Lung tissue change & cancer-risk pipeline",
    description=DISCLAIMER,
    version="0.1.0",
)

# The viewer is served by Vite on another port in development. In production it
# is static files behind the same origin and this does nothing.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class InferRequest(BaseModel):
    patient_id: str
    tasks: list[Literal["risk", "change", "saliency"]] = Field(
        default_factory=lambda: ["risk", "change", "saliency"]
    )


def reporter_for(job: Job) -> pipeline.Reporter:
    """Wire pipeline progress into the job record."""
    log = progress_reporter(job)

    def on_stage(stage: str, message: str = "", progress: float | None = None) -> None:
        log(stage, message)
        job.progress = progress

    return pipeline.Reporter(on_stage=on_stage)


# ---------------------------------------------------------------- diagnostics


@app.get("/health")
def health() -> dict:
    return {"ok": True, "busy": registry.busy()}


@app.get("/system")
def system() -> dict:
    """What this deployment can actually do.

    The UI reads this at startup so it can show the risk panel's real state
    instead of a spinner that never resolves when no model is installed.
    """
    backend = load_backend()
    cache = SeriesCache()
    return {
        "risk": {"available": backend.available(), "name": backend.name, **backend.describe()},
        "cohort": {
            "built": paths.SERIES_PARQUET.exists(),
            "splits": paths.SPLITS_PARQUET.exists(),
        },
        "cache": {**cache.status(), "root": str(cache.status()["root"])},
        "provenance": stamp("service"),
    }


# -------------------------------------------------------------------- cohort


@app.get("/cohort")
def cohort(
    split: Literal["train", "val", "test"] | None = None,
    limit: int = Query(50, ge=1, le=1000),
    min_rounds: int = Query(1, ge=1, le=3),
    q: str | None = None,
    label: int | None = Query(None, ge=-1, le=1, description="1 cancer, 0 none, -1 unknown"),
) -> dict:
    try:
        patients = pipeline.cohort_listing(
            split=split, limit=limit, min_rounds=min_rounds, query=q, label=label
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    return {"patients": patients, "count": len(patients), "split": split}


@app.get("/random-patient")
def random_patient(
    split: Literal["train", "val", "test"] | None = "test",
    min_rounds: int = Query(2, ge=1, le=3),
) -> dict:
    """A random patient — held-out and multi-round by default.

    Those defaults are the honest ones for a demo: the model never trained on a
    test patient, and change detection needs at least two rounds to say anything.
    """
    try:
        patient_id = pipeline.random_patient(split=split, min_rounds=min_rounds)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except LookupError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    return pipeline.patient_overview(patient_id)


@app.get("/patient/{patient_id}")
def patient(patient_id: str) -> dict:
    try:
        return pipeline.patient_overview(patient_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


# ---------------------------------------------------------------------- jobs


@app.post("/patient/{patient_id}/fetch", status_code=202)
def fetch(patient_id: str, masks: bool = True) -> dict:
    """Stream and preprocess one patient. Returns immediately with a job id."""
    if registry.busy():
        raise HTTPException(
            status_code=409,
            detail="Another job is already running. This server runs one heavy job at a "
                   "time so concurrent fetches cannot evict each other from the cache.",
        )
    try:
        pipeline.patient_overview(patient_id)  # 404 now rather than inside the job
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    job = registry.submit(
        "fetch",
        lambda j: pipeline.fetch_patient(patient_id, reporter_for(j), with_masks=masks),
        patient_id=patient_id,
    )
    return job.as_dict()


@app.post("/infer", status_code=202)
def infer(request: InferRequest) -> dict:
    if registry.busy():
        raise HTTPException(status_code=409, detail="Another job is already running.")
    try:
        pipeline.patient_overview(request.patient_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    job = registry.submit(
        "infer",
        lambda j: pipeline.infer(request.patient_id, list(request.tasks), reporter_for(j)),
        patient_id=request.patient_id,
    )
    return job.as_dict()


@app.get("/jobs")
def jobs(limit: int = Query(20, ge=1, le=100)) -> dict:
    return {"jobs": [job.as_dict() for job in registry.list(limit)]}


@app.get("/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    job = registry.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"No job {job_id}")
    return job.as_dict()


@app.get("/jobs/{job_id}/stream")
async def job_stream(job_id: str) -> StreamingResponse:
    """Server-sent events for one job.

    §4 asks for progress that is "SSE friendly", and the loading state matters
    here: fetching is tens of seconds and §8.4 wants the *stage* shown, not a
    spinner. Polling would work; this just makes the UI's job simpler.
    """
    if registry.get(job_id) is None:
        raise HTTPException(status_code=404, detail=f"No job {job_id}")

    async def events():
        last = None
        while True:
            job = registry.get(job_id)
            if job is None:
                break
            payload = job.as_dict()
            # Only push when something actually changed, so an idle job does not
            # stream one event per tick.
            fingerprint = (payload["status"], payload["stage"], payload["message"], payload["progress"])
            if fingerprint != last:
                last = fingerprint
                yield f"data: {json.dumps(payload, default=str)}\n\n"
            if payload["status"] in ("done", "error", "cancelled"):
                break
            await asyncio.sleep(0.4)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ------------------------------------------------------------------- results


@app.get("/results/{patient_id}")
def results(patient_id: str) -> dict:
    path = pipeline.results_path(patient_id)
    if not path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No results for patient {patient_id}. POST /infer to produce them.",
        )
    return jsonio.read(path)


@app.on_event("shutdown")
def shutdown() -> None:
    registry.shutdown()
