import { useEffect, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { extractSlice, visitLabel } from "../data";
import type { VolumeState } from "../hooks";
import type { HUWindow, SeriesMeta, Volume } from "../types";
import { fragmentShader, vertexShader } from "../volume/shaders";
import { PRESETS, windowToImageData } from "../windowing";
import { WindowControls } from "./WindowControls";

type Mode = "mip" | "surface" | "layers";
const MODE_INFO: Record<Mode, { name: string; blurb: string }> = {
  layers: { name: "Layers", blurb: "Lungs in blue, soft tissue in orange, bone in white — like a translucent model. Adjust each layer's opacity below." },
  surface: { name: "Surface", blurb: "Solid surface at a chosen density: skin (about −300 HU) or the skeleton (about +300 HU)." },
  mip: { name: "X-ray", blurb: "Maximum-intensity projection: each pixel shows the densest thing along its line of sight, like a see-through X-ray. Uses the window on the right." },
};

const MAX_TEXTURE_WIDTH = 256;

interface Props {
  meta: SeriesMeta | null;
  state: VolumeState;
  /** Window used for the image on the cutting plane; shared with the browse view. */
  window: HUWindow;
  /** Axial slice used for the cutting plane; shared with the browse view. */
  sliceIndex: number;
  onSliceIndexChange: (i: number) => void;
}

interface Three {
  renderer: THREE.WebGLRenderer;
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  controls: OrbitControls;
  material: THREE.ShaderMaterial;
  volumeMesh: THREE.Mesh;
  outline: THREE.LineSegments;
  planeMesh: THREE.Mesh;
  planeCanvas: HTMLCanvasElement;
  planeTexture: THREE.CanvasTexture;
  texture: THREE.Data3DTexture | null;
  raf: number;
}

const huToNorm = (hu: number) => (hu + 1000) / 2550;

/** Downsample the volume in-plane to at most 256 px and pack HU into uint8 (10 HU per level). */
function buildTexture(vol: Volume): THREE.Data3DTexture {
  const [d, h, w] = vol.meta.shape;
  const s = Math.max(1, Math.ceil(Math.max(w, h) / MAX_TEXTURE_WIDTH));
  const tw = Math.floor(w / s);
  const th = Math.floor(h / s);
  const out = new Uint8Array(tw * th * d);
  let o = 0;
  for (let z = 0; z < d; z++) {
    const zb = z * h * w;
    for (let y = 0; y < th; y++) {
      const yb = zb + y * s * w;
      for (let x = 0; x < tw; x++) {
        const v = (vol.data[yb + x * s] + 1000) / 10;
        out[o++] = v < 0 ? 0 : v > 255 ? 255 : v;
      }
    }
  }
  const tex = new THREE.Data3DTexture(out, tw, th, d);
  tex.format = THREE.RedFormat;
  tex.type = THREE.UnsignedByteType;
  tex.minFilter = THREE.LinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.unpackAlignment = 1;
  tex.needsUpdate = true;
  return tex;
}

/** Physical extent of the scan as (X = cols, Y = slices, Z = rows), normalised so the longest side is 1. */
function extent(meta: SeriesMeta): THREE.Vector3 {
  const [d, h, w] = meta.shape;
  const [sz, sy, sx] = meta.spacing;
  const v = new THREE.Vector3(w * sx, d * sz, h * sy);
  return v.divideScalar(Math.max(v.x, v.y, v.z));
}

export function VolumeView({ meta, state, window: win, sliceIndex, onSliceIndexChange }: Props) {
  const mount = useRef<HTMLDivElement>(null);
  const three = useRef<Three | null>(null);

  const [mode, setMode] = useState<Mode>("layers");
  const [isoHU, setIsoHU] = useState(300);
  const [opacity, setOpacity] = useState({ lung: 1, soft: 1, bone: 1 });
  const [steps, setSteps] = useState(350);
  const [cut, setCut] = useState(false);
  // The X-ray projection needs a wide window (the densest thing on every line of sight is bone).
  const [mipWindow, setMipWindow] = useState<HUWindow>(PRESETS.Bone);

  const volume = state.volume;
  const depth = meta?.shape[0] ?? 1;
  const slice = Math.min(sliceIndex, depth - 1);

  const requestRender = () => {
    const t = three.current;
    if (!t || t.raf) return;
    t.raf = requestAnimationFrame(() => {
      t.raf = 0;
      t.renderer.render(t.scene, t.camera);
    });
  };

  // One-time scene setup.
  useEffect(() => {
    const el = mount.current;
    if (!el) return;

    const renderer = new THREE.WebGLRenderer({ antialias: false, alpha: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
    renderer.setClearColor(0x000000, 1);
    renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
    el.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(38, 1, 0.05, 20);
    camera.position.set(1.4, 0.9, 2.1);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.minDistance = 1.1;
    controls.maxDistance = 6;
    controls.addEventListener("change", requestRender);

    const material = new THREE.ShaderMaterial({
      glslVersion: THREE.GLSL3,
      vertexShader,
      fragmentShader,
      uniforms: {
        uVolume: { value: null },
        uMode: { value: 2 },
        uSteps: { value: 350 },
        uIso: { value: huToNorm(300) },
        uWindow: { value: new THREE.Vector2(0, 1) },
        uOpacity: { value: new THREE.Vector3(1, 1, 1) },
        uClipOn: { value: false },
        uClipY: { value: 0 },
      },
      transparent: true,
      premultipliedAlpha: true,
      depthWrite: false,
      side: THREE.FrontSide,
    });
    const volumeMesh = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), material);
    volumeMesh.visible = false;
    volumeMesh.renderOrder = 2;
    scene.add(volumeMesh);

    const outline = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1)),
      new THREE.LineBasicMaterial({ color: 0x3a4a5c }),
    );
    outline.visible = false;
    scene.add(outline);

    const planeCanvas = document.createElement("canvas");
    const planeTexture = new THREE.CanvasTexture(planeCanvas);
    planeTexture.minFilter = THREE.LinearFilter;
    const planeMesh = new THREE.Mesh(
      new THREE.PlaneGeometry(1, 1),
      new THREE.MeshBasicMaterial({ map: planeTexture, side: THREE.DoubleSide }),
    );
    planeMesh.rotation.x = Math.PI / 2; // image top (anterior) -> +Z
    planeMesh.visible = false;
    planeMesh.renderOrder = 1;
    scene.add(planeMesh);

    const t: Three = { renderer, scene, camera, controls, material, volumeMesh, outline, planeMesh, planeCanvas, planeTexture, texture: null, raf: 0 };
    three.current = t;

    const ro = new ResizeObserver(() => {
      const { clientWidth: w, clientHeight: h } = el;
      if (!w || !h) return;
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      requestRender();
    });
    ro.observe(el);

    return () => {
      ro.disconnect();
      if (t.raf) cancelAnimationFrame(t.raf);
      controls.dispose();
      t.texture?.dispose();
      planeTexture.dispose();
      material.dispose();
      renderer.dispose();
      el.removeChild(renderer.domElement);
      three.current = null;
    };
  }, []);

  // New volume: rebuild the 3-D texture and resize the cube.
  useEffect(() => {
    const t = three.current;
    if (!t) return;
    t.texture?.dispose();
    t.texture = null;
    if (!volume) {
      t.volumeMesh.visible = t.outline.visible = t.planeMesh.visible = false;
      requestRender();
      return;
    }
    t.texture = buildTexture(volume);
    t.material.uniforms.uVolume.value = t.texture;
    const e = extent(volume.meta);
    t.volumeMesh.scale.copy(e);
    t.outline.scale.copy(e);
    t.planeMesh.scale.set(e.x, e.z, 1);
    t.volumeMesh.visible = t.outline.visible = true;
    requestRender();
  }, [volume]);

  // Render settings -> uniforms.
  useEffect(() => {
    const t = three.current;
    if (!t) return;
    const u = t.material.uniforms;
    u.uMode.value = mode === "mip" ? 0 : mode === "surface" ? 1 : 2;
    u.uSteps.value = steps;
    u.uIso.value = huToNorm(isoHU);
    u.uOpacity.value.set(opacity.lung, opacity.soft, opacity.bone);
    u.uWindow.value.set(huToNorm(mipWindow.center - mipWindow.width / 2), huToNorm(mipWindow.center + mipWindow.width / 2));
    requestRender();
  }, [mode, steps, isoHU, opacity, mipWindow]);

  // Cutting plane: position + the windowed slice image drawn on it.
  useEffect(() => {
    const t = three.current;
    if (!t) return;
    t.material.uniforms.uClipOn.value = cut && !!volume;
    t.planeMesh.visible = cut && !!volume;
    if (volume && cut) {
      const [d, h, w] = volume.meta.shape;
      const s = extractSlice(volume, "axial", slice);
      if (t.planeCanvas.width !== w || t.planeCanvas.height !== h) {
        t.planeCanvas.width = w;
        t.planeCanvas.height = h;
      }
      const img = new ImageData(w, h);
      windowToImageData(s, win, img);
      t.planeCanvas.getContext("2d")!.putImageData(img, 0, 0);
      t.planeTexture.needsUpdate = true;
      const yUnit = (slice + 0.5) / d - 0.5;
      t.material.uniforms.uClipY.value = yUnit;
      t.planeMesh.position.y = yUnit * t.volumeMesh.scale.y;
    }
    requestRender();
  }, [volume, cut, slice, win]);

  const lookFrom = (x: number, y: number, z: number) => {
    const t = three.current;
    if (!t) return;
    t.camera.position.set(x, y, z).normalize().multiplyScalar(2.6);
    t.controls.target.set(0, 0, 0);
    t.controls.update();
    requestRender();
  };

  if (!meta) return <div className="empty">Pick a scan on the left.</div>;

  return (
    <div className="view">
      <div className="stage">
        <div className="volume-box" ref={mount}>
          {!volume && !state.error && (
            <div className="slice-overlay">
              <div className="progress"><div style={{ width: `${Math.round(state.progress * 100)}%` }} /></div>
              <span>Loading… {Math.round(state.progress * 100)}%</span>
            </div>
          )}
          {state.error && <div className="slice-overlay error">{state.error}</div>}
          <div className="slice-label">
            <b>Patient {meta.patient}</b> · {visitLabel(meta)}
            <br />
            drag to rotate · wheel to zoom · right-drag to pan
          </div>
        </div>
        <div className="slider-row">
          <label className="toggle">
            <input type="checkbox" checked={cut} onChange={(e) => setCut(e.target.checked)} /> Cut at slice
          </label>
          <input type="range" min={0} max={depth - 1} value={slice} disabled={!cut} onChange={(e) => onSliceIndexChange(+e.target.value)} />
          <span className="slider-value">{slice + 1} / {depth}</span>
        </div>
      </div>

      <aside className="side">
        <div className="panel">
          <h3>Rendering</h3>
          <div className="chips">
            {(Object.keys(MODE_INFO) as Mode[]).map((m) => (
              <button key={m} className={`chip ${mode === m ? "active" : ""}`} onClick={() => setMode(m)}>{MODE_INFO[m].name}</button>
            ))}
          </div>
          <p className="hint">{MODE_INFO[mode].blurb}</p>

          {mode === "surface" && (
            <>
              <div className="chips">
                <button className="chip" onClick={() => setIsoHU(-300)}>Skin</button>
                <button className="chip" onClick={() => setIsoHU(300)}>Skeleton</button>
              </div>
              <label className="field">
                <span>Threshold <b>{isoHU} HU</b></span>
                <input type="range" min={-800} max={1200} step={10} value={isoHU} onChange={(e) => setIsoHU(+e.target.value)} />
              </label>
            </>
          )}

          {mode === "layers" && (
            <>
              {(["lung", "soft", "bone"] as const).map((k) => (
                <label className="field" key={k}>
                  <span>{k === "lung" ? "Lungs" : k === "soft" ? "Soft tissue" : "Bone"} <b>{opacity[k].toFixed(1)}×</b></span>
                  <input type="range" min={0} max={4} step={0.1} value={opacity[k]} onChange={(e) => setOpacity({ ...opacity, [k]: +e.target.value })} />
                </label>
              ))}
            </>
          )}

          <label className="field">
            <span>Quality <b>{steps} steps</b></span>
            <input type="range" min={120} max={700} step={10} value={steps} onChange={(e) => setSteps(+e.target.value)} />
          </label>
        </div>

        <div className="panel">
          <h3>View from</h3>
          <div className="chips">
            <button className="chip" onClick={() => lookFrom(0, 0.15, 1)}>Front</button>
            <button className="chip" onClick={() => lookFrom(1, 0.15, 0)}>Left side</button>
            <button className="chip" onClick={() => lookFrom(0, 1, 0.001)}>Top</button>
            <button className="chip" onClick={() => lookFrom(1.4, 0.9, 2.1)}>Angle</button>
          </div>
          <p className="hint">Head is up. Turn on "Cut at slice" to remove everything above the chosen axial slice and see where that 2-D picture sits inside the body.</p>
        </div>

        {mode === "mip" && <WindowControls value={mipWindow} onChange={setMipWindow} />}
      </aside>
    </div>
  );
}
