"""NLST outcomes must be real, complete, and correctly decoded.

These read IDC's clinical tables, which are a separate (small, fast) download,
so they skip when it is absent rather than failing.
"""

from __future__ import annotations

import pytest

from cohort import clinical, labels

def _clinical_available() -> bool:
    """Whether IDC's clinical tables are on disk, without trying to fetch them."""
    try:
        clinical.clinical_dir(fetch=False)
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _clinical_available(),
    reason="IDC clinical tables not downloaded; call cohort.clinical.clinical_dir()",
)


class TestParticipantTable:
    def test_covers_the_whole_trial(self):
        people = clinical.load_table(clinical.PARTICIPANTS)
        # NLST enrolled ~53,000 across both arms.
        assert 53_000 <= len(people) <= 54_000
        assert people["pid"].is_unique

    def test_cancer_table_matches_the_participant_flags(self):
        """The two tables must agree on who had cancer."""
        people = clinical.load_table(clinical.PARTICIPANTS)
        cancers = clinical.load_table("nlst_canc")

        flagged = set(people.loc[~people["candx_days"].map(clinical.is_missing), "pid"])
        in_cancer_table = set(cancers["pid"].astype(str))
        assert flagged == in_cancer_table


class TestOutcomes:
    def test_labels_are_binary_and_complete(self):
        frame = clinical.outcomes()
        assert set(frame["label"].unique()) <= {0, 1}
        assert frame["PatientID"].is_unique
        assert (frame["label_source"] == "idc").all()

    def test_positives_match_the_published_trial_total(self):
        """NLST reports ~2,050 lung cancers across both arms."""
        frame = clinical.outcomes()
        positives = int((frame["label"] == 1).sum())
        assert 2_000 <= positives <= 2_100

    def test_a_positive_has_a_diagnosis_time(self):
        frame = clinical.outcomes()
        positives = frame[frame["label"] == 1]
        assert positives["days_to_diagnosis"].notna().all()

    def test_a_negative_has_follow_up_instead(self):
        """A 0 is only honest if we know how long they were followed."""
        frame = clinical.outcomes()
        negatives = frame[frame["label"] == 0]
        assert negatives["days_to_diagnosis"].isna().all()
        assert negatives["cancer_free_days"].notna().mean() > 0.99


class TestDecoding:
    def test_coded_values_become_text(self):
        assert clinical.decode(clinical.PARTICIPANTS, "gender", "1") == "Male"
        assert clinical.decode(clinical.PARTICIPANTS, "gender", "2") == "Female"
        assert clinical.decode(clinical.PARTICIPANTS, "cigsmok", "1") == "Current"

    def test_missing_markers_become_none(self):
        for marker in ("", ".N", ".M", None):
            assert clinical.decode(clinical.PARTICIPANTS, "cancyr", marker) is None

    def test_histology_codes_are_named(self):
        """`de_type` is bare ICD-O-3; the NLST dictionary has no mapping for it."""
        decoded = clinical.decode(clinical.PARTICIPANTS, "de_type", "8140")
        assert "Adenocarcinoma" in decoded
        assert "8140" in decoded

    def test_an_unknown_code_falls_through_rather_than_vanishing(self):
        assert clinical.decode(clinical.PARTICIPANTS, "de_type", "9999") == "9999"

    def test_stage_is_decoded_from_the_dictionary(self):
        assert clinical.decode(clinical.PARTICIPANTS, "de_stag", "110") == "Stage IA"


class TestPatientRecord:
    @staticmethod
    def a_positive() -> str:
        frame = clinical.outcomes()
        return frame.loc[frame["label"] == 1, "PatientID"].iloc[0]

    def test_positive_record_carries_tumour_detail(self):
        record = clinical.patient_record(self.a_positive())
        assert record["found"] is True
        assert record["label"] == 1
        assert record["outcome"]["diagnosed"] is True
        assert record["outcome"]["days_to_diagnosis"] is not None
        assert record["tumour"], "a diagnosed patient should have tumour fields"

    def test_negative_record_has_no_tumour_detail(self):
        frame = clinical.outcomes()
        negative = frame.loc[frame["label"] == 0, "PatientID"].iloc[0]
        record = clinical.patient_record(negative)
        assert record["label"] == 0
        assert record["outcome"]["diagnosed"] is False
        assert record["tumour"] == []
        assert record["locations"] == []

    def test_unknown_patient_is_reported_not_raised(self):
        record = clinical.patient_record("not-a-real-pid")
        assert record["found"] is False

    def test_screening_rounds_are_listed_with_results(self):
        record = clinical.patient_record(self.a_positive())
        assert record["screening"], "expected per-round screening results"
        for entry in record["screening"]:
            assert entry["round"] in ("T0", "T1", "T2")


class TestLabelIntegration:
    def test_from_idc_produces_a_mergeable_table(self):
        table = labels.from_idc()
        assert list(table.columns) == ["label", "label_source"]
        assert table.index.name == "PatientID"

    def test_merge_prefers_idc_over_an_annotation_list(self):
        """Annotation sets are positives-only; IDC knows about negatives too."""
        import pandas as pd

        ids = ["100002", "100004"]
        idc = labels.from_idc(ids)
        sybil_boxes = pd.DataFrame(
            {"label": [labels.POSITIVE], "label_source": ["sybil"]},
            index=pd.Index(["100002"], name="PatientID"),
        )
        merged = labels.merge(ids, sybil_boxes, idc)
        # `idc` outranks `sybil` in SOURCE_PRIORITY, so it wins the overlap.
        assert merged.frame.loc["100002", "label_source"] == "idc"
        assert merged.n_unknown == 0
