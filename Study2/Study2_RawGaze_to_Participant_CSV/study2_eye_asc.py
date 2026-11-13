import os
import re
from collections import Counter

import numpy as np
import pandas as pd


# ============================================================
# STUDY 1 — RAW ASC -> PARTICIPANT EYE CSVs
#
# mode = 1 : trial onset -> response       (main response gaze)
# mode = 2 : feedback period              (feedback gaze)
# mode = 3 : all valid fixations in trial
#
# IMPORTANT:
# ------------------------------------------------------------
# - Raw ASC files are READ ONLY.
# - Every TRIALID occurrence becomes one separate output row.
# - raw_id is retained exactly as EyeLink logged it.
# - raw_id is NOT used as the merge key.
# - Reused raw_id values therefore cannot collapse/replace rows here.
# - start_ts and asc_line retain the raw-file location of each event.
#
# Trial identity for later merging:
#     pid + Phase + phase_trial
#
# IMPORTANT OUTPUT RULE:
# Training trials are parsed internally ONLY so that the raw event stream
# and the transition into the main experiment can be audited. Training
# rows are NOT written to the participant eye CSVs.
#
# Main output:
#     mode 1 -> LE + ES + EE response-gaze rows only (660 if complete)
#     mode 2 -> LE feedback rows only (160 if complete)
#     mode 3 -> LE + ES + EE all-fixation rows only (660 if complete)
#
# Provenance columns retained in the main-task CSVs:
#     pid         participant number
#     raw_id      EyeLink TrialID exactly as recorded
#     trial_seq   chronological TRIALID occurrence in the ASC file
#     phase_trial chronological trial number within Phase
#     start_ts    timestamp of the TRIALID event
#     asc_line    source ASC line containing that TRIALID
#
# HOW PHASE LABELS ARE ASSIGNED
# ------------------------------------------------------------
# The experiment writes Training/Task metadata AFTER the trial activity
# and before the next TRIALID. Therefore metadata are attached to the
# currently open trial, not carried forward to the following trial.
#
# Main experimental phase assignment:
#     final Phase is assigned from the programmed chronological design:
#       LE = first 160 main trials
#       ES = next 360 main trials
#       EE = final 140 main trials
#
# Raw Training/Task messages are preserved separately as provenance.
# This is necessary because P1-P3 contain two displaced Task metadata
# labels at the ES/EE boundary; independent raw-response vs behaviour
# validation showed 0 mismatches under the programmed chronology.
#
# Training:
# The first training task writes its metadata only at block endings, so
# not every early training trial contains its own Training/Task message.
# The experiment design places the complete training half before the
# first confirmed Training 0 trial. Therefore every TRIALID event before
# the first confirmed main-experiment trial is classified TRAIN FOR AUDIT
# PURPOSES ONLY, then excluded from saved participant eye datasets.
#
# Crucially, this does NOT infer or alter raw TrialID values.
# It only labels chronological trial events that are already present.
# ============================================================


# ============================================================
# SETTINGS
# ============================================================

mode = 3

SOURCE_ROOT = r"D:/Aberdeen_Uni_June24/cap/THESIS/OV_Analysis/data"
OUTPUT_ROOT = r"D:/Birmingham_Uni_Oct25/Eye_Tracking_Project/Study2/Study2_Data"

# Process all participants by default.
# Set to an integer (e.g. 18) for a single-participant test.
ONLY_PID = None

# Participant 6 is known to be incomplete at the study level.
# It remains in the parser audit so incompleteness is transparent,
# but expected-count failure alone is labelled KNOWN_INCOMPLETE rather
# than being mistaken for a parser failure.
EXCLUDE = {2, 6, 9, 14, 18, 20, 26}

KNOWN_INCOMPLETE = set()
# One concise audit file is maintained across all three parser modes.
AUDIT_FILENAME = "Study2_eye_PARSE_AUDIT.csv"
AUDIT_VERSION = "study2_main_only_design_phase_v1"


# ============================================================
# EXPECTED DESIGN COUNTS
# ============================================================

# Eye-tracked choice trials in the training half:
# Task 1 = 3 blocks x 8 = 24
# Task 2 = 2 blocks x 5 = 10
# Task 3 = 2 blocks x 4 = 8
# Total = 42
#
# The 6-item training slider exists in the experiment but does not create
# TRIALID events in this eye-tracking stream.
EXPECTED_TRAIN = 42

EXPECTED_MAIN = {
    "LE": 160,
    "ES": 360,
    "EE": 140,
}

EXPECTED_TOTAL_EYE_TRIALS = (
    EXPECTED_TRAIN
    + EXPECTED_MAIN["LE"]
    + EXPECTED_MAIN["ES"]
    + EXPECTED_MAIN["EE"]
)  # 702


# ============================================================
# SCREEN / ROI SETTINGS
# ============================================================

screen_width = 1919
screen_height = 1079
screen_center = (screen_width / 2, screen_height / 2)

middle_margin = 50
middle_left_bound = screen_center[0] - middle_margin
middle_right_bound = screen_center[0] + middle_margin


# ============================================================
# HELPERS
# ============================================================

def extract_pid(path):
    """Extract participant number from a path containing e.g. sub-18."""
    match = re.search(
        r"(?:^|[\\/])sub-(\d+)(?:[\\/]|$)",
        path,
        flags=re.I,
    )

    if not match:
        raise ValueError(
            f"Could not identify participant from path: {path}"
        )

    return int(match.group(1))


def roi_from_x(avg_x):
    """Physical ROI only: 1 = LEFT, 2 = RIGHT."""
    if avg_x < middle_left_bound:
        return 1

    if avg_x > middle_right_bound:
        return 2

    return None


def middle_summary(fixations):
    if len(fixations) <= 2:
        return None, 0

    mids = fixations[1:-1]

    left_dur = sum(
        f["duration"]
        for f in mids
        if f["roi"] == 1
    )

    right_dur = sum(
        f["duration"]
        for f in mids
        if f["roi"] == 2
    )

    if left_dur > right_dur:
        return 1, left_dur

    if right_dur > left_dur:
        return 2, right_dur

    return 0, left_dur


def gaze_switches(fixations):
    return sum(
        1
        for i in range(1, len(fixations))
        if fixations[i]["roi"] != fixations[i - 1]["roi"]
    )


def explicit_main_phase(train, task):
    """
    Return LE/ES/EE only when the trial itself explicitly contains
    Training 0 + Task 1/2/3 metadata.
    """
    if pd.isna(train):
        return None

    try:
        train = int(train)
    except (TypeError, ValueError):
        return None

    if train != 0:
        return None

    task = str(task).strip().lower()

    if task == "task 1":
        return "LE"

    if task == "task 2":
        return "ES"

    if task == "task 3":
        return "EE"

    return None


def add_response_metrics(record, fixations):
    """Mode 1: trial onset -> response."""
    mid_loc, mid_dur = middle_summary(fixations)

    dwell_left = sum(
        f["duration"]
        for f in fixations
        if f["roi"] == 1
    )

    dwell_right = sum(
        f["duration"]
        for f in fixations
        if f["roi"] == 2
    )

    dwell_total = sum(
        f["duration"]
        for f in fixations
    )

    first_dur = (
        fixations[0]["duration"]
        if fixations
        else None
    )

    final_dur = (
        fixations[-1]["duration"]
        if fixations
        else None
    )

    middle_dur = max(
        0,
        dwell_total
        - ((first_dur or 0) + (final_dur or 0))
    )

    record.update({
        "DwellLeft": dwell_left,
        "DwellRight": dwell_right,
        "DwellTotal": dwell_total,
        "Nfix": len(fixations),
        "FirstFixLoc": (
            fixations[0]["roi"]
            if fixations else None
        ),
        "FirstFixDur": first_dur,
        "FinalFixLoc": (
            fixations[-1]["roi"]
            if fixations else None
        ),
        "FinalFixDur": final_dur,
        "MiddleFixDur": middle_dur,
        "eachMiddleFixDur": (
            middle_dur / max(1, len(fixations) - 2)
            if len(fixations) > 2
            else 0
        ),
        "MiddleDominantLoc": mid_loc,
        "MiddleDominantDur": mid_dur,
        "GazeSwitch": gaze_switches(fixations),
        "LeftFixNR": sum(
            1 for f in fixations if f["roi"] == 1
        ),
        "RightFixNR": sum(
            1 for f in fixations if f["roi"] == 2
        ),
        "GazeDiff": abs(dwell_left - dwell_right),
        "last_roi": (
            fixations[-1]["roi"]
            if fixations else None
        ),
        "Fixations": list(fixations),
    })

    return record


def add_allfix_metrics(record, fixations):
    """Mode 3: all valid fixations associated with the trial block."""
    dwell_left = sum(
        f["duration"]
        for f in fixations
        if f["roi"] == 1
    )

    dwell_right = sum(
        f["duration"]
        for f in fixations
        if f["roi"] == 2
    )

    dwell_total = sum(
        f["duration"]
        for f in fixations
    )

    first_dur = (
        fixations[0]["duration"]
        if fixations
        else None
    )

    final_dur = (
        fixations[-1]["duration"]
        if fixations
        else None
    )

    middle_dur = max(
        0,
        dwell_total
        - ((first_dur or 0) + (final_dur or 0))
    )

    record.update({
        "DwellLeft_allfix": dwell_left,
        "DwellRight_allfix": dwell_right,
        "DwellTotal_allfix": dwell_total,
        "Nfix_allfix": len(fixations),
        "FirstFixLoc_allfix": (
            fixations[0]["roi"]
            if fixations else None
        ),
        "FirstFixDur_allfix": first_dur,
        "FinalFixLoc_allfix": (
            fixations[-1]["roi"]
            if fixations else None
        ),
        "FinalFixDur_allfix": final_dur,
        "MiddleFixDur_allfix": middle_dur,
        "eachMiddleFixDur_allfix": (
            middle_dur / max(1, len(fixations) - 2)
            if len(fixations) > 2
            else 0
        ),
        "GazeSwitch_allfix": gaze_switches(fixations),
        "LeftFixNR_allfix": sum(
            1 for f in fixations if f["roi"] == 1
        ),
        "RightFixNR_allfix": sum(
            1 for f in fixations if f["roi"] == 2
        ),
        "GazeDiff_allfix": abs(
            dwell_left - dwell_right
        ),
        "last_roi_allfix": (
            fixations[-1]["roi"]
            if fixations else None
        ),
        "Fixations_allfix": list(fixations),
    })

    return record


def add_feedback_metrics(record, fixations):
    """Mode 2: feedback window only."""
    dwell_left = sum(
        f["duration"]
        for f in fixations
        if f["roi"] == 1
    )

    dwell_right = sum(
        f["duration"]
        for f in fixations
        if f["roi"] == 2
    )

    dwell_total = dwell_left + dwell_right

    first_dur = (
        fixations[0]["duration"]
        if fixations
        else 0
    )

    final_dur = (
        fixations[-1]["duration"]
        if fixations
        else 0
    )

    middle_dur = max(
        0,
        dwell_total - (first_dur + final_dur)
    )

    record.update({
        "DwellLeft_Feed": dwell_left,
        "DwellRight_Feed": dwell_right,
        "DwellTotal_Feed": dwell_total,
        "Nfix_Feed": len(fixations),
        "FirstFixLoc_Feed": (
            fixations[0]["roi"]
            if fixations else None
        ),
        "FirstFixDur_Feed": first_dur,
        "FinalFixLoc_Feed": (
            fixations[-1]["roi"]
            if fixations else None
        ),
        "FinalFixDur_Feed": final_dur,
        "MiddleFixDur_Feed": middle_dur,
        "eachMiddleFixDur_Feed": (
            middle_dur / max(1, len(fixations) - 2)
            if len(fixations) > 2
            else 0
        ),
        "GazeSwitch_Feed": gaze_switches(fixations),
        "LeftFixNR_Feed": sum(
            1 for f in fixations if f["roi"] == 1
        ),
        "RightFixNR_Feed": sum(
            1 for f in fixations if f["roi"] == 2
        ),
        "GazeDiff_Feed": abs(
            dwell_left - dwell_right
        ),
        "last_roi_Feed": (
            fixations[-1]["roi"]
            if fixations else None
        ),
        "Fixations_Feed": list(fixations),
    })

    return record


def classify_phases(df, pid):
    """
    Assign analysis Phase from the PROGRAMMED CHRONOLOGICAL TASK ORDER,
    while preserving the raw EyeLink Training/Task metadata separately.

    Why:
    ----
    Independent validation showed that for participants 1-3 the raw
    Training/Task metadata are displaced at the ES/EE boundary:

        metadata mapping -> many raw-response/behaviour mismatches
        design chronology -> 0 mismatches across all 660 main trials

    The experiment program executes:
        42 eye-tracked training trials
        160 LE trials
        360 ES trials
        140 EE trials

    Therefore:
      - raw Task metadata remain provenance only;
      - final Phase / phase_trial follow the validated programmed order;
      - raw metadata discrepancies are explicitly retained and audited.
    """
    df = df.copy()

    # Preserve exactly what the raw ASC trial block said.
    df["Task_raw"] = df["Task"]

    df["Phase_metadata"] = [
        explicit_main_phase(train, task)
        for train, task in zip(
            df["train"],
            df["Task_raw"],
        )
    ]

    explicit_main_positions = np.flatnonzero(
        df["Phase_metadata"]
        .isin(["LE", "ES", "EE"])
        .to_numpy()
    )

    if len(explicit_main_positions) == 0:
        raise ValueError(
            f"participant {pid}: no explicit Training 0 main trial found"
        )

    first_main_pos = int(
        explicit_main_positions[0]
    )

    # The first confirmed main trial must occur immediately after the
    # 42 eye-tracked training trials.
    if first_main_pos != EXPECTED_TRAIN:
        raise ValueError(
            f"participant {pid}: first main trial occurs at zero-based "
            f"position {first_main_pos}, expected {EXPECTED_TRAIN}"
        )

    expected_total = (
        EXPECTED_TRAIN
        + sum(EXPECTED_MAIN.values())
    )

    if len(df) != expected_total:
        raise ValueError(
            f"participant {pid}: parsed {len(df)} raw trial events, "
            f"expected {expected_total}; refusing to infer phase boundaries"
        )

    # Final analysis labels: programmed chronological order.
    df["Phase"] = "TRAIN"
    df["phase_trial"] = np.nan
    df["main_trial_seq"] = np.nan

    main_idx = df.index[
        first_main_pos:
    ]

    if len(main_idx) != sum(EXPECTED_MAIN.values()):
        raise ValueError(
            f"participant {pid}: expected 660 main events after training, "
            f"found {len(main_idx)}"
        )

    le_end = EXPECTED_MAIN["LE"]
    es_end = le_end + EXPECTED_MAIN["ES"]

    le_idx = main_idx[:le_end]
    es_idx = main_idx[le_end:es_end]
    ee_idx = main_idx[es_end:]

    df.loc[le_idx, "Phase"] = "LE"
    df.loc[es_idx, "Phase"] = "ES"
    df.loc[ee_idx, "Phase"] = "EE"

    df.loc[le_idx, "phase_trial"] = np.arange(
        1, EXPECTED_MAIN["LE"] + 1
    )
    df.loc[es_idx, "phase_trial"] = np.arange(
        1, EXPECTED_MAIN["ES"] + 1
    )
    df.loc[ee_idx, "phase_trial"] = np.arange(
        1, EXPECTED_MAIN["EE"] + 1
    )

    df.loc[main_idx, "main_trial_seq"] = np.arange(
        1,
        sum(EXPECTED_MAIN.values()) + 1,
    )

    df["phase_assignment"] = "programmed_chronology"

    # Raw metadata disagreement is kept visible rather than silently
    # rewritten. Missing metadata is not counted as disagreement.
    metadata_known = df["Phase_metadata"].notna()

    df["phase_metadata_mismatch"] = False

    df.loc[
        metadata_known,
        "phase_metadata_mismatch",
    ] = (
        df.loc[metadata_known, "Phase_metadata"]
        != df.loc[metadata_known, "Phase"]
    )

    # Final merge key must be unique.
    main = df[
        df["Phase"].isin(["LE", "ES", "EE"])
    ]

    if main.duplicated(
        ["pid", "Phase", "phase_trial"]
    ).any():
        raise ValueError(
            f"participant {pid}: duplicate pid + Phase + phase_trial"
        )

    return df


def build_audit(
    full_df,
    saved_df,
    pid,
    label,
    trialid_events_found,
):
    """
    One concise participant-level audit row.

    full_df:
        All chronological TRIALID events parsed from the raw ASC,
        including training. Used ONLY for integrity/structure checks.

    saved_df:
        The analysis-relevant output that will actually be written:
        LE+ES+EE for response/allfix, LE only for feedback.

    The audit is designed to make the TrialID issue easy to interpret:
    reused numeric raw_id labels are reported, while separate timestamps
    and ASC source lines demonstrate whether they are distinct raw events.
    """
    phase_counts = (
        full_df["Phase"]
        .value_counts()
        .reindex(
            ["TRAIN", "LE", "ES", "EE", "UNKNOWN"],
            fill_value=0,
        )
    )

    raw_events = int(len(full_df))
    train_n = int(phase_counts["TRAIN"])
    le_n = int(phase_counts["LE"])
    es_n = int(phase_counts["ES"])
    ee_n = int(phase_counts["EE"])
    unknown_n = int(phase_counts["UNKNOWN"])
    main_n = le_n + es_n + ee_n

    metadata_mismatch = full_df[
        full_df["phase_metadata_mismatch"]
    ].copy()

    metadata_mismatch_n = int(
        len(metadata_mismatch)
    )

    if metadata_mismatch_n:
        metadata_mismatch_examples = ";".join(
            (
                str(r.Phase)
                + str(int(r.phase_trial))
                + ":raw="
                + str(r.Phase_metadata)
                + "/"
                + str(r.Task_raw)
            )
            for r in metadata_mismatch.head(10).itertuples()
        )
    else:
        metadata_mismatch_examples = ""

    # -------------------------
    # Raw-event preservation
    # -------------------------
    one_row_per_raw_event = (
        int(trialid_events_found) == raw_events
    )

    raw_seq_ok = np.array_equal(
        full_df["trial_seq"].to_numpy(),
        np.arange(1, raw_events + 1),
    )

    ts_unique = bool(full_df["start_ts"].is_unique)
    ts_order_ok = bool(
        full_df["start_ts"].is_monotonic_increasing
    )

    line_unique = bool(full_df["asc_line"].is_unique)
    line_order_ok = bool(
        full_df["asc_line"].is_monotonic_increasing
    )

    raw_event_ok = all([
        one_row_per_raw_event,
        raw_seq_ok,
        ts_unique,
        ts_order_ok,
        line_unique,
        line_order_ok,
    ])

    # -------------------------
    # Raw TrialID reuse
    # -------------------------
    raw_counts = full_df["raw_id"].value_counts()
    reused = raw_counts[raw_counts > 1]

    reused_ids_n = int(len(reused))
    reused_events_n = int(reused.sum())

    if reused_ids_n:
        reused_rows = full_df[
            full_df["raw_id"].isin(reused.index)
        ]

        reuse_distinct_ok = all(
            g["start_ts"].is_unique
            and g["asc_line"].is_unique
            for _, g in reused_rows.groupby("raw_id")
        )

        reuse_examples = ";".join(
            str(int(x))
            for x in list(reused.index[:5])
        )
    else:
        reuse_distinct_ok = True
        reuse_examples = ""

    # -------------------------
    # Experimental structure
    # -------------------------
    training_count_ok = (
        train_n == EXPECTED_TRAIN
    )

    main_counts_ok = (
        le_n == EXPECTED_MAIN["LE"]
        and
        es_n == EXPECTED_MAIN["ES"]
        and
        ee_n == EXPECTED_MAIN["EE"]
    )

    total_raw_count_ok = (
        raw_events == EXPECTED_TOTAL_EYE_TRIALS
    )

    unknown_ok = (
        unknown_n == 0
    )

    main_full = full_df[
        full_df["Phase"].isin(["LE", "ES", "EE"])
    ].copy()

    expected_phase_sequence = (
        ["LE"] * EXPECTED_MAIN["LE"]
        + ["ES"] * EXPECTED_MAIN["ES"]
        + ["EE"] * EXPECTED_MAIN["EE"]
    )

    main_phase_order_ok = (
        main_full["Phase"].tolist()
        == expected_phase_sequence
    )

    main_key_unique = not main_full.duplicated(
        ["Phase", "phase_trial"]
    ).any()

    phase_assignment_ok = bool(
        (
            main_full["phase_assignment"]
            == "programmed_chronology"
        ).all()
    )

    structure_ok = all([
        training_count_ok,
        main_counts_ok,
        total_raw_count_ok,
        unknown_ok,
        main_phase_order_ok,
        main_key_unique,
        phase_assignment_ok,
    ])

    # -------------------------
    # Saved analysis output
    # -------------------------
    if label == "feedback":
        expected_saved = EXPECTED_MAIN["LE"]
        expected_saved_phases = {"LE"}
    else:
        expected_saved = sum(EXPECTED_MAIN.values())
        expected_saved_phases = {"LE", "ES", "EE"}

    saved_rows_n = int(len(saved_df))

    saved_count_ok = (
        saved_rows_n == expected_saved
    )

    saved_phases_ok = (
        set(saved_df["Phase"].dropna().unique())
        <= expected_saved_phases
    )

    saved_key_unique = not saved_df.duplicated(
        ["pid", "Phase", "phase_trial"]
    ).any()

    saved_output_ok = all([
        saved_count_ok,
        saved_phases_ok,
        saved_key_unique,
    ])

    # -------------------------
    # Overall status
    # -------------------------
    if (
        raw_event_ok
        and reuse_distinct_ok
        and structure_ok
        and saved_output_ok
    ):
        status = "PASS"

    elif (
        pid in KNOWN_INCOMPLETE
        and raw_event_ok
        and reuse_distinct_ok
        and main_key_unique
        and saved_key_unique
    ):
        status = "KNOWN_INCOMPLETE"

    else:
        status = "CHECK"

    # Reader-friendly interpretation.
    if reused_ids_n > 0:
        id_text = (
            f"{reused_ids_n} numeric raw TrialID label(s) were reused "
            f"across {reused_events_n} raw events. The repeated labels "
            f"occur at distinct timestamps/source lines={reuse_distinct_ok}; "
            "raw_id is retained only as provenance and is never used as "
            "the merge key."
        )
    else:
        id_text = (
            "No reused numeric raw TrialID labels were detected."
        )

    if metadata_mismatch_n > 0:
        metadata_text = (
            f" Raw Training/Task metadata disagree with the validated "
            f"programmed phase assignment on {metadata_mismatch_n} "
            f"main-trial boundary event(s): {metadata_mismatch_examples}. "
            f"The raw metadata are retained as provenance and are not "
            f"used as the merge key."
        )
    else:
        metadata_text = (
            " Raw Training/Task metadata agree with the programmed "
            "phase assignment on all explicitly labelled main trials."
        )

    if label == "feedback":
        save_text = (
            f"{saved_rows_n} LE feedback rows saved; ES/EE have no "
            "analysis feedback window and are intentionally not written "
            "to the feedback CSV."
        )
    else:
        save_text = (
            f"{saved_rows_n} main-task rows saved "
            f"(LE={le_n}, ES={es_n}, EE={ee_n}); "
            f"{train_n} training rows were used only for audit and "
            "excluded from the saved participant CSV."
        )

    if status == "PASS":
        summary = (
            "PASS: raw ASC trial events were preserved one-to-one; "
            + save_text
            + " "
            + id_text
            + metadata_text
            + " No evidence of row loss/collapse from raw TrialID reuse "
              "was detected by these checks."
        )
    elif status == "KNOWN_INCOMPLETE":
        summary = (
            "KNOWN INCOMPLETE PARTICIPANT: raw-event provenance checks "
            "passed, but expected study counts are not required to match "
            "for this participant. "
            + id_text
        )
    else:
        summary = (
            "CHECK REQUIRED: at least one raw-event, phase-structure, "
            "or saved-output check failed. "
            + id_text
        )

    print(
        f"PID {pid:>3} | {label:<8} | {status:<16} | "
        f"raw={raw_events} train={train_n} "
        f"LE={le_n} ES={es_n} EE={ee_n} "
        f"saved={saved_rows_n} reused_ids={reused_ids_n}"
    )

    return {
        "audit_version": AUDIT_VERSION,
        "pid": pid,
        "mode": label,
        "status": status,

        "raw_events": raw_events,
        "training_ignored": train_n,
        "main_detected": main_n,
        "saved_rows": saved_rows_n,

        "LE_n": le_n,
        "ES_n": es_n,
        "EE_n": ee_n,
        "UNKNOWN_n": unknown_n,

        "raw_event_one_to_one": bool(one_row_per_raw_event),
        "raw_sequence_ok": bool(raw_seq_ok),
        "timestamp_unique": bool(ts_unique),
        "timestamp_order_ok": bool(ts_order_ok),
        "source_line_unique": bool(line_unique),
        "source_line_order_ok": bool(line_order_ok),

        "raw_id_reused_n": reused_ids_n,
        "raw_id_reused_events_n": reused_events_n,
        "raw_id_reuse_distinct_events": bool(reuse_distinct_ok),
        "raw_id_examples": reuse_examples,

        "training_count_ok": bool(training_count_ok),
        "main_counts_ok": bool(main_counts_ok),
        "raw_total_count_ok": bool(total_raw_count_ok),
        "main_phase_order_ok": bool(main_phase_order_ok),
        "main_key_unique": bool(main_key_unique),
        "phase_assignment_ok": bool(phase_assignment_ok),

        "metadata_phase_mismatch_n": metadata_mismatch_n,
        "metadata_phase_mismatch_examples": metadata_mismatch_examples,

        "saved_count_ok": bool(saved_count_ok),
        "saved_phases_ok": bool(saved_phases_ok),
        "saved_key_unique": bool(saved_key_unique),

        "summary": summary,
    }


# ============================================================
# FIND RAW ASC FILES
# ============================================================

asc_files = []

for root, dirs, files in os.walk(SOURCE_ROOT):
    for filename in files:
        if filename.lower().endswith(".asc"):

            path = os.path.join(
                root,
                filename,
            )

            pid = extract_pid(
                path
            )

            # Study 2 exclusions + ignore IDs above 27
            if pid in EXCLUDE or pid > 27:
                continue

            if (
                ONLY_PID is not None
                and pid != ONLY_PID
            ):
                continue

            asc_files.append(
                (pid, path)
            )

asc_files.sort(
    key=lambda x: (x[0], x[1])
)

if not asc_files:
    raise FileNotFoundError(
        f"No .asc files found under {SOURCE_ROOT}"
    )


# Do not silently choose one if multiple ASC files exist for a participant.
pid_counts = Counter(
    pid for pid, _ in asc_files
)

multiple = [
    pid
    for pid, n in pid_counts.items()
    if n > 1
]

if multiple:
    raise ValueError(
        "More than one .asc file found for participant(s): "
        + ", ".join(map(str, sorted(multiple)))
        + ". Resolve this explicitly before parsing."
    )


# ============================================================
# PROCESS PARTICIPANTS
# ============================================================

audit_rows = []

for pid, file_path in asc_files:

    print("\n" + "#" * 76)
    print(f"PARTICIPANT {pid}")
    print(file_path)
    print("#" * 76)

    participant_output_dir = os.path.join(
        OUTPUT_ROOT,
        f"Participant_{pid}",
    )

    if not os.path.isdir(
        participant_output_dir
    ):
        raise FileNotFoundError(
            "Output folder does not exist: "
            + participant_output_dir
        )

    rows = []

    current = None
    fixations = []

    fixation_start_time = None
    response_time = None

    feedback_start_time = None
    feedback_end_time = None
    feedback_fixations = []

    trial_seq = 0
    trialid_events_found = 0

    # --------------------------------------------------------
    # Helper used when an open trial block is closed.
    # Returns the finalized record; it does not mutate parser state.
    # --------------------------------------------------------

    def finalize_record(record, trial_fixations):
        if record is None:
            return None

        if mode == 1:
            add_response_metrics(
                record,
                trial_fixations,
            )

        elif mode == 2:
            if "DwellTotal_Feed" not in record:
                add_feedback_metrics(
                    record,
                    [],
                )

        elif mode == 3:
            add_allfix_metrics(
                record,
                trial_fixations,
            )

        return record

    # --------------------------------------------------------
    # Parse ASC chronologically
    # --------------------------------------------------------

    with open(
        file_path,
        "r",
        errors="replace",
    ) as asc:

        for line_no, line in enumerate(
            asc,
            start=1,
        ):

            # =================================================
            # NEW TRIAL
            # =================================================

            trial_match = re.search(
                r"MSG\s+(\d+)\s+TRIALID\s+(\d+)",
                line,
            )

            if trial_match:

                if current is not None:
                    rows.append(
                        finalize_record(
                            current,
                            fixations,
                        )
                    )
                    current = None

                trial_seq += 1
                trialid_events_found += 1

                current = {
                    "pid": pid,
                    "raw_id": int(
                        trial_match.group(2)
                    ),
                    "trial_seq": trial_seq,
                    "train": np.nan,
                    "Task": pd.NA,
                    "Messages": pd.NA,
                    "start_ts": int(
                        trial_match.group(1)
                    ),
                    "asc_line": int(
                        line_no
                    ),
                }

                fixations = []
                fixation_start_time = None
                response_time = None

                feedback_start_time = None
                feedback_end_time = None
                feedback_fixations = []

                continue

            # =================================================
            # TRIAL METADATA
            #
            # These messages belong to the CURRENT trial block.
            # They are NOT carried into the next trial.
            # =================================================

            msg_match = re.search(
                r"MSG\s+\d+\s+!V TRIAL_VAR\s+(.*)",
                line,
            )

            if (
                msg_match
                and current is not None
            ):

                msg = (
                    msg_match.group(1)
                    .strip()
                )

                training_match = re.search(
                    r"\bTraining\s+(-?\d+)\b",
                    msg,
                    flags=re.I,
                )

                if training_match:
                    current["train"] = int(
                        training_match.group(1)
                    )

                task_match = re.search(
                    r"\bTask\s+([123])\b",
                    msg,
                    flags=re.I,
                )

                if task_match:
                    current["Task"] = (
                        f"Task {task_match.group(1)}"
                    )

                # Keep a compact trace of the meaningful metadata.
                if (
                    training_match
                    or task_match
                ):
                    existing = current.get(
                        "Messages",
                        pd.NA,
                    )

                    if pd.isna(existing):
                        current["Messages"] = msg
                    else:
                        current["Messages"] = (
                            str(existing)
                            + " | "
                            + msg
                        )

                continue

            # =================================================
            # RESPONSE — mode 1
            # =================================================

            if mode == 1:

                response_match = re.search(
                    r"MSG\s+(\d+)\s+"
                    r"(ResponseLeft|ResponseRight)",
                    line,
                )

                if response_match:
                    response_time = int(
                        response_match.group(1)
                    )
                    continue

            # =================================================
            # FEEDBACK — mode 2
            # =================================================

            if mode == 2:

                feedback_on_match = re.search(
                    r"MSG\s+(\d+)\s+FeedbackOn",
                    line,
                )

                if feedback_on_match:
                    feedback_start_time = int(
                        feedback_on_match.group(1)
                    )
                    feedback_end_time = None
                    feedback_fixations = []
                    fixation_start_time = None
                    continue

                feedback_off_match = re.search(
                    r"MSG\s+(\d+)\s+FeedbackOff",
                    line,
                )

                if feedback_off_match:
                    feedback_end_time = int(
                        feedback_off_match.group(1)
                    )

                    if current is not None:

                        valid_feedback = [
                            f
                            for f in feedback_fixations
                            if (
                                feedback_start_time
                                is not None
                                and
                                f["start_time"]
                                >= feedback_start_time
                                and
                                f["end_time"]
                                <= feedback_end_time
                            )
                        ]

                        add_feedback_metrics(
                            current,
                            valid_feedback,
                        )

                    feedback_start_time = None
                    feedback_end_time = None
                    feedback_fixations = []
                    fixation_start_time = None

                    continue

            # =================================================
            # FIXATION START
            # =================================================

            sfix_match = re.search(
                r"SFIX\s+\w+\s+(\d+)",
                line,
            )

            if sfix_match:

                if (
                    mode == 2
                    and feedback_start_time is None
                ):
                    continue

                fixation_start_time = int(
                    sfix_match.group(1)
                )

                continue

            # =================================================
            # FIXATION END
            # =================================================

            efix_match = re.search(
                r"EFIX\s+\w+\s+"
                r"(\d+)\s+(\d+)\s+(\d+)\s+"
                r"(\d+\.\d+)\s+(\d+\.\d+)",
                line,
            )

            if (
                efix_match
                and current is not None
                and fixation_start_time is not None
            ):

                fixation_end_time = int(
                    efix_match.group(2)
                )

                fixation_duration = int(
                    efix_match.group(3)
                )

                avg_x = float(
                    efix_match.group(4)
                )

                avg_y = float(
                    efix_match.group(5)
                )

                roi = roi_from_x(
                    avg_x
                )

                if roi is not None:

                    fix = {
                        "start_time": fixation_start_time,
                        "end_time": fixation_end_time,
                        "duration": fixation_duration,
                        "roi": roi,
                        "avg_x": avg_x,
                    }

                    if mode == 2:
                        fix["avg_y"] = avg_y

                    if mode == 1:
                        # Same response-window rule as before:
                        # include fixations beginning before/at response.
                        if (
                            response_time is None
                            or
                            fixation_start_time <= response_time
                        ):
                            fixations.append(
                                fix
                            )

                    elif mode == 2:
                        feedback_fixations.append(
                            fix
                        )

                    elif mode == 3:
                        fixations.append(
                            fix
                        )

                fixation_start_time = None

    # Final open trial at EOF.
    if current is not None:
        rows.append(
            finalize_record(
                current,
                fixations,
            )
        )
        current = None

    # ========================================================
    # DATAFRAME + PHASE CLASSIFICATION
    # ========================================================

    trial_df = pd.DataFrame(
        rows
    )

    if len(trial_df) == 0:
        raise ValueError(
            f"participant {pid}: no trials parsed"
        )

    if trial_df["trial_seq"].duplicated().any():
        raise ValueError(
            f"participant {pid}: duplicate trial_seq"
        )

    trial_df = classify_phases(
        trial_df,
        pid,
    )

    # Only main-task rows need a unique Phase + phase_trial key.
    # Training rows intentionally have phase_trial = NaN and are excluded
    # from the saved analysis files, so they must not be included in this
    # duplicate-key check.
    main_key_check = trial_df[
        trial_df["Phase"].isin(["LE", "ES", "EE"])
    ]

    if main_key_check.duplicated(
        ["Phase", "phase_trial"]
    ).any():
        raise ValueError(
            f"participant {pid}: duplicate Phase + phase_trial "
            "within main experimental trials"
        )

    # ========================================================
    # KEEP ONLY ANALYSIS-RELEVANT MAIN TASKS
    # ========================================================

    main_df = trial_df[
        trial_df["Phase"].isin(["LE", "ES", "EE"])
    ].copy()

    # Response/allfix: retain all three real experimental tasks.
    # Feedback: only LE actually contains the outcome-feedback window
    # used by this preprocessing mode.
    if mode == 2:
        save_df = main_df[
            main_df["Phase"] == "LE"
        ].copy()
    else:
        save_df = main_df.copy()

    # Training flag and compact raw-message string are not needed in the
    # analysis CSV. Raw task/phase metadata ARE retained explicitly for
    # provenance, including the known P1-P3 ES/EE boundary discrepancy.
    save_df = save_df.drop(
        columns=["train", "Messages", "Task"],
        errors="ignore",
    )

    # Defensive checks on the actual file we are about to save.
    if save_df.duplicated(
        ["pid", "Phase", "phase_trial"]
    ).any():
        raise ValueError(
            f"participant {pid}: duplicate pid + Phase + phase_trial "
            "in analysis output"
        )

    # ========================================================
    # AUDIT
    # ========================================================

    label = {
        1: "response",
        2: "feedback",
        3: "allfix",
    }[mode]

    audit_row = build_audit(
        full_df=trial_df,
        saved_df=save_df,
        pid=pid,
        label=label,
        trialid_events_found=trialid_events_found,
    )

    audit_rows.append(
        audit_row
    )

    # ========================================================
    # SAVE PARTICIPANT MAIN-TASK CSV
    # ========================================================

    suffix = {
        1: "eye_response_phase",
        2: "eye_feedback_phase",
        3: "eye_allfix_phase",
    }[mode]

    output_file = os.path.join(
        participant_output_dir,
        f"Participant_{pid}_{suffix}.csv",
    )

    save_df.to_csv(
        output_file,
        index=False,
    )

    if audit_row["status"] == "CHECK":
        print("  -> CHECK audit before using this participant.")


# ============================================================
# SAVE / UPDATE ONE COMBINED AUDIT FILE
# ============================================================

new_audit_df = pd.DataFrame(
    audit_rows
)

audit_file = os.path.join(
    OUTPUT_ROOT,
    AUDIT_FILENAME,
)

# The same concise audit file is maintained across mode 1/2/3 runs.
# Re-running a mode replaces only the matching participant+mode rows,
# so the file stays current instead of accumulating duplicate audits.
if os.path.exists(audit_file):
    old_audit_df = pd.read_csv(
        audit_file
    )

    if {
        "pid",
        "mode",
    }.issubset(old_audit_df.columns):

        current_keys = set(
            zip(
                new_audit_df["pid"].astype(int),
                new_audit_df["mode"].astype(str),
            )
        )

        keep_mask = [
            (int(pid), str(mode_name))
            not in current_keys
            for pid, mode_name in zip(
                old_audit_df["pid"],
                old_audit_df["mode"],
            )
        ]

        old_audit_df = old_audit_df.loc[
            keep_mask
        ].copy()

        audit_df = pd.concat(
            [
                old_audit_df,
                new_audit_df,
            ],
            ignore_index=True,
        )

    else:
        # Old audit with an incompatible structure: replace it rather
        # than silently mixing different audit definitions.
        audit_df = new_audit_df.copy()

else:
    audit_df = new_audit_df.copy()

audit_df = audit_df.sort_values(
    ["pid", "mode"]
).reset_index(
    drop=True
)

audit_df.to_csv(
    audit_file,
    index=False,
)

print("\n" + "=" * 76)
print("ONE COMBINED AUDIT FILE:")
print(audit_file)

print("\nSTATUS COUNTS:")
print(
    audit_df["status"]
    .value_counts(dropna=False)
    .to_string()
)

check_rows = audit_df[
    audit_df["status"] == "CHECK"
]

if len(check_rows):
    print("\nROWS REQUIRING ATTENTION:")
    print(
        check_rows[
            [
                "pid",
                "mode",
                "raw_events",
                "training_ignored",
                "LE_n",
                "ES_n",
                "EE_n",
                "UNKNOWN_n",
                "saved_rows",
                "raw_id_reused_n",
            ]
        ].to_string(index=False)
    )
else:
    print("\nNo unexpected CHECK rows in the current audit.")

print("=" * 76)
