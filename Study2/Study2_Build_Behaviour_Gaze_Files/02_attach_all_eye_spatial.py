from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# 02 — FINAL VALIDATED BEHAVIOUR + EYE MERGE — STUDY 1
# ============================================================
#
# This script is designed for the eye files produced by:
#
#     study1_eye_asc.py
#
# and validated by:
#
#     study1_parsing_safety_audit.csv
#
# CRITICAL MERGE KEY
# ------------------
# Behaviour:
#     sub_id + phase + trial
#
# Eye:
#     pid + Phase + phase_trial
#
# raw_id / TrialID is NEVER used as a merge key.
#
# Behaviour and gaze remain in PHYSICAL LEFT / RIGHT coordinates.
# NO E/S recoding happens here.
#
# Training is not merged.
# Response/allfix merge to LE + ES + EE.
# Feedback merges to LE only.
#
# Raw EyeLink metadata are retained as provenance.
# ============================================================


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(
    "D:/Birmingham_Uni_Oct25/Eye_Tracking_Project/Study2"
)

DATA_DIR = PROJECT_ROOT / "data_sets"
EYE_ROOT = PROJECT_ROOT / "Study2_Data"

BEHAVIOUR_FILE = (
    DATA_DIR / "Study2_Behaviour.csv"
)

FINAL_PARSER_AUDIT_FILE = (
    DATA_DIR / "Study2_FINAL_EYE_PARSING_SAFETY_AUDIT.csv"
)

OUTPUT_FILE = (
    DATA_DIR / "Study2_Behaviour_with_Gaze.csv"
)

MERGE_AUDIT_FILE = (
    DATA_DIR / "Study2_Behaviour_with_Gaze_MERGE_AUDIT.csv"
)

# ============================================================
# EXPECTED DESIGN
# ============================================================

EXPECTED_MAIN = {
    "LE": 160,
    "ES": 360,
    "EE": 140,
}

MAIN_PHASES = [
    "LE",
    "ES",
    "EE",
]


# ============================================================
# LOAD + VALIDATE BEHAVIOUR MASTER
# ============================================================

beh = pd.read_csv(
    BEHAVIOUR_FILE
)

master = beh.copy()

required_beh = {
    "sub_id",
    "phase",
    "trial",
}

missing = required_beh - set(
    master.columns
)

if missing:
    raise KeyError(
        "Behavioural master missing columns: "
        + str(sorted(missing))
    )

master["sub_id"] = pd.to_numeric(
    master["sub_id"],
    errors="raise",
).astype(int)

master["phase"] = (
    master["phase"]
    .astype(str)
    .str.strip()
)

# trial is required to be numeric for all main-task rows.
main_beh_mask = master["phase"].isin(
    MAIN_PHASES
)

master.loc[
    main_beh_mask,
    "trial",
] = pd.to_numeric(
    master.loc[
        main_beh_mask,
        "trial",
    ],
    errors="raise",
)

# Exact behavioural merge key must be unique for main tasks.
main_beh = master[
    main_beh_mask
].copy()

if main_beh.duplicated(
    ["sub_id", "phase", "trial"]
).any():

    bad = main_beh.loc[
        main_beh.duplicated(
            ["sub_id", "phase", "trial"],
            keep=False,
        ),
        ["sub_id", "phase", "trial"],
    ]

    raise ValueError(
        "Duplicate behavioural merge keys found:\n"
        + bad.head(30).to_string(index=False)
    )

# For each participant/phase that exists, trial must be exactly 1..N.
for (
    pid,
    phase,
), g in main_beh.groupby(
    ["sub_id", "phase"],
    sort=True,
):

    observed = (
        pd.to_numeric(
            g["trial"],
            errors="raise",
        )
        .astype(int)
        .sort_values()
        .to_numpy()
    )

    expected = np.arange(
        1,
        len(g) + 1,
    )

    if not np.array_equal(
        observed,
        expected,
    ):
        raise ValueError(
            f"Behaviour participant {pid}, phase {phase}: "
            "trial is not exactly 1..N."
        )


# ============================================================
# LOAD FINAL PARSER SAFETY AUDIT
# ============================================================

if not FINAL_PARSER_AUDIT_FILE.exists():
    raise FileNotFoundError(
        "Final parser safety audit not found:\n"
        + str(FINAL_PARSER_AUDIT_FILE)
    )

parser_audit = pd.read_csv(
    FINAL_PARSER_AUDIT_FILE
)

required_audit = {
    "pid",
    "status",
    "raw_events",
    "main_events",
    "behaviour_keys_match",
    "raw_response_choice_mismatch_n",
    "response_structure_ok",
    "response_count_ok",
    "response_matches_raw",
    "feedback_structure_ok",
    "feedback_count_ok",
    "feedback_matches_raw",
    "allfix_structure_ok",
    "allfix_count_ok",
    "allfix_matches_raw",
    "response_allfix_provenance_ok",
    "response_feedback_provenance_ok",
}

missing = required_audit - set(
    parser_audit.columns
)

if missing:
    raise KeyError(
        "Final parser audit missing columns: "
        + str(sorted(missing))
    )

parser_audit["pid"] = pd.to_numeric(
    parser_audit["pid"],
    errors="raise",
).astype(int)

# Never merge data from a participant whose FINAL parser audit failed.
bad_parser = parser_audit[
    parser_audit["status"] != "PASS"
]

if len(
    bad_parser
):
    raise ValueError(
        "Final parser audit contains non-PASS participant(s):\n"
        + bad_parser[
            ["pid", "status"]
        ].to_string(index=False)
    )

validated_eye_pids = set(
    parser_audit["pid"].astype(int)
)


# ============================================================
# HELPERS
# ============================================================

def validated_pid(pid):
    return int(pid) in validated_eye_pids


def prepare_eye_file(
    path,
    pid,
    label,
    allowed_phases,
):
    """
    Read and validate one FINAL parser output.

    The eye merge key is:
        pid + Phase + phase_trial

    raw_id is provenance only.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"{label}, participant {pid}: missing file:\n{path}"
        )

    df = pd.read_csv(
        path
    )

    required = {
        "pid",
        "raw_id",
        "trial_seq",
        "phase_trial",
        "main_trial_seq",
        "start_ts",
        "asc_line",
        "Task_raw",
        "Phase_metadata",
        "Phase",
        "phase_assignment",
        "phase_metadata_mismatch",
    }

    missing = required - set(
        df.columns
    )

    if missing:
        raise KeyError(
            f"{label}, participant {pid}: "
            f"missing FINAL parser columns {sorted(missing)}"
        )

    out = df.copy()

    # --------------------------------------------------------
    # Numeric provenance
    # --------------------------------------------------------

    numeric_cols = [
        "pid",
        "raw_id",
        "trial_seq",
        "phase_trial",
        "main_trial_seq",
        "start_ts",
        "asc_line",
    ]

    for col in numeric_cols:

        out[col] = pd.to_numeric(
            out[col],
            errors="coerce",
        )

        if out[col].isna().any():
            raise ValueError(
                f"{label}, participant {pid}: "
                f"missing/non-numeric {col}"
            )

    out["pid"] = out[
        "pid"
    ].astype(int)

    out["phase_trial"] = out[
        "phase_trial"
    ].astype(int)

    out["main_trial_seq"] = out[
        "main_trial_seq"
    ].astype(int)

    # --------------------------------------------------------
    # Participant identity
    # --------------------------------------------------------

    pids_inside = set(
        out["pid"].unique()
    )

    if pids_inside != {
        int(pid)
    }:
        raise ValueError(
            f"{label}, participant {pid}: "
            f"file contains pid(s) {sorted(pids_inside)}"
        )

    # --------------------------------------------------------
    # Final validated phase identity
    # --------------------------------------------------------

    phases_inside = set(
        out["Phase"]
        .dropna()
        .astype(str)
        .unique()
    )

    if not phases_inside.issubset(
        set(allowed_phases)
    ):
        raise ValueError(
            f"{label}, participant {pid}: "
            f"unexpected Phase values {sorted(phases_inside)}"
        )

    # The final parser must explicitly state that phase was assigned
    # from the validated programmed chronology.
    assignments = set(
        out["phase_assignment"]
        .dropna()
        .astype(str)
        .unique()
    )

    if assignments != {
        "programmed_chronology"
    }:
        raise ValueError(
            f"{label}, participant {pid}: "
            f"unexpected phase_assignment {sorted(assignments)}"
        )

    # --------------------------------------------------------
    # Key uniqueness
    # --------------------------------------------------------

    if out.duplicated(
        ["pid", "Phase", "phase_trial"]
    ).any():

        bad = out.loc[
            out.duplicated(
                ["pid", "Phase", "phase_trial"],
                keep=False,
            ),
            [
                "pid",
                "Phase",
                "phase_trial",
                "raw_id",
                "trial_seq",
                "start_ts",
                "asc_line",
            ],
        ]

        raise ValueError(
            f"{label}, participant {pid}: "
            "duplicate eye merge key:\n"
            + bad.head(30).to_string(index=False)
        )

    # --------------------------------------------------------
    # Exact within-phase sequence
    # --------------------------------------------------------

    for phase in sorted(
        phases_inside
    ):

        p = (
            out[
                out["Phase"] == phase
            ]
            .sort_values(
                "phase_trial"
            )
        )

        observed = p[
            "phase_trial"
        ].to_numpy()

        expected = np.arange(
            1,
            len(p) + 1,
        )

        if not np.array_equal(
            observed,
            expected,
        ):
            raise ValueError(
                f"{label}, participant {pid}, phase {phase}: "
                "phase_trial is not exactly 1..N."
            )

    return out


def exact_key_match(
    beh_df,
    eye_df,
    pid,
    phase,
):
    """
    Compare the behavioural and eye trial numbers for one phase.
    """

    b = (
        beh_df[
            (beh_df["sub_id"] == int(pid))
            & (beh_df["phase"] == phase)
        ]
        .copy()
        .sort_values(
            "trial"
        )
    )

    e = (
        eye_df[
            eye_df["Phase"] == phase
        ]
        .copy()
        .sort_values(
            "phase_trial"
        )
    )

    beh_trials = (
        pd.to_numeric(
            b["trial"],
            errors="raise",
        )
        .astype(int)
        .to_numpy()
    )

    eye_trials = (
        e["phase_trial"]
        .astype(int)
        .to_numpy()
    )

    return (
        len(b) > 0
        and len(b) == len(e)
        and np.array_equal(
            beh_trials,
            eye_trials,
        )
    )


def rename_eye_for_merge(
    df,
    mode,
):
    """
    Rename only provenance/key columns that would otherwise collide.

    Response gaze metrics keep their familiar unsuffixed names.
    Feedback metrics already carry *_Feed.
    Allfix metrics already carry *_allfix.
    """

    out = df.copy()

    # --------------------------------------------------------
    # Merge keys
    # --------------------------------------------------------

    out = out.rename(
        columns={
            "pid": "sub_id",
            "Phase": "phase",
            "phase_trial": "trial",
        }
    )

    # --------------------------------------------------------
    # Provenance columns
    # --------------------------------------------------------

    if mode == "response":

        provenance_map = {
            "raw_id": "eye_raw_id",
            "trial_seq": "eye_trial_seq",
            "main_trial_seq": "eye_main_trial_seq",
            "start_ts": "eye_start_ts",
            "asc_line": "eye_asc_line",
            "Task_raw": "eye_Task_raw",
            "Phase_metadata": "eye_Phase_metadata",
            "phase_assignment": "eye_phase_assignment",
            "phase_metadata_mismatch":
                "eye_phase_metadata_mismatch",
        }

    elif mode == "feedback":

        provenance_map = {
            "raw_id": "eye_raw_id_Feed",
            "trial_seq": "eye_trial_seq_Feed",
            "main_trial_seq": "eye_main_trial_seq_Feed",
            "start_ts": "eye_start_ts_Feed",
            "asc_line": "eye_asc_line_Feed",
            "Task_raw": "eye_Task_raw_Feed",
            "Phase_metadata": "eye_Phase_metadata_Feed",
            "phase_assignment": "eye_phase_assignment_Feed",
            "phase_metadata_mismatch":
                "eye_phase_metadata_mismatch_Feed",
        }

    elif mode == "allfix":

        provenance_map = {
            "raw_id": "eye_raw_id_allfix",
            "trial_seq": "eye_trial_seq_allfix",
            "main_trial_seq": "eye_main_trial_seq_allfix",
            "start_ts": "eye_start_ts_allfix",
            "asc_line": "eye_asc_line_allfix",
            "Task_raw": "eye_Task_raw_allfix",
            "Phase_metadata": "eye_Phase_metadata_allfix",
            "phase_assignment": "eye_phase_assignment_allfix",
            "phase_metadata_mismatch":
                "eye_phase_metadata_mismatch_allfix",
        }

    else:
        raise ValueError(
            f"Unknown mode: {mode}"
        )

    out = out.rename(
        columns=provenance_map
    )

    return out


def safe_left_merge(
    left,
    right,
    indicator_name,
):
    """
    Exact left merge with row-count and key safeguards.
    """

    n_before = len(
        left
    )

    merged = left.merge(
        right,
        on=[
            "sub_id",
            "phase",
            "trial",
        ],
        how="left",
        validate="one_to_one",
        indicator=indicator_name,
        sort=False,
    )

    if len(
        merged
    ) != n_before:
        raise ValueError(
            f"{indicator_name}: merge changed master row count "
            f"from {n_before} to {len(merged)}"
        )

    return merged


# ============================================================
# READ + VALIDATE ALL EYE FILES
# ============================================================

response_parts = []
feedback_parts = []
allfix_parts = []

merge_audit_rows = []

behaviour_pids = sorted(
    master["sub_id"]
    .dropna()
    .astype(int)
    .unique()
)

for pid in behaviour_pids:

    # --------------------------------------------------------
    # Participant exists in final validated eye audit?
    # --------------------------------------------------------

    if not validated_pid(
        pid
    ):

        merge_audit_rows.append({
            "sub_id": pid,
            "parser_validated": False,
            "eye_files_loaded": False,
            "response_LE_key_ok": False,
            "response_ES_key_ok": False,
            "response_EE_key_ok": False,
            "feedback_LE_key_ok": False,
            "allfix_LE_key_ok": False,
            "allfix_ES_key_ok": False,
            "allfix_EE_key_ok": False,
            "status": "NO_VALIDATED_EYE_DATA",
            "note": (
                "Participant is present in behaviour but absent from "
                "the final PASS parser audit; gaze is left missing."
            ),
        })

        continue

    pdir = (
        EYE_ROOT
        / f"Participant_{pid}"
    )

    response_file = (
        pdir
        / f"Participant_{pid}_eye_response_phase.csv"
    )

    feedback_file = (
        pdir
        / f"Participant_{pid}_eye_feedback_phase.csv"
    )

    allfix_file = (
        pdir
        / f"Participant_{pid}_eye_allfix_phase.csv"
    )

    response = prepare_eye_file(
        response_file,
        pid,
        "response",
        {"LE", "ES", "EE"},
    )

    feedback = prepare_eye_file(
        feedback_file,
        pid,
        "feedback",
        {"LE"},
    )

    allfix = prepare_eye_file(
        allfix_file,
        pid,
        "allfix",
        {"LE", "ES", "EE"},
    )

    # --------------------------------------------------------
    # Exact behaviour <-> eye key checks
    # --------------------------------------------------------

    r_le = exact_key_match(
        master,
        response,
        pid,
        "LE",
    )

    r_es = exact_key_match(
        master,
        response,
        pid,
        "ES",
    )

    r_ee = exact_key_match(
        master,
        response,
        pid,
        "EE",
    )

    f_le = exact_key_match(
        master,
        feedback,
        pid,
        "LE",
    )

    a_le = exact_key_match(
        master,
        allfix,
        pid,
        "LE",
    )

    a_es = exact_key_match(
        master,
        allfix,
        pid,
        "ES",
    )

    a_ee = exact_key_match(
        master,
        allfix,
        pid,
        "EE",
    )

    all_keys_ok = all([
        r_le,
        r_es,
        r_ee,
        f_le,
        a_le,
        a_es,
        a_ee,
    ])

    if not all_keys_ok:
        raise ValueError(
            f"Participant {pid}: final parser PASS exists, "
            "but behavioural/eye phase_trial keys do not match exactly."
        )

    # --------------------------------------------------------
    # Exact expected eye counts
    # --------------------------------------------------------

    response_counts = (
        response[
            "Phase"
        ]
        .value_counts()
        .to_dict()
    )

    allfix_counts = (
        allfix[
            "Phase"
        ]
        .value_counts()
        .to_dict()
    )

    feedback_counts = (
        feedback[
            "Phase"
        ]
        .value_counts()
        .to_dict()
    )

    expected_response = {
        "LE": 160,
        "ES": 360,
        "EE": 140,
    }

    expected_feedback = {
        "LE": 160,
    }

    if response_counts != expected_response:
        raise ValueError(
            f"Participant {pid}: response counts "
            f"{response_counts} != {expected_response}"
        )

    if allfix_counts != expected_response:
        raise ValueError(
            f"Participant {pid}: allfix counts "
            f"{allfix_counts} != {expected_response}"
        )

    if feedback_counts != expected_feedback:
        raise ValueError(
            f"Participant {pid}: feedback counts "
            f"{feedback_counts} != {expected_feedback}"
        )

    response_parts.append(
        rename_eye_for_merge(
            response,
            "response",
        )
    )

    feedback_parts.append(
        rename_eye_for_merge(
            feedback,
            "feedback",
        )
    )

    allfix_parts.append(
        rename_eye_for_merge(
            allfix,
            "allfix",
        )
    )

    metadata_note_n = int(
        pd.to_numeric(
            response[
                "phase_metadata_mismatch"
            ],
            errors="coerce",
        )
        .fillna(0)
        .astype(bool)
        .sum()
    )

    merge_audit_rows.append({
        "sub_id": pid,
        "parser_validated": True,
        "eye_files_loaded": True,

        "response_LE_key_ok": r_le,
        "response_ES_key_ok": r_es,
        "response_EE_key_ok": r_ee,

        "feedback_LE_key_ok": f_le,

        "allfix_LE_key_ok": a_le,
        "allfix_ES_key_ok": a_es,
        "allfix_EE_key_ok": a_ee,

        "metadata_boundary_notes_n": metadata_note_n,

        "status": "READY_TO_MERGE",
        "note": (
            "Final parser audit PASS and all explicit behavioural/eye "
            "keys match."
        ),
    })


# ============================================================
# CONCATENATE VALIDATED EYE DATA
# ============================================================

if not response_parts:
    raise ValueError(
        "No validated response eye files were available to merge."
    )

response_all = pd.concat(
    response_parts,
    ignore_index=True,
)

feedback_all = pd.concat(
    feedback_parts,
    ignore_index=True,
)

allfix_all = pd.concat(
    allfix_parts,
    ignore_index=True,
)


# ============================================================
# DROP COLUMNS WE DO NOT WANT TO COPY INTO MASTER
# ============================================================

# The final files may contain parser helper columns not needed downstream.
# Keep the explicit key, provenance, and gaze variables.
#
# We intentionally do NOT keep duplicate parser-only columns such as
# a second copy of Phase after it has become the merge key.

def select_response_columns(df):

    wanted = [
        # key
        "sub_id",
        "phase",
        "trial",

        # provenance
        "eye_raw_id",
        "eye_trial_seq",
        "eye_main_trial_seq",
        "eye_start_ts",
        "eye_asc_line",
        "eye_Task_raw",
        "eye_Phase_metadata",
        "eye_phase_assignment",
        "eye_phase_metadata_mismatch",

        # gaze
        "DwellLeft",
        "DwellRight",
        "DwellTotal",
        "Nfix",
        "FirstFixLoc",
        "FirstFixDur",
        "FinalFixLoc",
        "FinalFixDur",
        "MiddleFixDur",
        "eachMiddleFixDur",
        "MiddleDominantLoc",
        "MiddleDominantDur",
        "GazeSwitch",
        "LeftFixNR",
        "RightFixNR",
        "GazeDiff",
        "last_roi",
        "Fixations",
    ]

    return df[
        [
            c
            for c in wanted
            if c in df.columns
        ]
    ].copy()


def select_feedback_columns(df):

    wanted = [
        # key
        "sub_id",
        "phase",
        "trial",

        # provenance
        "eye_raw_id_Feed",
        "eye_trial_seq_Feed",
        "eye_main_trial_seq_Feed",
        "eye_start_ts_Feed",
        "eye_asc_line_Feed",
        "eye_Task_raw_Feed",
        "eye_Phase_metadata_Feed",
        "eye_phase_assignment_Feed",
        "eye_phase_metadata_mismatch_Feed",

        # gaze
        "DwellLeft_Feed",
        "DwellRight_Feed",
        "DwellTotal_Feed",
        "Nfix_Feed",
        "FirstFixLoc_Feed",
        "FirstFixDur_Feed",
        "FinalFixLoc_Feed",
        "FinalFixDur_Feed",
        "MiddleFixDur_Feed",
        "eachMiddleFixDur_Feed",
        "GazeSwitch_Feed",
        "LeftFixNR_Feed",
        "RightFixNR_Feed",
        "GazeDiff_Feed",
        "last_roi_Feed",
        "Fixations_Feed",
    ]

    return df[
        [
            c
            for c in wanted
            if c in df.columns
        ]
    ].copy()


def select_allfix_columns(df):

    wanted = [
        # key
        "sub_id",
        "phase",
        "trial",

        # provenance
        "eye_raw_id_allfix",
        "eye_trial_seq_allfix",
        "eye_main_trial_seq_allfix",
        "eye_start_ts_allfix",
        "eye_asc_line_allfix",
        "eye_Task_raw_allfix",
        "eye_Phase_metadata_allfix",
        "eye_phase_assignment_allfix",
        "eye_phase_metadata_mismatch_allfix",

        # gaze
        "DwellLeft_allfix",
        "DwellRight_allfix",
        "DwellTotal_allfix",
        "Nfix_allfix",
        "FirstFixLoc_allfix",
        "FirstFixDur_allfix",
        "FinalFixLoc_allfix",
        "FinalFixDur_allfix",
        "MiddleFixDur_allfix",
        "eachMiddleFixDur_allfix",
        "GazeSwitch_allfix",
        "LeftFixNR_allfix",
        "RightFixNR_allfix",
        "GazeDiff_allfix",
        "last_roi_allfix",
        "Fixations_allfix",
    ]

    return df[
        [
            c
            for c in wanted
            if c in df.columns
        ]
    ].copy()


response_all = select_response_columns(
    response_all
)

feedback_all = select_feedback_columns(
    feedback_all
)

allfix_all = select_allfix_columns(
    allfix_all
)


# ============================================================
# FINAL GLOBAL KEY UNIQUENESS BEFORE MERGE
# ============================================================

for label, df in [
    ("response", response_all),
    ("feedback", feedback_all),
    ("allfix", allfix_all),
]:

    if df.duplicated(
        ["sub_id", "phase", "trial"]
    ).any():

        raise ValueError(
            f"{label}: duplicate global merge key "
            "sub_id + phase + trial"
        )


# ============================================================
# MERGE RESPONSE
# ============================================================

master = safe_left_merge(
    master,
    response_all,
    "_merge_response",
)

master[
    "response_eye_attached"
] = (
    master["_merge_response"]
    == "both"
)

master = master.drop(
    columns=[
        "_merge_response"
    ]
)


# ============================================================
# MERGE FEEDBACK
# ============================================================

master = safe_left_merge(
    master,
    feedback_all,
    "_merge_feedback",
)

master[
    "feedback_eye_attached"
] = (
    master["_merge_feedback"]
    == "both"
)

master = master.drop(
    columns=[
        "_merge_feedback"
    ]
)


# ============================================================
# MERGE ALLFIX
# ============================================================

master = safe_left_merge(
    master,
    allfix_all,
    "_merge_allfix",
)

master[
    "allfix_eye_attached"
] = (
    master["_merge_allfix"]
    == "both"
)

master = master.drop(
    columns=[
        "_merge_allfix"
    ]
)


# ============================================================
# POST-MERGE KEY CHECK
# ============================================================

merged_main = master[
    master["phase"].isin(
        MAIN_PHASES
    )
]

if merged_main.duplicated(
    ["sub_id", "phase", "trial"]
).any():
    raise ValueError(
        "Final merged master contains duplicate main-task keys."
    )


# ============================================================
# CROSS-WINDOW PROVENANCE CHECKS
# ============================================================

# ------------------------------------------------------------
# Response vs allfix — same raw trial event
# ------------------------------------------------------------

both = (
    master[
        "response_eye_attached"
    ]
    & master[
        "allfix_eye_attached"
    ]
)

if both.any():

    provenance_pairs = [
        (
            "eye_raw_id",
            "eye_raw_id_allfix",
        ),
        (
            "eye_trial_seq",
            "eye_trial_seq_allfix",
        ),
        (
            "eye_main_trial_seq",
            "eye_main_trial_seq_allfix",
        ),
        (
            "eye_start_ts",
            "eye_start_ts_allfix",
        ),
        (
            "eye_asc_line",
            "eye_asc_line_allfix",
        ),
    ]

    for (
        response_col,
        allfix_col,
    ) in provenance_pairs:

        a = pd.to_numeric(
            master.loc[
                both,
                response_col,
            ],
            errors="coerce",
        ).to_numpy()

        b = pd.to_numeric(
            master.loc[
                both,
                allfix_col,
            ],
            errors="coerce",
        ).to_numpy()

        if not np.array_equal(
            a,
            b,
        ):
            raise ValueError(
                "Response/allfix raw provenance mismatch: "
                f"{response_col} != {allfix_col}"
            )


# ------------------------------------------------------------
# Response vs feedback — LE same raw trial event
# ------------------------------------------------------------

both = (
    master[
        "response_eye_attached"
    ]
    & master[
        "feedback_eye_attached"
    ]
)

if both.any():

    if not (
        master.loc[
            both,
            "phase",
        ]
        == "LE"
    ).all():
        raise ValueError(
            "Feedback attached outside LE."
        )

    provenance_pairs = [
        (
            "eye_raw_id",
            "eye_raw_id_Feed",
        ),
        (
            "eye_trial_seq",
            "eye_trial_seq_Feed",
        ),
        (
            "eye_main_trial_seq",
            "eye_main_trial_seq_Feed",
        ),
        (
            "eye_start_ts",
            "eye_start_ts_Feed",
        ),
        (
            "eye_asc_line",
            "eye_asc_line_Feed",
        ),
    ]

    for (
        response_col,
        feed_col,
    ) in provenance_pairs:

        a = pd.to_numeric(
            master.loc[
                both,
                response_col,
            ],
            errors="coerce",
        ).to_numpy()

        b = pd.to_numeric(
            master.loc[
                both,
                feed_col,
            ],
            errors="coerce",
        ).to_numpy()

        if not np.array_equal(
            a,
            b,
        ):
            raise ValueError(
                "Response/feedback raw provenance mismatch: "
                f"{response_col} != {feed_col}"
            )


# ============================================================
# BEHAVIOUR KEY == EYE KEY CHECK AFTER MERGE
# ============================================================

# This is intentionally redundant. The merge already uses these keys,
# but we keep an explicit check for auditability.

for attached_col, expected_phases in [
    (
        "response_eye_attached",
        {"LE", "ES", "EE"},
    ),
    (
        "allfix_eye_attached",
        {"LE", "ES", "EE"},
    ),
    (
        "feedback_eye_attached",
        {"LE"},
    ),
]:

    attached = master[
        attached_col
    ]

    if attached.any():

        unexpected = set(
            master.loc[
                attached,
                "phase",
            ].unique()
        ) - expected_phases

        if unexpected:
            raise ValueError(
                f"{attached_col}: unexpected attached phases "
                f"{sorted(unexpected)}"
            )


# ============================================================
# COMPLETE MERGE AUDIT
# ============================================================

merge_audit = pd.DataFrame(
    merge_audit_rows
)

# Add actual attachment counts after the merge.
for i, row in merge_audit.iterrows():

    pid = int(
        row[
            "sub_id"
        ]
    )

    p = master[
        master[
            "sub_id"
        ] == pid
    ]

    merge_audit.loc[
        i,
        "response_attached_n",
    ] = int(
        p[
            "response_eye_attached"
        ].sum()
    )

    merge_audit.loc[
        i,
        "feedback_attached_n",
    ] = int(
        p[
            "feedback_eye_attached"
        ].sum()
    )

    merge_audit.loc[
        i,
        "allfix_attached_n",
    ] = int(
        p[
            "allfix_eye_attached"
        ].sum()
    )

    if bool(
        row[
            "parser_validated"
        ]
    ):

        expected_response_n = sum(
            EXPECTED_MAIN.values()
        )

        expected_feedback_n = EXPECTED_MAIN[
            "LE"
        ]

        attached_ok = (
            int(
                merge_audit.loc[
                    i,
                    "response_attached_n",
                ]
            )
            == expected_response_n
            and
            int(
                merge_audit.loc[
                    i,
                    "feedback_attached_n",
                ]
            )
            == expected_feedback_n
            and
            int(
                merge_audit.loc[
                    i,
                    "allfix_attached_n",
                ]
            )
            == expected_response_n
        )

        if attached_ok:
            merge_audit.loc[
                i,
                "status",
            ] = "PASS"

            merge_audit.loc[
                i,
                "note",
            ] = (
                "PASS: final parser audit validated; explicit "
                "participant+phase+trial keys matched; response, "
                "feedback and allfix were attached with expected counts."
            )

        else:
            merge_audit.loc[
                i,
                "status",
            ] = "CHECK"

            merge_audit.loc[
                i,
                "note",
            ] = (
                "CHECK: validated eye participant did not receive "
                "the expected number of attached rows."
            )


# ============================================================
# FINAL GLOBAL SANITY CHECKS
# ============================================================

validated_beh_rows = master[
    master[
        "sub_id"
    ].isin(
        validated_eye_pids
    )
]

# Every validated participant's main task must have response + allfix.
validated_main = validated_beh_rows[
    validated_beh_rows[
        "phase"
    ].isin(
        MAIN_PHASES
    )
]

if not validated_main[
    "response_eye_attached"
].all():
    raise ValueError(
        "At least one validated main-task behavioural row "
        "is missing response eye data."
    )

if not validated_main[
    "allfix_eye_attached"
].all():
    raise ValueError(
        "At least one validated main-task behavioural row "
        "is missing allfix eye data."
    )

# Every validated LE row must have feedback.
validated_le = validated_beh_rows[
    validated_beh_rows[
        "phase"
    ] == "LE"
]

if not validated_le[
    "feedback_eye_attached"
].all():
    raise ValueError(
        "At least one validated LE behavioural row "
        "is missing feedback eye data."
    )

# Feedback nowhere else.
bad_feedback = (
    master[
        "feedback_eye_attached"
    ]
    & (
        master[
            "phase"
        ] != "LE"
    )
)

if bad_feedback.any():
    raise ValueError(
        "Feedback eye data attached outside LE."
    )


# ============================================================
# SAVE
# ============================================================

master.to_csv(
    OUTPUT_FILE,
    index=False,
)

merge_audit.to_csv(
    MERGE_AUDIT_FILE,
    index=False,
)


# ============================================================
# CONCISE CONSOLE REPORT
# ============================================================

print(
    "\n"
    + "=" * 82
)

print(
    "FINAL BEHAVIOUR + EYE MERGE — STUDY 1"
)

print(
    "=" * 82
)

print(
    "\nMERGE STATUS COUNTS:"
)

print(
    merge_audit[
        "status"
    ]
    .value_counts(
        dropna=False
    )
    .to_string()
)

not_pass = merge_audit[
    merge_audit[
        "status"
    ] != "PASS"
]

if len(
    not_pass
):

    print(
        "\nPARTICIPANTS WITHOUT COMPLETE VALIDATED EYE MERGE:"
    )

    print(
        not_pass[
            [
                "sub_id",
                "status",
                "response_attached_n",
                "feedback_attached_n",
                "allfix_attached_n",
                "note",
            ]
        ].to_string(
            index=False
        )
    )

else:

    print(
        "\nAll behavioural participants have a complete validated eye merge."
    )

print(
    "\nGLOBAL ATTACHMENT TOTALS:"
)

print(
    "Response:",
    int(
        master[
            "response_eye_attached"
        ].sum()
    )
)

print(
    "Feedback:",
    int(
        master[
            "feedback_eye_attached"
        ].sum()
    )
)

print(
    "Allfix:",
    int(
        master[
            "allfix_eye_attached"
        ].sum()
    )
)

print(
    "\nMerge key:"
)

print(
    "  behaviour = sub_id + phase + trial"
)

print(
    "  eye       = pid + Phase + phase_trial"
)

print(
    "  raw_id    = provenance only; NEVER a merge key"
)

print(
    "\nSaved combined master:"
)

print(
    OUTPUT_FILE
)

print(
    "\nSaved merge audit:"
)

print(
    MERGE_AUDIT_FILE
)

print(
    "=" * 82
)
