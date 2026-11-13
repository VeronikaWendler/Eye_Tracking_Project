from pathlib import Path
import re
import numpy as np
import pandas as pd


# ============================================================
# 01 — BUILD STUDY 1 BEHAVIOURAL MASTER IN PHYSICAL COORDINATES
#
# INPUT:
#   Study1_Data/
#       Participant_1/
#           Study1_LE_Phase_Participant_1_with_learners.csv
#           Study1_ES_Phase_Participant_1.csv
#           Study1_EE_Phase_Participant_1.csv
#           Study1_SP_Phase_Participant_1.csv
#       Participant_2/
#           ...
#
# IMPORTANT:
#   For ES we ALWAYS use the NON-_re file.
#   p1/op1/ev1 = physical LEFT
#   p2/op2/ev2 = physical RIGHT
#
# OUTPUT:
#   Study1/data_sets/Study1_Behaviour.csv
# ============================================================


# ============================================================
# PATHS
# ============================================================

STUDY_ROOT = Path(
    "D:/Birmingham_Uni_Oct25/Eye_Tracking_Project/Study1"
)

INPUT_ROOT = (
    STUDY_ROOT /
    "Study1_Data"
)

OUTPUT_DIR = (
    STUDY_ROOT /
    "data_sets"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = (
    OUTPUT_DIR /
    "Study1_Behaviour.csv"
)


# ============================================================
# PARTICIPANTS
# ============================================================

# Participant 6 was already excluded in the previous Study 1
# behavioural-master script because the dataset is incomplete.
EXCLUDE = {6}


def participant_number(path):
    m = re.fullmatch(
        r"Participant_(\d+)",
        path.name
    )

    if not m:
        return None

    return int(
        m.group(1)
    )


participant_folders = []

for folder in INPUT_ROOT.glob("Participant_*"):

    if not folder.is_dir():
        continue

    pid = participant_number(folder)

    if pid is None:
        continue

    if pid in EXCLUDE:
        continue

    participant_folders.append(
        (pid, folder)
    )


participant_folders = sorted(
    participant_folders,
    key=lambda x: x[0]
)


if not participant_folders:
    raise FileNotFoundError(
        f"No Participant_* folders found in:\n{INPUT_ROOT}"
    )


print(
    "Participants found:",
    [pid for pid, _ in participant_folders]
)


# ============================================================
# STANDARD COLUMNS USED ACROSS PHASES
# ============================================================

columns_to_keep = [
    "ev1",
    "ev2",
    "p1",
    "p2",
    "cho",
    "corr",
    "trial",
    "cond",
    "out",
    "cfout",
    "phase",
    "sess",
    "op1",
    "op2",
    "rew",
    "chose_right",
    "rtime",
]


EXPECTED_PHASE_COUNTS = {
    "LE": 160,
    "ES": 360,
    "EE": 140,
    "SP": 51,
}


# ============================================================
# HELPERS
# ============================================================

def load_phase_file(
    path,
    pid,
    expected_phase
):

    if not path.exists():
        raise FileNotFoundError(
            f"\nMissing {expected_phase} file for participant {pid}:\n"
            f"{path}"
        )

    df = pd.read_csv(
        path
    )

    # ------------------------------
    # Required standard columns
    # ------------------------------
    missing = [
        c for c in columns_to_keep
        if c not in df.columns
    ]

    if missing:
        raise KeyError(
            f"\nParticipant {pid}, phase {expected_phase}:\n"
            f"missing required columns {missing}\n"
            f"File: {path}"
        )

    # ------------------------------
    # Participant identity
    # ------------------------------
    if "SubID" in df.columns:

        ids = (
            pd.to_numeric(
                df["SubID"],
                errors="coerce"
            )
            .dropna()
            .astype(int)
            .unique()
        )

        if (
            len(ids) != 1
            or
            ids[0] != pid
        ):
            raise ValueError(
                f"\nParticipant-ID mismatch in {path}\n"
                f"Folder says {pid}, file contains {ids}"
            )

    # ------------------------------
    # Phase identity
    # ------------------------------
    phase_values = (
        df["phase"]
        .dropna()
        .astype(str)
        .unique()
    )

    if not (
        len(phase_values) == 1
        and
        phase_values[0] == expected_phase
    ):
        raise ValueError(
            f"\nPhase mismatch for participant {pid}.\n"
            f"Expected {expected_phase}, found {phase_values}\n"
            f"File: {path}"
        )

    # ------------------------------
    # Trial sequence
    # ------------------------------
    trials = pd.to_numeric(
        df["trial"],
        errors="coerce"
    )

    expected_trials = np.arange(
        1,
        len(df) + 1
    )

    if not np.array_equal(
        trials.to_numpy(),
        expected_trials
    ):
        raise ValueError(
            f"\nTrial sequence is not 1..N for participant {pid}, "
            f"phase {expected_phase}.\nFile: {path}"
        )

    # ------------------------------
    # Expected trial count
    # ------------------------------
    expected_n = EXPECTED_PHASE_COUNTS[
        expected_phase
    ]

    if len(df) != expected_n:
        raise ValueError(
            f"\nUnexpected number of rows for participant {pid}, "
            f"phase {expected_phase}:\n"
            f"expected {expected_n}, found {len(df)}\n"
            f"File: {path}"
        )

    # ========================================================
    # IMPORTANT SMALL CONSISTENCY FIX FOR ES
    #
    # The current spatial ES converter contains scoreArrayEXP2
    # but assigns `rew` from scoreEXP2.
    # LE and EE use their trial-wise score arrays.
    #
    # Therefore, if scoreArrayEXP2 exists, standardise `rew`
    # to that trial-wise score array here.
    # ========================================================

    if (
        expected_phase == "ES"
        and
        "scoreArrayEXP2" in df.columns
    ):
        df["rew"] = df[
            "scoreArrayEXP2"
        ]

    # Keep only common analysis columns.
    df = df[
        columns_to_keep
    ].copy()

    # Explicit provenance.
    if expected_phase in {
        "LE",
        "ES",
        "EE",
    }:
        df["coordinate_system"] = (
            "physical_left_right"
        )
    else:
        df["coordinate_system"] = (
            "single_option"
        )

    return df


# ============================================================
# BUILD PARTICIPANT BY PARTICIPANT
# ============================================================

all_participants = []

for pid, participant_folder in participant_folders:

    print(
        "\n" + "=" * 70
    )

    print(
        f"PARTICIPANT {pid}"
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # LE
    #
    # Prefer the version containing the learner classification
    # if it exists, otherwise use the base LE file.
    # The common master columns are identical either way.
    # --------------------------------------------------------

    le_with_learners = (
        participant_folder /
        f"Study1_LE_Phase_Participant_{pid}_with_learners.csv"
    )

    le_base = (
        participant_folder /
        f"Study1_LE_Phase_Participant_{pid}.csv"
    )

    if le_with_learners.exists():
        le_file = le_with_learners
    else:
        le_file = le_base

    le = load_phase_file(
        le_file,
        pid,
        "LE"
    )

    # --------------------------------------------------------
    # ES
    #
    # CRITICAL: NON-_re file only.
    # This preserves actual physical left/right positions.
    # --------------------------------------------------------

    es_file = (
        participant_folder /
        f"Study1_ES_Phase_Participant_{pid}.csv"
    )

    es_re_file = (
        participant_folder /
        f"Study1_ES_Phase_Participant_{pid}_re.csv"
    )

    if not es_file.exists() and es_re_file.exists():
        raise FileNotFoundError(
            f"\nParticipant {pid}: only the _re ES file exists.\n"
            f"Do NOT use it for the spatial master.\n"
            f"Expected spatial file:\n{es_file}"
        )

    es = load_phase_file(
        es_file,
        pid,
        "ES"
    )

    # Every ES trial must contain exactly one E and one S.
    valid_es = (
        (
            (es["op1"] == "E")
            &
            (es["op2"] == "S")
        )
        |
        (
            (es["op1"] == "S")
            &
            (es["op2"] == "E")
        )
    )

    if not valid_es.all():
        bad = es.loc[
            ~valid_es,
            [
                "trial",
                "op1",
                "op2",
            ]
        ]

        raise ValueError(
            f"\nParticipant {pid}: invalid ES option identities.\n"
            + bad.head(20).to_string(index=False)
        )

    # --------------------------------------------------------
    # EE
    # --------------------------------------------------------

    ee_file = (
        participant_folder /
        f"Study1_EE_Phase_Participant_{pid}.csv"
    )

    ee = load_phase_file(
        ee_file,
        pid,
        "EE"
    )

    # --------------------------------------------------------
    # SP
    # --------------------------------------------------------

    sp_file = (
        participant_folder /
        f"Study1_SP_Phase_Participant_{pid}.csv"
    )

    sp = load_phase_file(
        sp_file,
        pid,
        "SP"
    )

    # --------------------------------------------------------
    # Combine participant in actual phase order
    # --------------------------------------------------------

    participant_df = pd.concat(
        [
            le,
            es,
            ee,
            sp,
        ],
        ignore_index=True
    )

    participant_df.insert(
        0,
        "sub_id",
        pid
    )

    participant_df.insert(
        1,
        "participant_row",
        np.arange(
            1,
            len(participant_df) + 1
        )
    )

    print(
        participant_df[
            "phase"
        ]
        .value_counts()
        .reindex(
            [
                "LE",
                "ES",
                "EE",
                "SP",
            ]
        )
    )

    print(
        "Total participant rows:",
        len(participant_df)
    )

    all_participants.append(
        participant_df
    )


# ============================================================
# COMBINE ALL PARTICIPANTS
# ============================================================

master = pd.concat(
    all_participants,
    ignore_index=True
)

master.insert(
    0,
    "index",
    np.arange(
        1,
        len(master) + 1
    )
)


# ============================================================
# GLOBAL SAFETY CHECKS
# ============================================================

# Unique participant x phase x trial key.
duplicates = master.duplicated(
    [
        "sub_id",
        "phase",
        "trial",
    ],
    keep=False
)

if duplicates.any():

    bad = master.loc[
        duplicates,
        [
            "sub_id",
            "phase",
            "trial",
        ]
    ]

    raise ValueError(
        "\nDuplicate participant/phase/trial keys found:\n"
        + bad.head(30).to_string(index=False)
    )


print(
    "\n=============================================="
)

print(
    "MASTER PHASE COUNTS BY PARTICIPANT"
)

print(
    "=============================================="
)

print(
    master.groupby(
        [
            "sub_id",
            "phase",
        ]
    )
    .size()
    .unstack(
        fill_value=0
    )
)


# ============================================================
# ES SPATIAL SANITY CHECK
# ============================================================

es = master[
    master["phase"] == "ES"
].copy()

print(
    "\n=============================================="
)

print(
    "ES PHYSICAL/SPATIAL SANITY CHECK"
)

print(
    "=============================================="
)

print(
    "\nPhysical arrangement counts:"
)

print(
    es.groupby(
        [
            "op1",
            "op2",
        ]
    )
    .size()
)

print(
    "\nFraction E physically LEFT:",
    (es["op1"] == "E").mean()
)

print(
    "Fraction E physically RIGHT:",
    (es["op2"] == "E").mean()
)


# ============================================================
# SP CHECK
# ============================================================

sp = master[
    master["phase"] == "SP"
]

print(
    "\nSP rows per participant:"
)

print(
    sp.groupby(
        "sub_id"
    )
    .size()
)


# ============================================================
# SAVE
# ============================================================

master.to_csv(
    OUTPUT_FILE,
    index=False
)

print(
    "\nSaved behavioural master:"
)

print(
    OUTPUT_FILE
)

print(
    "\nTotal rows:",
    len(master)
)