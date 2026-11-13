from pathlib import Path
import re
import numpy as np
import pandas as pd


# ============================================================
# STUDY 1 — FINAL EYE PARSING SAFETY AUDIT
# ============================================================
#
# READ ONLY.
#
# Run this AFTER regenerating response / feedback / allfix files with:
#     study1_eye_parser_MAIN_ONLY_FINAL.py
#
# One row per participant. One concise audit file.
#
# Final phase identity is checked against the programmed chronology:
#   42 training
#   160 LE
#   360 ES
#   140 EE
#
# Raw EyeLink Training/Task metadata are audited separately.
# They are provenance, not the merge key.
#
# The strongest alignment check compares raw ASC ResponseLeft/Right
# against behavioural cho for every main trial.
# ============================================================

PROJECT_ROOT = Path(
    "D:/Birmingham_Uni_Oct25/Eye_Tracking_Project/Study2"
)

RAW_ROOT = Path(
    "D:/Aberdeen_Uni_June24/cap/THESIS/OV_Analysis/data"
)

EYE_ROOT = PROJECT_ROOT / "Study2_Data"
DATA_DIR = PROJECT_ROOT / "data_sets"

BEHAVIOUR_FILE = DATA_DIR / "Study2_Behaviour.csv"

OUTPUT_FILE = (
    DATA_DIR
    / "Study2_FINAL_EYE_PARSING_SAFETY_AUDIT.csv"
)

EXPECTED_TRAIN = 42

EXPECTED_MAIN = {
    "LE": 160,
    "ES": 360,
    "EE": 140,
}

EXPECTED_RAW = 702
EXPECTED_MAIN_N = 660

# Study 2 participants excluded from all gaze-audit checks.
EXCLUDE = {2, 6, 9, 14, 18, 20, 26}


def extract_pid(path):
    m = re.search(
        r"(?:^|[\\/])sub-(\d+)(?:[\\/]|$)",
        str(path),
        flags=re.I,
    )

    if not m:
        raise ValueError(
            f"Could not extract participant ID from {path}"
        )

    return int(m.group(1))


def explicit_phase(train, task):
    if pd.isna(train):
        return None

    try:
        train = int(train)
    except (TypeError, ValueError):
        return None

    if train != 0:
        return None

    if pd.isna(task):
        return None

    task = str(task).strip()

    return {
        "Task 1": "LE",
        "Task 2": "ES",
        "Task 3": "EE",
    }.get(task)


def scan_raw(path, pid):
    rows = []
    current = None
    trial_seq = 0

    with open(path, "r", errors="replace") as f:

        for line_no, line in enumerate(f, start=1):

            trial_match = re.search(
                r"MSG\s+(\d+)\s+TRIALID\s+(\d+)",
                line,
            )

            if trial_match:

                if current is not None:
                    rows.append(current)

                trial_seq += 1

                current = {
                    "pid": pid,
                    "trial_seq": trial_seq,
                    "raw_id": int(trial_match.group(2)),
                    "start_ts": int(trial_match.group(1)),
                    "asc_line": line_no,
                    "train": np.nan,
                    "Task_raw": pd.NA,
                    "response_side": pd.NA,
                    "response_n": 0,
                }

                continue

            if current is None:
                continue

            response_match = re.search(
                r"MSG\s+(\d+)\s+(ResponseLeft|ResponseRight)",
                line,
            )

            if response_match:
                current["response_n"] += 1

                if pd.isna(current["response_side"]):
                    current["response_side"] = (
                        response_match.group(2)
                    )

                continue

            msg_match = re.search(
                r"MSG\s+\d+\s+!V TRIAL_VAR\s+(.*)",
                line,
            )

            if not msg_match:
                continue

            msg = msg_match.group(1).strip()

            tm = re.search(
                r"\bTraining\s+(-?\d+)\b",
                msg,
                flags=re.I,
            )

            if tm:
                current["train"] = int(tm.group(1))

            taskm = re.search(
                r"\bTask\s+([123])\b",
                msg,
                flags=re.I,
            )

            if taskm:
                current["Task_raw"] = (
                    f"Task {taskm.group(1)}"
                )

    if current is not None:
        rows.append(current)

    df = pd.DataFrame(rows)

    if len(df) != EXPECTED_RAW:
        raise ValueError(
            f"Participant {pid}: raw ASC has {len(df)} trial events, "
            f"expected {EXPECTED_RAW}"
        )

    df["Phase_metadata"] = [
        explicit_phase(train, task)
        for train, task in zip(
            df["train"],
            df["Task_raw"],
        )
    ]

    # Independent design-based phase identity.
    df["Phase"] = "TRAIN"
    df["phase_trial"] = np.nan
    df["main_trial_seq"] = np.nan

    main_idx = df.index[EXPECTED_TRAIN:]

    le_idx = main_idx[:160]
    es_idx = main_idx[160:520]
    ee_idx = main_idx[520:660]

    df.loc[le_idx, "Phase"] = "LE"
    df.loc[es_idx, "Phase"] = "ES"
    df.loc[ee_idx, "Phase"] = "EE"

    df.loc[le_idx, "phase_trial"] = np.arange(1, 161)
    df.loc[es_idx, "phase_trial"] = np.arange(1, 361)
    df.loc[ee_idx, "phase_trial"] = np.arange(1, 141)

    df.loc[main_idx, "main_trial_seq"] = np.arange(1, 661)

    df["raw_cho"] = df["response_side"].map({
        "ResponseLeft": 1,
        "ResponseRight": 2,
    })

    known = df["Phase_metadata"].notna()

    df["metadata_phase_mismatch"] = False
    df.loc[known, "metadata_phase_mismatch"] = (
        df.loc[known, "Phase_metadata"]
        != df.loc[known, "Phase"]
    )

    return df


def read_generated(path, pid, allowed_phases):
    if not path.exists():
        return None, False, f"missing {path.name}"

    df = pd.read_csv(path)

    required = {
        "pid",
        "raw_id",
        "trial_seq",
        "phase_trial",
        "start_ts",
        "asc_line",
        "Phase",
    }

    missing = required - set(df.columns)

    if missing:
        return df, False, (
            "missing columns: "
            + ",".join(sorted(missing))
        )

    if set(df["Phase"].astype(str).unique()) - set(allowed_phases):
        return df, False, "unexpected phase"

    if set(pd.to_numeric(df["pid"], errors="coerce").dropna().astype(int)) != {pid}:
        return df, False, "pid mismatch"

    if df.duplicated(
        ["pid", "Phase", "phase_trial"]
    ).any():
        return df, False, "duplicate merge key"

    for phase in sorted(set(df["Phase"].astype(str))):
        p = df[
            df["Phase"] == phase
        ].sort_values("phase_trial")

        obs = pd.to_numeric(
            p["phase_trial"],
            errors="coerce",
        ).to_numpy()

        exp = np.arange(1, len(p) + 1)

        if not np.array_equal(obs, exp):
            return df, False, f"{phase} phase_trial not 1..N"

    return df, True, "OK"


def provenance_match(raw, generated, phases):
    cols = [
        "Phase",
        "phase_trial",
        "raw_id",
        "trial_seq",
        "start_ts",
        "asc_line",
    ]

    a = raw[
        raw["Phase"].isin(phases)
    ][cols].copy()

    b = generated[
        generated["Phase"].isin(phases)
    ][cols].copy()

    a = a.sort_values(
        ["Phase", "phase_trial"]
    ).reset_index(drop=True)

    b = b.sort_values(
        ["Phase", "phase_trial"]
    ).reset_index(drop=True)

    if len(a) != len(b):
        return False

    for col in cols:

        if col == "Phase":
            if not np.array_equal(
                a[col].astype(str).to_numpy(),
                b[col].astype(str).to_numpy(),
            ):
                return False

        else:
            if not np.array_equal(
                pd.to_numeric(
                    a[col], errors="coerce"
                ).to_numpy(),
                pd.to_numeric(
                    b[col], errors="coerce"
                ).to_numpy(),
            ):
                return False

    return True


def cross_generated_match(a, b, phases):
    cols = [
        "Phase",
        "phase_trial",
        "raw_id",
        "trial_seq",
        "start_ts",
        "asc_line",
    ]

    x = a[
        a["Phase"].isin(phases)
    ][cols].sort_values(
        ["Phase", "phase_trial"]
    ).reset_index(drop=True)

    y = b[
        b["Phase"].isin(phases)
    ][cols].sort_values(
        ["Phase", "phase_trial"]
    ).reset_index(drop=True)

    return x.equals(y)


# ============================================================
# BEHAVIOUR
# ============================================================

beh = pd.read_csv(BEHAVIOUR_FILE)

required_beh = {
    "sub_id",
    "phase",
    "trial",
    "cho",
}

missing = required_beh - set(beh.columns)

if missing:
    raise KeyError(
        f"Behaviour file missing: {sorted(missing)}"
    )

beh["sub_id"] = pd.to_numeric(
    beh["sub_id"], errors="raise"
).astype(int)

beh = beh[
    beh["sub_id"].between(1, 27)
    & ~beh["sub_id"].isin(EXCLUDE)
].copy()

beh["trial"] = pd.to_numeric(
    beh["trial"], errors="raise"
).astype(int)

beh["cho"] = pd.to_numeric(
    beh["cho"], errors="raise"
).astype(int)

beh = beh[
    beh["phase"].isin(["LE", "ES", "EE"])
].copy()


# ============================================================
# RAW FILES
# ============================================================

asc_files = []


for path in RAW_ROOT.rglob("*.asc"):
    pid = extract_pid(path)

    if pid in EXCLUDE or pid > 27:
        continue

    asc_files.append(
        (pid, path)
    )

asc_files.sort(key=lambda x: x[0])

rows = []

for pid, path in asc_files:

    raw = scan_raw(path, pid)

    raw_main = raw[
        raw["Phase"].isin(["LE", "ES", "EE"])
    ].copy()

    # Raw event integrity.
    raw_sequence_ok = np.array_equal(
        raw["trial_seq"].to_numpy(),
        np.arange(1, EXPECTED_RAW + 1),
    )

    timestamps_ok = (
        raw["start_ts"].is_unique
        and raw["start_ts"].is_monotonic_increasing
    )

    lines_ok = (
        raw["asc_line"].is_unique
        and raw["asc_line"].is_monotonic_increasing
    )

    raw_counts = raw["raw_id"].value_counts()
    reused = raw_counts[raw_counts > 1]

    if len(reused):
        repeated = raw[
            raw["raw_id"].isin(reused.index)
        ]

        reuse_distinct_ok = all(
            g["start_ts"].is_unique
            and g["asc_line"].is_unique
            for _, g in repeated.groupby("raw_id")
        )
    else:
        reuse_distinct_ok = True

    metadata_mismatch = raw[
        raw["metadata_phase_mismatch"]
    ]

    metadata_mismatch_n = len(metadata_mismatch)

    if metadata_mismatch_n:
        metadata_examples = ";".join(
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
        metadata_examples = ""

    # Raw response vs behaviour.
    b = beh[
        beh["sub_id"] == pid
    ][
        ["sub_id", "phase", "trial", "cho"]
    ].copy()

    b = b.rename(
        columns={
            "sub_id": "pid",
            "phase": "Phase",
            "trial": "phase_trial",
            "cho": "beh_cho",
        }
    )

    rb = b.merge(
        raw_main[
            [
                "pid",
                "Phase",
                "phase_trial",
                "raw_cho",
                "response_n",
            ]
        ],
        on=["pid", "Phase", "phase_trial"],
        how="outer",
        indicator=True,
        validate="one_to_one",
    )

    all_beh_keys_match = bool(
        (rb["_merge"] == "both").all()
    )

    response_missing_n = int(
        rb["raw_cho"].isna().sum()
    )

    response_multiple_n = int(
        (
            rb["response_n"].fillna(0)
            != 1
        ).sum()
    )

    comparable = rb[
        rb["_merge"] == "both"
    ].dropna(
        subset=["raw_cho"]
    )

    response_choice_mismatch_n = int(
        (
            comparable["beh_cho"].astype(int)
            != comparable["raw_cho"].astype(int)
        ).sum()
    )

    # Generated files.
    pdir = EYE_ROOT / f"Participant_{pid}"

    response, response_ok, response_note = read_generated(
        pdir / f"Participant_{pid}_eye_response_phase.csv",
        pid,
        {"LE", "ES", "EE"},
    )

    feedback, feedback_ok, feedback_note = read_generated(
        pdir / f"Participant_{pid}_eye_feedback_phase.csv",
        pid,
        {"LE"},
    )

    allfix, allfix_ok, allfix_note = read_generated(
        pdir / f"Participant_{pid}_eye_allfix_phase.csv",
        pid,
        {"LE", "ES", "EE"},
    )

    response_count_ok = bool(
        response_ok
        and len(response) == 660
        and response["Phase"].value_counts().to_dict()
        == {"ES": 360, "LE": 160, "EE": 140}
    )

    allfix_count_ok = bool(
        allfix_ok
        and len(allfix) == 660
        and allfix["Phase"].value_counts().to_dict()
        == {"ES": 360, "LE": 160, "EE": 140}
    )

    feedback_count_ok = bool(
        feedback_ok
        and len(feedback) == 160
        and set(feedback["Phase"].astype(str)) == {"LE"}
    )

    response_raw_ok = bool(
        response_ok
        and provenance_match(
            raw, response, {"LE", "ES", "EE"}
        )
    )

    allfix_raw_ok = bool(
        allfix_ok
        and provenance_match(
            raw, allfix, {"LE", "ES", "EE"}
        )
    )

    feedback_raw_ok = bool(
        feedback_ok
        and provenance_match(
            raw, feedback, {"LE"}
        )
    )

    response_allfix_ok = bool(
        response_ok
        and allfix_ok
        and cross_generated_match(
            response,
            allfix,
            {"LE", "ES", "EE"},
        )
    )

    response_feedback_ok = bool(
        response_ok
        and feedback_ok
        and cross_generated_match(
            response,
            feedback,
            {"LE"},
        )
    )

    all_safe = all([
        len(raw) == 702,
        len(raw_main) == 660,
        raw_sequence_ok,
        timestamps_ok,
        lines_ok,
        reuse_distinct_ok,
        all_beh_keys_match,
        response_missing_n == 0,
        response_multiple_n == 0,
        response_choice_mismatch_n == 0,
        response_ok,
        feedback_ok,
        allfix_ok,
        response_count_ok,
        feedback_count_ok,
        allfix_count_ok,
        response_raw_ok,
        feedback_raw_ok,
        allfix_raw_ok,
        response_allfix_ok,
        response_feedback_ok,
    ])

    status = (
        "PASS"
        if all_safe
        else "CHECK"
    )

    if status == "PASS":
        if metadata_mismatch_n:
            summary = (
                "PASS: raw events, behaviour responses, and all generated "
                "eye files align exactly under the programmed chronology. "
                f"{metadata_mismatch_n} raw Task metadata boundary "
                f"discrepancy/discrepancies are retained as provenance: "
                f"{metadata_examples}."
            )
        else:
            summary = (
                "PASS: raw events, behaviour responses, raw Task metadata, "
                "and all generated eye files align under the programmed "
                "chronology."
            )
    else:
        summary = (
            "CHECK REQUIRED: at least one raw-event, response-alignment, "
            "generated-file count, or provenance check failed."
        )

    rows.append({
        "pid": pid,
        "status": status,

        "raw_events": len(raw),
        "main_events": len(raw_main),

        "raw_sequence_ok": raw_sequence_ok,
        "timestamps_ok": bool(timestamps_ok),
        "source_lines_ok": bool(lines_ok),

        "raw_id_reused_n": int(len(reused)),
        "raw_id_reuse_distinct_events": bool(reuse_distinct_ok),

        "metadata_phase_mismatch_n": metadata_mismatch_n,
        "metadata_phase_mismatch_examples": metadata_examples,

        "behaviour_keys_match": all_beh_keys_match,
        "raw_response_missing_n": response_missing_n,
        "raw_response_multiple_n": response_multiple_n,
        "raw_response_choice_mismatch_n": response_choice_mismatch_n,

        "response_structure_ok": response_ok,
        "response_count_ok": response_count_ok,
        "response_matches_raw": response_raw_ok,

        "feedback_structure_ok": feedback_ok,
        "feedback_count_ok": feedback_count_ok,
        "feedback_matches_raw": feedback_raw_ok,

        "allfix_structure_ok": allfix_ok,
        "allfix_count_ok": allfix_count_ok,
        "allfix_matches_raw": allfix_raw_ok,

        "response_allfix_provenance_ok": response_allfix_ok,
        "response_feedback_provenance_ok": response_feedback_ok,

        "response_note": response_note,
        "feedback_note": feedback_note,
        "allfix_note": allfix_note,

        "summary": summary,
    })


audit = pd.DataFrame(rows).sort_values(
    "pid"
).reset_index(drop=True)

audit.to_csv(
    OUTPUT_FILE,
    index=False,
)

print("\n" + "=" * 84)
print("FINAL EYE PARSING SAFETY AUDIT")
print("=" * 84)

print("\nSTATUS COUNTS:")
print(
    audit["status"]
    .value_counts(dropna=False)
    .to_string()
)

bad = audit[
    audit["status"] != "PASS"
]

if len(bad):
    print("\nPARTICIPANTS REQUIRING REVIEW:")
    print(
        bad[
            [
                "pid",
                "raw_events",
                "main_events",
                "raw_response_choice_mismatch_n",
                "response_count_ok",
                "feedback_count_ok",
                "allfix_count_ok",
            ]
        ].to_string(index=False)
    )
else:
    print("\nAll participants PASS.")

notes = audit[
    audit["metadata_phase_mismatch_n"] > 0
]

if len(notes):
    print("\nPASSING PARTICIPANTS WITH RAW METADATA BOUNDARY NOTES:")
    print(
        notes[
            [
                "pid",
                "metadata_phase_mismatch_n",
                "metadata_phase_mismatch_examples",
            ]
        ].to_string(index=False)
    )

print("\nSaved:")
print(OUTPUT_FILE)
print("=" * 84)
