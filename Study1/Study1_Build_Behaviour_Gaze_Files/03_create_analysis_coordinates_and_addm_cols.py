from pathlib import Path
import numpy as np
import pandas as pd


# ============================================================
# 03 — FINAL COORDINATE + GAZE + aDDM FEATURE BUILD — STUDY 1
# ============================================================
# Input:  Study1_Behaviour_with_Gaze.csv
# Output: Study1_Behaviour_with_Gaze_AnalysisReady.csv
#
# Original variables always retain PHYSICAL meaning:
#   op1/p1/ev1/gaze-left  = LEFT
#   op2/p2/ev2/gaze-right = RIGHT
#
# Nothing is reordered. E/S-relative variables are ADDED from op1/op2.
# ============================================================

ROOT = Path("D:/Birmingham_Uni_Oct25/Eye_Tracking_Project/Study1")
DATA_DIR = ROOT / "data_sets"
INPUT_FILE = DATA_DIR / "Study1_Behaviour_with_Gaze.csv"
OUTPUT_FILE = DATA_DIR / "Study1_Behaviour_with_Gaze_AnalysisReady.csv"
AUDIT_FILE = DATA_DIR / "Study1_Behaviour_with_Gaze_AnalysisReady_AUDIT.csv"
MAIN_PHASES = ["LE", "ES", "EE"]

df = pd.read_csv(INPUT_FILE)
n_input = len(df)

required = {
    "sub_id", "phase", "trial", "cho", "op1", "op2", "p1", "p2", "ev1", "ev2",
    "DwellLeft", "DwellRight", "DwellTotal", "FirstFixLoc", "FinalFixLoc",
    "LeftFixNR", "RightFixNR",
}
missing = required - set(df.columns)
if missing:
    raise KeyError(f"Missing required columns: {sorted(missing)}")

df["sub_id"] = pd.to_numeric(df["sub_id"], errors="raise").astype(int)
df["phase"] = df["phase"].astype(str).str.strip()
for c in ["trial", "cho", "p1", "p2", "ev1", "ev2"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

main_mask = df["phase"].isin(MAIN_PHASES)
is_es = df["phase"].eq("ES")
le_mask = df["phase"].eq("LE")


# ============================================================
# SAFETY CHECKS
# ============================================================

main = df.loc[main_mask]
if main.duplicated(["sub_id", "phase", "trial"]).any():
    bad = main.loc[
        main.duplicated(["sub_id", "phase", "trial"], keep=False),
        ["sub_id", "phase", "trial"],
    ]
    raise ValueError("Duplicate main behavioural keys:\n" + bad.head(30).to_string(index=False))

if "coordinate_system" in df.columns:
    coord = set(df.loc[main_mask, "coordinate_system"].dropna().astype(str).str.lower().unique())
    bad_coord = [x for x in coord if "physical" not in x]
    if bad_coord:
        raise ValueError(f"Unexpected non-physical coordinate labels: {bad_coord}")

invalid_choice = main_mask & ~df["cho"].isin([1, 2])
if invalid_choice.any():
    bad = df.loc[invalid_choice, ["sub_id", "phase", "trial", "cho", "op1", "op2"]]
    raise ValueError(
        "Invalid choice code in two-option phase; expected cho=1 LEFT or cho=2 RIGHT.\n"
        + bad.head(30).to_string(index=False)
    )

valid_es_identity = ((df["op1"] == "E") & (df["op2"] == "S")) | ((df["op1"] == "S") & (df["op2"] == "E"))
if (is_es & ~valid_es_identity).any():
    bad = df.loc[is_es & ~valid_es_identity, ["sub_id", "trial", "op1", "op2", "cho"]]
    raise ValueError(
        "Every ES trial must contain exactly one E and one S.\n"
        + bad.head(30).to_string(index=False)
    )

es = df.loc[is_es]
E_LEFT_FRACTION = float(es["op1"].eq("E").mean())
if E_LEFT_FRACTION < 0.02 or E_LEFT_FRACTION > 0.98:
    raise ValueError(
        f"ES positions look canonicalized (E-left fraction={E_LEFT_FRACTION:.4f}); "
        "this resembles the old `_re` data."
    )

if "chose_right" in df.columns:
    observed = pd.to_numeric(df["chose_right"], errors="coerce")
    expected = pd.Series(np.nan, index=df.index, dtype=float)
    expected.loc[main_mask & df["cho"].eq(1)] = 0
    expected.loc[main_mask & df["cho"].eq(2)] = 1
    comparable = main_mask & observed.notna() & expected.notna()
    if not np.array_equal(observed.loc[comparable].to_numpy(), expected.loc[comparable].to_numpy()):
        raise ValueError("chose_right does not agree with physical cho coding.")


# ============================================================
# HELPERS
# ============================================================

def num(x):
    return pd.to_numeric(x, errors="coerce")


def safe_prop(numer, denom):
    numer, denom = num(numer), num(denom)
    return (numer / denom).where(denom > 0, np.nan)


def higher_side(left, right):
    left, right = num(left), num(right)
    out = pd.Series(np.nan, index=left.index, dtype=float)
    out.loc[left > right] = 1
    out.loc[right > left] = 2
    return out


def opposite_side(side):
    out = pd.Series(np.nan, index=side.index, dtype=float)
    out.loc[side == 1] = 2
    out.loc[side == 2] = 1
    return out


def pick_side(left, right, side):
    left, right = num(left), num(right)
    out = pd.Series(np.nan, index=left.index, dtype=float)
    out.loc[side == 1] = left.loc[side == 1]
    out.loc[side == 2] = right.loc[side == 2]
    return out


def fix_matches(fix_loc, target_side):
    fix, target = num(fix_loc), num(target_side)
    out = pd.Series(np.nan, index=fix.index, dtype=float)
    valid = fix.isin([1, 2]) & target.isin([1, 2])
    out.loc[valid] = fix.loc[valid].eq(target.loc[valid]).astype(int)
    return out


def fix_identity(fix_loc, op_left, op_right):
    fix = num(fix_loc)
    out = pd.Series(pd.NA, index=fix.index, dtype="object")
    out.loc[fix == 1] = op_left.loc[fix == 1]
    out.loc[fix == 2] = op_right.loc[fix == 2]
    return out


# ============================================================
# PHYSICAL CHOICE + ES IDENTITY COORDINATES
# ============================================================

df["chose_left_spatial"] = np.nan
df["chose_right_spatial"] = np.nan
df.loc[main_mask, "chose_left_spatial"] = df.loc[main_mask, "cho"].eq(1).astype(int)
df.loc[main_mask, "chose_right_spatial"] = df.loc[main_mask, "cho"].eq(2).astype(int)

for c in ["E_is_left", "E_is_right", "S_is_left", "S_is_right", "E_side", "S_side"]:
    df[c] = np.nan

df.loc[is_es, "E_is_left"] = df.loc[is_es, "op1"].eq("E").astype(int)
df.loc[is_es, "E_is_right"] = df.loc[is_es, "op2"].eq("E").astype(int)
df.loc[is_es, "S_is_left"] = df.loc[is_es, "op1"].eq("S").astype(int)
df.loc[is_es, "S_is_right"] = df.loc[is_es, "op2"].eq("S").astype(int)
df.loc[is_es & df["op1"].eq("E"), "E_side"] = 1
df.loc[is_es & df["op2"].eq("E"), "E_side"] = 2
df.loc[is_es & df["op1"].eq("S"), "S_side"] = 1
df.loc[is_es & df["op2"].eq("S"), "S_side"] = 2

df["chosen_stimulus"] = pd.Series(pd.NA, index=df.index, dtype="object")
df.loc[main_mask & df["cho"].eq(1), "chosen_stimulus"] = df.loc[main_mask & df["cho"].eq(1), "op1"]
df.loc[main_mask & df["cho"].eq(2), "chosen_stimulus"] = df.loc[main_mask & df["cho"].eq(2), "op2"]
df["stim_chosen"] = df["chosen_stimulus"].copy()  # old compatibility name

for c in ["chose_E", "chose_S"]:
    df[c] = np.nan
valid_es_choice = is_es & df["chosen_stimulus"].isin(["E", "S"])
df.loc[valid_es_choice, "chose_E"] = df.loc[valid_es_choice, "chosen_stimulus"].eq("E").astype(int)
df.loc[valid_es_choice, "chose_S"] = df.loc[valid_es_choice, "chosen_stimulus"].eq("S").astype(int)

# Explicit E/S values. These are the safe replacements for old code that
# interpreted p1 as E and p2 as S after using the `_re` behavioural files.
for c in [
    "p_E", "p_S", "p_E_pct", "p_S_pct",
    "ev_E", "ev_S", "ValueDifference_ES", "ValueDifference_ES_pct"
]:
    df[c] = np.nan

df.loc[is_es, "p_E"] = pick_side(df.loc[is_es, "p1"], df.loc[is_es, "p2"], df.loc[is_es, "E_side"])
df.loc[is_es, "p_S"] = pick_side(df.loc[is_es, "p1"], df.loc[is_es, "p2"], df.loc[is_es, "S_side"])
df.loc[is_es, "p_E_pct"] = df.loc[is_es, "p_E"] * 100.0
df.loc[is_es, "p_S_pct"] = df.loc[is_es, "p_S"] * 100.0
df.loc[is_es, "ev_E"] = pick_side(df.loc[is_es, "ev1"], df.loc[is_es, "ev2"], df.loc[is_es, "E_side"])
df.loc[is_es, "ev_S"] = pick_side(df.loc[is_es, "ev1"], df.loc[is_es, "ev2"], df.loc[is_es, "S_side"])
df.loc[is_es, "ValueDifference_ES"] = df.loc[is_es, "p_S"] - df.loc[is_es, "p_E"]
df.loc[is_es, "ValueDifference_ES_pct"] = df.loc[is_es, "p_S_pct"] - df.loc[is_es, "p_E_pct"]


# ============================================================
# CORE VALUE / aDDM VARIABLES
# ============================================================

# Keep original p1/p2 unchanged; percentage copies exist only for old compatibility.
df["p1_pct"] = df["p1"] * 100.0
df["p2_pct"] = df["p2"] * 100.0

for c in ["aDDM_value_left", "aDDM_value_right", "aDDM_choice_right"]:
    df[c] = np.nan
df.loc[main_mask, "aDDM_value_left"] = df.loc[main_mask, "ev1"]
df.loc[main_mask, "aDDM_value_right"] = df.loc[main_mask, "ev2"]
df.loc[main_mask, "aDDM_choice_right"] = df.loc[main_mask, "chose_right_spatial"]

df["ev_opt"] = np.nan
df["ev_sub"] = np.nan
df.loc[main_mask, "ev_opt"] = df.loc[main_mask, ["ev1", "ev2"]].max(axis=1)
df.loc[main_mask, "ev_sub"] = df.loc[main_mask, ["ev1", "ev2"]].min(axis=1)

df["BetterOption"] = higher_side(df["ev1"], df["ev2"])
df.loc[~main_mask, "BetterOption"] = np.nan  # ties also remain NaN
df["WorseOption"] = opposite_side(df["BetterOption"])

for c in ["V_corr", "V_sub", "p_high", "p_low"]:
    df[c] = np.nan
df.loc[main_mask, "V_corr"] = df.loc[main_mask, ["p1_pct", "p2_pct"]].max(axis=1)
df.loc[main_mask, "V_sub"] = df.loc[main_mask, ["p1_pct", "p2_pct"]].min(axis=1)
df.loc[main_mask, "p_high"] = df.loc[main_mask, ["p1", "p2"]].max(axis=1)
df.loc[main_mask, "p_low"] = df.loc[main_mask, ["p1", "p2"]].min(axis=1)


# ============================================================
# VALUE-RANKED + PHYSICAL GAZE FEATURES FOR EACH WINDOW
# ============================================================

def add_window_features(frame, suffix, use):
    """suffix: '' (response), '_Feed', or '_allfix'."""
    dl, dr, dt = f"DwellLeft{suffix}", f"DwellRight{suffix}", f"DwellTotal{suffix}"
    ff, lf = f"FirstFixLoc{suffix}", f"FinalFixLoc{suffix}"
    lnr, rnr = f"LeftFixNR{suffix}", f"RightFixNR{suffix}"

    if not {dl, dr, dt, ff, lf}.issubset(frame.columns):
        return

    pl = "PropDwell_Left" if not suffix else f"PropDwell_Left{suffix}"
    pr = "PropDwell_Right" if not suffix else f"PropDwell_Right{suffix}"
    frame[pl] = np.nan
    frame[pr] = np.nan
    frame.loc[use, pl] = safe_prop(frame.loc[use, dl], frame.loc[use, dt])
    frame.loc[use, pr] = safe_prop(frame.loc[use, dr], frame.loc[use, dt])

    if not suffix:
        frame["aDDM_gaze_left"] = np.nan
        frame["aDDM_gaze_right"] = np.nan
        frame.loc[use, "aDDM_gaze_left"] = frame.loc[use, pl]
        frame.loc[use, "aDDM_gaze_right"] = frame.loc[use, pr]

    time_adv = "DwellTimeAdvantage" if not suffix else f"DwellTimeAdvantage{suffix}"
    prop_adv = "DwellPropAdvantage" if not suffix else f"DwellPropAdvantage{suffix}"
    frame[time_adv] = np.nan
    frame[prop_adv] = np.nan
    frame.loc[use, time_adv] = num(frame.loc[use, dr]) - num(frame.loc[use, dl])
    frame.loc[use, prop_adv] = frame.loc[use, pr] - frame.loc[use, pl]

    opt_time = "DwellTime_opt" if not suffix else f"DwellTime_opt{suffix}"
    sub_time = "DwellTime_sub" if not suffix else f"DwellTime_sub{suffix}"
    opt_prop = "PropDwell_opt" if not suffix else f"PropDwell_opt{suffix}"
    sub_prop = "PropDwell_sub" if not suffix else f"PropDwell_sub{suffix}"
    for c in [opt_time, sub_time, opt_prop, sub_prop]:
        frame[c] = np.nan

    better, worse = frame["BetterOption"], frame["WorseOption"]
    frame.loc[use, opt_time] = pick_side(frame.loc[use, dl], frame.loc[use, dr], better.loc[use])
    frame.loc[use, sub_time] = pick_side(frame.loc[use, dl], frame.loc[use, dr], worse.loc[use])
    frame.loc[use, opt_prop] = safe_prop(frame.loc[use, opt_time], frame.loc[use, dt])
    frame.loc[use, sub_prop] = safe_prop(frame.loc[use, sub_time], frame.loc[use, dt])

    attention = "AttentionW" if not suffix else f"AttentionW{suffix}"
    inattention = "InattentionW" if not suffix else f"InattentionW{suffix}"
    frame[attention] = np.nan
    frame[inattention] = np.nan
    frame.loc[use, attention] = (
        frame.loc[use, opt_prop] * frame.loc[use, "V_corr"]
        - frame.loc[use, sub_prop] * frame.loc[use, "V_sub"]
    )
    frame.loc[use, inattention] = (
        frame.loc[use, sub_prop] * frame.loc[use, "V_corr"]
        - frame.loc[use, opt_prop] * frame.loc[use, "V_sub"]
    )

    first_corr = "FixLocFirstCorr" if not suffix else f"FixLocFirstCorr{suffix}"
    last_corr = "FixLocLastCorr" if not suffix else f"FixLocLastCorr{suffix}"
    frame[first_corr] = np.nan
    frame[last_corr] = np.nan
    frame.loc[use, first_corr] = fix_matches(frame.loc[use, ff], better.loc[use])
    frame.loc[use, last_corr] = fix_matches(frame.loc[use, lf], better.loc[use])

    if suffix:
        frame[f"BetterOption{suffix}"] = np.nan
        frame.loc[use, f"BetterOption{suffix}"] = better.loc[use]

    if lnr in frame.columns and rnr in frame.columns:
        opt_fix = "Fix_opt" if not suffix else f"Fix_opt{suffix}"
        sub_fix = "Fix_sub" if not suffix else f"Fix_sub{suffix}"
        phys_fix_adv = "FixationAdvantage" if not suffix else f"FixationAdvantage{suffix}"
        corr_fix_adv = "FixationAdvantage_Corr" if not suffix else f"FixationAdvantage_Corr{suffix}"
        for c in [opt_fix, sub_fix, phys_fix_adv, corr_fix_adv]:
            frame[c] = np.nan

        frame.loc[use, opt_fix] = pick_side(frame.loc[use, lnr], frame.loc[use, rnr], better.loc[use])
        frame.loc[use, sub_fix] = pick_side(frame.loc[use, lnr], frame.loc[use, rnr], worse.loc[use])
        frame.loc[use, phys_fix_adv] = num(frame.loc[use, rnr]) - num(frame.loc[use, lnr])
        frame.loc[use, corr_fix_adv] = frame.loc[use, opt_fix] - frame.loc[use, sub_fix]

    corr_time = "DwelltimeAdvantageCorrect" if not suffix else f"DwelltimeAdvantageCorrect{suffix}"
    corr_prop = "DwellPropAdvantageCorrect" if not suffix else f"DwellPropAdvantageCorrect{suffix}"
    frame[corr_time] = np.nan
    frame[corr_prop] = np.nan
    frame.loc[use, corr_time] = frame.loc[use, opt_time] - frame.loc[use, sub_time]
    frame.loc[use, corr_prop] = frame.loc[use, opt_prop] - frame.loc[use, sub_prop]


add_window_features(df, "", main_mask)
add_window_features(df, "_Feed", le_mask)
add_window_features(df, "_allfix", main_mask)


# ============================================================
# ES IDENTITY-ALIGNED RESPONSE GAZE
# ============================================================

for c in ["Dwell_E", "Dwell_S", "PropDwell_E", "PropDwell_S", "DTA_ES", "DTA_spatial",
          "absDTA_ES", "absDTA_spatial", "DTA_ES2", "DTA_spatial2"]:
    df[c] = np.nan

df.loc[is_es, "Dwell_E"] = pick_side(
    df.loc[is_es, "DwellLeft"], df.loc[is_es, "DwellRight"], df.loc[is_es, "E_side"]
)
df.loc[is_es, "Dwell_S"] = pick_side(
    df.loc[is_es, "DwellLeft"], df.loc[is_es, "DwellRight"], df.loc[is_es, "S_side"]
)
df.loc[is_es, "PropDwell_E"] = safe_prop(df.loc[is_es, "Dwell_E"], df.loc[is_es, "DwellTotal"])
df.loc[is_es, "PropDwell_S"] = safe_prop(df.loc[is_es, "Dwell_S"], df.loc[is_es, "DwellTotal"])
df.loc[is_es, "DTA_ES"] = df.loc[is_es, "PropDwell_S"] - df.loc[is_es, "PropDwell_E"]
df.loc[is_es, "DTA_spatial"] = df.loc[is_es, "PropDwell_Right"] - df.loc[is_es, "PropDwell_Left"]
df.loc[is_es, "absDTA_ES"] = df.loc[is_es, "DTA_ES"].abs()
df.loc[is_es, "absDTA_spatial"] = df.loc[is_es, "DTA_spatial"].abs()
df.loc[is_es, "DTA_ES2"] = df.loc[is_es, "DTA_ES"] ** 2
df.loc[is_es, "DTA_spatial2"] = df.loc[is_es, "DTA_spatial"] ** 2

for c in ["FixNR_E", "FixNR_S", "FixAdvantage_SE"]:
    df[c] = np.nan
df.loc[is_es, "FixNR_E"] = pick_side(
    df.loc[is_es, "LeftFixNR"], df.loc[is_es, "RightFixNR"], df.loc[is_es, "E_side"]
)
df.loc[is_es, "FixNR_S"] = pick_side(
    df.loc[is_es, "LeftFixNR"], df.loc[is_es, "RightFixNR"], df.loc[is_es, "S_side"]
)
df["Fix_E"] = df["FixNR_E"].copy()
df["Fix_S"] = df["FixNR_S"].copy()
df.loc[is_es, "FixAdvantage_SE"] = df.loc[is_es, "FixNR_S"] - df.loc[is_es, "FixNR_E"]

# Clear identity-relative names used by the corrected R analyses.
df["DwellTimeAdvantage_ES"] = np.nan       # positive = more ms on S
df["DwellPropAdvantage_ES"] = np.nan       # positive = greater proportional dwell on S
df["FixationAdvantage_ES"] = np.nan        # positive = more fixations on S
df.loc[is_es, "DwellTimeAdvantage_ES"] = df.loc[is_es, "Dwell_S"] - df.loc[is_es, "Dwell_E"]
df.loc[is_es, "DwellPropAdvantage_ES"] = df.loc[is_es, "DTA_ES"]
df.loc[is_es, "FixationAdvantage_ES"] = df.loc[is_es, "FixAdvantage_SE"]

# Reconstruct the old "remove first fixation" variables without touching the
# audited dwell columns. The physical versions remain explicitly spatial.
for c in [
    "DwellLeft_noFirst", "DwellRight_noFirst", "DwellTotal_noFirst",
    "PropDwell_Left_noFirst", "PropDwell_Right_noFirst",
    "DwellTimeAdvantage_noFirst",
    "Dwell_E_noFirst", "Dwell_S_noFirst", "PropDwell_E_noFirst",
    "PropDwell_S_noFirst", "DwellTimeAdvantage_ES_noFirst",
    "DwellPropAdvantage_ES_noFirst",
]:
    df[c] = np.nan

if "FirstFixDur" in df.columns:
    ff_dur = num(df["FirstFixDur"])
    dl = num(df["DwellLeft"])
    dr = num(df["DwellRight"])
    first_loc = num(df["FirstFixLoc"])

    df.loc[main_mask, "DwellLeft_noFirst"] = (
        dl.loc[main_mask] - np.where(first_loc.loc[main_mask].eq(1), ff_dur.loc[main_mask], 0)
    ).clip(lower=0)
    df.loc[main_mask, "DwellRight_noFirst"] = (
        dr.loc[main_mask] - np.where(first_loc.loc[main_mask].eq(2), ff_dur.loc[main_mask], 0)
    ).clip(lower=0)
    df.loc[main_mask, "DwellTotal_noFirst"] = (
        df.loc[main_mask, "DwellLeft_noFirst"] + df.loc[main_mask, "DwellRight_noFirst"]
    )
    df.loc[main_mask, "PropDwell_Left_noFirst"] = safe_prop(
        df.loc[main_mask, "DwellLeft_noFirst"], df.loc[main_mask, "DwellTotal_noFirst"]
    )
    df.loc[main_mask, "PropDwell_Right_noFirst"] = safe_prop(
        df.loc[main_mask, "DwellRight_noFirst"], df.loc[main_mask, "DwellTotal_noFirst"]
    )
    df.loc[main_mask, "DwellTimeAdvantage_noFirst"] = (
        df.loc[main_mask, "DwellRight_noFirst"] - df.loc[main_mask, "DwellLeft_noFirst"]
    )

    df.loc[is_es, "Dwell_E_noFirst"] = pick_side(
        df.loc[is_es, "DwellLeft_noFirst"], df.loc[is_es, "DwellRight_noFirst"], df.loc[is_es, "E_side"]
    )
    df.loc[is_es, "Dwell_S_noFirst"] = pick_side(
        df.loc[is_es, "DwellLeft_noFirst"], df.loc[is_es, "DwellRight_noFirst"], df.loc[is_es, "S_side"]
    )
    df.loc[is_es, "PropDwell_E_noFirst"] = safe_prop(
        df.loc[is_es, "Dwell_E_noFirst"], df.loc[is_es, "DwellTotal_noFirst"]
    )
    df.loc[is_es, "PropDwell_S_noFirst"] = safe_prop(
        df.loc[is_es, "Dwell_S_noFirst"], df.loc[is_es, "DwellTotal_noFirst"]
    )
    df.loc[is_es, "DwellTimeAdvantage_ES_noFirst"] = (
        df.loc[is_es, "Dwell_S_noFirst"] - df.loc[is_es, "Dwell_E_noFirst"]
    )
    df.loc[is_es, "DwellPropAdvantage_ES_noFirst"] = (
        df.loc[is_es, "PropDwell_S_noFirst"] - df.loc[is_es, "PropDwell_E_noFirst"]
    )


# ============================================================
# ES FIRST / FINAL / MIDDLE FIXATION IDENTITY
# ============================================================

df["FirstFixStimulus"] = pd.Series(pd.NA, index=df.index, dtype="object")
df["FinalFixStimulus"] = pd.Series(pd.NA, index=df.index, dtype="object")
df.loc[is_es, "FirstFixStimulus"] = fix_identity(
    df.loc[is_es, "FirstFixLoc"], df.loc[is_es, "op1"], df.loc[is_es, "op2"]
)
df.loc[is_es, "FinalFixStimulus"] = fix_identity(
    df.loc[is_es, "FinalFixLoc"], df.loc[is_es, "op1"], df.loc[is_es, "op2"]
)

for c in ["FirstFix_E", "FirstFix_S", "FinalFix_E", "FinalFix_S", "chose_final_fixated"]:
    df[c] = np.nan

first_valid = is_es & df["FirstFixStimulus"].isin(["E", "S"])
final_valid = is_es & df["FinalFixStimulus"].isin(["E", "S"])
df.loc[first_valid, "FirstFix_E"] = df.loc[first_valid, "FirstFixStimulus"].eq("E").astype(int)
df.loc[first_valid, "FirstFix_S"] = df.loc[first_valid, "FirstFixStimulus"].eq("S").astype(int)
df.loc[final_valid, "FinalFix_E"] = df.loc[final_valid, "FinalFixStimulus"].eq("E").astype(int)
df.loc[final_valid, "FinalFix_S"] = df.loc[final_valid, "FinalFixStimulus"].eq("S").astype(int)
df["FirstFixOnS"] = df["FirstFix_S"].copy()
df["FinalFixOnS"] = df["FinalFix_S"].copy()

valid_final_choice = final_valid & df["chosen_stimulus"].isin(["E", "S"])
df.loc[valid_final_choice, "chose_final_fixated"] = (
    df.loc[valid_final_choice, "FinalFixStimulus"]
    == df.loc[valid_final_choice, "chosen_stimulus"]
).astype(int)

# Identity-coded location variables for legacy R analyses:
#   1 = E, 2 = S, NaN = unavailable.
df["FirstFixLoc_ES"] = np.nan
df["FinalFixLoc_ES"] = np.nan
df.loc[is_es & df["FirstFixStimulus"].eq("E"), "FirstFixLoc_ES"] = 1
df.loc[is_es & df["FirstFixStimulus"].eq("S"), "FirstFixLoc_ES"] = 2
df.loc[is_es & df["FinalFixStimulus"].eq("E"), "FinalFixLoc_ES"] = 1
df.loc[is_es & df["FinalFixStimulus"].eq("S"), "FinalFixLoc_ES"] = 2

df["MiddleDominantStimulus"] = pd.Series(pd.NA, index=df.index, dtype="object")
df["MiddleDominantLoc_ES"] = np.nan
if "MiddleDominantLoc" in df.columns:
    df.loc[is_es, "MiddleDominantStimulus"] = fix_identity(
        df.loc[is_es, "MiddleDominantLoc"], df.loc[is_es, "op1"], df.loc[is_es, "op2"]
    )
    df.loc[is_es & df["MiddleDominantStimulus"].eq("E"), "MiddleDominantLoc_ES"] = 1
    df.loc[is_es & df["MiddleDominantStimulus"].eq("S"), "MiddleDominantLoc_ES"] = 2

# Correct identity-based sequence flags that replace the old ESE/ESS/... flags
# which had actually been built from physical LEFT/RIGHT fixation locations.
def sequence_flag(a, b, c=None):
    valid = a.notna() & b.notna() if c is None else a.notna() & b.notna() & c.notna()
    out = pd.Series(np.nan, index=df.index, dtype=float)
    if c is None:
        out.loc[valid] = (a.loc[valid] & b.loc[valid]).astype(int)
    else:
        out.loc[valid] = (a.loc[valid] & b.loc[valid] & c.loc[valid]).astype(int)
    return out

fE, fS = df["FirstFixLoc_ES"].eq(1), df["FirstFixLoc_ES"].eq(2)
mE, mS = df["MiddleDominantLoc_ES"].eq(1), df["MiddleDominantLoc_ES"].eq(2)
lE, lS = df["FinalFixLoc_ES"].eq(1), df["FinalFixLoc_ES"].eq(2)

loc_valid3 = df["FirstFixLoc_ES"].notna() & df["MiddleDominantLoc_ES"].notna() & df["FinalFixLoc_ES"].notna()
for name, cond in {
    "EEE": fE & mE & lE, "EES": fE & mE & lS, "ESE": fE & mS & lE, "ESS": fE & mS & lS,
    "SEE": fS & mE & lE, "SES": fS & mE & lS, "SSE": fS & mS & lE, "SSS": fS & mS & lS,
}.items():
    df[name] = np.nan
    df.loc[is_es & loc_valid3, name] = cond.loc[is_es & loc_valid3].astype(int)

pair_specs = {
    "EE_first": (fE, mE, df["FirstFixLoc_ES"].notna() & df["MiddleDominantLoc_ES"].notna()),
    "ES_first": (fE, mS, df["FirstFixLoc_ES"].notna() & df["MiddleDominantLoc_ES"].notna()),
    "SE_first": (fS, mE, df["FirstFixLoc_ES"].notna() & df["MiddleDominantLoc_ES"].notna()),
    "SS_first": (fS, mS, df["FirstFixLoc_ES"].notna() & df["MiddleDominantLoc_ES"].notna()),
    "EE_final": (fE, lE, df["FirstFixLoc_ES"].notna() & df["FinalFixLoc_ES"].notna()),
    "ES_final": (fE, lS, df["FirstFixLoc_ES"].notna() & df["FinalFixLoc_ES"].notna()),
    "SE_final": (fS, lE, df["FirstFixLoc_ES"].notna() & df["FinalFixLoc_ES"].notna()),
    "SS_final": (fS, lS, df["FirstFixLoc_ES"].notna() & df["FinalFixLoc_ES"].notna()),
    "EE_middle": (mE, lE, df["MiddleDominantLoc_ES"].notna() & df["FinalFixLoc_ES"].notna()),
    "ES_middle": (mE, lS, df["MiddleDominantLoc_ES"].notna() & df["FinalFixLoc_ES"].notna()),
    "SE_middle": (mS, lE, df["MiddleDominantLoc_ES"].notna() & df["FinalFixLoc_ES"].notna()),
    "SS_middle": (mS, lS, df["MiddleDominantLoc_ES"].notna() & df["FinalFixLoc_ES"].notna()),
}
for name, (a, b, valid) in pair_specs.items():
    df[name] = np.nan
    mask = is_es & valid
    df.loc[mask, name] = (a.loc[mask] & b.loc[mask]).astype(int)


# ============================================================
# ES ALLFIX IDENTITY VARIABLES
# ============================================================

allfix_req = {"DwellLeft_allfix", "DwellRight_allfix", "DwellTotal_allfix",
              "FirstFixLoc_allfix", "FinalFixLoc_allfix"}

if allfix_req.issubset(df.columns):
    for c in ["Dwell_E_allfix", "Dwell_S_allfix", "PropDwell_E_allfix", "PropDwell_S_allfix",
              "DTA_ES_allfix", "DTA_spatial_allfix"]:
        df[c] = np.nan

    df.loc[is_es, "Dwell_E_allfix"] = pick_side(
        df.loc[is_es, "DwellLeft_allfix"], df.loc[is_es, "DwellRight_allfix"], df.loc[is_es, "E_side"]
    )
    df.loc[is_es, "Dwell_S_allfix"] = pick_side(
        df.loc[is_es, "DwellLeft_allfix"], df.loc[is_es, "DwellRight_allfix"], df.loc[is_es, "S_side"]
    )
    df.loc[is_es, "PropDwell_E_allfix"] = safe_prop(
        df.loc[is_es, "Dwell_E_allfix"], df.loc[is_es, "DwellTotal_allfix"]
    )
    df.loc[is_es, "PropDwell_S_allfix"] = safe_prop(
        df.loc[is_es, "Dwell_S_allfix"], df.loc[is_es, "DwellTotal_allfix"]
    )
    df.loc[is_es, "DTA_ES_allfix"] = (
        df.loc[is_es, "PropDwell_S_allfix"] - df.loc[is_es, "PropDwell_E_allfix"]
    )

    if {"PropDwell_Left_allfix", "PropDwell_Right_allfix"}.issubset(df.columns):
        df.loc[is_es, "DTA_spatial_allfix"] = (
            df.loc[is_es, "PropDwell_Right_allfix"] - df.loc[is_es, "PropDwell_Left_allfix"]
        )

    if {"LeftFixNR_allfix", "RightFixNR_allfix"}.issubset(df.columns):
        df["FixNR_E_allfix"] = np.nan
        df["FixNR_S_allfix"] = np.nan
        df.loc[is_es, "FixNR_E_allfix"] = pick_side(
            df.loc[is_es, "LeftFixNR_allfix"], df.loc[is_es, "RightFixNR_allfix"], df.loc[is_es, "E_side"]
        )
        df.loc[is_es, "FixNR_S_allfix"] = pick_side(
            df.loc[is_es, "LeftFixNR_allfix"], df.loc[is_es, "RightFixNR_allfix"], df.loc[is_es, "S_side"]
        )

    df["FirstFixStimulus_allfix"] = pd.Series(pd.NA, index=df.index, dtype="object")
    df["FinalFixStimulus_allfix"] = pd.Series(pd.NA, index=df.index, dtype="object")
    df.loc[is_es, "FirstFixStimulus_allfix"] = fix_identity(
        df.loc[is_es, "FirstFixLoc_allfix"], df.loc[is_es, "op1"], df.loc[is_es, "op2"]
    )
    df.loc[is_es, "FinalFixStimulus_allfix"] = fix_identity(
        df.loc[is_es, "FinalFixLoc_allfix"], df.loc[is_es, "op1"], df.loc[is_es, "op2"]
    )

    for c in ["FirstFix_E_allfix", "FirstFix_S_allfix", "FinalFix_E_allfix",
              "FinalFix_S_allfix", "chose_final_fixated_allfix"]:
        df[c] = np.nan

    first_af = is_es & df["FirstFixStimulus_allfix"].isin(["E", "S"])
    final_af = is_es & df["FinalFixStimulus_allfix"].isin(["E", "S"])
    df.loc[first_af, "FirstFix_E_allfix"] = df.loc[first_af, "FirstFixStimulus_allfix"].eq("E").astype(int)
    df.loc[first_af, "FirstFix_S_allfix"] = df.loc[first_af, "FirstFixStimulus_allfix"].eq("S").astype(int)
    df.loc[final_af, "FinalFix_E_allfix"] = df.loc[final_af, "FinalFixStimulus_allfix"].eq("E").astype(int)
    df.loc[final_af, "FinalFix_S_allfix"] = df.loc[final_af, "FinalFixStimulus_allfix"].eq("S").astype(int)

    valid_final_af = final_af & df["chosen_stimulus"].isin(["E", "S"])
    df.loc[valid_final_af, "chose_final_fixated_allfix"] = (
        df.loc[valid_final_af, "FinalFixStimulus_allfix"]
        == df.loc[valid_final_af, "chosen_stimulus"]
    ).astype(int)


# ============================================================
# LE MODEL-CONVENIENCE VARIABLES
# ============================================================

df["feedback"] = np.nan
if "out" in df.columns:
    out_num = num(df["out"])
    df.loc[le_mask, "feedback"] = out_num.loc[le_mask].replace({-1: 0, 1: 1})

df["q_init"] = np.nan
df.loc[le_mask, "q_init"] = 0.5

df["split_by"] = np.nan
if "cond" in df.columns:
    df.loc[le_mask, "split_by"] = num(df.loc[le_mask, "cond"])


# ============================================================
# OVERALL VALUE / VALUE-DIFFERENCE VARIABLES
# ============================================================

for c in ["OV_num", "AbsValueDiff", "RLdiff", "OV_num_prob", "AbsValueDiff_prob", "RLdiff_prob"]:
    df[c] = np.nan

df.loc[main_mask, "OV_num"] = df.loc[main_mask, "p1_pct"] + df.loc[main_mask, "p2_pct"]
df.loc[main_mask, "AbsValueDiff"] = (df.loc[main_mask, "p1_pct"] - df.loc[main_mask, "p2_pct"]).abs()
df.loc[main_mask, "RLdiff"] = df.loc[main_mask, "p1_pct"] - df.loc[main_mask, "p2_pct"]
df.loc[main_mask, "OV_num_prob"] = df.loc[main_mask, "p1"] + df.loc[main_mask, "p2"]
df.loc[main_mask, "AbsValueDiff_prob"] = (df.loc[main_mask, "p1"] - df.loc[main_mask, "p2"]).abs()
df.loc[main_mask, "RLdiff_prob"] = df.loc[main_mask, "p1"] - df.loc[main_mask, "p2"]

# Compatibility aliases used extensively by the existing Rmd. In the old
# pipeline these were separate copies of the same continuous quantities.
df["OV_num_2"] = df["OV_num"].copy()
df["AbsValueDiff_2"] = df["AbsValueDiff"].copy()


# ============================================================
# OPTIONAL DESCRIPTIVE VALUE CATEGORIES
# ============================================================

for c in ["OV", "VD", "OV_2", "VD_2"]:
    df[c] = np.nan
for c in ["OVcate", "Abscate", "OVcate_2", "Abscate_2"]:
    df[c] = pd.Series(pd.NA, index=df.index, dtype="object")

cat_map = {"low": 1, "medium": 2, "high": 3}


def assign_categories(frame, value_col, text_col, numeric_col, q_low, q_high):
    groups = frame.loc[main_mask].groupby(["sub_id", "phase"], sort=True).groups
    for _, idx in groups.items():
        values = num(frame.loc[idx, value_col])
        valid = values.dropna()
        if valid.empty:
            continue
        lo, hi = valid.quantile(q_low), valid.quantile(q_high)
        cats = pd.Series(pd.NA, index=values.index, dtype="object")
        cats.loc[values.notna() & (values <= lo)] = "low"
        cats.loc[values.notna() & (values > lo) & (values <= hi)] = "medium"
        cats.loc[values.notna() & (values > hi)] = "high"
        frame.loc[idx, text_col] = cats
        frame.loc[idx, numeric_col] = cats.map(cat_map)


assign_categories(df, "OV_num", "OVcate", "OV", 0.25, 0.75)
assign_categories(df, "AbsValueDiff", "Abscate", "VD", 0.25, 0.75)
assign_categories(df, "OV_num", "OVcate_2", "OV_2", 0.33, 0.66)
assign_categories(df, "AbsValueDiff", "Abscate_2", "VD_2", 0.33, 0.66)


# ============================================================
# FINAL VALIDATION
# ============================================================

choice_identity_ok = df.loc[is_es, "chosen_stimulus"].isin(["E", "S"]).all()

dwell_valid = is_es & df["DwellLeft"].notna() & df["DwellRight"].notna()
dwell_conservation_ok = np.allclose(
    df.loc[dwell_valid, "Dwell_E"] + df.loc[dwell_valid, "Dwell_S"],
    df.loc[dwell_valid, "DwellLeft"] + df.loc[dwell_valid, "DwellRight"],
    equal_nan=True,
)

prop_valid = is_es & df["PropDwell_E"].notna() & df["PropDwell_S"].notna()
prop_sum_ok = (
    np.allclose(
        (df.loc[prop_valid, "PropDwell_E"] + df.loc[prop_valid, "PropDwell_S"]).to_numpy(),
        np.ones(int(prop_valid.sum())),
        atol=1e-10,
        rtol=0,
    )
    if prop_valid.any() else True
)

dta_valid = is_es & df["absDTA_ES"].notna() & df["absDTA_spatial"].notna()
abs_dta_invariant_ok = np.allclose(
    df.loc[dta_valid, "absDTA_ES"],
    df.loc[dta_valid, "absDTA_spatial"],
    equal_nan=True,
)

addm_valid = main_mask & df["aDDM_gaze_left"].notna() & df["aDDM_gaze_right"].notna()
addm_gaze_sum_ok = (
    np.allclose(
        (df.loc[addm_valid, "aDDM_gaze_left"] + df.loc[addm_valid, "aDDM_gaze_right"]).to_numpy(),
        np.ones(int(addm_valid.sum())),
        atol=1e-10,
        rtol=0,
    )
    if addm_valid.any() else True
)

main_key_unique_ok = ~df.loc[main_mask].duplicated(["sub_id", "phase", "trial"]).any()


# ============================================================
# PARTICIPANT AUDIT
# ============================================================

audit_rows = []
for pid in sorted(df["sub_id"].dropna().astype(int).unique()):
    p = df[df["sub_id"] == pid]
    pes = p[p["phase"] == "ES"]

    pdwell = pes["DwellLeft"].notna() & pes["DwellRight"].notna()
    p_dwell_ok = (
        np.allclose(
            pes.loc[pdwell, "Dwell_E"] + pes.loc[pdwell, "Dwell_S"],
            pes.loc[pdwell, "DwellLeft"] + pes.loc[pdwell, "DwellRight"],
            equal_nan=True,
        )
        if pdwell.any() else True
    )

    pdta = pes["absDTA_ES"].notna() & pes["absDTA_spatial"].notna()
    p_dta_ok = (
        np.allclose(
            pes.loc[pdta, "absDTA_ES"], pes.loc[pdta, "absDTA_spatial"], equal_nan=True
        )
        if pdta.any() else True
    )

    p_choice_ok = pes["chosen_stimulus"].isin(["E", "S"]).all() if len(pes) else True
    status = "PASS" if all([p_dwell_ok, p_dta_ok, p_choice_ok]) else "CHECK"

    audit_rows.append({
        "sub_id": pid,
        "status": status,
        "main_rows": int(p["phase"].isin(MAIN_PHASES).sum()),
        "ES_rows": len(pes),
        "ES_E_left_fraction": float(pes["op1"].eq("E").mean()) if len(pes) else np.nan,
        "ES_choice_identity_ok": bool(p_choice_ok),
        "ES_dwell_conservation_ok": bool(p_dwell_ok),
        "ES_abs_DTA_invariant_ok": bool(p_dta_ok),
        "ES_DTA_nonmissing_n": int(pes["DTA_ES"].notna().sum()),
        "ES_chose_S_nonmissing_n": int(pes["chose_S"].notna().sum()),
        "value_tie_n_main": int(
            (p["phase"].isin(MAIN_PHASES) & num(p["ev1"]).eq(num(p["ev2"]))).sum()
        ),
        "response_eye_attached_n": (
            int(p["response_eye_attached"].sum()) if "response_eye_attached" in p.columns else np.nan
        ),
    })

audit = pd.DataFrame(audit_rows)

global_checks = {
    "row_count_preserved": len(df) == n_input,
    "main_key_unique": bool(main_key_unique_ok),
    "ES_choice_identity_ok": bool(choice_identity_ok),
    "ES_dwell_conservation_ok": bool(dwell_conservation_ok),
    "ES_prop_sum_ok": bool(prop_sum_ok),
    "ES_abs_DTA_invariant_ok": bool(abs_dta_invariant_ok),
    "aDDM_gaze_sum_ok": bool(addm_gaze_sum_ok),
    "participant_audits_all_PASS": bool((audit["status"] == "PASS").all()),
}

failed = [name for name, passed in global_checks.items() if not passed]
if failed:
    raise ValueError("Final feature-building audit failed: " + ", ".join(failed))


# ============================================================
# SAVE
# ============================================================

df.to_csv(OUTPUT_FILE, index=False)
audit.to_csv(AUDIT_FILE, index=False)

print("\n" + "=" * 86)
print("FINAL COORDINATE + GAZE + aDDM FEATURE BUILD — STUDY 1")
print("=" * 86)
print("\nGLOBAL CHECKS:")
for name, passed in global_checks.items():
    print(f"  {name}: {passed}")

print(f"\nE physically LEFT on {E_LEFT_FRACTION:.4f} of ES trials.")
print("\nOriginal L/R columns were not swapped or reordered.")
print("E/S columns were derived trial-by-trial from op1/op2.")
print("No participants were excluded during feature construction.")
print(f"\nRows: {n_input} -> {len(df)}")
print("\nSaved analysis-ready master:")
print(OUTPUT_FILE)
print("\nSaved feature audit:")
print(AUDIT_FILE)
print("=" * 86)
