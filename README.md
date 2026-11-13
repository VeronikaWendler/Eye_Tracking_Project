# Eye Tracking Project

This repository processes behavioural and eye-tracking data from Study 1 and Study 2 and produces the figures and statistical tables for both studies. The task phases are learning (LE), experiential–symbolic choice (ES), experiential–experiential choice (EE), and stated probability ratings (SP).

## Preprocessing order

Start in `Study1/` or `Study2/` and follow the table from top to bottom. The studies can run independently.

| Step | Study 1 | Study 2 |
|---|---|---|
| 1. Convert MATLAB behaviour to participant CSVs | In `Study1_Matlab_to_Python_Conversion/`, run the `study1_mat_to_py_*.py` scripts in order: LE → ES → EE → SP. | In `Study2_Matlab_to_Python_Conversion/`, run the `study2_mat_to_py_*.py` scripts in the same order. |
| 2. Parse EyeLink recordings | `Study1_RawGaze_to_Participant_CSV/study1_eye_asc.py` | `Study2_RawGaze_to_Participant_CSV/study2_eye_asc.py` |
| 3. Combine the four behavioural phases | `Study1_Build_Behaviour_Gaze_Files/01_build_behaviour_spatial.py` | `Study2_Build_Behaviour_Gaze_Files/01_build_behaviour_spatial.py` |
| 4. Check gaze and trial alignment | `Study1_RawGaze_to_Participant_CSV/study1_parsing_safety_audit.py` | `Study2_RawGaze_to_Participant_CSV/study2_parsing_safety_audit.py` |
| 5. Attach validated gaze to behavioural trials | `Study1_Build_Behaviour_Gaze_Files/02_attach_all_eye_spatial.py` | `Study2_Build_Behaviour_Gaze_Files/02_attach_all_eye_spatial.py` |
| 6. Add analysis variables and aDDM regressors | `Study1_Build_Behaviour_Gaze_Files/03_create_analysis_coordinates_and_addm_cols.py` | `Study2_Build_Behaviour_Gaze_Files/03_create_analysis_coordinates_and_addm_cols.py` |

In step 2, run each gaze parser three times by changing `mode` in the script. Mode `1` extracts response-period gaze for the main analyses, mode `2` extracts learning-feedback gaze, and mode `3` extracts all valid fixations. The audit in step 4 needs both the parsed gaze and the behavioural table from step 3.

The final datasets and audit files are written to each study’s `data_sets/` folder. The main analysis inputs are `Study1_Behaviour_with_Gaze_AnalysisReady.csv` and `Study2_Behaviour_with_Gaze_AnalysisReady.csv`.

## Analyses

| Analysis | Study 1 | Study 2 |
|---|---|---|
| Accuracy, reaction time, choice, and value sensitivity | `Study1_Analysis/Study1_Behavioural_Analyses.Rmd` | `Study2_Analysis/Study2_Behavioural_Analyses.Rmd` |
| Dwell time, fixations, and gaze switching | `Study1_Analysis/Study1_Main_Gaze_Analyses.Rmd` | `Study2_Analysis/Study2_Main_Gaze_Analyses.Rmd` |

Figures and tables are saved in each study’s `results/figures/` and `results/tables/` folders.

## Model fitting and follow-up

Use the two analysis-ready CSVs as inputs to the companion [ESaDDM repository](https://github.com/VeronikaWendler/esaddm) for model fitting. The optional `exploratory_corrs.Rmd` combines ESaDDM posterior summaries with the participant-level EE slopes from the behavioural notebooks. Before running it, update its input paths for both studies. The current Study 2 slopes path points to the Study 1 table.

`DDM_image.py` generates the standalone drift-diffusion schematic.