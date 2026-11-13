# Mat file to CSV conversion scripts for Study2

# This folder contains conversion scripts that transform raw MATLAB files collected from Study 2 experiments to trial-level CSV files across all phases (LE, ES, EE, SP)

All scripts:
- iterate over participant .mat files
- extract task-specific variables
- recreate columns used in Garcia's original analyses
- output one CSV per participant and phase

## File overview & run order

### study2_mat_to_py_LE.py
- Converts .mat files from teh first phase: LE (LE – learning phase) into trial-level CSVs

Output (for every participant):
'Study2_LE_Phase_Participant_ID.csv'  
'Study2_LE_Phase_Participant_ID_info.csv' -> contains info on the stimuli and probability pairings for that participant

### study2_mat_to_py_ES.py 
- Converts .mat files from the second phase: ES (ES – experiential vs symbolic phase) into
CSV format + columns of interest

Output:  
'Study2_ES_Phase_Participant_ID.csv'

### study2_mat_to_py_EE.py
- Converts .mat files from the third EE phase (EE – experiential vs experiential phase)
into CSVs

Output:  
'Study2_EE_Phase_Participant_ID.csv'

### study2_mat_to_py_SP.py
- Converts slider-rating .mat files from the SP (slider / probability
estimation) phase into CSV

Output: 
'Study2_SP_Phase_Participant_ID.csv'