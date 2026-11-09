Metric 4a — post-run verification on your own Anaconda installation

1. Extract the ZIP into C:\Users\Deepak\Desktop\bird_layout_opt\
   The ZIP contains metric4a_preflight\metric4a_postrun_checks.py only.
   Do not move, delete or overwrite the completed metric_4a_72sectors folder.

2. In Anaconda Prompt:

   conda activate floris-env
   cd /d C:\Users\Deepak\Desktop\bird_layout_opt\metric4a_preflight
   python metric4a_postrun_checks.py endpoints

3. Send ChatGPT the terminal output plus, if requested:
   C:\Users\Deepak\Desktop\bird_layout_opt\metric_4a_72sectors_postrun\endpoint_aep_1deg.json

4. The script reproduces the archived 5-degree AEP for both ends,
   computes their 1-degree AEP, compares against the <0.2% threshold,
   and writes corridor margins and aligned pair counts.

5. If either endpoint deviates by 0.2% or more, run separately:

   python metric4a_postrun_checks.py fullfront1deg

   This performs 1-degree AEP evaluations on all 38 SAVED Pareto layouts,
   without changing NSGA-II results.

Reports go to a NEW folder named metric_4a_72sectors_postrun.
The original completed optimisation archive is strictly read-only.
Independent two-end bird confirmation and figures are a separate stage.
