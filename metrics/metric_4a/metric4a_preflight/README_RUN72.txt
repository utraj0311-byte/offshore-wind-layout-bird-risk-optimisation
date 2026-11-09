METRIC 4a — 72-SECTOR RUNNER ADD-ON (18 OCT 2026)
================================================

This adds ONE Python file into your existing pre-flight directory. It does
not replace config.py, floris_aep.py, model8_code or either original archive.

1. Extract Metric4a_72sector_Runner_Addon.zip INTO:
   C:\Users\Deepak\Desktop\bird_layout_opt
   Extract with folder paths (it contains metric4a_preflight/metric4a_run72.py).

2. Start Anaconda Prompt:
   conda activate floris-env
   cd /d C:\Users\Deepak\Desktop\bird_layout_opt\metric4a_preflight
   python metric4a_run72.py verify

   Send the entire output. The program DOES NOT create the production run
   directory, save checkpoints or run NSGA-II in this stage.

3. ONLY after the 'verify' result passes, run the small CHECKPOINT TEST:
   python metric4a_run72.py smoke

   This small verification DOES run NSGA-II for just 8 layouts x 3 generations
   on 120 birds/batch, twice. It compares uninterrupted minimise() with a
   pickle checkpoint / resume run, checking EXACT final X, F and hypervolume.
   It is NOT the 8,000-evaluation production run. DO NOT continue if FAIL.
   Send the full output.

4. PRODUCTION (ONLY after explicit user approval, not part of pre-flight):
   python metric4a_run72.py run

   Output directory is NEW:
   C:\Users\Deepak\Desktop\bird_layout_opt\metric_4a_72sectors
   The command refuses to overwrite if that folder already exists.
   Saves a checkpoint after setup and at every 5 completed generations.
   Leave the prompt open. Never run another run command into the same folder.

5. RESUME if Windows/laptop restarts:
   conda activate floris-env
   cd /d C:\Users\Deepak\Desktop\bird_layout_opt\metric4a_preflight
   python metric4a_run72.py resume

   All hashes and package versions are checked before resuming. If a checkpoint
   cannot be loaded, the previous checkpoint is attempted. It may repeat up
   to 5 generations after interruption, but DOES NOT change saved input data.
   Source files MUST NOT be changed between start and resume.

6. The simulation is entirely local. The optimiser uses ONE Python worker
   process, though NumPy/FLORIS may use additional native threads. Plan
   about 7-9 hours (measurement-based estimate, not a guarantee).

7. During the long run: Windows Settings > System > Power & battery > Screen,
   sleep & hibernate timeouts (or Power & sleep). Set plugged-in device sleep
   to NEVER, and leave the laptop plugged in with cooling/ventilation. Screen
   may turn off; PC sleep/hibernate must be off. If lid-closed sleep applies,
   leave the lid open. Restore your normal settings when finished.

SCIENTIFIC SETTINGS (FROZEN)
- Exactly the original NSGA-II(pop_size=80, sampling=original class,
  eliminate_duplicates=True), gen=100, seed=42, same implied default
  crossover and mutation; no new operator, no repair, no parallel workers.
- 5 x 2,000 training trajectories, seeds 42-46; p_response = 0.6;
  encounter scaling = 0.6854; AEP loss scaling = 137.96914764866568 GWh;
  wake-free AEP = 854.4956585617186 GWh/year.
- Original FLORIS YAML unchanged; D=242.24 m, site 6000 m square, 4D spacing.
- Only n_sectors changes 12 -> 72 (84 -> 504 wind conditions derived).
- Historical pymoo version was not archived. Laptop pymoo 0.6.1.5 is required.
- No change to bird corridor or behaviour; including the known corner issue.

PRODUCTION OUTPUTS
- run_settings.json, settings_diff.json, file_hashes.json
- initial_population.csv, checkpoint_latest.pkl (and previous backup)
- pareto_decision_vectors.npy, pareto_normalised_objectives.npy
- pareto_points.csv, final_population.npz (complete last generation X,F,G,CV)
- all_evaluated_candidates.csv, hypervolume.csv
- highest-AEP_layout.csv, lowest-risk_layout.csv, training_summary.json

The independent 20 x 5,000-bird confirmation, 1-degree checks, corridor margin
check, plots, and original 3 x 10,000 fresh-front verification are POST-RUN
steps, NOT done during this production runner. Their scripts will be prepared
once the saved Pareto front is available. This avoids losing the optimisation
archive because of an error in a later validation/plotting step.

SAFETY
- This script never writes to or renames the original two ZIP archives.
- The script does not launch 8,000 evaluations without the explicit 'run' command.
- If verify or smoke fails: STOP, do not execute 'run'.
