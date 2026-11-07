# AI Town: A Living Autonomous Society

AI Town is a headless multi-agent life simulation with a cozy procedural desktop viewer. Citizen life choices are represented as learned policy actions; the graphical application can load a PPO or MAPPO checkpoint or display clearly labelled random and rule-based baselines.

## Install (Windows PowerShell)

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Run

```powershell
python -m pytest -q
python train.py --algorithm PPO --episodes 100 --seed 29 --days 30 --output models/ppo.pt
python train.py --algorithm MAPPO --episodes 100 --seed 23 --days 30 --output models/mappo.pt
python evaluate.py --mode rule --episodes 5 --days 30 --seed 1001
python evaluate.py --mode learned --checkpoint models/mappo.pt --episodes 5 --days 30 --seed 1001
python -m ai_town.app --mode rule
python -m ai_town.app --mode learned --checkpoint models/mappo.pt
python -m ai_town.app --mode ppo
python -m ai_town.app --mode mappo
python -m experiments.research_dashboard
python plot_results.py
```

In the viewer, Space pauses, R resets, F follows the selected citizen, A shows Town Pulse, C toggles citizen comparison, B toggles wellbeing bars for all citizens, Tab changes the comparison citizen, and M toggles synthesized UI sound effects. The visible controls are Play/Pause, Reset, Follow, Pulse, Graphs, Compare, policy cycle, and speed cycle. The policy chip cycles Random, Rule, PPO, and MAPPO; key 1/2/4/5 selects those modes, and key 3 loads the `--checkpoint` policy. Graphs builds `results/research_dashboard.html` from saved evaluation episode JSON and training JSONL, then opens it in the system browser. The dashboard reports source files, seeds, horizons, and recorded evaluation/training metrics; it does not claim convergence. Arrows or right-button drag pan, `+`/`-` cycles 1x/2x/5x/10x, mouse wheel zooms over town and scrolls the timeline over the panel, and clicking a citizen or building opens its inspector.

`config/default.yaml` contains simulation, economy, reward, and optimizer defaults. Evaluation commands write machine-readable results under `results/` by mode. Training metrics and checkpoints are generated locally. Both learned algorithms load through the same evaluation and deployment interfaces; checkpoint metadata distinguishes them.

The viewer starts paused. At 1×, one simulated hour takes one real second (a full day takes 24 seconds); 2×/5×/10× multiply that pace. Space toggles playback; the mouse wheel zooms over the town and scrolls the citizen timeline when the pointer is over that panel. The viewer runs open-ended past Day 30/31; training and evaluation still use their configured finite horizons. Learned deployment samples the trained categorical policy; pass `--deterministic` to evaluation for argmax diagnostics. The inspector timeline records decisions, rewards, earnings, purchases, and clinic costs. The deployment label identifies the checkpoint algorithm. A quiet synthesized ambient loop and UI chimes play when audio is available; the speaker button or `M` mutes both. The renderer continues silently when an audio device is unavailable. Terrain detail, foreground shadows, civic landmarks, patrol-car beacon, calendar observances, and ambient safety reports do not alter RL transitions or rewards.

After evaluating random, rule-based, PPO and MAPPO modes with matching held-out seeds and 30-day horizons, build the comparison table and chart with `python plot_results.py`. Use `python evaluate.py --mode learned --checkpoint models/ppo.pt --episodes 5 --days 30 --seed 1001` for PPO results; evaluation samples the policy by default and checkpoint metadata determines the algorithm and result filename. Add `--deterministic` to inspect argmax behavior.

See `docs/` for architecture, RL formulation, roadmap, engineering decisions, testing/experiment methodology, and the academic demo guide. Current implementation is an initial vertical slice; read the roadmap for incomplete acceptance criteria. Experimental superiority and convergence have not been established. Evaluation reports per-seed survival, wellbeing, employment, wealth inequality, health, stress, social satisfaction, and hunger incidents. Results must be regenerated after environment or reward changes.

The short food-price robustness records are in `results/economy_stress_*.json`. To run the reward-ablation workflow, use module form: `python -m experiments.run_reward_ablation ...`; direct script-path invocation does not resolve the project-root training module. See `docs/Testing and Experiments.md` for the tested command and evidence limits.

After navigation retargeting changes, matched seed evaluations are retained under `results/post_navigation_fix/`. Reproduce the 60-day movement/occupancy audit with `python -m experiments.audit_navigation --mode learned --checkpoint models/mappo.pt --episodes 3 --days 60 --seed 6001 --output results/navigation_mappo.json`. Measure CPU headless throughput with `python -m experiments.profile_headless --mode learned --checkpoint models/mappo.pt --episodes 3 --days 30 --seed 8101 --output results/headless_mappo_profile_eval.json`. These scripts do not train or modify checkpoints.

The saved PPO and MAPPO actors can also be evaluated at alternate population sizes using the same local observations. Existing sweep artifacts cover populations 1, 5, and 10 from the previous map/configuration; regenerate evaluations for the current 17-citizen town before making claims about its outcomes. Interpret population sweeps as sensitivity checks, not training or generalization guarantees.

To rerun the paired 10-seed high-food-price stress evaluation for Random, Rule, PPO, and MAPPO, use `python -m experiments.run_economy_sweep --episodes 10 --days 30 --seed 9001`. Raw rows and summary are written to `results/economy_stress_expanded/`; the summary describes fixed-checkpoint variation across environment seeds, not training-seed uncertainty.

## Viewer sections and town records

The map is a 50×32 grid with 17 named citizens, including Police Officer Amara Okeke. All citizens use the same selected policy; her Work action targets the police station. Incident reports remain stochastic narrative telemetry: they have no attacker action, damage, theft, police dispatch, or effect on citizen state/reward. The town has separated lots, road-cleared landscaped parcels, and a distinct police uniform. Safety includes a 30-day reported incident count and a rate per 1,000 resident-days. Configure incident rates and type weights, calendar start date, and observances under `town_events` in `config/default.yaml`.

The Graphs control opens a local, self-contained research dashboard at `results/research_dashboard.html`. It summarizes saved evaluation rows and PPO/MAPPO training metrics, including seeds and run conditions. Current retained evaluation/training artifacts predate the 17-citizen town; the dashboard labels their recorded conditions and does not present them as results for this layout. Regenerate evaluations before reporting current-town outcomes. Fine grass, shaded buildings, game HUD, gentle procedural music, and UI chimes are presentation-only. Music and effects can be toggled together with the speaker button or `M`.

To regenerate headless UI screenshots for the sections and Data pages:

```powershell
$env:SDL_VIDEODRIVER = 'dummy'
python -m ai_town.app --mode rule --smoke-frames 3 --preview-section Town --screenshot results/continuation_qa/section_Town.png
python -m ai_town.app --mode rule --smoke-frames 3 --preview-section People --screenshot results/continuation_qa/section_People.png
python -m ai_town.app --mode rule --smoke-frames 3 --preview-section Places --screenshot results/continuation_qa/section_Places.png
python -m ai_town.app --mode rule --smoke-frames 3 --preview-section Data --preview-data Reports --screenshot results/continuation_qa/data_Reports.png
python -m ai_town.app --mode rule --smoke-frames 3 --preview-section Places --preview-building police_station_01 --screenshot results/continuation_qa/police_station.png
python -m ai_town.app --mode rule --smoke-frames 3 --preview-section Data --preview-data Social --screenshot results/continuation_qa/social_safety.png
python -m ai_town.app --mode rule --smoke-frames 3 --preview-section Data --preview-data Events --screenshot results/continuation_qa/events_safety_calendar.png
```

The other Data previews are Economy, Social, Events, and RL Lab; select each with `--preview-data`. These smoke captures validate rendering for a selected view; live mouse interaction is a separate check. To rebuild the three-training-seed financial-stability ablation summary from retained raw evaluations, run `python -m experiments.summarize_financial_ablation`. The result is `results/ablations/financial_stability_three_seed_summary.json`; see `docs/Testing and Experiments.md` for its small-sample interpretation.

