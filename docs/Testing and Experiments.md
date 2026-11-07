# Testing and experiments

## Automated checks

Run `python -m pytest -q`. The current 70 tests cover PettingZoo parallel lifecycle, observation bounds, finite reward, seeded transitions and stochastic evaluation, state bounds, purchase/restock and financial-history invariants, navigation, batch-social reward scoring, 30-day viability, GAE, optimizer parameter updates, checkpoint round trips, bounded playback-clock updates, renderer composition, foreground depth layers and wellbeing, renderer building-placement clearance, hover and wellbeing-bar visibility, keyboard mappings, arrow camera deltas, right-drag deltas, checkpoint population compatibility, route retargeting, navigation-audit output, and stress-sweep summaries. On 2026-11-01, the Windows Python 3.13 venv passed: 70 passed, with two PettingZoo old-API deprecation warnings. In the shared global Python install, an unrelated broken global pytest plugin may require `$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'`.

## Training and evaluation

Training commands used for the current checkpoints:

```powershell
python train.py --algorithm PPO --episodes 100 --seed 29 --days 30 --output models/ppo.pt
python train.py --algorithm MAPPO --episodes 100 --seed 23 --days 30 --output models/mappo.pt
```

Evaluation: `python evaluate.py --mode rule --episodes 5 --days 30 --seed 1001`

Learned evaluation: `python evaluate.py --mode learned --checkpoint models/mappo.pt --episodes 5 --days 30 --seed 1001`

PPO evaluation: `python evaluate.py --mode learned --checkpoint models/ppo.pt --episodes 5 --days 30 --seed 1001`

Training writes per-algorithm JSONL diagnostics to `logs/train_ppo.jsonl` or `logs/train_mappo.jsonl`; evaluation emits JSON with configuration, seeds, decision mode, reward, wellbeing, survival, employment, health, stress, social satisfaction, wealth and hunger incidents. Learned deployment and primary evaluation sample the categorical policy; use `--deterministic` for argmax diagnostics. Always use identical episode lengths, seeds and reward configuration when comparing. A small set of evaluation seeds is exploratory, not evidence of learning superiority or convergence. `python plot_results.py` writes `results/summary.csv` and `results/comparison.png` after all four result files are available.

Reward ablation workflow (run as a module so root-level `train.py` is importable): `python -m experiments.run_reward_ablation --algorithm MAPPO --component financial_stability --episodes 100 --eval-episodes 5 --days 30 --seed 23`. On 2026-11-01, this matched the primary MAPPO training seed (23), trained 100 episodes × 30 days with the financial-stability reward set to zero, and evaluated the full and ablated checkpoints on held-out seeds 1001–1005. Across those five episodes, full/ablated mean wellbeing was 83.70/82.86, health 94.99/96.64, stress 13.48/11.60, and social satisfaction 88.68/85.81; survival and employment were 100% for both. Reward totals are not directly comparable because one reward component is absent in the ablation. Artifacts are the ablated YAML, `models/ablations/mappo_without_financial_stability.pt`, `logs/replicates/ablation_mappo_seed23.jsonl`, and `results/ablations/mappo_{full,without_financial_stability}.json`. One training seed and one ablated component are exploratory, not robust causal evidence. The workflow's `--seed` controls ablation training; evaluation uses fixed seed 1001. Direct `python experiments/run_reward_ablation.py` is not supported because it fails to import root-level `train`.

Higher-food-price robustness run: compare policies with `--config experiments/configs/stress_food_price.yaml --seed 2001`. Three 30-day episodes per mode were evaluated on 2026-11-01. Mean survival was 0% random, 100% rule, 86.7% PPO, and 100% MAPPO. Raw results are stored under `results/economy_stress_*.json`. This small diagnostic set does not establish robustness.

Expanded paired-seed high-food-price check: run `python -m experiments.run_economy_sweep --episodes 10 --days 30 --seed 9001`. It evaluates Random, Rule, PPO, and MAPPO in the same environment seed range using `experiments/configs/stress_food_price.yaml`, saving each raw policy record plus `results/economy_stress_expanded/summary.json`. The 2026-11-01 run covered seeds 9001–9010: survival was 0% Random, 100% Rule, 97% PPO, and 98% MAPPO; mean wellbeing was 53.37, 66.80, 66.62, and 88.69, respectively. MAPPO mean health was 97.70 versus 46.79 for PPO; the PPO survival SD was 4.8 percentage points and MAPPO SD 4.2 points. This is a 10-environment-seed comparison of fixed checkpoints; it does not quantify training-seed variation or establish generalization.

Long-horizon viability check: both training-seed checkpoints of PPO/MAPPO ran three 60-day episodes at environment seeds 3001–3003; the rule baseline also ran those seeds. Raw outputs are `results/long_run_{rule,ppo,mappo}.json` for primary seeds 29/23 and `results/long_run_{ppo_seed129,mappo_seed123}.json` for replicate seeds 129/123. Survival was 100% rule, 80% primary PPO, 100% replicate PPO, and 100% for both MAPPO seeds. Primary PPO's per-seed outcomes were 80%, 90%, and 70%. Primary MAPPO averaged wellbeing 87.72, health 98.89, and stress 13.95; the rule baseline averaged wellbeing 65.66, health 100, and stress 67.45. This shows a PPO training-seed sensitivity signal rather than a consistent PPO failure. Four trained checkpoints and three environment seeds remain too few for a robust estimate.

## Clean Windows install and smoke commands

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest -q
python -m compileall -q ai_town train.py evaluate.py plot_results.py experiments
python -m ai_town.app --mode rule
python -m ai_town.app --mode learned --checkpoint models/ppo.pt
```

Interactive Windows QA on 2026-11-01 launched the saved MAPPO checkpoint. Play advanced the displayed clock from Day 1 00:00 to Day 1 04:00; citizen activities changed autonomously, salary earnings appeared, and the life timeline populated. Pause froze the clock; Town pulse and the visible Compare button opened their panels. The live QA exposed a frame-gating bug that stopped after its first tick; elapsed-time accumulation now schedules bounded updates at 1x/2x/5x/10x and has focused unit coverage. Expanded follow, reset, speed, policy-switch, and long-duration control checks are still outstanding.

## Reproducibility limits

Validation was performed on Windows 11 / Python 3.13 with CPU-only execution (no CUDA device). A fresh `.venv` editable installation passed `pip check`; its full test suite passed without disabling external pytest plugins. The shared global Python installation has an unrelated broken LangSmith pytest plugin and needed `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` for tests.

Current training and baseline comparisons are CPU runs on the configured Python/PyTorch stack. Report checkpoint metadata, environment YAML, training seed, evaluation seeds, episode length, and raw JSON. The main comparison uses five held-out seeds (1001–1005), outside training seeds, with 30 days per episode and the default configuration. Learned policies sample from their categorical distribution; Torch is seeded per evaluation episode. Repeating the five-seed MAPPO primary evaluation reproduced every episode record exactly. Primary/replicate PPO training seeds 29/129 averaged reward 32.59/36.01 and survival 94%/100%; MAPPO seeds 23/123 averaged reward 22.34/27.63 and survival 100%/100%. Primary/replicate MAPPO wellbeing averaged 85.85/86.71. These are descriptive results from two checkpoints per method and five evaluation seeds each, not evidence of convergence or broad superiority. Rerun every policy after changes to rewards, needs, navigation, or economy. Do not infer that higher episode reward alone means better citizen outcomes.

## Third independent training-seed replicates (2026-11-01)

To avoid overwriting primary checkpoints or their metric traces, the `train.py` CLI accepts `--metrics-output` for a separate trace. New 100-episode × 30-day CPU runs used:

```powershell
python train.py --algorithm PPO --episodes 100 --seed 229 --days 30 --output models/replicates/ppo_seed229.pt --metrics-output logs/replicates/train_ppo_seed229.jsonl
python train.py --algorithm MAPPO --episodes 100 --seed 223 --days 30 --output models/replicates/mappo_seed223.pt --metrics-output logs/replicates/train_mappo_seed223.jsonl
python evaluate.py --mode learned --checkpoint models/replicates/ppo_seed229.pt --episodes 5 --days 30 --seed 1001 --output results/ppo_seed229_eval.json
python evaluate.py --mode learned --checkpoint models/replicates/mappo_seed223.pt --episodes 5 --days 30 --seed 1001 --output results/mappo_seed223_eval.json
python evaluate.py --mode learned --checkpoint models/replicates/ppo_seed229.pt --episodes 3 --days 60 --seed 3004 --output results/long_run_ppo_seed229.json
python evaluate.py --mode learned --checkpoint models/replicates/mappo_seed223.pt --episodes 3 --days 60 --seed 3004 --output results/long_run_mappo_seed223.json
```

On five matched 30-day held-out environment seeds 1001–1005, PPO seed 229 averaged reward 37.02, wellbeing 74.15, and 100% survival/employment. MAPPO seed 223 averaged reward 31.34, wellbeing 89.06, and 100% survival/employment. For 60-day seeds 3004–3006, PPO survival was 90%, 100%, 100%; MAPPO survival was 100% on all three. These add a third independent training seed per algorithm, but the small sample remains exploratory and does not establish convergence or superiority.

`python -m pytest -q` passes 70 tests (two PettingZoo deprecated-API warnings). A 240-frame learned-MAPPO dummy-display run at 1280×800 reported mean frame time 16.62 ms, p95 17.98 ms, and 60.2 mean FPS; this includes the Pygame 60 FPS limiter and is a smoke-level observation, not a hardware-independent benchmark. The latest 180-frame run measured 59.3 FPS mean and 17.8 ms p95. `results/visual_final.png` shows the all-bars and Town Pulse view; `results/depth_layer_validation.png` verifies foreground depth composition; `results/placement_validation.png` verifies the current road/building layout.

## Navigation, population, and matched-seed follow-up (2026-11-01)

The 60-day navigation audit found a genuine crowded-entrance stall: a citizen could keep targeting an office access cell after another citizen occupied it. `Town._choose_destination` now reserves occupied cells and in-progress destinations, and `step_agent` retargets an occupied entrance. This preserves observations, action meanings, reward computation, and checkpoint formats, but it changes simulation path selection; older evaluation files therefore describe the pre-fix environment and should not be mixed with the new runs.

Three 60-day runs each for Rule, primary PPO, and primary MAPPO (environment seeds 6001–6003) are recorded in `results/navigation_{rule,ppo,mappo}.json`. They had zero duplicate-cell ticks and zero off-grid agent ticks. Mean no-progress-on-pending-route rates were 0.2% Rule, 4.1% PPO, and 1.5% MAPPO; worst per-episode consecutive no-progress streaks were 5, 29, and 25 ticks, respectively. This diagnostic counts a tick where an agent has a distinct outstanding target but its position does not change; some such ticks are ordinary crowd avoidance, so interpret it as a conservative mobility signal. The audited scenarios showed 100% Rule survival, 93.3% PPO survival, and 100% MAPPO survival. Raw wellbeing, hunger, cash, visited-cell, and route data are retained per episode.

The evaluator's `--population` sweep now works with saved MAPPO checkpoints below their training population: it retains the saved centralized critic shape for deserialization while inference uses the shared actor on the requested number of agents. Rule, PPO, and MAPPO each completed five 30-day episodes at populations 1, 5, and 10 (seeds 5001–5005). The matched means are in `results/post_navigation_fix/population_summary.csv`; raw episodes are in `results/post_navigation_fix/population_sweeps/`. Rule survived all episodes at all three sizes. MAPPO survival was 100%, 100%, and 98%; PPO survival was 100%, 84%, and 100%. This limited sample shows population sensitivity for PPO at five citizens and is not a population-generalization claim.

After the navigation fix, all baselines and six existing learned checkpoints were evaluated on the same 15 stochastic environment seeds (1001–1005 and 7001–7010), 30 days each. Each checkpoint still represents one of three training seeds per algorithm. Raw rows are in `results/post_navigation_fix/` and `results/post_navigation_fix/seeds_7001_7010/`; the reproducible summary command is `python -m experiments.summarize_seed_sweep`. The summary reports per-checkpoint means, standard deviations, and Student-t 95% intervals across environment seeds; these intervals do not measure variation across training seeds and do not establish convergence or algorithm superiority. Across the three PPO checkpoints, survival ranged from 98.0% to 98.7% and wellbeing means from 67.63 to 73.96. MAPPO survival ranged from 98.0% to 100%, with wellbeing means from 88.01 to 89.59. The rule baseline survived all 15 episodes, while random survived none; these remain descriptive outcomes under this configuration.

Performance commands:

```powershell
python -m experiments.profile_headless --mode learned --checkpoint models/mappo.pt --episodes 3 --days 30 --seed 8101 --output results/headless_mappo_profile_eval.json
$env:SDL_VIDEODRIVER = 'dummy'
python -m ai_town.app --mode learned --checkpoint models/mappo.pt --smoke-frames 240 --profile --screenshot results/visual_final.png --preview-analytics --preview-all-bars
```

The recorded CPU run evaluated 2,160 environment steps / 21,600 citizen decisions in 7.50 s (288 environment steps/s, 2,880 decisions/s). The Pygame smoke reported 60.2 mean FPS under its 60 FPS limiter. Both are local smoke measurements rather than hardware-independent benchmarks.

For a repeatable movement/occupancy audit, run `python -m experiments.audit_navigation --mode learned --checkpoint models/mappo.pt --episodes 3 --days 60 --seed 6001 --output results/navigation_mappo.json`; use `--mode rule` for the transparent rule baseline and `--checkpoint models/ppo.pt` for PPO. Reproduce the 1/5/10 population sweep and summary with `python -m experiments.run_population_sweep --episodes 5 --days 30 --seed 5001` and `python -m experiments.summarize_population_sweep`.

Interactive follow-up after these changes launched a separate `models/mappo.pt` window, leaving the existing minimized window untouched. Visible Play/Pause, Reset, all four speeds, Random/Rule/MAPPO mode cycling, citizen selection, Follow, Compare, Town Pulse collapse, and wellbeing bar toggles were observed. Mouse-wheel zoom-in/out worked. At 5× the learned-policy viewer completed one 30-day episode and auto-paused at Day 31 00:00; Reset returned to Day 1 00:00. Desktop keyboard injection did not change the visible state for Space, B, or Tab, so shortcuts and right-button drag are still unverified rather than claimed to work. The visible mouse controls remained available.

Research-facing viewer updates add a local HTML dashboard at `results/research_dashboard.html`; build it with `python -m experiments.research_dashboard`. It reads saved per-episode rows in `results/{random,rule,ppo,mappo}.json` and training traces in `logs/train_{ppo,mappo}.jsonl`. Evaluation tables include mean ± SD across environment seeds and checkpoint training-seed metadata; the dashboard labels these as distinct sources of variation. The Graphs control opens this file in the system browser. A click on the sound icon or `M` toggles the generated ambient loop and UI tones; if audio cannot initialize the viewer remains silent. Fine grass detail is fixed-seed, cached renderer art; shaded facades, raised curb lips, and painter-sorted layers provide a 2.5D depth effect without changing the navigation grid. Fresh 1280×800 captures of all four sections, all Data pages, and clinic/home inspectors are in `results/continuation_qa/final_audit_*.png` and `audit_*_final.png`. The text overlay pass found and corrected a clinic-inspector row collision, an events filter label, and footer/Places directory clipping. `compileall` passed; live Graphs activation remains unverified.

Continuation verification on 2026-11-01 used the project venv: focused app/town-info/visual/simulation/environment tests passed (53), the full suite passed (70), `compileall` passed for `ai_town`, `tests`, and `experiments`, and `pip check` found no broken requirements. Fresh dummy-display screenshots for Town, People, Places, and all five Data pages are under `results/continuation_qa/`. Visual review found and fixed overlapping People controls in selected-citizen mode; the refreshed `People_fixed.png` shows the identity card, controls, wellbeing bars, and timeline without collisions. The day-boundary snapshot regression verifies captured population, employment, recorded salary/spending/social events, and grocery stock; the app starts paused, so a full interactive day was not advanced during this verification.

The financial-stability ablation has a three-training-seed summary at `results/ablations/financial_stability_three_seed_summary.json`, reproducible with `python -m experiments.summarize_financial_ablation`. It averages the five held-out environment episodes within each checkpoint before comparing the three training-seed means (23, 123, 223). Keep the training seed as the unit (`n=3`): seed 223 shows substantial ablation sensitivity, so do not pool raw episodes or describe the output as a stable causal estimate.





