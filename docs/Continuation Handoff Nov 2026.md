# AI Town continuation handoff (1 Nov 2026)

## Objective

Continue the existing AI Town project in `F:\RL Project` through a polished, usable visual and information experience. Preserve the simulation and RL system: do not rebuild the application, alter transition/reward/policy logic, replace learned MAPPO/PPO checkpoints, or compromise headless operation. The user's current request also calls out fixing houses/buildings placed on roads. The governing project rules are in `AGENTS.md`.

The latest detailed user brief is the pasted file titled **“AI Town: Interactive World Identity and Complete UI Upgrade”** (most recent attachment from the user). Earlier active requests additionally require natural pixel-art terrain, readable citizens and directional movement, balanced town layout, single wellbeing bars that track citizens, compact/collapsible Town Pulse, ambient/day-night polish, improved UI sections and reports, and correcting bad placements.

## Work completed in the current continuation

- Added configured town identity and distinct names for homes, workplaces, shops, clinic and public spaces in `config/default.yaml`.
- Added renderer-only building placement in `ai_town/buildings.py`; moved visual lots off the roads without changing simulation building/state coordinates. Added footprint/bounds/road/collision checks in `tests/test_visuals.py`.
- Added renderer-independent building/catalog/statistics/inspection data in `ai_town/town_info.py`. It reports explicit unavailable/unmodeled values instead of inventing capacity, office assignments, restaurant stock, or visitors.
- Added daily state snapshots in `ai_town/analytics.py` and structured business/life events in `ai_town/simulation.py` (employment, wages, purchases, clinic use, social interactions, deaths, grocery restocks). Data collection is intended not to affect RNG, rewards, actions, or transitions.
- Added configurable building signs and named map labels in `ai_town/visuals.py`.
- Added Town / People / Places / Data navigation and People/Places directories, building inspection, Economy/Social/Events/RL Lab/Reports views, search, map selection/focus, labels toggle, and preview flags in `ai_town/app.py`.
- Added tests in `tests/test_town_info.py`, `tests/test_visuals.py`, and `tests/test_app.py`.
- Visual review artifacts already inspected: `results/places_directory_validation.png`, `results/building_inspector_validation.png`, `results/economy_view_validation.png`, `results/rl_lab_validation.png`. Town overview needs a fresh screenshot after a recent text-wrap patch.

## Immediate continuation steps

1. Inspect `git status` (this workspace may not be a Git repository) and the changed files before editing. Worktree state is shared; do not discard local changes.
2. Rerun focused tests after the latest `pygame.font.init()` test fix:

   ```powershell
   .\.venv\Scripts\python.exe -m pytest tests/test_app.py tests/test_town_info.py tests/test_visuals.py tests/test_simulation.py tests/test_environment.py -q
   ```

   Then run the complete test suite. Use the project venv because global pytest has a broken `idna` plugin. Do not run training as part of ordinary regression validation.
3. Run `compileall` on `ai_town`, `tests`, `experiments`, and `pip check` in the venv.
4. Launch the actual application and inspect a fresh screenshot; inspect `--preview-section town` and the data/people/place previews. Fix visual defects before claiming completion. Check the app's actual launch command in `README.md`; preview CLI switches are `--preview-section`, `--preview-data`, and `--preview-building`.
5. Exercise/verify one daily rollover and the relevant in-app interactions. Unit tests do not establish that all OS mouse/keyboard interactions work end to end.
6. Review each new event field against actual transitions. The current event feed intentionally omits generic decision-per-tick noise. Eating is not yet a dedicated event; if added, emit it only when food consumption/meal actually occurs. Do not infer employment loss, capacity, visits, friendship, or treasury when the simulation does not track them.
7. Update README and focused docs (architecture, testing/experiments, implementation roadmap, engineering decisions) to explain the renderer/data split, event and daily snapshot semantics, UI navigation, launch commands, tests and known limits.
8. Preserve and report limitations candidly: map labels/building signs, resident search, actual workplace assignment/capacity, visitors, friendships, seasons/weather, and some interactions may not be modeled. Validate no unintended sim/RL/checkpoint behavior changes.

## Files to inspect first

- `ai_town/app.py`
- `ai_town/town_info.py`
- `ai_town/analytics.py`
- `ai_town/simulation.py`
- `ai_town/visuals.py`
- `ai_town/buildings.py`
- `config/default.yaml`
- `tests/test_app.py`, `tests/test_town_info.py`, `tests/test_visuals.py`
- README and `docs/`

## Ablation evidence already collected

An additional MAPPO `financial_stability` reward-component ablation run completed for training seeds 123 and 223 (100 episodes × 30 days; five held-out evaluation episodes on seeds 1001–1005). Raw outputs, checkpoints, per-run evaluations, and the two-seed summary are under `results/ablations/financial_stability_replicates/`. Existing seed-23 paired data are in `results/ablations/mappo_full.json` and `results/ablations/mappo_without_financial_stability.json`.

The additional seeds show strong seed sensitivity: seed 223 ablated performance had very low health and survival, whereas the full models remained strong. Do **not** present this as a stable causal estimate from only one training seed or blend raw episodes incorrectly. Next, build a small reproducible summarizer that aggregates the three training-seed-level means (23, 123, 223), preserves per-seed results and evaluation conditions, and clearly labels n=3 and the sensitivity. No need to retrain these runs.

## Acceptance bar

Keep the user's existing town recognizable and functional; town should dominate the viewport; natural grass must not read as a checkerboard; citizens and their single wellbeing bars must remain readable/follow correctly; structures must not occupy roads; UI sections must show actual simulation data; headless imports/tests and all checkpoint metadata/compatibility must remain intact. Finish with tests, inspected screenshots, docs, and a concise summary including concrete evidence and any remaining limitations.
