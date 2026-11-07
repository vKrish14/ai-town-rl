# Engineering Decisions and Improvements

## Initial implementation

- **Original proposal:** use Python, PettingZoo, PyTorch, and a separate 2D renderer.
- **Implementation:** Gymnasium/PettingZoo Parallel API, PyTorch shared actor plus centralized critic, and Pygame CE procedural graphics.
- **Reason:** this matches the requested stack while keeping training headless and making the policy math inspectable.
- **Advantages:** low integration overhead, portable Windows dependencies, deterministic procedural art, clear module boundaries.
- **Disadvantages:** hand-built renderer and MAPPO require careful validation; current content is visually modest versus a finished commercial indie game.
- **Validation:** headless suite and dummy-video smoke runs pass; interactive desktop visual QA remains outstanding.

## Fresh repository decision

At initial inspection of this checkout, only `AGENTS.md` was present. The implementation was built from that project constitution. The current audit should be updated if a fuller project checkout becomes available; do not overwrite any later-discovered user work.

## Navigation commitment and action mask

- **Original issue:** policy actions could change each hour during long walks, reversing routes and causing citizens to fail to reach destinations.
- **Change:** keep the chosen activity while in transit, expose it in the observation, and provide a one-action mask until arrival.
- **Reason:** this preserves navigation as support for RL activity decisions and makes the decision/execution boundary explicit to PPO.
- **Trade-offs:** choices occur less frequently during travel; the per-agent mask is part of the environment contract and must be supplied during training and deployment.
- **Validation:** added mask/commitment tests and reran PettingZoo API checks.

## Transaction and pathfinding separation

- **Change:** moved citizen state, map data, A*, and atomic shared-economy transactions into `citizen.py`, `buildings.py`, `navigation.py`, and `economy.py`.
- **Reason:** allow simulation rules, pathfinding, and economic invariants to be understood and tested independently.
- **Validation:** navigation and transaction regression tests plus the full headless suite.

## Long-horizon life viability

- **Observed failure:** initial 30-day policy comparisons ended with zero survival across random, rule, and trained policies. Citizens spent too much of each hour in cross-town transit, work/sleep imposed excessive per-hour need costs, and finite food stock had no replenishment.
- **Change:** movement throughput, work/sleep costs, hunger drift, shop restocking, and default vacancies were tuned and made configurable. The viewer interpolates between discrete movement states.
- **Reason:** make daily needs and travel compatible with a 24-tick day and provide a viable economic loop without choosing citizen actions for the policy.
- **Trade-offs:** default availability is more generous (10 vacancies and daily supply replenishment), and the faster grid movement is an abstraction rather than a walking-speed model.
- **Validation at the time of tuning:** three 30-day rule baseline seeds after the change showed all citizens alive and employed with no high-hunger incidents. Subsequent learned-policy validation is recorded in the reproducible stochastic deployment section below; the baseline result itself is not evidence of learned-policy quality.

## Reproducible stochastic deployment and corrected evaluation

- **Observed issue:** deterministic MAPPO argmax inference collapsed on the held-out seed sample, while categorical policy sampling yielded viable behavior. Evaluation also seeded the town and NumPy but not Torch's categorical sampler, making stochastic results non-repeatable.
- **Change:** learned deployment and primary evaluation sample policy actions by default; `--deterministic` remains available for argmax diagnostics. Torch is seeded independently for each evaluation episode, and result JSON records the decision mode and checkpoint metadata.
- **Reason:** deployment should preserve the learned stochastic policy while comparisons remain reproducible.
- **Validation:** five-seed MAPPO output reproduced exactly across two runs. On default held-out seeds 1001–1005, primary/replicate PPO (training seeds 29/129) averaged reward 32.59/36.01 and survival 94%/100%; primary/replicate MAPPO (seeds 23/123) averaged reward 22.34/27.63 and survival 100%/100%. Across those runs, MAPPO average wellbeing was 85.85/86.71. Two training seeds per algorithm and five evaluation seeds per checkpoint are descriptive evidence, not proof of convergence or general superiority.

## Additional validation and workflow findings

- **Economic stress:** three 30-day evaluation episodes per mode at food price 7 (seed range 2001–2003) produced survival averages of 0% random, 100% rule, 86.7% PPO, and 100% MAPPO. Raw rows are retained in `results/economy_stress_*.json`; the small set is diagnostic only.
- **Reward-ablation command:** direct script execution could not import root-level `train.py`; running as `python -m experiments.run_reward_ablation` succeeds. A matched MAPPO financial-stability ablation trained 100 episodes × 30 days at training seed 23 and evaluated the full/ablated checkpoints on seeds 1001–1005. Both had 100% survival/employment; full/ablated wellbeing was 83.70/82.86, health 94.99/96.64, stress 13.48/11.60, and social satisfaction 88.68/85.81. Reward totals are not comparable after a reward term is removed. One training seed/component remains exploratory, not robust causal evidence. Evaluation uses fixed seed 1001 independently of the training `--seed`.
- **Environment verification:** a clean Python 3.13 Windows venv installed editable dependencies and passed `pip check`; the current suite passes 27 tests (two PettingZoo deprecated-API warnings). Training and evaluation were CPU-only.
- **Plot export portability:** the first clean-environment chart run selected TkAgg and failed because the installed Python had no usable Tcl/Tk runtime. `plot_results.py` now selects Matplotlib's noninteractive Agg backend before importing pyplot. Re-running the comparison generator wrote both CSV and PNG successfully.
- **Interactive visual review:** the first desktop-control attempt was interrupted by the computer-use surface's Escape stop. The user restored access later in the session, allowing the live deployment checks documented below.
- **Live playback and controls:** once desktop access returned, learned MAPPO playback revealed the simulation tick counter was incorrectly used as a frame gate, so 1x advanced once and stalled. Replaced it with a bounded elapsed-time accumulator, added a pure helper with validation and tests, replaced the unsupported play/pause glyph, and added a visible Compare button. Live desktop verification showed time progression, autonomous activity changes, salary events, populated citizen timeline, pause, statistics and comparison. Updated suite: 27 passed; two PettingZoo deprecation warnings. Longer desktop control sessions remain to be checked.
- **Procedural art pass:** added mown park lawns, meadow tufts, a sandy pond bank with water ripples, storefront awnings, a clinic cross, a town-hall clock, clearer citizen silhouettes, and warm lamp halos. These are deterministic renderer-only shapes; they do not affect observations, transitions, navigation, or rewards. Dummy-render preview confirms the details remain legible at the default window size. The art remains a lightweight procedural style, not a finished commercial sprite set.
- **60-day robustness:** on environment seeds 3001–3003, the primary MAPPO checkpoint, its independent training-seed replicate, the PPO replicate, and rule baseline all had 100% survival. Primary PPO survival was 80%, 90%, and 70%. This indicates training-seed sensitivity rather than consistent PPO failure. Preserve raw runs in `results/long_run_*.json`; increase training/evaluation seeds before drawing conclusions or retuning.

## Cached pixel-inspired renderer pass

- **Change:** moved original tile-map and citizen-sprite construction into `ai_town.visuals`, added cached map surfaces and four-direction/four-frame characters, landscaped fixed-coordinate routes and neighbourhoods, added an inspector portrait and render-time pond shimmer, and smoothed/clamped camera controls.
- **Reason:** improve world readability and character identity while keeping the simulation's 36×24 coordinate contract and renderer-only boundary intact.
- **Trade-offs:** artwork remains procedural and intentionally modest; the map is rebuilt only for a newly requested display tile size, and sprite frames are cached by citizen/direction/frame/size. No third-party asset or new runtime dependency was introduced.
- **Validation:** renderer surface/cache/direction/walk-cycle tests and the full suite pass; a learned-MAPPO dummy-display screenshot was inspected at 1280×800. Interactive live-window verification of the new controls is still outstanding because the existing desktop window was minimized and was left untouched.

## Combined wellbeing indicator

- **Change:** added a renderer-only weighted wellbeing calculation with configurable weights, inverted hunger/stress, value clamping, explicit critical/low/moderate/good thresholds, interpolated bar color, and a single floating bar per shown citizen. Bars fade and interpolate with frame time; selected/hovered is the default and the B key or panel toggle shows all citizens.
- **Reason:** make overall life quality readable in-world without crowding the town with six separate need bars or feeding presentation data back into the RL system.
- **Trade-offs:** default weights are equal and the calculation mirrors `Town._wellbeing`; it is a display indicator only. Thresholds are presentation bands, not reward cutoffs.
- **Validation:** unit tests cover inversion, bounds, weighting, validation, threshold boundaries, and color values; full suite and learned-policy renderer smoke pass. The headless screenshot shows the numeric score in the inspector and the selected citizen's bar.

## Visual refinement pass (2026-11-01)

- **Change:** replaced per-cell grass fills with deterministic, overlapping low-contrast ground patches and fine grass marks; moved presentation-only building footprints off the boulevard; shifted the residential frontage clear of the north-south street; increased citizen sprite width; raised the default camera zoom; moved wellbeing bars into a crisp UI layer above night lighting; and made Town Pulse compact and collapsible from its header.
- **Reason:** remove the checkerboard effect and road/building collisions while improving map use of the viewport and keeping citizen state legible.
- **Boundary:** presentation lots do not alter `BUILDINGS`, agent destinations, pathfinding, environment transitions, rewards, learned policies, checkpoints, analytics, or headless imports.
- **Validation:** dummy-display application smoke generated and visually inspected `results/visual_refined_final.png`; subsequent regression runs are documented below. The user's minimized interactive window was left untouched, so live-window interaction was not rechecked in that pass.

## Occupied-entrance retargeting (2026-11-01)

- **Observed issue:** long navigation audits found citizens preserving routes into office access cells that had become occupied. A 60-day PPO episode had a 878-tick consecutive no-progress streak at an occupied work entrance.
- **Change:** destination selection now avoids currently occupied cells and destinations reserved by other in-progress travelers; a moving citizen retargets to another available access point if its chosen entrance becomes occupied.
- **Reason:** prevent indefinite route stalls while preserving the policy's requested life activity.
- **Boundary:** this changes deterministic path selection in the simulation. It does not change observations, action space/masks, rewards, state transitions after arrival, the PPO/MAPPO algorithms, or checkpoint data. Existing evaluation files from before this change are historical and are not directly comparable to post-fix results.
- **Validation:** regression test covers occupied entrance retargeting. In three 60-day seeds per policy, no duplicate-cell or off-grid ticks occurred; worst no-progress streaks fell to 5 Rule, 29 PPO, and 25 MAPPO ticks. Matched 30-day evaluations were rerun for baselines and all six saved checkpoints on 15 environment seeds; raw data and per-checkpoint Student-t intervals are retained under `results/post_navigation_fix/`.

## Checkpoint actor population compatibility

- **Observed issue:** evaluating a saved MAPPO actor at a different environment population failed while loading the centralized critic, whose dimensions correctly encode its training population.
- **Change:** checkpoint deserialization reconstructs the critic from the saved population metadata; inference then uses the shared actor on any supported evaluator population.
- **Boundary:** no policy weights or checkpoint format changed; the critic remains the original trained critic and is not used to choose deployment actions.
- **Validation:** MAPPO checkpoint population compatibility regression and matched five-seed Rule/PPO/MAPPO 30-day runs at populations 1, 5, and 10 pass. PPO survival was 84% at population 5 in this sample, while MAPPO survival was 98–100%. These runs identify sensitivity; they do not establish population generalization.

## Post-refinement runtime and controls

- **Validation:** a fresh learned-MAPPO window verified Play/Pause, Reset, 1×/2×/5×/10× speed cycling, visible Random/Rule/MAPPO policy labels and cycling, citizen selection, follow-camera, Compare, Town Pulse collapse, and all-bars versus selected-only display. At 5×, the saved MAPPO policy advanced through the complete 30-day episode and auto-paused at Day 31 00:00; Reset returned to Day 1 00:00. The existing minimized window remained untouched. Mouse-wheel zoom-in/out worked. Keyboard shortcut injection did not visibly change controls through the desktop automation helper, so keyboard shortcuts and right-button drag remain unverified.
- **Performance:** a 240-frame 1280×800 renderer smoke measured 60.2 mean FPS and 17.98 ms p95 including the 60 FPS limiter. A 3×30-day CPU MAPPO evaluation ran at 288 environment steps/s and 2,880 citizen decisions/s. These are local smoke observations, not portable benchmarks.
- **UI follow-up:** the live Compare panel exposed binary floating-point formatting noise. Comparison values now round to one decimal place, covered by a formatting unit test.

## Procedural identity variations

- **Change:** added deterministic facade color combinations, a few house dormers/chimneys, and per-citizen skin-tone variation while keeping clothing color, hair/accessory selection, four facing directions, and movement frames in the cached sprite pipeline.
- **Reason:** reduce repeated identical housing and strengthen citizen identity without external asset dependencies.
- **Validation:** regenerated and inspected `results/visual_final.png`; nearest-neighbor scaling and cached-surface behavior remain covered by visual tests. The overall art is still intentionally procedural and does not yet match the detail of a professional hand-authored sprite set.

## Foreground depth composition (2026-11-01)

- **Observed issue:** the cached town surface flattened houses, trees, and citizens into a fixed order, so citizens could never pass behind foreground scenery.
- **Change:** terrain remains a cached base surface; houses and trees are cached as separate cropped layers and composited in foot-position order with citizens. Citizen UI badges and wellbeing bars remain above the world layers.
- **Boundary:** this is renderer-only; no simulation data, policy decisions, checkpoints, controls, or headless imports change.
- **Placement correction:** a new geometric regression exposed overlapping restaurant/grocery/town-hall footprints and a clinic façade near the north homes. Presentation lots are now centralized, the restaurant sits in the clear service-row gap, and the clinic frontage moved below the homes with a shorter planted edge above the boulevard. The regression checks all building visual footprints against roads, map bounds, and one another.
- **Wellbeing display coverage:** the pointer hit region and selected/hovered/all visibility rule are now explicit renderer helpers covered by boundary and mode tests.
- **Control coverage:** moved keyboard action names, arrow-key camera deltas, and incremental right-drag deltas into renderer-only helpers used by `app.py`; tests cover shortcuts and drag start/move/release. Physical desktop key injection remains unverified because the automation surface did not deliver those events visibly.
- **Validation:** cache/depth, placement, hover/bar-mode, and control-mapping regressions pass; 59 tests pass. A learned-MAPPO 90-frame screenshot smoke at 1280×800 was visually inspected at `results/placement_validation.png`; a separate 180-frame render profile measured 59.3 FPS mean and 17.8 ms p95 under the frame limiter.

## Research-facing game presentation (2026-11-01)

- **Change:** added deterministic fine grass marks, shaded building sides and raised curb highlights for a stronger 2.5D look, strengthened the dark forest HUD and active section/policy states, exposed saved Random/Rule/PPO/MAPPO policies, and added synthesized ambient music and UI chimes with a shared mute control. The Graphs button builds and opens a self-contained local HTML dashboard from saved evaluation episode rows and PPO/MAPPO JSONL training metrics.
- **Reason:** make the viewer read as a complete town game while giving a university audience a direct path from play to reproducible research evidence.
- **Boundary:** terrain texture, UI, audio, policy selection, and report launching do not modify observations, rewards, transitions, or checkpoint contents. The dashboard labels sources, seeds, and run conditions and keeps training diagnostics separate from evaluation outcomes.
- **Validation:** compileall and headless screenshots of all UI sections plus clinic/home inspectors completed. Visual audit caught a missing sound-font glyph, a clinic-inspector row collision, an overlong Events filter label, and footer/Places clipping; these were corrected and recaptured under `results/continuation_qa/`. The local audio mixer initialized and accepted a synthesized chime. The low-battery overlay cleared after power was connected. A live Graphs click and audible physical speaker-button check remain unverified.

## Open-ended campaign and civic event feed (2026-11-01)

- **Change:** the interactive viewer continues after the standard 30-day horizon while training/evaluation keep finite episodes. Added a renderer-only Willowbrook Safety Office, a small animated beacon patrol prop, configurable annual calendar observances, ambient safety incident reports, and rolling 30-day count/rate display in Social. Event feed filters include Safety and Calendar.
- **Model boundary:** civic facilities are presentation landmarks; named crime entries are story reports without offender actions, damage, theft, state changes, or reward effects. The crime RNG is separately seeded. UI history is bounded during infinite runs.
- **Validation:** compileall passed; 32 elapsed-day paired environment smoke remained active beyond the 30-day training horizon and citizen positions/health/money/hunger matched between safety telemetry enabled/disabled. It reached Day 33 / Feb 2, 2026 with two ambient reports overall and one inside the rolling 30-day window. Headless screenshots for Social, Events, and police-station inspector were regenerated; filter labels and inspector status text were wrapped/shortened after visual audit. The patrol prop and safety-stat panel are visible in `results/continuation_qa/`. The updated MAPPO desktop demo is running; actual audible playback and Graphs launch were not exercised through the desktop UI.

## East-quarter expansion and named incident cards (2026-11-01)

- **Change:** expanded the grid to 50×32 and the default roster to 16 named residents. Extended the east/south roads and added six homes plus offices, a market, restaurant, and clinic for the new district. Events now show red STORY cards for named assault, home burglary, robbery, theft, and vandalism reports. Music level was raised modestly.
- **Compatibility:** individual observation size (28) and action set (8) are unchanged. The shared actor can infer over the larger roster; MAPPO inference retains the checkpoint's critic shape, while a new 16-resident training run builds a 16-agent critic. Retained training/evaluation artifacts still use the former 10-resident, 36×24 setup and must be regenerated before reporting outcomes for this map.
- **Validation:** compileall passed; the saved 10-resident MAPPO checkpoint produced 16 valid actions in a 16-resident 50×32 environment, with 16 unique names, 16 homes, 3 food stores, and 31 named facilities. Seed 10 reproducibly creates a story-only assault and the configured New Year's observance. Headless Town and Events screenshots were inspected; citizen/home labels, red card text, and event disclaimer fit within the viewport. The audio mixer played the 0.20-volume loop and stopped it on mute.

## District spacing and architectural silhouettes (2026-11-01)

- **Change:** regrouped the visible homes and services into separated residential, civic, and east-side blocks; repositioned the riverside park and safety station; replaced dotted stepping stones with continuous footpaths; removed a second layer of garden strips that duplicated each home's yard; and gave offices a flat roof and repeated glazing. Building labels now appear on hover or selection so the default town view stays open.
- **Road clearance correction:** the lake now ends west of the east boulevard, and the east park was moved and resized to keep a grass verge from both roads. Park hit-testing uses the same rendered footprint.
- **Boundary:** all placements and facade changes are renderer-only. Simulation destinations, navigation, rewards, policies, and resident state remain unchanged.
- **Validation:** `compileall` and a MAPPO Town-view screenshot smoke completed successfully; the resulting layout was visually inspected at 1280×800.

## Policy-controlled police resident and road-cleared parcels (2026-11-01)

- **Change:** added Amara Okeke as the 17th citizen, gave her a distinct police uniform, assigned her a home and the police station as her workplace, and selected her in the starting citizen inspector. The officer still selects from the shared eight-action RL policy; Work routes to the police station. Crime reports remain narrative-only and do not trigger dispatch. Moved the rendered and simulation building lots onto one aligned coordinate layout. The visible footprint helper now drives building hit-testing, camera framing, and the selected outline.
- **Reason:** show the authority as a real, inspectable citizen and remove drift between building clicks, outlines, simulation destinations, and the roads drawn beneath them.
- **Research impact:** the default roster is now 17, and building destination coordinates changed. Actor observation/action sizes are unchanged, but town transitions and population-dependent critic shape differ; retained checkpoints/evaluation rows remain historical and must not be described as results for this town.
- **Validation:** compiled the modules; constructed a 17-agent environment; confirmed Amara's Work target is a legal police-station entrance; audited all 33 rendered footprints against roads, the lake, map edges, and one another; checked fresh MAPPO Town, People, and selected-station screenshots; and passed 46 focused tests across visuals, town information, simulation, and evaluation.
- **Selection and lot alignment:** standardized every structure's outline/hit-test footprint and every park's six-by-five-cell lot; moved a south residential lot to clear the park. The station's patrol vehicle is a renderer-only prop and has a clearer sedan silhouette; Amara remains the actual citizen agent. Refreshed Town and selected-station captures in `results/continuation_qa/`.
- **Outline correction:** the first shared footprint included the full garden apron, leaving conspicuous empty space in front of each building. Selection and hit testing now follow each rendered building silhouette (including chimney/office roof differences), ending at the façade instead of the garden edge. Verified tight outlines on the station and a house in fresh MAPPO captures.
- **World alignment correction:** town art/building sprites were rasterized at `int(tile)` while the world origin, citizen positions, clicks, and selection outlines used fractional `tile`, causing position-dependent drift across the map. The app now uses the same integer tile scale for every world-space element. Verified aligned house and station captures and reran 29 visual/town-info tests.

## Slower default viewer clock (2026-11-01)

- **Change:** changed the playback base interval from 0.2 to 1.0 real seconds per simulated hour. 2×, 5× and 10× remain proportional. Citizen movement interpolation now occupies most of the selected simulated-hour interval at each speed.
- **Reason:** at the previous default, five town hours elapsed per real second, making the 1× label feel too fast to watch.
- **Boundary:** affects only viewer playback pacing; headless environment stepping, training, evaluation horizons, and policy behavior per simulated hour are unchanged.
- **Validation:** focused playback-clock tests verify one hour per second at 1×, proportional higher speed, fractional time accumulation, and bounded catch-up.
