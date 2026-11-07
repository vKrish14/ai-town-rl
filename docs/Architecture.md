# Architecture

## Runtime boundaries

- `ai_town.simulation`: citizen state, building map, movement/pathfinding, needs, economic transitions, and event records. It imports no graphics or RL package.
- `ai_town.citizen`, `ai_town.buildings`, `ai_town.navigation`, `ai_town.economy`, `ai_town.rewards`: owned state and focused domain components used by the simulation engine.
- `ai_town.environment`: PettingZoo Parallel API, agent lifecycle, observations, and joint-step reward finalization from shared pre/post snapshots. It is headless.
- `ai_town.rl`: parameter-shared categorical actor, local critic for PPO, centralized global-state critic for MAPPO, action masking, GAE, checkpointing, and clipped PPO optimization.
- `ai_town.baselines`: explicitly labelled random and heuristic comparison policies.
- `train.py` / `evaluate.py`: independent headless collection, training, checkpoint and evaluation entry points.
- `ai_town.app`: Pygame presentation and user controls. It asks an injected policy for actions and advances the environment; it contains no reward or domain transition rules.
- `ai_town.audio`: synthesized ambient music and UI tones. The viewer may mute both; an unavailable mixer leaves the game silent and does not affect simulation.
- `experiments.research_dashboard`: creates a self-contained local HTML dashboard from saved per-episode evaluation JSON and PPO/MAPPO training JSONL. The viewer opens it in the system browser from the Graphs control; it is a separate research artifact, not a renderer dependency.
- `ai_town.town_info`: renderer-independent named building catalog, town identity/statistics, and type-aware inspection payloads. Unsupported model data is represented explicitly as unavailable rather than estimated.
- `ai_town.analytics`: end-of-day snapshots computed from simulation state and its recorded events. The app's Reports and trend views read these snapshots; they are not training diagnostics.
- `town_events` configuration supplies campaign dates, annual observances, and ambient safety-report rates. `Town` uses an independent seeded RNG for report generation. The named Police Officer is a regular policy-controlled citizen whose Work action targets the police station; incident reports still have no dispatch outcome, criminal action, reward, or citizen-state effect.
- `ai_town.visuals`: renderer-only pixel-inspired town illustration and citizen sprite construction. It consumes static grid coordinates and UI colors; it cannot alter walkability or simulation state. Presentation lots may be offset to avoid a drawn footprint overlapping a road; simulation building/action coordinates remain unchanged. Scaled map surfaces and directional citizen frames are cached.

## Data flow

The GUI or headless runner obtains per-citizen observations from the environment, selects one action per agent through a policy interface, and passes the actions to `step`. The environment delegates state changes to the simulation, returns rewards and termination flags, and exposes concatenated observations as the critic's global state. The renderer reads simulation state for presentation.

The renderer uses the expanded 50×32 navigation grid. `visuals.town_surface()` draws an authored tile-based illustration with seeded fine grass texture and caches it by display tile size; `visuals.citizen_sprite()` caches four-way, four-frame citizen sprites, including a distinct police uniform and hair/accessory variants derived from citizen identity/personality. `app` positions those assets using interpolated simulation positions and render-time animation, with foreground depth sorting and contact shadows giving the top-down scene a layered 2.5D look. Camera pan, zoom, lighting, water shimmer, audio cues, and UI are presentation-only. No external art assets are used.

Crowd navigation avoids cells already occupied by another citizen and cells reserved as an in-progress destination. If an entrance becomes occupied after a route was selected, the citizen chooses an available access point for the same activity. This only changes path selection; the action set, action masks, observations, reward, and decision policy remain the same. Saved MAPPO critics have a population-dependent input/output shape; `load_policy_checkpoint` reconstructs that critic using the checkpoint's recorded training population while its shared local actor can infer over a different environment population.

Citizen wellbeing bars use `visuals.citizen_wellbeing()`, a weighted average of health, energy, `100-hunger`, happiness, `100-stress`, and social satisfaction after clamping each input to [0, 100]. Default weights are all 1 in `config/default.yaml` under `visualization.wellbeing_weights`; partial overrides replace only named weights. The inspector shows the numeric score and tier. The map shows exactly one combined slim bar for a hovered/selected citizen, or for every citizen when `B`/the visible toggle is enabled. Alpha and score animate toward current state using render delta time. This score does not enter observations, reward, transitions, or policy decisions.

## Town records and viewer navigation

`Town._append_event` stores sequence ID, tick/time, category, citizen/building references, a short description, event-specific data, and reward when applicable. Decision events are retained for the citizen timeline; the town event feed filters them to avoid per-tick noise. Economics, employment, clinic use, social interactions, deaths, and restocks are recorded where the corresponding transition occurs. There is no separate meal event: eating food carried by a citizen is not represented as a purchase. Daily snapshots are created after the clock crosses a day boundary and derive population, wellbeing, employment, wealth, social satisfaction, recorded income/spending and grocery stock from state and that day's event records.

The viewer campaign is open-ended while normal training/evaluation remain finite-horizon. Willowbrook Safety Office and Maple Town Hall are simulation buildings; the patrol prop is renderer-only. Named assault, burglary, robbery, theft, and vandalism descriptions are separately seeded story reports; there is no suspect action, damage, property/state change, incident dispatch, or reward effect. Observances use `calendar_start` and advance one calendar date per simulated day. The Social page reports last-30-day reports and a rate per 1,000 resident-days using stored daily living-population exposure plus the current partial day.

Rendered lots and simulation destinations now share one road-cleared coordinate source. Hit-testing, camera focus, and selected-building outlines use the same visible footprint bounds. A home is associated with its resident via the citizen's `home`; office assignment to an individual workplace, capacity, restaurant inventory, and persistent visitor counts are not simulation state. Place inspectors therefore show town employment and entrance-area occupancy with an explanatory scope, and mark missing capacity/inventory data as not modeled. The Town/People/Places/Data navigation, directories, inspectors, and Reports panel are presentation-only. Event/snapshot collection does not select actions or change rewards. 

## Decisions

Use Pygame CE for a modest, portable 2D renderer and PyTorch for an inspectable MAPPO implementation. Keep the initial town and its assets procedural to avoid asset licensing and packaging complexity. See [Engineering Decisions and Improvements](Engineering%20Decisions%20and%20Improvements.md).
