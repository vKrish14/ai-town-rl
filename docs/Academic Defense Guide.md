# Demonstration and viva guide

## Demonstration sequence

1. Start the town in rule mode, unpause it, and show the configurable clock, buildings, citizen activity, needs, and movement.
2. Select a citizen. Scroll the life timeline to show activity decisions and rewards, then show the money balance and town statistics.
3. Load PPO and MAPPO separately. Explain that the actor is shared and decentralized at execution, while MAPPO uses a centralized value critic during training.
4. Run headless evaluation with the same seeds and 30-day horizon for all four policy types. Open the raw JSON before the summary plot.
5. Show the YAML, training JSONL, and checkpoint metadata for the run. Explain the reward components and the continuing death penalty.

Use `python -m ai_town.app --mode learned --checkpoint models/ppo.pt` or substitute `models/mappo.pt`. The viewer is paused on launch; press Space to run. Press A for town statistics and use the mouse wheel over the timeline to inspect older events.

## Questions to be ready for

- **Why MAPPO?** It keeps a parameter-shared decentralized actor while allowing a centralized critic to use ordered joint observations during training.
- **What makes this genuine RL?** PPO updates the categorical actor from collected environment rollouts using GAE and a clipped policy objective. Baselines are explicitly named, separately implemented, and never loaded as trained checkpoints.
- **What does the policy control?** One of eight life activities. A* resolves movement to the chosen destination and action masks prevent activity changes while in transit.
- **How reproducible are results?** Training and evaluation record seeds/configuration; evaluations retain per-seed JSON. Repeated training seeds and larger confidence samples are still needed for research-strength claims.
- **What remains uncertain?** One training seed and 100 episodes per checkpoint are still limited evidence. Do not claim convergence or general policy superiority from these runs. Navigation lacks a reservation system, and visual polish remains a work in progress.

Always distinguish a training trace, a single held-out comparison, and a replicated statistical result. Present only values that match the checked-in result files and checkpoint metadata.
