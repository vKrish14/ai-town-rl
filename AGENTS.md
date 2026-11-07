# AI Town: Project Constitution

These rules govern implementation in this repository. Keep this file concise; put detailed setup and user documentation in the README and `docs/`.

## Vision and objectives

Build a cozy, polished, observable town simulation where citizens' consequential life choices come from learned multi-agent policies. Balance a watchable game experience with a reproducible, explainable RL research project. Prioritize a complete, testable vertical slice over breadth.

## Architecture

- Keep simulation/state transitions, PettingZoo-compatible RL environment, training/evaluation, rendering/UI, and analytics as separate modules with explicit interfaces.
- The environment must run headlessly. The renderer consumes simulation state and events; it must not define transitions, rewards, or policy decisions. Deployment loads a saved policy and requests actions through the policy interface.
- Keep configuration, assets, checkpoints, logs, and experiment outputs out of source modules. Make maps, economy, needs, reward weights, and seeds configurable.
- Use small, cohesive modules and documented data contracts. Add dependencies only when they serve a clear requirement; record substantial architecture or algorithm deviations in `docs/Engineering Decisions and Improvements.md` with rationale and validation.

## Genuine RL

- Citizens' life decisions must be selected by learned policies in deployment. Navigation/pathfinding and low-level animation may be deterministic support systems, but must not silently choose life activities.
- Preserve an auditable multi-agent formulation: documented observation/action spaces, transition and termination semantics, reward components, policy checkpoint/version metadata, and reproducible training/evaluation entry points.
- Keep training separate from inference and graphics. Maintain transparent baselines and evaluate learned policies against them; inspect reward exploitation and test reward components/ablations before making learning claims.
- Never present scripted behavior, random actions, or a demonstration policy as trained RL. Label fallback/demo modes clearly in the UI and documentation.

## Visual and gameplay standards

- Aim for a cohesive, original, cozy top-down town: readable silhouettes, restrained natural palette, clear roads/building functions, distinguishable citizens, smooth movement, and a camera/UI that keeps the town dominant.
- Prefer original procedural or properly licensed assets; record third-party asset attribution. Do not copy proprietary game assets.
- Provide legible citizen inspection and town-level trends without crowding the play view. Distinguish simulation outcomes from training diagnostics.
- Visual polish must not obscure agent state, decisions, or causal outcomes.

## Code quality and modularity

- Use type hints for public interfaces, clear names, small functions, explicit validation, deterministic seeding, and structured logging. Avoid hidden global state and duplicated domain logic.
- Keep rendering and optional analytics dependencies out of headless environment imports. Make modules independently understandable and replaceable (especially assets, policies, maps, and renderers).
- Document non-obvious reward, observation, action, and time-step choices next to their implementation and in the technical documentation.

## Testing and evidence

- Add focused automated tests with each feature: state bounds/transitions, transactions/employment, navigation, observation/action validity, reward components, episode lifecycle, seeded determinism, checkpoint round-trips, and renderer/integration behavior as applicable.
- Validate the headless environment independently of the GUI. Include smoke-level end-to-end checks for runnable milestones.
- Do not claim a feature works unless it was run or the testing limitation is stated. Never fabricate training curves, metrics, comparisons, convergence, or experimental conclusions; retain seeds, configs, checkpoints, and raw metrics for reported results.

## Performance and operations

- Target ordinary Windows hardware and CPU operation first; CUDA is optional. Keep networks and rendering effects modest and configurable.
- Avoid blocking training in the render loop. Use bounded update rates, batched/vectorized work where useful, and profile before optimizing. Preserve a responsive UI and viable headless throughput.
- Provide exact PowerShell commands for installation, tests, training, evaluation, and launch. Checkpointing and fresh-town policy deployment must be reproducible.

## Documentation

- Keep README focused on project purpose, setup, and common commands. Maintain architecture/RL formulation, configuration, experiment methodology, asset attribution, and engineering decisions in focused `docs/` pages.
- Update documentation and configuration examples with behavior or interface changes. Record experiment conditions and limitations alongside results.
