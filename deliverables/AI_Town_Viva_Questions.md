# AI Town viva questions and answers

Answers describe the current implementation. Treat evaluation values as exploratory and population-10 unless stated otherwise.

## 1. What problem does AI Town study?

How multiple citizens can choose daily activities in a coupled, resource-limited town while balancing personal needs. It is a synthetic simulation, not an empirical model of human behaviour.

## 2. Why use a multi-agent formulation?

Each citizen has an individual observation, action, reward and trajectory. Citizens share food and vacancies, and nearby social actions can affect peers, so state transitions couple across agents.

## 3. Is the formal model an MDP or POSG?

From one citizen’s perspective it is partially observed. A Markov-game/POSG description is more precise: joint actions affect shared state, each agent sees a local observation and receives its own reward.

## 4. What does N mean?

The current default viewer has 17 citizens. Retained main evaluation files used population 10, which must be reported with those results.

## 5. What is the observation dimension?

28 float32 features per citizen, in the order described on slide 6: six needs, cash, employment, time features, position, shared stock/vacancies, food, peer distance, four traits and eight pending-action indicators.

## 6. Are all observations truly bounded between zero and one?

The declared Gymnasium Box is [0,1] with shape 28, and most fields are capped or normalized. However the expanded map still divides x by 35 and y by 23. Coordinates can exceed those denominators, so the implementation can emit values above 1. This is a known interface mismatch.

## 7. What does the MAPPO critic observe?

The code concatenates the 28-value local observation for each citizen in fixed order. At 17 agents the input has 476 values and the critic outputs 17 values. This is a concatenated observation vector, not every hidden physical state variable.

## 8. What are the eight actions?

Work, Sleep, Eat, Socialise, Recreation, Shop, Exercise and Find Employment. They select high-level activities. A* navigation supplies routes to destinations.

## 9. How does action masking work?

While travelling to a committed activity, TownParallelEnv exposes only the pending action. The actor masks other logits to −1e9 before constructing a categorical distribution. Outside this transit state all eight are available; transaction logic handles unavailable purchases/jobs.

## 10. Is navigation learned?

No. A* is deterministic support. The policy chooses the life activity; path planning finds a route to its destination.

## 11. How does an Eat action work?

At an eligible destination, if the citizen carries food, food decreases by one and hunger decreases by 35. Ordinary hourly drift then adds 0.7 hunger and subtracts 1.4 energy, .35 social and adds .1 stress. Without carried food the restaurant path can buy a meal if funds allow.

## 12. How do shared interactions occur in a step?

The environment accepts a joint action dictionary, then applies Town.step_agent in citizen order. Shared social effects can therefore affect peers. Batch rewards are finalized using snapshots taken before the transitions and the resulting post-transition state.

## 13. What is the wellbeing measure?

(health + energy + 100−hunger + happiness + 100−stress + social)/6. It is a state measure used in reports and the delta term of reward.

## 14. What are the reward components?

For living citizens: .05 times wellbeing change; −.30 times excess hunger above 50 divided by 50; +.08 conditional sustainable work; .15 times capped-at-$60 cash change divided by 60; +.25 newly employed. The final reward for a dead citizen is only −.50.

## 15. Why cap financial stability?

The money delta is computed from min(money,60), so reward cannot grow without bound from accumulating large cash balances.

## 16. What is sustainable work exactly?

The citizen is employed, money is below $50, selected Work, and arrived at the destination during the step.

## 17. Give a numerical reward calculation.

For the constructed Eat example, wellbeing rises from 63.33 to 68.74. ΔW≈5.408; hunger ends below 50; no job, money or death term applies. Reward=.05×5.408≈+.2704. This is a calculation from transition rules, not a measured episode.

## 18. What is the RL objective?

Maximize expected discounted return J(θ)=E[Σ γ^t r_t], with γ=.99 per simulated hour and finite training/evaluation episodes.

## 19. What does PPO’s ratio represent?

ρ_t is the new policy probability for the recorded action divided by the old/behavior policy probability. It indicates how much the action probability changed.

## 20. What does clipping do?

The clipped surrogate removes additional incentive when the probability ratio moves beyond [1−ε,1+ε] in a direction that would improve the surrogate. Here ε=.2. It discourages large updates but is not a strict trust-region constraint.

## 21. How is the actor loss implemented?

The code minimizes the negative mean clipped surrogate minus .01 times mean categorical entropy. Advantage is normalized across the rollout with a small epsilon.

## 22. What is GAE?

A discounted sum of temporal-difference residuals: δ_t=r_t+γ(1−d_t)V_{t+1}−V_t, and Â_t=δ_t+γλ(1−d_t)Â_{t+1}. λ=.95 balances shorter low-variance TD information with longer-horizon returns.

## 23. How does terminal handling work?

The configured episode ends at 30 days with terminations true and truncations false. GAE uses zero next value at the last rollout step and a nonterminal mask. It does not bootstrap beyond the finite horizon.

## 24. How does shared parameterization work?

One actor network maps every citizen’s local observation to eight logits. Shared weights reduce duplication; different observations and personality features yield different action distributions. Rewards remain individual.

## 25. How does MAPPO differ from the project’s shared PPO?

Both use a shared decentralized actor. Shared PPO has a local value critic over one 28-feature observation. MAPPO uses a centralized value critic over concatenated citizen observations and outputs one value per citizen.

## 26. What happens at deployment?

The app loads the saved actor through the policy interface and asks it to choose actions. The MAPPO critic is not needed for decentralized action selection. The renderer reads environment state and events without defining rewards or transitions.

## 27. What hyperparameters were used in the documented training runs?

Learning rate 3e−4, γ=.99, λ=.95, clip=.2, entropy coefficient .01, value coefficient .5, four update epochs, minibatches up to 64 and gradient norm clipping at .5. Documented retained runs trained 100 episodes×30 days. The CLI default is 20 episodes.

## 28. How many samples are in a 30-day training episode?

720 environment steps at 24 steps per day. At the current population 17, rollout observation shape is [720,17,28] and critic state shape [720,476].

## 29. What is the dataset?

The environment generates RL experience online. There is no bundled external human dataset or single canonical experience dataset. Per-episode evaluation JSON, aggregate CSVs, training JSONL and checkpoints with metadata are retained.

## 30. How should the primary results be described?

The post-navigation-fix comparison uses 15 environment seeds per checkpoint, 30 days, population 10. PPO three-seed final wellbeing ranges 67.63–73.96, survival 98.0–98.7%. MAPPO ranges 88.01–89.59 and 98.0–100%. This is descriptive evidence from three training seeds, not proof of universal superiority or convergence.

## 31. What did the high-price stress test show?

Across matched seeds 9001–9010 for fixed checkpoints, survival was 0% random, 100% rule, 97% PPO and 98% MAPPO. Mean wellbeing was 53.37, 66.80, 66.62 and 88.69. It measures environment-seed variation for the tested checkpoints and one stress config, not training-seed uncertainty.

## 32. What did the financial-stability ablation show?

Three MAPPO training seeds were trained with the financial-stability reward component removed and evaluated on five common environment seeds. Full checkpoints had 100% survival. Ablated survival was 100%,100%,20%; mean wellbeing change was −19.83 with between-training-seed SD 17.70. Seed 223 was highly sensitive, so this is exploratory.

## 33. Why is population drift important?

Checkpoint evaluations and summary tables were created for population 10, before the current 17-citizen layout. Population affects critic dimensions, environment interactions and outcomes. Those numbers cannot be presented as current-town results.

## 34. What does crime mean in the current simulation?

Crime reports are stochastic narrative events. They do not model an offender action, damage, theft, dispatch, reward or citizen-state change. The police officer is a real policy-controlled citizen, but dispatch mechanics are not implemented.

## 35. What is the most immediate technical improvement?

Correct position normalization and the declared observation bounds for the expanded map, then rerun seeded evaluation for the 17-citizen town.

## 36. How can a result be reproduced?

Use the recorded config, training checkpoint and metadata, algorithm, training seed, evaluation seed set, horizon and stochastic/deterministic mode. `docs/Testing and Experiments.md` gives commands and source result files.