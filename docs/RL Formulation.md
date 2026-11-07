# RL formulation

## Environment

The town is a cooperative, partially observed multi-agent process with one policy-controlled citizen per agent. One environment step represents one simulated hour. An episode ends after the configured number of days. `citizen_i` IDs are stable in index order. The environment accepts a joint action dictionary, applies all agent transitions, then scores each citizen from shared pre-step and post-step snapshots so social effects do not bias rewards by agent iteration order.

## Observation

Each actor receives 28 float32 values in `[0,1]`: health, energy, hunger, happiness, stress, social satisfaction (six needs normalized by 100); money capped at $200 and normalized; employment flag; hour normalized by day length; day divided by episode length; x/y divided by map bounds; least stocked shop fraction; vacancies fraction; personal food capped at five; mean Manhattan distance to other citizens scaled by 50; four personality traits; and an eight-value one-hot pending activity. The critic receives the ordered concatenation of all individual observations (476 values at the current default population of 17; 280 at population 10). No future state is exposed.

## Actions

Discrete actions: 0 work, 1 sleep, 2 eat, 3 socialise, 4 recreation, 5 shop, 6 exercise, 7 find employment. Work by the named Police Officer citizen routes to the police station; all other workers route to offices. The officer uses the same shared policy and reward formulation; incident dispatch is not modeled. Navigation is a deterministic A* helper and is not a life decision. An activity remains committed while the citizen travels to its destination so hourly policy changes cannot cause endless route reversal. The environment supplies an action mask that exposes only the committed activity in transit and all choices at a destination. Rendering and pathfinding remain outside the policy.

## Rewards and transitions

The reward has separately reported components: change in average citizen wellbeing (health, energy, inverse hunger, happiness, inverse stress and social satisfaction), a starvation penalty above hunger 50, a survival penalty on the death transition and every remaining timestep, conditional sustainable-work bonus, capped financial-stability change, and one-time employment acquisition bonus. The continuing death penalty prevents a dead agent from becoming reward-neutral for the rest of an episode. Wealth has a capped marginal utility and does not grow reward indefinitely. Component weights are in `config/default.yaml` and can be overridden when constructing the environment.

For citizen (i), define wellbeing

\[
W_i(s_t)=\frac{H_i+E_i+(100-G_i)+J_i+(100-X_i)+S_i}{6},
\]

where (H,E,G,J,X,S\) denote health, energy, hunger, happiness, stress, and social satisfaction. The implemented reward is

\[
r_{i,t}=w_W\Delta W_i-w_G\frac{\max(0,G_i-50)}{50}-w_D\mathbf{1}[H_i=0]
+w_{work}\mathbf{1}[\text{employed, low cash, working at office}]
+w_F\frac{\min(M_{i,t},60)-\min(M_{i,t-1},60)}{60}
+w_J\mathbf{1}[\text{newly employed}].
\]

The weights (w_W,w_G,w_D,w_{work},w_F,w_J) are configurable in YAML. The capped money delta makes funds useful while preventing unlimited wealth accumulation from producing unlimited reward.

Movement advances up to the configured number of grid cells per simulated hour along an A* route (default eight); the renderer interpolates between simulation positions. The activity remains committed during travel and the action mask prevents policy changes from reversing the route. Need drift occurs hourly. Activities apply their effects while at their destination. Work pays two half-salary installments per simulated day when the citizen is working at the office. Eating uses carried food, or purchases a meal only at the restaurant; groceries require a separate shopping action. Shared food stock and vacancies constrain purchases and employment; shops restock daily up to configured capacity.

## Learning

The actor is parameter-shared and decentralized at inference. `SharedPPOPolicy` uses an individual-observation value critic; `MAPPOPolicy` uses a centralized critic mapping concatenated global observations to one value per agent. Both use categorical PPO clipping, GAE, entropy regularization, value regression and gradient clipping. The actor applies the environment's activity mask while a citizen is navigating. Training, deployment, and primary evaluation sample the categorical policy. Use `evaluate.py --deterministic` for an argmax diagnostic; each result records its decision mode. Training and graphics run separately. Baselines are labelled and not represented as learned.

The clipped policy objective is

\[
L^{clip}(\theta)=\mathbb{E}_t\left[\min\left(\rho_t(\theta)\hat A_t,
\operatorname{clip}(\rho_t(\theta),1-\epsilon,1+\epsilon)\hat A_t\right)\right],
\quad \rho_t(\theta)=\frac{\pi_\theta(a_t|o_t)}{\pi_{old}(a_t|o_t)}.
\]

GAE uses

\[
\delta_t=r_t+\gamma(1-d_t)V(s_{t+1})-V(s_t),\qquad
\hat A_t=\delta_t+\gamma\lambda(1-d_t)\hat A_{t+1}.
\]

In MAPPO, the decentralized actor uses only (o_{i,t}) at inference. During training, the centralized value network receives the ordered concatenation (s_t=[o_{1,t},\dots,o_{N,t}]) and predicts one value per citizen. Shared PPO instead predicts values from each local observation.

## Known gaps

The current route planner replans around occupied cells but has no reservation system, so dense traffic can still cause delays. The action abstraction applies activity effects hourly and has not been calibrated against a target real-time economy. Checkpoints include algorithm and training configuration metadata, but full experiment manifests and confidence intervals need more consistent reporting. API and optimizer tests are implemented; policy quality and convergence are not established by the available training budget.
