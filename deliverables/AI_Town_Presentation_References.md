# AI Town references

## Reinforcement learning and multi-agent methods

1. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). “Proximal Policy Optimization Algorithms.” arXiv:1707.06347. https://arxiv.org/abs/1707.06347
2. Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2016). “High-Dimensional Continuous Control Using Generalized Advantage Estimation.” ICLR 2016. https://arxiv.org/abs/1506.02438
3. Yu, C., Velu, A., Vinitsky, E., Gao, J., Wang, Y., Bayen, A., & Wu, Y. (2022). “The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games.” Advances in Neural Information Processing Systems 35, Datasets and Benchmarks Track. https://proceedings.neurips.cc/paper_files/paper/2022/hash/9c1535a02f0ce079433344e14d910597-Abstract.html
4. Terry, J. K., Black, B., Grammel, N., Jayakumar, M., Hari, A., Sullivan, R., Santos, L., Dieffendahl, C., Horsch, C., Perez-Vicente, R., Williams, N. L., Lokesh, Y., & Ravi, P. (2021). “PettingZoo: Gym for Multi-Agent Reinforcement Learning.” Advances in Neural Information Processing Systems 34. https://proceedings.neurips.cc/paper/2021/hash/803f7c4c3ff61b71be53a0c803bfb57f-Abstract.html

## Software references

5. Farama Foundation. Gymnasium API, fundamental spaces. https://gymnasium.farama.org/v1.2.1/api/spaces/fundamental/
6. Paszke, A. et al. (2019). “PyTorch: An Imperative Style, High-Performance Deep Learning Library.” Advances in Neural Information Processing Systems 32. https://proceedings.neurips.cc/paper_files/paper/2019/file/bdbca288fee7f92f2bfa9f7012727740-Abstract.html

## Project source of truth

7. Current implementation: AGENTS.md, README.md, config/default.yaml, ai_town/simulation.py, ai_town/environment.py, ai_town/rewards.py, ai_town/rl.py, train.py, evaluate.py, and ai_town/app.py.
8. Evaluation comparison: results/post_navigation_fix/summary.csv and raw runs in results/post_navigation_fix/.
9. High-food-price evaluation: results/economy_stress_expanded/summary.json, config experiments/configs/stress_food_price.yaml.
10. Reward ablation: results/ablations/financial_stability_three_seed_summary.json and per-seed records under results/ablations/.

No external human-behaviour dataset is included. The town generates synthetic experience through seeded simulator rollouts.
