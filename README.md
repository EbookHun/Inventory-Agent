```markdown

# 📦 Sequential Supply Chain Agent (2-Month Lag)

This repository contains a reinforcement learning solution for retail inventory replenishment. Rather than a simple immediate-feedback environment, this simulator enforces a **two-month logistics arrival lag**, requiring the agent to make sequential, long-term hedging decisions to avoid the bullwhip effect.

## 🎯 Problem Framing (Markov Decision Process)

| MDP Component | Definition | Business Logic |
| :--- | :--- | :--- |
| **State (S)** | `s_t = (I_t, P1_t, P2_t)` | A flattened 3D state tracking current inventory (`I`), stock arriving next month (`P1`), and stock arriving in two months (`P2`). Space size: 396 states. |
| **Action (A)** | `a_t` ∈ `[0, 1, 2, 3, 4, 5]` | The quantity of new inventory ordered today. |
| **Reward (R)** | `R_t = (Rev) - (Cost) - (Hold) - (Pen)` | Net Margin: Revenue ($50) minus unit cost ($20), holding cost ($2), and stockout penalties ($15). |
| **Transition (P)**| `I_{t+1} = max(0, I_t + P1_t - D_t)` | Demand (`D`) is uniformly stochastic (0-4). Pipeline shifts deterministically: `P1_{t+1} = P2_t` and `P2_{t+1} = a_t`. |
| **Horizon (T)** | `T = 24` | Evaluated over 24-month financial cycles. |

## 🧠 Policy Designs
1. **Baseline (Naive Rule-Based):** Orders a fixed batch whenever on-hand inventory drops below 3. It acts myopically, ignoring the 2-month pipeline, leading to massive overstock holding costs.
2. **Learning Agent (Tabular Q-Learning):** An off-policy TD control algorithm. It learns the temporal delay, ordering smaller, smoothed batches to rely on in-transit goods.

## 🚀 Execution & Reproducibility

To run the simulation and generate the learning curve:
```bash
pip install numpy matplotlib
python inventory_agent.py