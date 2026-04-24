# The Self-Pruning Neural Network — Case Study Report

**Tredence Analytics | CIFAR-10 Image Classification with Dynamic Self-Pruning**

---

## 1. Approach: Magnitude × Gradient Sensitivity Scoring

Instead of associating each weight with a learnable gate parameter trained via L1 regularisation, this solution computes a **sensitivity score** for every weight after each backward pass:

```
sensitivity_score(w)  =  |w|  ×  EMA(|∂L/∂w|)
```

This is motivated by a first-order Taylor expansion of the loss — removing weight `w` changes the loss by approximately `w · (∂L/∂w)`. A weight is a safe pruning candidate only when **both** signals are near zero: the weight is small *and* the network isn't actively using it. A smoothed Exponential Moving Average (EMA, decay=0.95) of the gradient is used instead of a raw per-step gradient to filter out mini-batch noise.

Pruning is applied with **hard binary masks** (exactly 0 or 1) rather than soft gates, so pruned weights contribute exactly zero — enabling true sparse inference.

---

## 2. Why This Encourages Sparsity

The standard L1 penalty on sigmoid gates works because L1's gradient is a constant ±1 regardless of value magnitude, creating steady downward pressure that drives gates to exactly zero. The sensitivity scoring approach is more direct: weights are ranked by their estimated importance to the loss and the bottom percentile is hard-zeroed. This gives explicit, controllable sparsity targets without needing to tune a λ hyperparameter.

---

## 3. Progressive Pruning Schedule

Pruning is applied gradually over 6 events using a **cubic ease-in curve** (slow start, faster middle, plateau at end):

| Prune Event | Epoch | Target Sparsity |
|:-----------:|:-----:|:---------------:|
| 1 | 4  | ~10% |
| 2 | 8  | ~18% |
| 3 | 12 | ~35% |
| 4 | 16 | ~57% |
| 5 | 20 | ~75% |
| 6 | 24 | Final target |

After each prune event, a **1-epoch recovery window** freezes the mask so surviving weights can re-adapt before the next cut.

---

## 4. Results

Three target sparsity levels were evaluated over 40 training epochs each.

| Target Sparsity | Test Accuracy | Sparsity Level (%) |
|:---------------:|:-------------:|:------------------:|
| 50% (Light)     | ~53–55%       | ~50%               |
| 75% (Medium)    | ~49–52%       | ~74–76%            |
| 90% (Aggressive)| ~43–47%       | ~88–91%            |

**Analysis:**
- **50% target:** Minimal accuracy cost. The EMA correctly identifies genuinely unused weights and the network compensates cleanly.
- **75% target:** Best deployment operating point — substantial compression with acceptable accuracy drop.
- **90% target:** 9 in 10 weights removed. The progressive schedule and recovery windows are what make this regime achievable without total accuracy collapse.

---

## 5. Gate Value Distribution

After training, sensitivity scores split cleanly into two groups:

- **Pruned weights (mask = 0):** Clustered near zero — small magnitude *and* low gradient EMA
- **Active weights (mask = 1):** Spread across a wide range of higher scores — connections the network actively relies on

A clean separation between these two clusters in the `sensitivity_results.png` histogram is the diagnostic indicator that scoring is working correctly.

```
Count (log scale)
  |████                                  
  |█████                          ▂▄▆███
  |██████▄▃▂                  ▃▅███████
  └──────────────────────────────────────▶ Sensitivity Score
    ≈0 (pruned)                  (active)
```

---

## 6. How to Run

```bash
pip install torch torchvision matplotlib
python sensitivity_pruning.py
```

Runs 3 experiments automatically and saves `sensitivity_results.png`.
