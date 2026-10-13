# Top-N Recommender Systems: Implicit ALS vs. SLIM

A from-scratch Python implementation and comparison of two classic Top-N recommendation algorithms on an implicit-feedback dataset.

| Model | Paper |
|-------|-------|
| **Implicit Feedback ALS** (weighted matrix factorization) | Hu, Koren & Volinsky, *Collaborative Filtering for Implicit Feedback Datasets*, ICDM 2008 |
| **SLIM** (Sparse Linear Methods) | Ning & Karypis, *SLIM: Sparse Linear Methods for Top-N Recommender Systems*, ICDM 2011 |

## How it works

**Dataset** – A synthetic implicit-feedback matrix (300 users x 200 items, ~8% density). Users belong to 5 interest groups, each preferring a block of items, plus some random noise interactions. Values are interaction counts (e.g. views), not ratings. The data is generated inside the code, so no dataset file is needed.

**Evaluation** – Leave-one-out: one interacted item per user is hidden, and the model must rank it inside the user's Top-10 list. Metrics:
- **Hit Rate@10** – fraction of users whose hidden item appears in the Top-10.
- **ARHR@10** – Average Reciprocal Hit Rank; rewards ranking the hidden item higher.

**Implicit ALS** – Binary preference `p_ui` with confidence `c_ui = 1 + alpha * r_ui` (alpha = 40), 20 latent factors, 15 ALS iterations.

**SLIM** – Learns a sparse item-item matrix `W` (score = `R * W`) with ElasticNet regression, non-negativity constraint and zero diagonal (`w_jj = 0`).

## Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

## Results

| Method | Hit Rate@10 | ARHR@10 |
|--------|-------------|---------|
| Implicit ALS [Paper 1] | 0.3200 | 0.0941 |
| SLIM [Paper 2] | **0.3533** | **0.1060** |

- SLIM achieved the higher Hit Rate and ARHR on this dataset.
- The learned SLIM matrix is **84.12% sparse** (6,351 non-zero coefficients out of 40,000).

Full program output:

![Program output](screenshots/output.png)

## Project structure

```
├── main.py            # Data generation, both models, evaluation
├── README.md          # This file
├── requirements.txt   # Dependencies
└── screenshots/
    └── output.png     # Program output
```

## Notes

Results vary slightly if the random seed (`np.random.seed(42)`) is changed. The data is synthetic, so results show the algorithms' behavior, not real-world performance.
