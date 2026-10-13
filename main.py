"""
=============================================================================
Top-N Recommender Systems: Implicit Feedback ALS vs. SLIM
=============================================================================

Implementation based on two base research papers:

  [1] Y. Hu, Y. Koren, and C. Volinsky,
      "Collaborative Filtering for Implicit Feedback Datasets,"
      ICDM 2008.
      -> Implemented as: Weighted Matrix Factorization with confidence
         weighting, trained using Alternating Least Squares (ALS).

  [2] X. Ning and G. Karypis,
      "SLIM: Sparse Linear Methods for Top-N Recommender Systems,"
      ICDM 2011.
      -> Implemented as: Sparse item-item coefficient matrix learned via
         ElasticNet regression (L1 + L2 regularization) with non-negativity.

Both models are trained on the same implicit-feedback dataset and evaluated
using Hit Rate@10 and ARHR@10, so their Top-N recommendation quality can
be compared directly.

Requirements: numpy, scipy, scikit-learn
    pip install numpy scipy scikit-learn
=============================================================================
"""

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import ElasticNet
import warnings

warnings.filterwarnings("ignore")
np.random.seed(42)


# =========================================================================
# STEP 1: DATASET
# =========================================================================
def generate_implicit_dataset(n_users=300, n_items=200, n_groups=5):
    """
    Creates a synthetic implicit-feedback dataset.

    Implicit feedback means we only observe ACTIONS (clicks, views, purchases)
    and never explicit star ratings - exactly the setting described in Paper [1].

    Users are placed into interest groups, and each group prefers a certain
    block of items. This creates learnable preference patterns.

    Values in the matrix = interaction COUNTS (e.g. number of views),
    not ratings. Paper [1] converts these counts into confidence values.
    """
    R = np.zeros((n_users, n_items))
    items_per_group = n_items // n_groups

    for u in range(n_users):
        group = u % n_groups
        start = group * items_per_group
        end = start + items_per_group

        # Main interest: items from the user's own group
        n_main = np.random.randint(8, 20)
        chosen = np.random.choice(range(start, end), size=n_main, replace=False)
        for i in chosen:
            R[u, i] = np.random.randint(1, 6)   # interaction count 1-5

        # Some random noise interactions outside the group
        n_noise = np.random.randint(1, 5)
        noise = np.random.choice(n_items, size=n_noise, replace=False)
        for i in noise:
            if R[u, i] == 0:
                R[u, i] = 1

    return R


def leave_one_out_split(R):
    """
    Standard Top-N evaluation protocol: for each user, hide ONE interacted
    item as the test item and train on the rest.

    At test time we check whether the model ranks that hidden item inside
    the user's Top-10 recommendation list.
    """
    R_train = R.copy()
    test_items = {}

    for u in range(R.shape[0]):
        interacted = np.where(R[u] > 0)[0]
        if len(interacted) < 3:
            continue
        held_out = np.random.choice(interacted)
        R_train[u, held_out] = 0
        test_items[u] = held_out

    return R_train, test_items


# =========================================================================
# STEP 2: PAPER [1] - IMPLICIT FEEDBACK ALS (Hu, Koren & Volinsky, 2008)
# =========================================================================
class ImplicitALS:
    """
    Weighted Matrix Factorization for implicit feedback.

    Key ideas taken from Paper [1]:

      1. PREFERENCE (p_ui): binary. 1 if the user interacted with the item
         at all, 0 otherwise. The raw count does NOT mean "how much" they
         like it.

      2. CONFIDENCE (c_ui = 1 + alpha * r_ui): how much we TRUST that
         preference. More interactions -> higher confidence. A single click
         might be accidental, so it gets low confidence.

      3. All zero entries are still treated as observations (preference = 0)
         but with the LOWEST confidence, which solves the missing-negative-
         feedback problem in implicit datasets.

      4. Optimization by Alternating Least Squares: hold item factors fixed
         and solve for user factors in closed form, then swap, and repeat.
    """

    def __init__(self, n_factors=20, alpha=40, reg=0.1, n_iters=15):
        self.n_factors = n_factors
        self.alpha = alpha      # confidence scaling from the paper
        self.reg = reg          # regularization
        self.n_iters = n_iters

    def fit(self, R):
        n_users, n_items = R.shape

        # P = binary preference matrix, C = confidence weights
        P = (R > 0).astype(float)
        C = 1.0 + self.alpha * R

        # Randomly initialize latent factor matrices
        self.X = np.random.normal(0, 0.01, (n_users, self.n_factors))  # users
        self.Y = np.random.normal(0, 0.01, (n_items, self.n_factors))  # items

        I = np.eye(self.n_factors)

        for iteration in range(self.n_iters):
            # ---- Solve for user factors (item factors fixed) ----
            YtY = self.Y.T @ self.Y
            for u in range(n_users):
                Cu = C[u]
                # (Y^T Cu Y + reg*I) x_u = Y^T Cu p_u
                Y_weighted = self.Y * (Cu - 1)[:, np.newaxis]
                A = YtY + self.Y.T @ Y_weighted + self.reg * I
                b = self.Y.T @ (Cu * P[u])
                self.X[u] = np.linalg.solve(A, b)

            # ---- Solve for item factors (user factors fixed) ----
            XtX = self.X.T @ self.X
            for i in range(n_items):
                Ci = C[:, i]
                X_weighted = self.X * (Ci - 1)[:, np.newaxis]
                A = XtX + self.X.T @ X_weighted + self.reg * I
                b = self.X.T @ (Ci * P[:, i])
                self.Y[i] = np.linalg.solve(A, b)

            if (iteration + 1) % 5 == 0:
                print(f"    ALS iteration {iteration + 1}/{self.n_iters} complete")

        return self

    def predict_all(self):
        """Predicted preference scores for every user-item pair."""
        return self.X @ self.Y.T


# =========================================================================
# STEP 3: PAPER [2] - SLIM (Ning & Karypis, 2011)
# =========================================================================
class SLIM:
    """
    Sparse LInear Method for Top-N recommendation.

    Key ideas taken from Paper [2]:

      1. The recommendation score is a LINEAR combination of the items a
         user has already interacted with:   score = R * W

      2. W is a learned ITEM-ITEM coefficient matrix. Unlike traditional
         item-based CF, W is not a fixed similarity measure (cosine etc.) -
         it is LEARNED from the data.

      3. Each column of W is learned independently by regressing item j
         against all other items, using ElasticNet (L1 + L2 penalty):
            - The L1 penalty forces SPARSITY, keeping only the strongest
              item relationships and discarding weak/noisy ones.
            - The L2 penalty prevents overfitting.

      4. Constraints from the paper:
            - w_jj = 0  (an item must not recommend itself)
            - W >= 0    (coefficients are non-negative, so they represent
                         positive relationships between items)
    """

    def __init__(self, alpha=0.001, l1_ratio=0.1):
        self.alpha = alpha          # overall regularization strength
        self.l1_ratio = l1_ratio    # balance between L1 (sparse) and L2

    def fit(self, R):
        n_users, n_items = R.shape
        R_binary = (R > 0).astype(float)
        self.W = np.zeros((n_items, n_items))

        model = ElasticNet(
            alpha=self.alpha,
            l1_ratio=self.l1_ratio,
            positive=True,          # non-negativity constraint from paper
            fit_intercept=False,
            max_iter=200,
            selection="random",
        )

        for j in range(n_items):
            # Target: column j (did users interact with item j?)
            target = R_binary[:, j].copy()

            # Temporarily zero out column j so item j cannot predict itself
            # This enforces the w_jj = 0 constraint from the paper
            col_backup = R_binary[:, j].copy()
            R_binary[:, j] = 0

            model.fit(R_binary, target)
            self.W[:, j] = model.coef_

            R_binary[:, j] = col_backup   # restore

            if (j + 1) % 50 == 0:
                print(f"    SLIM: learned coefficients for {j + 1}/{n_items} items")

        # Report how sparse the learned matrix is (a key claim of the paper)
        total = self.W.size
        nonzero = np.count_nonzero(self.W)
        self.sparsity = 100.0 * (1 - nonzero / total)
        print(f"    Learned W matrix sparsity: {self.sparsity:.2f}% zeros "
              f"({nonzero} non-zero coefficients out of {total})")

        return self

    def predict_all(self, R):
        """score = R * W  (the linear model from the paper)"""
        return (R > 0).astype(float) @ self.W


# =========================================================================
# STEP 4: EVALUATION (Top-N metrics)
# =========================================================================
def evaluate(scores, R_train, test_items, N=10):
    """
    Hit Rate@N: fraction of users whose hidden test item appears in their
                Top-N recommendation list.
    ARHR:       Average Reciprocal Hit Rank - rewards ranking the correct
                item nearer the top of the list. (Used in the SLIM paper.)
    """
    hits = 0
    arhr = 0.0
    n_evaluated = 0

    for u, true_item in test_items.items():
        user_scores = scores[u].copy()

        # Never recommend items the user has already interacted with
        user_scores[R_train[u] > 0] = -np.inf

        top_n = np.argsort(-user_scores)[:N]

        if true_item in top_n:
            hits += 1
            rank = int(np.where(top_n == true_item)[0][0]) + 1
            arhr += 1.0 / rank

        n_evaluated += 1

    return {
        "hit_rate": hits / n_evaluated,
        "arhr": arhr / n_evaluated,
        "users_evaluated": n_evaluated,
    }


# =========================================================================
# MAIN
# =========================================================================
def main():
    print("=" * 74)
    print("TOP-N RECOMMENDER SYSTEMS - IMPLEMENTATION OF TWO BASE PAPERS")
    print("=" * 74)

    # ---------------- Dataset ----------------
    print("\n[1] Preparing implicit-feedback dataset")
    R = generate_implicit_dataset(n_users=300, n_items=200)
    R_train, test_items = leave_one_out_split(R)

    density = 100.0 * np.count_nonzero(R) / R.size
    print(f"    Users: {R.shape[0]}, Items: {R.shape[1]}")
    print(f"    Total interactions: {np.count_nonzero(R)}")
    print(f"    Matrix density: {density:.2f}%  (sparse, as in real datasets)")
    print(f"    Test set: 1 held-out item per user "
          f"({len(test_items)} users evaluated)")

    # ---------------- Model 1: Implicit ALS ----------------
    print("\n[2] Training Model 1: Implicit Feedback ALS")
    print("    Based on: Hu, Koren & Volinsky (ICDM 2008)")
    print("    Confidence weighting: c_ui = 1 + alpha * r_ui, with alpha = 40")
    als = ImplicitALS(n_factors=20, alpha=40, reg=0.1, n_iters=15)
    als.fit(R_train)
    als_scores = als.predict_all()
    als_results = evaluate(als_scores, R_train, test_items, N=10)

    # ---------------- Model 2: SLIM ----------------
    print("\n[3] Training Model 2: SLIM")
    print("    Based on: Ning & Karypis (ICDM 2011)")
    print("    Learning sparse item-item matrix W via ElasticNet (L1 + L2)")
    slim = SLIM(alpha=0.001, l1_ratio=0.1)
    slim.fit(R_train)
    slim_scores = slim.predict_all(R_train)
    slim_results = evaluate(slim_scores, R_train, test_items, N=10)

    # ---------------- Results ----------------
    print("\n" + "=" * 74)
    print("RESULTS (Top-10 recommendation, leave-one-out evaluation)")
    print("=" * 74)
    print(f"{'Method':<34}{'Hit Rate@10':>16}{'ARHR@10':>16}")
    print("-" * 74)
    print(f"{'Implicit ALS  [Paper 1]':<34}"
          f"{als_results['hit_rate']:>16.4f}{als_results['arhr']:>16.4f}")
    print(f"{'SLIM          [Paper 2]':<34}"
          f"{slim_results['hit_rate']:>16.4f}{slim_results['arhr']:>16.4f}")
    print("-" * 74)

    better = "SLIM" if slim_results["hit_rate"] > als_results["hit_rate"] else "Implicit ALS"
    print(f"Higher Hit Rate@10 on this dataset: {better}")
    print(f"SLIM item-item matrix sparsity: {slim.sparsity:.2f}% zeros")

    # ---------------- Sample recommendations ----------------
    print("\n" + "=" * 74)
    print("SAMPLE TOP-5 RECOMMENDATIONS FOR 3 USERS")
    print("=" * 74)

    for u in list(test_items.keys())[:3]:
        seen = np.where(R_train[u] > 0)[0]

        a = als_scores[u].copy()
        a[R_train[u] > 0] = -np.inf
        s = slim_scores[u].copy()
        s[R_train[u] > 0] = -np.inf

        print(f"\nUser {u}")
        print(f"  Already interacted with items : {[int(x) for x in seen[:8]]}"
              f"{' ...' if len(seen) > 8 else ''}")
        print(f"  Implicit ALS recommends       : {[int(x) for x in np.argsort(-a)[:5]]}")
        print(f"  SLIM recommends               : {[int(x) for x in np.argsort(-s)[:5]]}")
        print(f"  Actual held-out test item     : {test_items[u]}")

    print("\n" + "=" * 74)
    print("Execution completed successfully.")
    print("=" * 74)


if __name__ == "__main__":
    main()
