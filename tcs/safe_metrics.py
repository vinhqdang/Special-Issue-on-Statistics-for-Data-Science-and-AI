"""Rank-graduation (SAFE-family) metrics, vectorised for use inside the bootstrap.

Verified sources (see docs/safe_extension.md for the exact status of each item):

  RGX(Y, Z) = 1 - CvM(Y, Z) / G(Y) = 1 - int|L_{Y,Z} - L_Y| / int|L^c_Y - L_Y|
  where L_Y is the Lorenz curve of Y, L^c_Y its dual, and L_{Y,Z} the concordance curve
  (Y cumulated in the ascending order induced by Z).  (Auricchio, Bernardelli, Giudici,
  Toscani, arXiv:2511.23100, Sec. 2.2 and 4; Giudici, SAFE AI in Finance slides.)

  RGA = RGX(y_true, y_hat)           accuracy; equals AUC for a binary response
  RGR = RGX(y_hat, y_hat_perturbed)  robustness
  RGE = RGX(y_hat, y_hat_without_feature_j)   (not implemented here, needs refits)
  RGA parity (fairness): difference of RGA between the two levels of a binary protected
  variable (documented function ``compute_rga_parity`` of the authors' ``safeaipackage``).

Closed form used.  With n observations, ascending midranks ry of y and rz of z,
    sum_k [D(k) - C_Z(k)] = sum_i y_i (ry_i + rz_i - n - 1)
    sum_k [D(k) - C_Y(k)] = sum_i y_i (2 ry_i - n - 1)
    RGX(y, z) = sum_i y_i (ry_i + rz_i - n - 1) / sum_i y_i (2 ry_i - n - 1)
where D, C_Z, C_Y are the cumulative sums defining the dual Lorenz, concordance and Lorenz
curves; the common factor 1/(n * mean(y)) cancels.  It is exactly the area-ratio formula
printed in the Giudici slides.

Orientation.  RGA, RGR in [0, 1], HIGHER is better (1 = perfectly concordant ranking,
0 = perfectly reversed, 0.5 = uninformative).  The TCS needs lower-is-better coordinates, so
the functions ``rga_loss`` = 1 - RGA, ``rgr_loss`` = 1 - RGR and ``rga_parity_gap`` =
|RGA_group0 - RGA_group1| are provided; all are lower-is-better.

Conventions NOT taken from a source (own choices, flagged in the docs): ties in z receive
midranks (this makes RGA equal AUC with half credit for ties in the binary case); y must be
non-negative (Lorenz curves); the absolute value in the parity gap.
"""
import numpy as np
from scipy.stats import rankdata


def _as2d(a):
    a = np.asarray(a, float)
    return a[:, None] if a.ndim == 1 else a


def rgx(y, z):
    """RGX(Y, Z).  y, z: (n,) or (n, M) (same shape, or y (n,) broadcast against z (n, M)).
    Returns a float for 1-D input, else an (M,) array.  Higher = better concordance.
    NaN if sum(y) == 0 or the denominator is 0 (constant y)."""
    squeeze = np.ndim(z) == 1 and np.ndim(y) == 1
    y2, z2 = _as2d(y), _as2d(z)
    if y2.shape[1] == 1 and z2.shape[1] > 1:
        y2 = np.broadcast_to(y2, z2.shape)
    if y2.shape != z2.shape:
        raise ValueError("y and z must have the same shape")
    if np.any(y2 < 0):
        raise ValueError("RGX requires a non-negative first argument (Lorenz curves)")
    n = y2.shape[0]
    ry = rankdata(y2, axis=0)
    rz = rankdata(z2, axis=0)
    num = (y2 * (ry + rz - n - 1)).sum(0)
    den = (y2 * (2 * ry - n - 1)).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 0, num / den, np.nan)
    return float(out[0]) if squeeze else out


def rga(y_true, y_score):
    """Rank Graduation Accuracy (higher is better); = AUC for a binary y_true."""
    return rgx(y_true, y_score)


def rgr(pred_original, pred_perturbed):
    """Rank Graduation Robustness: RGX(prediction, prediction on perturbed input)."""
    return rgx(pred_original, pred_perturbed)


def rga_loss(y_true, y_score):
    """1 - RGA, lower is better (accuracy coordinate)."""
    return 1.0 - rga(y_true, y_score)


def rgr_loss(pred_original, pred_perturbed):
    """1 - RGR, lower is better (robustness coordinate)."""
    return 1.0 - rgr(pred_original, pred_perturbed)


def rga_parity_gap(y_true, y_score, group):
    """|RGA(group==0) - RGA(group==1)|, lower is better (fairness coordinate).
    y_score: (n,) or (n, M); group: (n,) binary."""
    g = np.asarray(group).astype(bool)
    y = np.asarray(y_true, float)
    s = np.asarray(y_score, float)
    return np.abs(rga(y[~g], s[~g]) - rga(y[g], s[g]))


def _selfcheck():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(1)
    y = rng.gamma(2.0, 1.0, 200)
    assert abs(rga(y, y) - 1) < 1e-12                      # perfect ranking
    assert abs(rga(y, -y)) < 1e-12                         # reversed ranking: minimum 0
    assert abs(rga(y, y ** 3 + 5) - 1) < 1e-12             # monotone transform invariance
    yb = (rng.random(300) < 0.3).astype(float)
    s = rng.random(300) + 0.4 * yb
    assert abs(rga(yb, s) - roc_auc_score(yb, s)) < 1e-12  # binary: RGA = AUC
    assert abs(rga(yb, np.ones(300)) - 0.5) < 1e-12        # constant score: 0.5
    # hand computation on the curves: y=(1,2,3), order induced by z = (obs1, obs3, obs2):
    # Lorenz cumsums 1,3,6 (sum 10); dual 3,5,6 (sum 14); concordance 1,4,6 (sum 11);
    # RGX = (14 - 11) / (14 - 10) = 0.75
    assert abs(rgx(np.array([1., 2, 3]), np.array([1., 3, 2])) - 0.75) < 1e-12
    M = rga(yb[:, None] * np.ones((1, 3)), np.column_stack([s, -s, yb]))
    assert M.shape == (3,) and abs(M[1] - (1 - roc_auc_score(yb, s))) < 1e-12
    assert abs(rgr(s, s) - 1) < 1e-12 and 0 <= rgr(s, s + rng.normal(0, .5, 300)) <= 1
    g = rng.integers(0, 2, 300)
    assert 0 <= rga_parity_gap(yb, s, g) <= 1
    return True


if __name__ == "__main__":
    print("selfcheck ok:", _selfcheck())
