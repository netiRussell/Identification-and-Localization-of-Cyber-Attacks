import numpy as np

def outlier_threshold(losses,
                      method="auto",         # 'auto' | 'iqr' | 'mad' | 'percentile'
                      perc=99.5,             # upper percentile for 'percentile' method
                      iqr_k=1.5,             # Tukey multiplier
                      mad_k=3.5,             # modified z-score cutoff
                      return_report=True):
    """
    Find an upper threshold to cut high-end outliers in a 1D array of losses.
    Returns: threshold, keep_mask, outlier_mask, cleaned_values, report(dict)
    """
    x = np.asarray(losses, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("Input has no finite values.")
    
    # Basic stats
    n   = x.size
    mu  = x.mean()
    sd  = x.std(ddof=0)
    sd  = sd if sd > 1e-12 else 1e-12  # avoid div-by-zero
    skew = np.mean(((x - mu) / sd) ** 3)

    # IQR rule (upper)
    q1, q3 = np.percentile(x, [25, 75])
    iqr = q3 - q1
    iqr_upper = q3 + iqr_k * iqr if iqr > 1e-12 else np.inf

    # MAD rule (upper) using modified z-score threshold
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad < 1e-12:
        mad_upper = np.inf
    else:
        c = 0.6744897501960817  # Φ^{-1}(0.75)
        mad_upper = med + (mad_k / c) * mad

    # Percentile rule (upper)
    perc_upper = np.percentile(x, perc)

    candidates = {"iqr": iqr_upper, "mad": mad_upper, "percentile": perc_upper}

    # Auto-pick (skew-aware, robust fallback)
    if method == "auto":
        # If heavily right-skewed, percentiles are safer; otherwise prefer IQR; fallback to MAD.
        if skew > 1:
            chosen = "percentile"
        elif np.isfinite(iqr_upper):
            chosen = "iqr"
        elif np.isfinite(mad_upper):
            chosen = "mad"
        else:
            chosen = "percentile"
    else:
        if method not in candidates:
            raise ValueError("method must be 'auto', 'iqr', 'mad', or 'percentile'")
        chosen = method

    thr = candidates[chosen]

    # Build masks (upper-tail only)
    out_mask  = x > thr
    keep_mask = ~out_mask
    cleaned   = x[keep_mask]

    report = {
        "n": n,
        "skew": float(skew),
        "candidates": {k: float(v) for k, v in candidates.items()},
        "chosen_method": chosen,
        "threshold": float(thr),
        "outlier_fraction": float(out_mask.mean())
    }
    return (thr, keep_mask, out_mask, cleaned, report) if return_report else (thr, keep_mask, out_mask, cleaned)

# ---- Example ----
if __name__ == "__main__":
    losses = np.load("./loss.npy")
    
    thr, keep, out, cleaned, info = outlier_threshold(losses, method="percentile")
    print(f"Chosen: {info['chosen_method']}, threshold={info['threshold']:.4f}, outliers={out.sum()} ({info['outlier_fraction']*100:.2f}%)")
