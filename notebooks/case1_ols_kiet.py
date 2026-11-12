"""
CASE 1 - CHỈ PHẦN OLS (baseline, Linear Probability Model)
Chạy: python case1_ols.py        (cần pandas numpy scipy scikit-learn)

Đầu ra: ols_coefficients.csv, ols_metrics.csv, ols_vif.csv, ols_summary.txt
Dùng cùng dữ liệu, cùng train/test (seed 42, stratify) với bước chọn biến.
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_squared_error, mean_absolute_error, roc_auc_score,
                             accuracy_score, precision_score, recall_score, f1_score)

DATA_PATH, TARGET, SEED = "train.csv", "SCA", 42
FEATURES = ['add_count', 'remove_count', 'view_count', 'add_to_detail_ratio',
            'search_count', 'unique_categories_viewed', 'avg_price_bucket',
            'avg_dwell_time', 'session_duration', 'hod', 'dow']


def prep(d):
    """hod -> sin/cos (biến chu kỳ), dow -> dummy (bỏ T0 làm nhóm gốc)."""
    d = d.copy()
    d["hod_sin"] = np.sin(2 * np.pi * d["hod"] / 24)
    d["hod_cos"] = np.cos(2 * np.pi * d["hod"] / 24)
    d = d.drop(columns="hod")
    return pd.get_dummies(d, columns=["dow"], drop_first=True, dtype=float)


def ols_hc3(X, y):
    """OLS + sai số chuẩn robust HC3 (giống statsmodels cov_type='HC3')."""
    Xc = np.column_stack([np.ones(len(X)), X.values])
    names = ["const"] + list(X.columns)
    XtX_inv = np.linalg.inv(Xc.T @ Xc)
    beta = XtX_inv @ Xc.T @ y.values
    resid = y.values - Xc @ beta
    h = np.einsum("ij,jk,ik->i", Xc, XtX_inv, Xc)               # đòn bẩy
    meat = (Xc * (resid / (1 - h))[:, None]).T @ (Xc * (resid / (1 - h))[:, None])
    se = np.sqrt(np.diag(XtX_inv @ meat @ XtX_inv))
    z = beta / se
    p = 2 * (1 - stats.norm.cdf(np.abs(z)))
    ss_res, ss_tot = (resid ** 2).sum(), ((y - y.mean()) ** 2).sum()
    n, k = Xc.shape
    r2 = 1 - ss_res / ss_tot
    out = pd.DataFrame({"coef": beta, "se_HC3": se, "z": z, "p_value": p,
                        "ci_low": beta - 1.96 * se, "ci_high": beta + 1.96 * se}, index=names)
    return out, r2, 1 - (1 - r2) * (n - 1) / (n - k), beta


def vif_table(X):
    """VIF_j = 1/(1-R2_j), hồi quy X_j theo các biến còn lại (có hằng số)."""
    res = {}
    for c in X.columns:
        others = np.column_stack([np.ones(len(X)), X.drop(columns=c).values])
        b = np.linalg.lstsq(others, X[c].values, rcond=None)[0]
        r = X[c].values - others @ b
        r2 = 1 - (r ** 2).sum() / ((X[c] - X[c].mean()) ** 2).sum()
        res[c] = 1 / (1 - r2)
    return pd.Series(res, name="VIF").sort_values(ascending=False)


def metrics(y_true, p):
    mse = mean_squared_error(y_true, p)
    yp = (p >= 0.5).astype(int)
    return {"RMSE": np.sqrt(mse), "MSE": mse, "MAE": mean_absolute_error(y_true, p),
            "AUC": roc_auc_score(y_true, p), "Accuracy": accuracy_score(y_true, yp),
            "Precision": precision_score(y_true, yp, zero_division=0),
            "Recall": recall_score(y_true, yp), "F1": f1_score(y_true, yp)}


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH, sep=None, engine="python").drop_duplicates().dropna().reset_index(drop=True)
    X, y = df[FEATURES], df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    print(f"Train {Xtr.shape}, Test {Xte.shape}, tỷ lệ SCA=1 (train) = {ytr.mean():.4f}")

    A, B = prep(Xtr), prep(Xte)
    B = B.reindex(columns=A.columns, fill_value=0.0)

    coef, r2, r2adj, beta = ols_hc3(A, ytr)
    coef.to_csv("ols_coefficients.csv")
    print(f"\nR2 = {r2:.4f}, Adj R2 = {r2adj:.4f}, n = {len(A)}")
    print(coef.round(4).to_string())

    vif = vif_table(A)
    vif.to_csv("ols_vif.csv")
    print("\nVIF (bộ biến OLS sau khi mã hoá hod/dow):\n", vif.round(2).to_string())

    p_tr = np.clip(np.column_stack([np.ones(len(A)), A.values]) @ beta, 0, 1)
    p_te_raw = np.column_stack([np.ones(len(B)), B.values]) @ beta
    p_te = np.clip(p_te_raw, 0, 1)
    m = pd.DataFrame({"Train": metrics(ytr, p_tr), "Test": metrics(yte, p_te)}).T
    m.to_csv("ols_metrics.csv")
    print("\nĐánh giá:\n", m.round(4).to_string())
    print(f"\nDự báo ngoài [0,1] (test, trước khi cắt): {((p_te_raw < 0) | (p_te_raw > 1)).mean():.2%}")

    with open("ols_summary.txt", "w", encoding="utf-8") as f:
        f.write(f"R2={r2:.4f} AdjR2={r2adj:.4f}\n{coef.round(4).to_string()}\n\nVIF\n{vif.round(2).to_string()}\n\n{m.round(4).to_string()}\n")
