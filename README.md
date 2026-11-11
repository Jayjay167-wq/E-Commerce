# So sánh OLS và ML (XGBoost, Random Forest) cho bài toán hồi quy

Repo dùng để các thành viên đăng kết quả chạy mô hình, sau đó gộp thành bảng so sánh chung.

## Phân công
| Case | Mô hình | Người phụ trách |
|------|---------|-----------------|
| Case 1 (11 biến) | OLS, XGBoost, Random Forest | Quốc Kiệt, Phan Lâm, NNgoc |
| Case 2 (16 biến) | XGBoost | Trân |
| Case 2 (16 biến) | Random Forest | Tú |

## Cấu trúc
```
results/case1/    file kết quả JSON của Case 1
results/case2/    file kết quả JSON của Case 2
notebooks/        notebook của từng người (đặt tên: case1_xgb_kiet.ipynb)
shap_figures/     hình SHAP và feature importance
data/raw/         dữ liệu gốc (KHÔNG commit, đã gitignore)
aggregate.py      gộp mọi JSON trong results/ thành bảng so sánh
results_template.json   mẫu file kết quả
```

## Cách đăng kết quả
1. Tạo nhánh riêng: `git checkout -b case1-xgb-kiet`
2. Chạy mô hình theo quy ước chung trong `CONTRIBUTING.md`.
3. Copy `results_template.json` vào `results/case1/` (hoặc `case2/`), đặt tên `<model>_<tên>.json`, điền số liệu.
4. Đưa notebook vào `notebooks/` và hình vào `shap_figures/`.
5. Commit, push, mở Pull Request. Một người khác trong nhóm xem qua rồi merge.

## Gộp kết quả
```
pip install pandas
python aggregate.py
```
Lệnh này tạo `summary.csv` và `summary.md` gồm RMSE, MSE, MAE, R² của mọi mô hình, kèm mức cải thiện so với OLS.
