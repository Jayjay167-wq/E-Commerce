# Quy ước chung (để so sánh công bằng)

- **Seed:** `random_state = 42` ở mọi nơi.
- **Chia dữ liệu:** dùng cùng một cách chia train/test cho tất cả mô hình (thống nhất trong nhóm, ghi vào `split` của file JSON). Nếu có nhiều session của cùng một user thì chia theo user hoặc theo thời gian, không chia ngẫu nhiên.
- **Cross-validation:** 5-fold trên tập train, báo cáo trung bình và độ lệch chuẩn.
- **Tuning:** XGBoost và Random Forest dùng cùng ngân sách tuning (ví dụ RandomizedSearchCV, 30 lần thử). Ghi tham số tốt nhất vào JSON.
- **Chỉ số:** RMSE, MSE, MAE, R², tính trên tập test.
- **Bộ biến:** ghi rõ danh sách biến dùng thực tế trong `features`.
- **Tên file:** `<model>_<tên>.json`, ví dụ `xgboost_tran.json`, `ols_kiet.json`.
- **Không commit dữ liệu gốc.**
