# Quy ước chung (để so sánh công bằng)

- **Seed:** `random_state = 42` ở mọi nơi.
- **Chia dữ liệu:** dùng cùng một cách chia train/test cho tất cả mô hình (thống nhất trong nhóm, ghi vào `split` của file JSON). Nếu có nhiều session của cùng một user thì chia theo user hoặc theo thời gian, không chia ngẫu nhiên.
- **Cross-validation:** 5-fold trên tập train, báo cáo trung bình và độ lệch chuẩn.
- **Tuning:** XGBoost và Random Forest dùng cùng ngân sách tuning (ví dụ RandomizedSearchCV, 30 lần thử). Ghi tham số tốt nhất vào JSON.
- **Chỉ số:** RMSE, MSE, MAE, R², tính trên tập test.
- **Bộ biến:** ghi rõ danh sách biến dùng thực tế trong `features`.
- **Tên file:** `<model>_<tên>.json`, ví dụ `xgboost_tran.json`, `ols_kiet.json`.
- **Không commit dữ liệu gốc.**


## Quy tắc dùng chung metric
Muốn so sánh các mô hình với nhau thì mọi người phải tính metric theo cùng một cách:

1. **Bộ metric cố định:** RMSE, MSE, MAE, R². Không thêm, không bớt, không đổi tên key trong `metrics_test`.
2. **Tính trên tập test:** dùng `sklearn.metrics` (`mean_squared_error`, `mean_absolute_error`, `r2_score`), RMSE = `sqrt(MSE)`. Với OLS, **không** lấy R² từ `statsmodels .summary()`, vì đó là R² trên tập train.
3. **Cùng dữ liệu đánh giá:** cùng cách chia train/test và cùng `random_state = 42` (chi tiết trong `CONTRIBUTING.md`).
4. **Cùng thang đo của y:** nếu train trên `log(y)` hoặc y đã chuẩn hoá thì phải đổi dự đoán về thang gốc trước khi tính metric.
5. **Tiền xử lý chỉ fit trên train:** scaler, encoder, imputer chỉ fit trên tập train rồi áp dụng cho tập test.
6. **Số liệu thô:** ghi đủ số thập phân, không làm tròn trong JSON. `aggregate.py` sẽ tự định dạng.
7. **Cross-validation:** báo cáo `rmse_mean` và `rmse_std` của 5-fold trên tập train. Không dùng các số này thay cho metric test.