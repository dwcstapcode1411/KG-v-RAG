#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
=============================================================================
MODULE DỰ BÁO GIÁ VÀNG (GOLD PRICE FORECASTING ENGINE)
Học phần: Phát triển các hệ thống thông minh (PTHTTM) - Mục 5.6+
Mô hình: ARIMA / Exponential Smoothing / Rolling Mean
=============================================================================

Cung cấp:
* `GoldForecaster` — class chính thực hiện dự báo chuỗi thời gian giá vàng.
* Hỗ trợ nhiều phương pháp: ARIMA, Exponential Smoothing, Rolling Mean Baseline.
* Tự động fallback nếu `statsmodels` chưa cài đặt.
* Tích hợp đánh giá backtest (MAE, RMSE, MAPE) trên tập kiểm thử.
* Lưu kết quả dự báo vào Neo4j dưới dạng node `GoldForecast`.
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from datetime import timedelta

warnings.filterwarnings("ignore")

# Đảm bảo in tiếng Việt chuẩn trên Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, Exception):
        pass

# Import statsmodels (graceful fallback)
_HAS_STATSMODELS = False
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    _HAS_STATSMODELS = True
except ImportError:
    pass


class GoldForecaster:
    """Bộ dự báo giá vàng đa phương pháp với đánh giá backtest tự động."""

    def __init__(self, df: pd.DataFrame = None, csv_path: str = None):
        """
        Khởi tạo bộ dự báo.

        Parameters
        ----------
        df : pd.DataFrame, optional
            DataFrame chứa cột 'date' và 'close'. Ưu tiên nếu cung cấp.
        csv_path : str, optional
            Đường dẫn tới file CSV dữ liệu giá vàng.
        """
        self.df = None
        self.train_df = None
        self.test_df = None
        self.forecast_df = None
        self.backtest_metrics = {}
        self.method_used = "N/A"

        if df is not None:
            self._prepare_data(df)
        elif csv_path and os.path.exists(csv_path):
            raw = pd.read_csv(csv_path)
            col_map = {c: c.strip().lower() for c in raw.columns}
            raw = raw.rename(columns=col_map)
            date_col = next((c for c in raw.columns if "date" in c or "time" in c), None)
            close_col = next((c for c in raw.columns if "close" in c or "price" in c), None)
            if date_col and close_col:
                raw = raw.rename(columns={date_col: "date", close_col: "close"})
                raw = raw.dropna(subset=["date", "close"])
                raw = raw.drop_duplicates(subset=["date"])
                self._prepare_data(raw)

    def _prepare_data(self, df: pd.DataFrame):
        """Chuẩn bị và chia dữ liệu train/test (80/20)."""
        ts = df[["date", "close"]].copy()
        ts["date"] = pd.to_datetime(ts["date"])
        ts = ts.sort_values("date").reset_index(drop=True)
        ts["close"] = ts["close"].astype(float)

        # Chia 80/20 cho backtest
        split_idx = int(len(ts) * 0.8)
        self.df = ts
        self.train_df = ts.iloc[:split_idx].copy()
        self.test_df = ts.iloc[split_idx:].copy()

    def forecast(self, steps: int = 30, method: str = "auto") -> pd.DataFrame:
        """
        Dự báo giá vàng cho N ngày tương lai.

        Parameters
        ----------
        steps : int
            Số ngày dự báo (mặc định: 30).
        method : str
            Phương pháp dự báo: 'arima', 'ets', 'rolling', 'auto'.
            'auto' sẽ chọn phương pháp tốt nhất dựa trên backtest.

        Returns
        -------
        pd.DataFrame
            Columns: date, forecast_close, lower_bound, upper_bound
        """
        if self.df is None or len(self.df) < 30:
            return pd.DataFrame(columns=["date", "forecast_close", "lower_bound", "upper_bound"])

        if method == "auto":
            method = self._select_best_method()

        series = self.df["close"].values
        last_date = self.df["date"].iloc[-1]

        if method == "arima" and _HAS_STATSMODELS:
            result = self._forecast_arima(series, steps, last_date)
        elif method == "ets" and _HAS_STATSMODELS:
            result = self._forecast_ets(series, steps, last_date)
        else:
            result = self._forecast_rolling(series, steps, last_date)

        self.forecast_df = result
        return result

    def _forecast_arima(self, series: np.ndarray, steps: int, last_date) -> pd.DataFrame:
        """Dự báo bằng ARIMA(2,1,2)."""
        self.method_used = "ARIMA(2,1,2)"
        try:
            model = ARIMA(series, order=(2, 1, 2))
            fit = model.fit()
            pred = fit.forecast(steps=steps)
            conf = fit.get_forecast(steps=steps).conf_int(alpha=0.05)

            dates = [last_date + timedelta(days=i) for i in range(1, steps + 1)]
            return pd.DataFrame({
                "date": dates,
                "forecast_close": np.round(pred, 2),
                "lower_bound": np.round(conf.iloc[:, 0].values, 2),
                "upper_bound": np.round(conf.iloc[:, 1].values, 2)
            })
        except Exception:
            return self._forecast_rolling(series, steps, last_date)

    def _forecast_ets(self, series: np.ndarray, steps: int, last_date) -> pd.DataFrame:
        """Dự báo bằng Exponential Smoothing (Holt-Winters)."""
        self.method_used = "Exponential Smoothing (Holt)"
        try:
            model = ExponentialSmoothing(series, trend="add", seasonal=None)
            fit = model.fit(optimized=True)
            pred = fit.forecast(steps=steps)

            # Tính khoảng tin cậy dựa trên residuals
            residuals = series - fit.fittedvalues
            std_resid = np.std(residuals)

            dates = [last_date + timedelta(days=i) for i in range(1, steps + 1)]
            return pd.DataFrame({
                "date": dates,
                "forecast_close": np.round(pred, 2),
                "lower_bound": np.round(pred - 1.96 * std_resid, 2),
                "upper_bound": np.round(pred + 1.96 * std_resid, 2)
            })
        except Exception:
            return self._forecast_rolling(series, steps, last_date)

    def _forecast_rolling(self, series: np.ndarray, steps: int, last_date) -> pd.DataFrame:
        """Dự báo đơn giản bằng Rolling Mean + Drift."""
        self.method_used = "Rolling Mean + Drift (Baseline)"
        window = min(20, len(series))
        recent = series[-window:]
        mean_val = np.mean(recent)
        std_val = np.std(recent)

        # Tính drift (xu hướng trung bình)
        daily_returns = np.diff(recent) / recent[:-1]
        drift = np.mean(daily_returns)

        predictions = []
        current = series[-1]
        for i in range(steps):
            current = current * (1 + drift)
            predictions.append(current)

        preds = np.array(predictions)
        dates = [last_date + timedelta(days=i) for i in range(1, steps + 1)]
        return pd.DataFrame({
            "date": dates,
            "forecast_close": np.round(preds, 2),
            "lower_bound": np.round(preds - 1.96 * std_val, 2),
            "upper_bound": np.round(preds + 1.96 * std_val, 2)
        })

    def _select_best_method(self) -> str:
        """Backtest trên tập kiểm thử để chọn phương pháp tốt nhất."""
        if self.train_df is None or self.test_df is None or len(self.test_df) < 5:
            return "rolling"

        train_series = self.train_df["close"].values
        test_actual = self.test_df["close"].values[:30]  # Giới hạn 30 ngày test
        n_test = len(test_actual)
        last_train_date = self.train_df["date"].iloc[-1]

        candidates = {}

        # Rolling baseline
        try:
            fc_r = self._forecast_rolling(train_series, n_test, last_train_date)
            mae_r = np.mean(np.abs(test_actual - fc_r["forecast_close"].values[:n_test]))
            candidates["rolling"] = mae_r
        except Exception:
            candidates["rolling"] = 1e9

        if _HAS_STATSMODELS:
            # ARIMA
            try:
                fc_a = self._forecast_arima(train_series, n_test, last_train_date)
                mae_a = np.mean(np.abs(test_actual - fc_a["forecast_close"].values[:n_test]))
                candidates["arima"] = mae_a
            except Exception:
                candidates["arima"] = 1e9

            # ETS
            try:
                fc_e = self._forecast_ets(train_series, n_test, last_train_date)
                mae_e = np.mean(np.abs(test_actual - fc_e["forecast_close"].values[:n_test]))
                candidates["ets"] = mae_e
            except Exception:
                candidates["ets"] = 1e9

        best = min(candidates, key=candidates.get)
        self.backtest_metrics = candidates
        return best

    def backtest(self) -> Dict[str, Any]:
        """
        Chạy backtest đầy đủ trên tập test 20% và tính các chỉ số đánh giá.

        Returns
        -------
        dict
            Chứa MAE, RMSE, MAPE, method, và chi tiết từng phương pháp.
        """
        if self.train_df is None or self.test_df is None or len(self.test_df) < 5:
            return {"error": "Không đủ dữ liệu để thực hiện backtest."}

        train_series = self.train_df["close"].values
        test_actual = self.test_df["close"].values[:30]
        n_test = len(test_actual)
        last_train_date = self.train_df["date"].iloc[-1]

        results = {}

        # Test tất cả các phương pháp có sẵn
        methods = {"Rolling Mean + Drift": ("rolling", self._forecast_rolling)}
        if _HAS_STATSMODELS:
            methods["ARIMA(2,1,2)"] = ("arima", self._forecast_arima)
            methods["Exponential Smoothing"] = ("ets", self._forecast_ets)

        for name, (key, fn) in methods.items():
            try:
                fc = fn(train_series, n_test, last_train_date)
                preds = fc["forecast_close"].values[:n_test]
                mae = float(np.mean(np.abs(test_actual - preds)))
                rmse = float(np.sqrt(np.mean((test_actual - preds) ** 2)))
                mape = float(np.mean(np.abs((test_actual - preds) / test_actual)) * 100)
                results[name] = {
                    "MAE": round(mae, 2),
                    "RMSE": round(rmse, 2),
                    "MAPE_pct": round(mape, 2),
                    "predictions": preds.tolist()
                }
            except Exception as e:
                results[name] = {"error": str(e)}

        # Chọn phương pháp tốt nhất
        valid = {k: v for k, v in results.items() if "MAE" in v}
        if valid:
            best_name = min(valid, key=lambda k: valid[k]["MAE"])
        else:
            best_name = "N/A"

        return {
            "best_method": best_name,
            "test_size": n_test,
            "actual_prices": test_actual.tolist(),
            "test_dates": self.test_df["date"].iloc[:n_test].dt.strftime("%Y-%m-%d").tolist(),
            "methods": results
        }

    def get_forecast_summary(self) -> Dict[str, Any]:
        """Trả về bản tóm tắt kết quả dự báo cho hiển thị trên UI."""
        if self.forecast_df is None or len(self.forecast_df) == 0:
            return {"available": False}

        fc = self.forecast_df
        last_actual = float(self.df["close"].iloc[-1])
        last_date = self.df["date"].iloc[-1].strftime("%Y-%m-%d")
        first_pred = float(fc["forecast_close"].iloc[0])
        last_pred = float(fc["forecast_close"].iloc[-1])
        min_pred = float(fc["forecast_close"].min())
        max_pred = float(fc["forecast_close"].max())
        change_pct = ((last_pred - last_actual) / last_actual) * 100

        return {
            "available": True,
            "method": self.method_used,
            "last_actual_date": last_date,
            "last_actual_close": round(last_actual, 2),
            "forecast_start": fc["date"].iloc[0].strftime("%Y-%m-%d"),
            "forecast_end": fc["date"].iloc[-1].strftime("%Y-%m-%d"),
            "forecast_days": len(fc),
            "first_predicted": round(first_pred, 2),
            "last_predicted": round(last_pred, 2),
            "min_predicted": round(min_pred, 2),
            "max_predicted": round(max_pred, 2),
            "change_pct": round(change_pct, 2),
            "trend": "Tăng" if change_pct > 1 else ("Giảm" if change_pct < -1 else "Đi ngang")
        }


def load_gold_data(csv_path: str = None) -> pd.DataFrame:
    """Tiện ích: Tải và làm sạch dữ liệu vàng từ file CSV (hỗ trợ cả môi trường cloud và local)."""
    candidates = [
        csv_path,
        os.path.join(os.path.dirname(__file__), "data", "gold_price_forecasting_dataset.csv"),
        "data/gold_price_forecasting_dataset.csv",
        r"C:\DATA\gold_price_forecasting_dataset.csv"
    ]
    path = None
    for cand in candidates:
        if cand and os.path.exists(cand):
            path = cand
            break

    if not path or not os.path.exists(path):
        # Tạo dữ liệu mẫu để demo nếu không tìm thấy file
        np.random.seed(2024)
        dates = pd.date_range("2021-01-04", periods=1259, freq="B")
        base = 1800
        returns = np.random.normal(0.0003, 0.012, len(dates))
        prices = [base]
        for r in returns[1:]:
            prices.append(prices[-1] * (1 + r))
        return pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "close": np.round(prices, 2)})

    df = pd.read_csv(path)
    col_map = {c: c.strip().lower() for c in df.columns}
    df = df.rename(columns=col_map)
    date_col = next((c for c in df.columns if "date" in c or "time" in c), "date")
    close_col = next((c for c in df.columns if "close" in c or "price" in c), "close")
    df = df.rename(columns={date_col: "date", close_col: "close"})
    df = df.dropna(subset=["date", "close"]).drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df[["date", "close"]]


def forecast_gold(df: pd.DataFrame, steps: int = 30) -> pd.DataFrame:
    """Hàm tương thích ngược với script kiểm thử: dự báo giá vàng."""
    forecaster = GoldForecaster(df=df)
    return forecaster.forecast(steps=steps, method="auto")


def save_forecast_to_neo4j(driver, database: str, forecast_df: pd.DataFrame) -> None:
    """Lưu trữ kết quả dự báo vào Neo4j dưới dạng node GoldForecast."""
    if driver is None:
        return
    with driver.session(database=database) as session:
        batch = []
        for _, row in forecast_df.iterrows():
            d_val = row["date"].strftime("%Y-%m-%d") if hasattr(row["date"], "strftime") else str(row["date"])
            batch.append({
                "date": d_val,
                "predicted_close": float(row["forecast_close"]),
                "lower_bound": float(row.get("lower_bound", row["forecast_close"])),
                "upper_bound": float(row.get("upper_bound", row["forecast_close"]))
            })
        cypher = """
        UNWIND $batch AS rec
        MERGE (f:GoldForecast {date: rec.date})
        SET f.predicted_close = rec.predicted_close,
            f.lower_bound = rec.lower_bound,
            f.upper_bound = rec.upper_bound
        WITH f
        MATCH (d:GoldDay)
        WHERE d.date < f.date
        WITH f, d ORDER BY d.date DESC LIMIT 1
        MERGE (f)-[:FORECAST_FOR]->(d)
        """
        session.run(cypher, batch=batch)


# =============================================================================
# CLI ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print(" DỰ BÁO GIÁ VÀNG - GOLD PRICE FORECASTING ENGINE")
    print("=" * 70)

    df = load_gold_data()
    print(f"[+] Loaded {len(df)} trading days: {df['date'].iloc[0]} -> {df['date'].iloc[-1]}")

    forecaster = GoldForecaster(df=df)

    # Backtest
    print("\n--- BACKTEST RESULTS ---")
    bt = forecaster.backtest()
    for method_name, metrics in bt.get("methods", {}).items():
        if "MAE" in metrics:
            print(f"  {method_name}: MAE=${metrics['MAE']:.2f}, RMSE=${metrics['RMSE']:.2f}, MAPE={metrics['MAPE_pct']:.2f}%")
        else:
            print(f"  {method_name}: {metrics.get('error', 'N/A')}")
    print(f"  Best method: {bt.get('best_method', 'N/A')}")

    # Forecast
    print("\n--- 30-DAY FORECAST ---")
    fc = forecaster.forecast(steps=30, method="auto")
    summary = forecaster.get_forecast_summary()
    print(f"  Method: {summary['method']}")
    print(f"  Last actual: ${summary['last_actual_close']} on {summary['last_actual_date']}")
    print(f"  Forecast range: ${summary['min_predicted']} - ${summary['max_predicted']}")
    print(f"  30-day change: {summary['change_pct']:+.2f}% ({summary['trend']})")
    print(f"\n  First 5 predictions:")
    print(fc.head().to_string(index=False))
