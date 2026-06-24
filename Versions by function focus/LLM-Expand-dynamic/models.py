# models.py
import logging
import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Tuple, Dict
from config import PipelineConfig

logger = logging.getLogger("QuantPipeline.Models")

class StatisticalAdaptiveEngine:
    def __init__(self, config: PipelineConfig):
        print("\nInitializing StatisticalAdaptiveEngine with provided configuration...")
        print("初始化统计适应引擎...")
        self.config = config
        # 为每个资产建立独立的不确定性量化回归器池
        self.models: Dict[str, Dict[str, lgb.LGBMRegressor]] = {}
        self.cqr_thresholds: Dict[str, float] = {}

    def fit_and_quantify(self, X_panel: pd.DataFrame, y_panel: pd.Series):
        print("\nFitting conformal quantile regression models and quantifying uncertainty...")
        print("拟合符合性分位数回归模型并量化不确定性...")
        """
        执行符合性分位数回归（CQR）训练，显式捕获条件异方差性。
        """
        self.models.clear()
        self.cqr_thresholds.clear()
        
        w_size = self.config.ROLLING_WINDOW_SIZE
        low_q = self.config.ALPHA / 2.0
        high_q = 1.0 - (self.config.ALPHA / 2.0)
        
        for symbol in self.config.SYMBOLS:
            # 抽取当前单资产切片
            X_symbol = X_panel.xs(symbol, level='Symbol')
            y_symbol = y_panel.xs(symbol, level='Symbol')
            
            X_train, y_train = X_symbol.iloc[-w_size:], y_symbol.iloc[-w_size:]
            
            # 1. 初始化 CQR 所需的三叉分流弱估计器
            model_low = lgb.LGBMRegressor(objective='quantile', alpha=low_q, n_estimators=40, max_depth=3, learning_rate=0.05, verbose=-1, random_state=42)
            model_high = lgb.LGBMRegressor(objective='quantile', alpha=high_q, n_estimators=40, max_depth=3, learning_rate=0.05, verbose=-1, random_state=42)
            model_mean = lgb.LGBMRegressor(objective='regression', n_estimators=40, max_depth=3, learning_rate=0.05, verbose=-1, random_state=42)
            
            model_low.fit(X_train, y_train)
            model_high.fit(X_train, y_train)
            model_mean.fit(X_train, y_train)
            
            # 2. 计算校准集上的异方差符合性非参数误差分数 E_i
            preds_low = model_low.predict(X_train)
            preds_high = model_high.predict(X_train)
            
            # E_i = max(q_low(X_i) - y_i, y_i - q_high(X_i))
            cqr_residuals = np.maximum(preds_low - y_train.values, y_train.values - preds_high)
            q_error_threshold = np.quantile(cqr_residuals, 1.0 - self.config.ALPHA)
            
            self.models[symbol] = {
                "low": model_low,
                "high": model_high,
                "mean": model_mean
            }
            self.cqr_thresholds[symbol] = q_error_threshold
            
        print("\nHeteroskedastic Conformal Quantile Regression (CQR) calibration matrix complete.")
        print("异方差符合性分位数回归（CQR）校准矩阵完成。")
        logger.info("\nHeteroskedastic Conformal Quantile Regression (CQR) calibration matrix complete.")
        return self

    def predict_with_bounds(self, current_X_dict: Dict[str, np.ndarray]) -> Dict[str, Tuple[float, float, float]]:
        print("\nPredicting with bounds...")
        print("进行带边界的预测...")
        """
        为所有资产的最新的单一观测截面输出条件点预测和高保真 CQR 符合性置信极值。
        """
        predictions = {}
        for symbol in self.config.SYMBOLS:
            if symbol not in self.models:
                raise RuntimeError(f"Asset model for {symbol} has not been calibrated yet.")
                
            x_inst = current_X_dict[symbol].reshape(1, -1)
            mods = self.models[symbol]
            q_err = self.cqr_thresholds[symbol]
            
            raw_mean = mods["mean"].predict(x_inst)[0]
            raw_low = mods["low"].predict(x_inst)[0]
            raw_high = mods["high"].predict(x_inst)[0]
            
            # 应用符合性单步带状平移守卫公式
            conformal_low = raw_low - q_err
            conformal_high = raw_high + q_err
            
            predictions[symbol] = (raw_mean, conformal_low, conformal_high)
            
        print("\nPrediction with bounds complete.")
        print("带边界的预测完成。")
        return predictions