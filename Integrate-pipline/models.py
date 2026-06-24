# models.py
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.kernel_ridge import KernelRidge
from config import PipelineConfig

class StatisticalAdaptiveEngine:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.models_low = {}
        self.models_mean_lgbm = {}
        self.models_mean_krr = {}
        self.models_high = {}
        self.cqr_thresholds = {}

    def fit_and_quantify(self, X_train: pd.DataFrame, y_train: pd.DataFrame):
        """Fits multi-model cascades and calibrates conformal quantile bounds."""
        symbols = X_train.index.get_level_values(1).unique()
        
        for sym in symbols:
            # Slicing asset cross sections safely
            X_sym = X_train.xs(sym, level='Symbol').values
            y_sym = y_train.xs(sym, level='Symbol').values
            
            # Define target quantiles matching user alpha metrics
            q_low_val = self.config.ALPHA / 2.0
            q_high_val = 1.0 - (self.config.ALPHA / 2.0)
            
            # 1. Instantiate Quantile Boundary Estimators
            m_low = lgb.LGBMRegressor(objective='quantile', alpha=q_low_val, n_estimators=50, verbose=-1)
            m_high = lgb.LGBMRegressor(objective='quantile', alpha=q_high_val, n_estimators=50, verbose=-1)
            
            # 2. Instantiate Mean Learners (Cascade Stacking Cluster)
            m_mean_lgb = lgb.LGBMRegressor(objective='regression', n_estimators=50, verbose=-1)
            m_mean_krr = KernelRidge(alpha=1.0, kernel='rbf')
            
            # Training Execution
            m_low.fit(X_sym, y_sym)
            m_high.fit(X_sym, y_sym)
            m_mean_lgb.fit(X_sym, y_sym)
            m_mean_krr.fit(X_sym, y_sym)
            
            # Extract historical in-sample performance limits
            pred_low = m_low.predict(X_sym)
            pred_high = m_high.predict(X_sym)
            
            # Compute Non-Parametric Calibration Errors
            residuals = np.maximum(pred_low - y_sym, y_sym - pred_high)
            self.cqr_thresholds[sym] = np.percentile(residuals, 100.0 * (1.0 - self.config.ALPHA))
            
            # Save instances to state registry mappings
            self.models_low[sym] = m_low
            self.models_high[sym] = m_high
            self.models_mean_lgbm[sym] = m_mean_lgb
            self.models_mean_krr[sym] = m_mean_krr

    def predict_with_bounds(self, current_X_dict: dict) -> dict:
        """Executes stacked prediction and estimates heteroskedastic width with epistemic penalties."""
        predictions_out = {}
        for sym, x_vector in current_X_dict.items():
            x_arr = x_vector.reshape(1, -1)
            
            # Evaluate cascaded mean learners
            pred_lgb = float(self.models_mean_lgbm[sym].predict(x_arr)[0])
            pred_krr = float(self.models_mean_krr[sym].predict(x_arr)[0])
            
            # Mathematical Mean Core Forecast
            meta_point_prediction = (pred_lgb + pred_krr) / 2.0
            
            # Compute Cross-Learner Divergence (Epistemic Disagreement Penalty)
            learner_disagreement = np.var([pred_lgb, pred_krr])
            
            # Extract basic distribution-free boundaries
            raw_low = float(self.models_low[sym].predict(x_arr)[0])
            raw_high = float(self.models_high[sym].predict(x_arr)[0])
            cal_err = self.cqr_thresholds[sym]
            
            # Injecting learner disagreement directly into the uncertainty interval inflation logic
            final_floor = raw_low - cal_err - np.sqrt(learner_disagreement)
            final_ceiling = raw_high + cal_err + np.sqrt(learner_disagreement)
            
            predictions_out[sym] = (meta_point_prediction, final_floor, final_ceiling)
        return predictions_out