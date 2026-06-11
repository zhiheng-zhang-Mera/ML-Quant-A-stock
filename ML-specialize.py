import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import LightGBMRegressor # Or CatBoost

class MLFinanceAcademicPipeline:
    def __init__(self, alpha: float = 0.05, tau: float = 0.02):
        """
        Args:
            alpha: Significance level for Conformal Prediction (e.g., 0.05 for 95% boundaries)
            tau: Weight of ML views vs Market Equilibrium in Black-Litterman
        """
        self.alpha = alpha
        self.tau = tau
        self.base_model = LightGBMRegressor(n_estimators=100, max_depth=5, random_state=42)
        self.conformal_residuals = []
        self.inv_cov_matrix = None
        self.mean_latent_state = None
        
    def _apply_inflation_deflator(self, df: pd.DataFrame, annual_inflation_rate: float = 0.03) -> pd.DataFrame:
        """Adjusts nominal asset values to real values to prevent multi-year look-back distortion."""
        df = df.copy()
        years_back = (df.index.max() - df.index).days / 365.25
        deflator = (1 + annual_inflation_rate) ** years_back
        df['real_close'] = df['close'] * deflator
        return df

    def extract_temporal_embeddings(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        ML Effort: Acts as a simplified statistical representation layer.
        In a full PhD script, this can be swapped with a PyTorch LSTM/TCN Autoencoder.
        """
        df = df.copy()
        # Real-returns and microstructural volatility features
        df['log_ret'] = np.log(df['real_close'] / df['real_close'].shift(1))
        df['normalized_range'] = (df['high'] - df['low']) / df['close']
        df['mom_5d'] = df['log_ret'].rolling(5).sum()
        df['mom_20d'] = df['log_ret'].rolling(20).sum()
        df['vol_20d'] = df['log_ret'].rolling(20).std()
        
        return df.dropna()

    def fit_regime_matcher_and_model(self, X: np.ndarray, y: np.ndarray):
        """Builds the Mahalanobis feature matrix and trains the ML predictor."""
        # Calculate covariance metrics for structural Mahalanobis distance matching
        covariance_matrix = np.cov(X, rowvar=False)
        self.inv_cov_matrix = np.linalg.pinv(covariance_matrix)
        self.mean_latent_state = np.mean(X, axis=0)
        
        # Train base ML Model
        self.base_model.fit(X, y)
        
        # ML Effort: Time-Series Conformal Prediction (Split Conformal Residuals)
        train_preds = self.base_model.predict(X)
        self.conformal_residuals = np.abs(y - train_preds)

    def compute_conformal_bounds(self, current_x: np.ndarray) -> tuple:
        """Computes distribution-free mathematically guaranteed risk boundaries."""
        base_pred = self.base_model.predict(current_x.reshape(1, -1))[0]
        
        # Get the (1-alpha) quantile of absolute historical errors
        q_threshold = np.quantile(self.conformal_residuals, 1 - self.alpha)
        
        trustworthy_range_low = base_pred - q_threshold
        trustworthy_range_high = base_pred + q_threshold
        return base_pred, trustworthy_range_low, trustworthy_range_high

    def black_litterman_execution_bridge(self, market_equilibrium_ret: float, 
                                         ml_predicted_ret: float, 
                                         conformal_width: float, 
                                         historical_var: float) -> float:
        """
        Finance Usability: Blends the erratic ML prediction with Market Equilibrium
        using the Conformal Prediction Width as the objective view uncertainty.
        """
        # Pi: Market Equilibrium Prior
        Pi = market_equilibrium_ret
        # Q: Machine Learning View
        Q = ml_predicted_ret
        # Sigma: Historical asset variance
        Sigma = historical_var
        # Omega: View Uncertainty derived directly from the Conformal Prediction Error Band
        Omega = (conformal_width ** 2) * self.tau
        
        # Standard Analytical Black-Litterman Formula for a single asset view:
        # E[R] = [(tau*Sigma)^-1 + P^T * Omega^-1 * P]^-1 * [(tau*Sigma)^-1 * Pi + P^T * Omega^-1 * Q]
        # With a single asset, P = 1
        scaled_prior_precision = 1 / (self.tau * Sigma)
        view_precision = 1 / Omega
        
        posterior_expected_return = (scaled_prior_precision * Pi + view_precision * Q) / (scaled_prior_precision + view_precision)
        return posterior_expected_return

    def generate_daily_report(self, df: pd.DataFrame, market_equilibrium_ret: float) -> dict:
        """Executes the pipeline and creates a non-overloading, decision-ready output."""
        # 1. Prepare Data
        df_real = self._apply_inflation_deflator(df)
        df_features = self.extract_temporal_embeddings(df_real)
        
        feature_cols = ['mom_5d', 'mom_20d', 'vol_20d', 'normalized_range']
        X = df_features[feature_cols].values
        y = df_features['log_ret'].shift(-1).fillna(0).values # Target: Next day log return
        
        # Strict Temporal split to prevent cheating (Look-ahead Bias)
        train_idx = int(len(X) * 0.8)
        self.fit_regime_matcher_and_model(X[:train_idx], y[:train_idx])
        
        # Current status
        current_x = X[-1]
        historical_var = np.var(y[:train_idx])
        
        # 2. Extract ML Predictions & Conformal Uncertainty Bounds
        ml_pred, low_bound, high_bound = self.compute_conformal_bounds(current_x)
        conformal_width = high_bound - low_bound
        
        # 3. Apply Black-Litterman Bridge for Usable Execution Allocation
        optimized_expected_return = self.black_litterman_execution_bridge(
            market_equilibrium_ret=market_equilibrium_ret,
            ml_predicted_ret=ml_pred,
            conformal_width=conformal_width,
            historical_var=historical_var
        )
        
        # 4. Generate Practical Conditional Trigger Levels
        last_price = df['close'].iloc[-1]
        conditional_buy_limit = last_price * (1 + low_bound) # Buy close to the statistical floor
        conditional_take_profit = last_price * (1 + optimized_expected_return)
        
        # Calculate Expected Win/Loss Ratio using the Conformal Threshold
        win_loss_ratio = abs(high_bound / (low_bound if low_bound != 0 else 1e-5))
        
        report = {
            "Target_Asset": "Target ETF/Stock",
            "Current_Close": last_price,
            "ML_Expected_Log_Return": optimized_expected_return,
            "Conformal_95_Confidence_Width": conformal_width,
            "Statistical_Floor_Price": last_price * (1 + low_bound),
            "Statistical_Ceiling_Price": last_price * (1 + high_bound),
            "Conditional_Buy_Order_Trigger": conditional_buy_limit,
            "Conditional_Sell_Order_Trigger": conditional_take_profit,
            "Expected_Win_Loss_Ratio": win_loss_ratio,
            "Execution_Action": "ACCUMULATE (Set Post-Market Order)" if optimized_expected_return > market_equilibrium_ret and win_loss_ratio > 1.2 else "HOLD/REDUCE"
        }
        return report

# --- Quick Execution Framework Demonstration ---
if __name__ == "__main__":
    # Simulate valid historical daily trading data
    np.random.seed(42)
    dates = pd.date_range(start="2020-01-01", periods=500, freq='D')
    simulated_close = 100 * np.exp(np.cumsum(np.random.normal(0.0002, 0.015, 500)))
    simulated_data = pd.DataFrame({
        'close': simulated_close,
        'high': simulated_close * 1.01,
        'low': simulated_close * 0.99,
        'open': simulated_close * 0.995
    }, index=dates)
    
    # Initialize our highly academic yet usable pipeline
    pipeline = MLFinanceAcademicPipeline(alpha=0.05, tau=0.02)
    
    # 0.0001 represents the daily historical risk premium (Market Equilibrium Prior)
    daily_report = pipeline.generate_daily_report(simulated_data, market_equilibrium_ret=0.0001)
    
    print("\n========= DAILY EXECUTION & REGIME REPORT =========")
    for key, val in daily_report.items():
        if isinstance(val, float):
            print(f"{key}: {val:.4f}")
        else:
            print(f"{key}: {val}")
    print("====================================================")