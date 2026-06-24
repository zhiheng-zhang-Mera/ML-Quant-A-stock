# portfolio.py
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from config import PipelineConfig

class BayesianExecutionBridge:
    def __init__(self, config: PipelineConfig):
        self.config = config
        
    def compute_matrix_bl_and_optimize(self, 
                                       historical_returns_df: pd.DataFrame, 
                                       ml_point_predictions: dict, 
                                       omega_matrix: pd.DataFrame, 
                                       llm_views: dict) -> tuple:
        """Performs structural Pricing Kernel contraction BL updates and solves MVO targets."""
        symbols = list(ml_point_predictions.keys())
        n_assets = len(symbols)
        
        # 1. Robust Ledoit-Wolf Shrinkage Covariance Matrix Estimation
        lw = LedoitWolf()
        lw.fit(historical_returns_df[symbols].values)
        sigma_robust = lw.covariance_
        
        # 2. Setup Equilibrium Baseline Prior Matrix Pi
        pi_prior = historical_returns_df[symbols].mean().values.reshape(-1, 1)
        
        # 3. Formulate Active View Vector Q with Domain-Specific L2 Contraction
        q_views = np.zeros((n_assets, 1))
        for i, sym in enumerate(symbols):
            raw_forecast = ml_point_predictions[sym]
            equilibrium_baseline = float(pi_prior[i])
            
            # Apply Pricing Kernel Regularization Operator
            q_views[i, 0] = ((1.0 - self.config.LAMBDA_KERNEL) * raw_forecast + 
                             self.config.LAMBDA_KERNEL * equilibrium_baseline)
            
        # Identity matrix structure mappings since views are directly aligned per asset
        P_matrix = np.eye(n_assets)
        Omega = omega_matrix.loc[symbols, symbols].values
        
        # 4. Standard high-dimensional Black-Litterman matrix derivation equations
        tau_sigma_inv = np.linalg.inv(self.config.TAU * sigma_robust)
        omega_inv = np.linalg.inv(Omega)
        
        posterior_covariance = np.linalg.inv(tau_sigma_inv + P_matrix.T @ omega_inv @ P_matrix)
        posterior_returns = posterior_covariance @ (tau_sigma_inv @ pi_prior + P_matrix.T @ omega_inv @ q_views)
        bl_returns_vector = posterior_returns.flatten()
        
        # 5. Non-Linear SLSQP Convex optimization execution solver
        def objective(w):
            port_return = np.dot(w, bl_returns_vector)
            port_vol = np.sqrt(np.dot(w.T, np.dot(sigma_robust, w)))
            # Maximize Sharpe ratio parameter boundary proxy
            return -(port_return - 0.0) / (port_vol + 1e-8)
            
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        bounds = [(0.0, 1.0) for _ in range(n_assets)]
        initial_guess = np.ones(n_assets) / n_assets
        
        opt_res = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)
        target_weights = opt_res.x if opt_res.success else initial_guess
        
        return bl_returns_vector, target_weights