# portfolio.py
import logging
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, Tuple
from config import PipelineConfig
from sklearn.covariance import LedoitWolf

logger = logging.getLogger("QuantPipeline.Portfolio")

class BayesianExecutionBridge:
    def __init__(self, config: PipelineConfig):
        print("\nInitializing BayesianExecutionBridge with provided configuration...")
        print("初始化贝叶斯执行桥...")
        self.config = config

    def compute_matrix_bl_and_optimize(
        self, 
        predictions_dict: Dict[str, Tuple[float, float, float]], 
        historical_returns_df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        基于高维标准矩阵级黑-利特曼公式求解后验收益分布，并直接进行均值-方差最大夏普比率优化。
        """
        print("\nComputing matrix-level Black-Litterman posterior and optimizing portfolio weights...")
        print("计算矩阵级 Black-Litterman 后验并优化投资组合权重...")
        symbols = self.config.SYMBOLS
        n_assets = len(symbols)
        w_size = self.config.ROLLING_WINDOW_SIZE
        
        # 1. 估计内生先验资产协方差矩阵 Sigma (NxN)
        hist_window = historical_returns_df[symbols].iloc[-w_size:]
        # Sigma = hist_window.cov().values.copy()
        # print("\nEstimating intrinsic prior asset covariance matrix Sigma...")
        # print("估计内生先验资产协方差矩阵 Sigma...")
        # 稳健性防御：加入微量岭正则项，防止极端市场条件下的奇异矩阵逆崩溃
        # Sigma += np.eye(n_assets) * 1e-6
        
        # 新逻辑：利用 Ledoit-Wolf 自适应估计最优收缩强度 alpha 并调整矩阵
        # 注：hist_window.values 的形状为 (n_samples, n_features)，即 (时间窗口, 资产数)
        lw = LedoitWolf()
        Sigma = lw.fit(hist_window.values).covariance_
        # 稳健性防御：由于 Ledoit-Wolf 已确保矩阵的正定性，此处的脊惩罚项可作为双重保险保留或调小
        Sigma += np.eye(n_assets) * 1e-8
        
        print("\nEstimating shrinkage prior asset covariance matrix Sigma via Ledoit-Wolf...")
        print("利用 Ledoit-Wolf 估计收缩先验资产协方差矩阵 Sigma...")
        
        
        
        # 2. 构造市场隐含均衡先验收益率向量 Pi (Nx1) - 采用经验滚动均值作为代理
        Pi = hist_window.mean().values.reshape(-1, 1)
        
        # 3. 提取 CQR 点预测作为 ML 观点向量 Q (Nx1)，并利用异方差幅宽构建不确定性协方差矩阵 Omega (NxN)
        Q = np.zeros((n_assets, 1))
        Omega = np.zeros((n_assets, n_assets))
        
        for i, symbol in enumerate(symbols):
            p_mean, p_low, p_high = predictions_dict[symbol]
            Q[i, 0] = p_mean
            
            # 条件幅宽物理测度
            cqr_width = p_high - p_low
            # Omega_ii = width^2 * tau
            Omega[i, i] = (cqr_width ** 2) * self.config.TAU
            if Omega[i, i] <= 0:
                Omega[i, i] = 1e-7
                
        # ==============================================================================
        # 【已修复】以下核心贝叶斯矩阵计算与优化逻辑已移出 for 循环，确保 Omega 填充完毕后执行
        # ==============================================================================
        
        # 4. 执行标准高维 Black-Litterman 贝叶斯共轭更新封闭解
        # P 矩阵在此定义为单位矩阵 I_N (代表对每个资产都存在独立的直接观点)
        inv_tau_Sigma = np.linalg.inv(self.config.TAU * Sigma)
        inv_Omega = np.linalg.inv(Omega)
        
        # 精度项叠加：(tau*Sigma)^-1 + P^T * Omega^-1 * P
        posterior_precision = np.linalg.inv(inv_tau_Sigma + inv_Omega)
        # 后验期望收益率矩阵公式
        BL_returns = np.dot(posterior_precision, np.dot(inv_tau_Sigma, Pi) + np.dot(inv_Omega, Q)).flatten()
        
        # 5. 马科维茨均值-方差框架下的鲁棒最大夏普比率非线性二次规划优化器 (MVO)
        def negative_sharpe(weights):
            port_return = np.dot(weights, BL_returns)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(Sigma, weights)))
            sharpe = (port_return - self.config.RISK_FREE_RATE) / (port_vol + 1e-8)
            return -sharpe

        # 现实交易约束：禁止任何形式的融券卖空 (Long-Only) 且全额满仓约束
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        bounds = [(0.0, 1.0) for _ in range(n_assets)]
        initial_weights = np.array([1.0 / n_assets] * n_assets)
        
        optimization_result = minimize(
            negative_sharpe, 
            initial_weights, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints
        )
        
        optimal_weights = optimization_result.x if optimization_result.success else initial_weights
        return BL_returns, optimal_weights, Sigma.diagonal()