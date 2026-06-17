# portfolio.py
import logging
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.linalg import block_diag
from typing import Dict, List, Tuple
from config import PipelineConfig
from sklearn.covariance import LedoitWolf

logger = logging.getLogger("QuantPipeline.Portfolio")

class BayesianExecutionBridge:
    def __init__(self, config: PipelineConfig):
        self.config = config

    def compute_matrix_bl_and_optimize(
        self,
        historical_returns_df: pd.DataFrame,
        ml_point_predictions: Dict[str, float],
        cqr_widths: Dict[str, float],
        llm_views: List[Tuple[str, float, float]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        白盒化可解释架构：
        1. 机器学习点预测与CQR置信幅宽构成资产内生贝叶斯先验底座(Pi, Sigma_ML)
        2. LLM大模型产生突发文本观点动态构建条件外部矩阵(P, Q, Omega_text)
        3. 实施高维标准贝叶斯封闭解共轭更新，并传递至有约束最大夏普比率非线性规划优化器。
        """
        symbols = self.config.SYMBOLS
        n_assets = len(symbols)
        w_size = self.config.ROLLING_WINDOW_SIZE
        
        # Layer 1: 基于 Ledoit-Wolf 收缩估计技术计算资产先验协方差矩阵 Sigma
        hist_window = historical_returns_df[symbols].iloc[-w_size:]
        lw = LedoitWolf()
        Sigma = lw.fit(hist_window.values).covariance_
        Sigma += np.eye(n_assets) * 1e-8  # 双重保险阻尼项，确保100%正定可逆
        
        # Layer 2: 将机器学习条件期望点预测，直接作为贝叶斯先验基准向量 Pi
        Pi = np.zeros((n_assets, 1))
        for i, symbol in enumerate(symbols):
            Pi[i, 0] = ml_point_predictions[symbol]
            
        # 同时根据 CQR 动态幅宽，对资产自身的先验方差进行条件异方差自适应缩放（构建 ML 先验方差大底）
        Sigma_ML = np.zeros((n_assets, n_assets))
        for i, symbol in enumerate(symbols):
            width_factor = (cqr_widths[symbol] * self.config.TAU) ** 2
            Sigma_ML[i, i] = Sigma[i, i] * max(width_factor, 1e-6)
            
        # Layer 3: 动态构建大模型外部观点矩阵 (P_text, Q_text, Omega_text)
        P_list = []
        Q_list = []
        Omega_list = []
        
        if len(llm_views) > 0:
            for asset, impact, text_uncertainty in llm_views:
                if asset in symbols:
                    idx = symbols.index(asset)
                    
                    # 映射该特定外部文本观点的资产挑选行矢量 (1 x N)
                    p_row = np.zeros((1, n_assets))
                    p_row[0, idx] = 1.0
                    P_list.append(p_row)
                    
                    # 填充文本预测的条件预期收益 Delta
                    Q_list.append(np.array([[impact]]))
                    
                    # 该新闻观点的方差锚定资产本身协方差乘以 LLM 识别的不确定性系数
                    text_variance = Sigma[idx, idx] * text_uncertainty
                    Omega_list.append(np.array([[max(text_variance, 1e-7)]]))

        # Layer 4: 贝叶斯条件共轭解融合结算
        if len(P_list) > 0:
            P_text = np.vstack(P_list)
            Q_text = np.vstack(Q_list)
            Omega_text = block_diag(*Omega_list)
            
            # 执行经典通用高维 Black-Litterman 后验推断封闭解
            inv_Sigma_ML = np.linalg.inv(Sigma_ML)
            inv_Omega_text = np.linalg.inv(Omega_text)
            
            # 后验精度矩阵计算
            posterior_precision = np.linalg.inv(inv_Sigma_ML + np.dot(np.dot(P_text.T, inv_Omega_text), P_text))
            # 联合条件期望推演
            BL_returns = np.dot(posterior_precision, np.dot(inv_Sigma_ML, Pi) + np.dot(np.dot(P_text.T, inv_Omega_text), Q_text)).flatten()
        else:
            # 机制防御：无外部重大突发新闻时，系统直接安全收缩回复回归树量价底座
            BL_returns = Pi.flatten()

        # Layer 5: 马科维茨均值-方差交易有约束非线性最大夏普优化器 (MVO)
        def negative_sharpe(weights):
            port_return = np.dot(weights, BL_returns)
            port_vol = np.sqrt(np.dot(weights.T, np.dot(Sigma, weights)))
            sharpe = (port_return - self.config.RISK_FREE_RATE) / (port_vol + 1e-8)
            return -sharpe

        # 严格执行实盘底线：禁止融券卖空 (Long-Only) 以及全额满仓配资限制
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