# portfolio.py (局部重构片段)
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
    historical_returns: np.ndarray,
    ml_point_predictions: np.ndarray, # 长度为 N 的 LightGBM 预测
    cqr_widths: np.ndarray,           # 长度为 N 的 CQR 宽度
    llm_views: List[Tuple[str, float, float]], # 新增传入：LLM 观点列表
    asset_symbols: List[str],
    config
):
    N = len(asset_symbols)
    
    # 1. 计算市场的先验均衡（Prior）与 Ledoit-Wolf 稳健协方差 Sigma
    # (保持您原有代码逻辑不变...)
    Sigma = config.covariance_estimator(historical_returns) 
    Pi = config.scale_factor * np.dot(Sigma, config.market_weights)
    
    # 2. 初始构建基础观点矩阵（来自 Layer 2 的 LightGBM + CQR）
    # 基础状态下：P_base 为 IxN 的单位阵，Q_base 为机器学习点预测，Omega_base 互不相关
    P_list = [np.eye(N)]
    Q_list = [ml_point_predictions.reshape(-1, 1)]
    
    # 原系统核心：将 CQR 宽度映射为量价方差
    omega_ml_diagonal = (cqr_widths * config.cqr_to_variance_scale) ** 2
    Omega_list = [np.diag(omega_ml_diagonal)]
    
    # 3. 硬核升级：动态注入 Layer 1.5 的 LLM 文本观点
    if len(llm_views) > 0:
        for asset, impact, text_uncertainty in llm_views:
            if asset in asset_symbols:
                idx = asset_symbols.index(asset)
                
                # 构建挑选矩阵的一行 (1xN)
                p_row = np.zeros((1, N))
                p_row[0, idx] = 1.0
                P_list.append(p_row)
                
                # 构建观点向量的一行 (1x1)
                q_row = np.array([[impact]])
                Q_list.append(q_row)
                
                # 动态计算该文本观点的贝叶斯先验方差
                # 基础方差锚定该资产自身的历史方差，乘以大模型输出的文本噪声放大系数
                base_asset_variance = Sigma[idx, idx]
                omega_text_val = base_asset_variance * text_uncertainty
                Omega_list.append(np.array([[omega_text_val]]))

    # 4. 垂直与对角矩阵大拼接
    P = np.vstack(P_list)      # 形状变为 (N + K) x N
    Q = np.vstack(Q_list)      # 形状变为 (N + K) x 1
    
    # Omega 是全对角阵，用 block 动态拼接保证数理独立性
    from scipy.linalg import block_diag
    Omega = block_diag(*Omega_list) # 形状变为 (N + K) x (N + K)
    
    # 5. 执行标准 Black-Litterman 贝叶斯共轭封闭解更新
    # 这里的数学公式完全兼容任意维度的 P, Q, Omega！不需要改动原有封闭解逻辑
    tau = config.tau
    tau_Sigma = tau * Sigma
    inv_tau_Sigma = np.linalg.inv(tau_Sigma)
    inv_Omega = np.linalg.inv(Omega)
    
    # 贝叶斯后验期望收益率计算
    BL_return_matrix = np.linalg.inv(inv_tau_Sigma + np.dot(np.dot(P.T, inv_Omega), P))
    BL_return_vector = np.dot(BL_return_matrix, (np.dot(inv_tau_Sigma, Pi) + np.dot(np.dot(P.T, inv_Omega), Q)))
    posterior_expected_returns = BL_return_vector.flatten()
    
    # 6. 将调整后的后验收益率传给马科维茨凸优化器（MVO）进行有约束配资
    # (保持原有的 Long-only 和 满仓约束不变...)
    optimal_weights = run_mvo_optimizer(posterior_expected_returns, Sigma, config)
    
    return posterior_expected_returns, optimal_weights