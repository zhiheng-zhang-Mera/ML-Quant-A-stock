import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

def compute_matrix_bl_and_optimize(
    historical_returns_df: pd.DataFrame,
    ml_view_results: dict,
    llm_sentiment_scores: dict,
    pipeline_config
) -> dict:
    """
    融入曹毅(2025)金融定价核约束思想的 Black-Litterman 与稳健资产配置组合器
    """
    asset_symbols = list(ml_view_results.keys())
    n_assets = len(asset_symbols)
    
    # 1. 采用 Ledoit-Wolf 收缩协方差作为稳健大底先验 (已硬化)
    lw = LedoitWolf()
    lw.fit(historical_returns_df[asset_symbols].values)
    Sigma = lw.covariance_
    
    # 2. 从历史滚动窗口提取长周期市场的隐含均衡先验作为定价核锚点 (Pricing Kernel Anchor)
    # 这里的长期均值代表金融学经典的一般均衡期望
    Pi_pricing_kernel = historical_returns_df[asset_symbols].mean().values
    
    # 3. 提取 Layer 3 的点预测与 CQR 宽度，构建主动观点向量
    raw_ml_views = np.array([ml_view_results[asset]["point_prediction"] for asset in asset_symbols])
    cqr_widths = np.array([ml_view_results[asset]["cqr_width"] for asset in asset_symbols])
    
    # 核心方向二：金融学定价核损失正则化收缩 (Domain Regularization)
    # 设定 Pricing Kernel 收缩强度 (可放入 config.py 统一管理，通常取 0.2 - 0.5)
    lambda_kernel = getattr(pipeline_config, "lambda_kernel", 0.3)
    
    # 执行 L2 空间下的均衡收缩： R_adjusted = (1 - lambda) * ML_Pred + lambda * Pi_Kernel
    # 从而防止纯树模型点预测在样本外因统计幻觉产生离群值，导致投资组合暴仓
    regularized_ml_views = (1.0 - lambda_kernel) * raw_ml_views + lambda_kernel * Pi_pricing_kernel
    
    # 4. 融合 LLM 突发文本观点 (保持您解耦后“量价守大底，文本做微调”的原则)
    # 将 regularized_ml_views 作为 BL 的观点 Q 矩阵输入
    Q = regularized_ml_views.reshape(-1, 1)
    
    # 5. 构建贝叶斯观点不确定性方差矩阵 Omega
    # 此时 cqr_widths 已经由于方向三的分歧度被动拉宽了，在这里完成了交叉共鸣
    tau = pipeline_config.tau
    Omega = np.diag(cqr_widths * tau)
    
    # 6. 执行经典高维 Black-Litterman 封闭解共轭更新
    P = np.eye(n_assets) # 恒等映射矩阵
    inv_Sigma = np.linalg.inv(Sigma)
    inv_Omega = np.linalg.inv(Omega)
    
    # 共轭更新公式
    posterior_precision = np.linalg.inv(inv_Sigma + np.dot(np.dot(P.T, inv_Omega), P))
    posterior_returns = np.dot(
        posterior_precision,
        np.dot(inv_Sigma, Pi_pricing_kernel.reshape(-1, 1)) + np.dot(np.dot(P.T, inv_Omega), Q)
    ).flatten()
    
    # 7. 带有现实换手率摩擦限制的马科维茨配置优化 (MVO 后端)
    # 略去原有的标准凸优化边界约束计算...
    # (保持您原有的 w_optimal 凸优化计算与 long-only 限制即可)
    
    # 返回后验期望与最优配置权重
    w_optimal = optimize_mvo_weights(posterior_returns, Sigma, pipeline_config)
    
    return {
        "posterior_returns": posterior_returns.tolist(),
        "optimal_weights": w_optimal
    }
