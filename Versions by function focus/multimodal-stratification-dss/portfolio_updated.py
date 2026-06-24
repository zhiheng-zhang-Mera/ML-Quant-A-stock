# old file update - multimodal-stratification-dss/portfolio_updated.py
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from stratification import MultimodalAssetStratifier

def compute_matrix_bl_stratified_optimize(
    returns_df, 
    base_omega_diag, 
    ensemble_views_dict, 
    cqr_widths_dict,
    lambda_kernel=0.1, 
    n_cohorts=3,
    cohort_correlation=0.25
):
    """
    共轭融合版优化器：集成了曹毅教授的定价核收缩(Plan 2)与徐亦达教授的多模态分层(Plan 2-Xu)
    """
    asset_list = list(returns_df.columns)
    n_assets = len(asset_list)
    
    # ---- Step 1: 基础大底构建 ----
    Sigma = returns_df.cov().values * 252  # 年化协方差
    # 假设无风险收益驱动下的市场隐含均衡先验 Pi (Pricing Kernel Base)
    Pi = np.mean(returns_df.values, axis=0) * 252 
    
    # ---- Step 2: 实施曹毅(2025)论文的金融学大底正则化收缩 (Plan 2) ----
    Q_raw = np.array([ensemble_views_dict[asset] for asset in asset_list])
    Q_kernel_shrunk = Q_raw - lambda_kernel * (Q_raw - Pi)
    
    # ---- Step 3: 实施徐亦达(2025)论文的跨模态决策分层 (Plan 2-Xu) ----
    stratifier = MultimodalAssetStratifier(n_cohorts=n_cohorts, cohort_correlation=cohort_correlation)
    cohort_assignments = stratifier.fit_stratify(cqr_widths_dict, ensemble_views_dict)
    
    # 重构非对角多模态共振不确定性矩阵 Omega
    Omega_stratified = stratifier.construct_stratified_omega(
        cqr_widths=cqr_widths_dict,
        cohort_assignments=cohort_assignments,
        base_omega_diag=base_omega_diag,
        asset_list=asset_list
    )
    
    # ---- Step 4: 贝叶斯后验推导 (Black-Litterman 公式扩展版) ----
    # 这里的 P 为单位阵（因为观点直接对应资产本身的预期收益）
    P = np.eye(n_assets)
    tau = 0.05
    
    inv_tau_Sigma = np.linalg.inv(tau * Sigma)
    inv_Omega = np.linalg.inv(Omega_stratified)
    
    # 后验精密度与后验收益率向量
    posterior_precision = inv_tau_Sigma + np.dot(np.dot(P.T, inv_Omega), P)
    posterior_variance = np.linalg.inv(posterior_precision)
    
    posterior_intermed = np.dot(inv_tau_Sigma, Pi) + np.dot(np.dot(P.T, inv_Omega), Q_kernel_shrunk)
    er_bl = np.dot(posterior_variance, posterior_intermed)
    
    # ---- Step 5: 下游优化器（带换手率限制的 MVO 或 CVaR 核心） ----
    # 此处省略常规优化器的繁琐边界设置，直接返回 BL 修正后的预期收益与结构化方差
    return er_bl, Sigma, cohort_assignments
