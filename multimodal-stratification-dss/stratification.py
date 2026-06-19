# new file added - multimodal-stratification-dss/stratification.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

class MultimodalAssetStratifier:
    """
    基于徐亦达教授团队 Yin et al. (2025) 决策支持系统思想构建的
    对比多模态资产分层引擎 (Contrastive Multimodal Asset Stratifier)
    """
    def __init__(self, n_cohorts=3, cohort_correlation=0.3, random_state=42):
        self.n_cohorts = n_cohorts
        self.cohort_correlation = cohort_correlation
        self.random_state = random_state
        self.scaler = StandardScaler()
        
    def fit_stratify(self, cqr_widths: dict, llm_views: dict) -> dict:
        """
        在横截面上对资产进行多模态分层
        :param cqr_widths: dict, 键为资产名称，值为当前步骤的 CQR 误差带宽度
        :param llm_views: dict, 键为资产名称，值为当前步骤的 LLM 情绪得分/观点值
        :return: dict, 键为资产名称，值为所属的队列标签 (0, 1, ..., n_cohorts-1)
        """
        assets = list(cqr_widths.keys())
        if len(assets) < self.n_cohorts:
            # 资产数量太少时退化为单队列
            return {asset: 0 for asset in assets}
        
        # 1. 抽取多模态异构特征
        features = []
        for asset in assets:
            w = cqr_widths[asset]
            v = llm_views[asset]
            features.append([w, v])
            
        features_arr = np.array(features)
        
        # 2. 跨模态特征标准化 (模拟隐空间对齐的前置缩放)
        scaled_features = self.scaler.fit_transform(features_arr)
        
        # 3. 部署高斯混合模型进行无监督风险-情绪分层 (Asset Stratification)
        gmm = GaussianMixture(n_components=self.n_cohorts, 
                              covariance_type='diag', 
                              random_state=self.random_state)
        cohort_labels = gmm.fit_predict(scaled_features)
        
        return {assets[i]: cohort_labels[i] for i in range(len(assets))}

    def construct_stratified_omega(self, cqr_widths: dict, cohort_assignments: dict, base_omega_diag: np.ndarray, asset_list: list) -> np.ndarray:
        """
        利用分层决策系统逻辑，重构 Black-Litterman 观点不确定性矩阵 Omega
        将组内资产赋予非对角的跨模态相关性共振系数
        """
        n_assets = len(asset_list)
        Omega = np.zeros((n_assets, n_assets))
        
        # 先填充对角线（由原本集成 CQR 与认知分歧度决定的基础方差）
        for i in range(n_assets):
            Omega[i, i] = base_omega_diag[i]
            
        # 注入组内横截面共振协方差 (Block-Diagonal Structuring)
        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                asset_a = asset_list[i]
                asset_b = asset_list[j]
                
                cohort_a = cohort_assignments.get(asset_a, -1)
                cohort_b = cohort_assignments.get(asset_b, -2)
                
                # 如果两只资产处于同一个决策分层队列（如高风险-高情绪迷茫组）
                if cohort_a == cohort_b:
                    # 协方差 = 风险相关性 * 标准差A * 标准差B
                    std_a = np.sqrt(Omega[i, i])
                    std_b = np.sqrt(Omega[j, j])
                    covariance = self.cohort_correlation * std_a * std_b
                    Omega[i, j] = covariance
                    Omega[j, i] = covariance
                    
        # 增加微小扰动确保矩阵绝对正定
        Omega += np.eye(n_assets) * 1e-6
        return Omega
