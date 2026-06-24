# stratification.py
import numpy as np
import pandas as pd
from sklearn.mixture import BayesianGaussianMixture
from config import PipelineConfig

class CrossModalStratifier:
    def __init__(self, config: PipelineConfig):
        self.config = config
        
    def compute_asset_cohorts(self, cqr_widths: dict, llm_views: dict) -> dict:
        """
        利用变分贝叶斯狄利克雷过程混合模型 (DP-GMM) 在共享隐空间中进行资产无监督聚类。
        对应徐亦达教授 (Li & Xu, 2026) 提出的无参数自适应机制分裂思想 (IGMVAE 生产级平替原型)。
        系统无需预设固定的 K 值，算法将根据资产的[不确定性幅宽, 文本语义视图]特征，
        自适应地推导并激活当前的横截面风险群集 (Active Cohorts)。
        """
        matrix_builder = []
        symbols = list(cqr_widths.keys())
        
        for sym in symbols:
            w_metric = cqr_widths[sym]
            v_metric = llm_views.get(sym, 0.0)
            matrix_builder.append([w_metric, v_metric])
            
        data_matrix = np.array(matrix_builder)
        
        # 设定全 Universe 资产数量作为机制成分数量的上限上限 (Upper Bound)
        max_components = len(symbols)
        
        # 实例化狄利克雷过程高斯混合模型 (Dirichlet Process GMM)
        # weight_concentration_prior 越低，模型越倾向于将不必要的群集权重收缩归零
        dp_gmm = BayesianGaussianMixture(
            n_components=max_components,
            weight_concentration_prior_type='dirichlet_process',
            weight_concentration_prior=1.0 / max_components,
            covariance_type='full',
            random_state=42,
            n_init=3,
            max_iter=200
        )
        
        # 执行变分贝叶斯期望最大化拟合，自动识别活跃机制群集
        labels = dp_gmm.fit_predict(data_matrix)
        
        # 格式化输出为资产映射字典
        return {symbols[i]: int(labels[i]) for i in range(len(symbols))}
        
    def generate_stratified_omega(self, cqr_widths: dict, cohorts: dict, llm_views: dict) -> pd.DataFrame:
        """
        构建多模态分层块状非对角主观观点误差方差矩阵 Omega。
        对应徐亦达教授决策支持系统 (Yin et al., 2025) 跨模态对齐思想。
        内生化测度量价时序不确定性与文本情绪大底之间的【跨模态认知撕裂度 (Cross-Modal Dissonance)】。
        """
        symbols = list(cqr_widths.keys())
        n = len(symbols)
        omega_mat = np.zeros((n, n))
        
        for i, s_i in enumerate(symbols):
            for j, s_j in enumerate(symbols):
                if i == j:
                    # 1. 基础方差校准（源自符合性预测异方差带宽）
                    base_variance = (cqr_widths[s_i] ** 2) * self.config.TAU
                    
                    # 2. 【核心理论加固】：计算跨模态认知撕裂度 (Cross-Modal Dissonance)
                    # 当文本主动观点极度激进(绝对值显著大于0)，但量价符合性安全带极其宽泛时，
                    # 意味着双模态信息在隐空间中发生严重冲突（小道消息或情绪幻觉）。
                    text_sentiment_intensity = np.abs(llm_views.get(s_i, 0.0))
                    statistical_uncertainty = cqr_widths[s_i]
                    
                    # 构造指数级互补对比惩罚算子 (Dissonance Multiplier)
                    # 100.0 为缩放系数，用于放大金融低信噪比环境下的微观撕裂特征
                    dissonance_penalty = np.exp(text_sentiment_intensity * statistical_uncertainty * 100.0)
                    
                    # 对角线主观方差根据撕裂度进行主动通胀 (方差扩张 -> 观点精度归零 -> 安全退守量价大底)
                    omega_mat[i, j] = base_variance * dissonance_penalty
                    
                elif cohorts[s_i] == cohorts[s_j]:
                    # 3. 横截面共振惩罚：如果两只资产被 DP-GMM 划分在同一个活跃隐性安全队列中，
                    # 表明它们正在经历相同的跨模态情绪-风险共振，赋予块状非对角交叉相关性系数 RHO_COHORT
                    base_co_variance = cqr_widths[s_i] * cqr_widths[s_j] * self.config.TAU
                    omega_mat[i, j] = base_co_variance * self.config.RHO_COHORT
                    
        return pd.DataFrame(omega_mat, index=symbols, columns=symbols)