# stratification.py
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from config import PipelineConfig

class CrossModalStratifier:
    def __init__(self, config: PipelineConfig):
        self.config = config
        
    def compute_asset_cohorts(self, cqr_widths: dict, llm_views: dict) -> dict:
        """Clusters assets into high-dimensional hidden cross-sectional risk cohorts."""
        matrix_builder = []
        symbols = list(cqr_widths.keys())
        
        for sym in symbols:
            w_metric = cqr_widths[sym]
            v_metric = llm_views.get(sym, 0.0)
            matrix_builder.append([w_metric, v_metric])
            
        data_matrix = np.array(matrix_builder)
        
        # Guard against sample size exceptions
        n_clusters = min(self.config.N_COHORTS, len(symbols))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(data_matrix)
        
        return {symbols[i]: int(labels[i]) for i in range(len(symbols))}
        
    #引入语义不确定性惩罚项
    def generate_stratified_omega(self, cqr_widths: dict, cohorts: dict, llm_views: dict) -> pd.DataFrame:
        symbols = list(cqr_widths.keys())
        n = len(symbols)
        omega_mat = np.zeros((n, n))
    
        for i, s_i in enumerate(symbols):
            for j, s_j in enumerate(symbols):
                if i == j:
                #若大模型观点绝对值过大（情绪激进）或语义模糊，强制引入方差乘数,对应“语义噪声主动防御/贝叶斯收缩机制”
                    llm_sentiment = llm_views.get(s_i, 0.0)
                    semantic_noise_multiplier = 1.0 + np.abs(llm_sentiment) * 2.0  # 模拟文本熵通胀
                    omega_mat[i, j] = (cqr_widths[s_i] ** 2) * self.config.TAU * semantic_noise_multiplier
                    
                elif cohorts[s_i] == cohorts[s_j]:
                    base_variance = cqr_widths[s_i] * cqr_widths[s_j] * self.config.TAU
                    omega_mat[i, j] = base_variance * self.config.RHO_COHORT
                
        return pd.DataFrame(omega_mat, index=symbols, columns=symbols)