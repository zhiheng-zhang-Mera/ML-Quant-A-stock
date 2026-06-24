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
        
    def generate_stratified_omega(self, cqr_widths: dict, cohorts: dict) -> pd.DataFrame:
        """Constructs a stratified block-diagonal view error matrix Omega."""
        symbols = list(cqr_widths.keys())
        n = len(symbols)
        omega_mat = np.zeros((n, n))
        
        for i, s_i in enumerate(symbols):
            for j, s_j in enumerate(symbols):
                if i == j:
                    # Baseline variance scaling via CQR widths
                    omega_mat[i, j] = (cqr_widths[s_i] ** 2) * self.config.TAU
                elif cohorts[s_i] == cohorts[s_j]:
                    # Asset risk-sentiment co-movement penalty tracking
                    base_variance = cqr_widths[s_i] * cqr_widths[s_j] * self.config.TAU
                    omega_mat[i, j] = base_variance * self.config.RHO_COHORT
                    
        return pd.DataFrame(omega_mat, index=symbols, columns=symbols)