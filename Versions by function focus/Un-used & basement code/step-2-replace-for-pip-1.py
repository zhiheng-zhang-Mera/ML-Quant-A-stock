import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.spatial.distance import mahalanobis

def step2_similarity_regime_fit(self, search_window_lead=10, history_pool_limit=180, similarity_threshold=0.7):
    """
    [步骤2终极升级版] 基于特征空间马氏距离的时序模式匹配与条件自适应训练
    学术意义：主动识别历史相似状态，打破单纯的时间线性权重，实现非平稳市场的跨期模式聚类。
    """
    print(f"--- [Step 2 Pattern Match] Initializing Cross-Intertemporal Matching ---")
    self._calculate_mechanistic_features()
    
    # 提取用于定义市场状态的白盒特征矩阵 (如 1D动量、5D动量、GK波动率)
    feature_cols = ['Mom_1D', 'Mom_5D', 'Mom_20D', 'GK_Vol']
    X = self.feature_df[feature_cols]
    y = self.feature_df['target_return']
    
    # 1. 定义“当前近期窗口”的特征行为轨迹 (Target Pattern)
    # 取最近的 search_window_lead 天作为我们要去历史中寻找的目标锚点
    current_pattern = X.iloc[-search_window_lead:]
    current_mean = current_pattern.mean().values
    
    # 计算协方差矩阵的逆，用于后续计算马氏距离（排除多重共线性干扰）
    cov_matrix = np.cov(X.iloc[:-search_window_lead].values, rowvar=False)
    inv_cov_matrix = np.linalg.pinv(cov_matrix)
    
    similar_indices = []
    
    # 2. 在历史长河中滚动检索相似的片段 (排除近期窗口本身，防止数据泄露)
    # 检索范围：从开头到（最新时间 - 2*search_window_lead）
    end_search_idx = len(X) - 2 * search_window_lead
    
    for t in range(0, end_search_idx):
        historical_segment = X.iloc[t : t + search_window_lead]
        hist_mean = historical_segment.mean().values
        
        # 计算当前模式与历史这一片段的马氏距离
        dist = mahalanobis(current_mean, hist_mean, inv_cov_matrix)
        
        # 将距离转化为相似度得分 (得分越高越相似)
        similarity_score = 1 / (1 + dist)
        
        if similarity_score > similarity_threshold:
            # 如果历史这段时期的市场状态与当下高度相似，记录其随后的收益表现索引
            # 注意：我们要抓取的是由于这种状态导致的“后市收益率”（t + search_window_lead 之后）
            similar_indices.extend(range(t, t + search_window_lead))
            
    # 去重
    similar_indices = list(set(similar_indices))
    
    # 3. 条件样本重组逻辑 (Conditional Sample Realignment)
    recent_indices = list(range(len(X) - history_pool_limit, len(X)))
    
    if len(similar_indices) > 5: # 设定最小相似样本门槛
        print(f"[模式触发] 成功在历史中检索到 {len(similar_indices)} 个周期的相似市场状态。")
        
        # 联合打包：近期窗口样本 + 历史相似状态样本
        combined_indices = list(set(recent_indices + similar_indices))
        X_train = X.iloc[combined_indices]
        y_train = y.iloc[combined_indices]
        
        # 赋予历史相似样本和近期样本不同的拟合权重（例如相似样本权重为 1.5）
        sample_weights = np.ones(len(combined_indices))
        for idx, global_idx in enumerate(combined_indices):
            if global_idx in similar_indices and global_idx not in recent_indices:
                sample_weights[idx] = 1.5 # 提高历史相似案例的样本话语权
    else:
        print("[模式退化] 历史长河中未发现高度相似的标的特征，回归纯局部近期窗口训练。")
        X_train = X.iloc[recent_indices]
        y_train = y.iloc[recent_indices]
        sample_weights = np.ones(len(recent_indices))
        
    # 4. 训练自适应白盒模型
    self.model = lgb.LGBMRegressor(n_estimators=50, max_depth=3, verbose=-1, random_state=42)
    self.model.fit(X_train, y_train, sample_weight=sample_weights)
    print("Regime-Similarity Adaptive Model training completed.")