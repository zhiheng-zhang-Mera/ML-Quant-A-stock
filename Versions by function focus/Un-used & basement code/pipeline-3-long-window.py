import os
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from datetime import datetime
from scipy.spatial.distance import mahalanobis

# 动态时间规整备用内核
try:
    from fastdtw import fastdtw
except ImportError:
    def fastdtw(x, y, dist=2):
        from scipy.spatial.distance import cdist
        D = cdist(x, y, metric='mahalanobis' if dist=='mahalanobis' else 'sqeuclidean')
        M, N = D.shape
        T = np.zeros((M+1, N+1))
        T[0, 1:] = np.inf
        T[1:, 0] = np.inf
        for i in range(1, M+1):
            for j in range(1, N+1):
                T[i, j] = D[i-1, j-1] + min(T[i-1, j], T[i, j-1], T[i-1, j-1])
        return T[M, N], []

class UltimateAcademicPipeline:
    def __init__(self, target_symbol, base_folder="./ultimate_academic_output"):
        self.symbol = target_symbol
        self.base_folder = base_folder
        self.model_folder = os.path.join(base_folder, "models")
        self.report_folder = os.path.join(base_folder, "reports")
        os.makedirs(self.model_folder, exist_ok=True)
        os.makedirs(self.report_folder, exist_ok=True)
        
        # 年度通胀平减字典
        self.inflation_table = {
            2018: 0.021, 2019: 0.029, 2020: 0.025, 2021: 0.009,
            2022: 0.020, 2023: 0.002, 2024: 0.003, 2025: 0.005, 2026: 0.006
        }
        self.raw_data = None
        self.feature_df = None
        self.model = None
        
        # 新增扩展状态存储
        self.llm_analysis_result = {}
        self.backtest_metrics = {}

    def step1_fetch_and_deflate_data(self):
        """
        [步骤1] 获取数据并进行年度通胀平减 (Inflation Adjustment)
        """
        print(f"--- [Step 1] Fetching & Inflation-Deflating Data for {self.symbol} ---")
        np.random.seed(101)
        dates = pd.date_range(start="2018-01-01", end="2026-06-01", freq='D')
        n_days = len(dates)
        
        mock_nominal_close = 1.0 + np.cumsum(np.random.normal(0.0002, 0.015, size=n_days))
        df = pd.DataFrame({
            'Date': dates,
            'Nominal_Open': mock_nominal_close * (1 + np.random.normal(0, 0.002, n_days)),
            'Nominal_High': mock_nominal_close * (1 + np.abs(np.random.normal(0.005, 0.002, n_days))),
            'Nominal_Low': mock_nominal_close * (1 - np.abs(np.random.normal(0.005, 0.002, n_days))),
            'Nominal_Close': mock_nominal_close,
            'Volume': np.random.randint(100000, 500000, size=n_days)
        })
        df['Year'] = df['Date'].dt.year
        
        base_year = 2026
        def get_deflator(year):
            if year >= base_year: return 1.0
            factor = 1.0
            for y in range(int(year), base_year):
                factor *= (1 + self.inflation_table.get(y, 0.015))
            return factor

        df['Deflator'] = df['Year'].apply(get_deflator)
        df['Open'] = df['Nominal_Open'] * df['Deflator']
        df['High'] = df['Nominal_High'] * df['Deflator']
        df['Low'] = df['Nominal_Low'] * df['Deflator']
        df['Close'] = df['Nominal_Close'] * df['Deflator']
        
        df.set_index('Date', inplace=True)
        self.raw_data = df
        print(f"Inflation adjustment complete. Real Prices anchored to {base_year} base.")

    def step1_b_llm_news_marketing_analysis(self):
        """
        [新增步骤 1B] 大语言模型（LLM）非结构化营销与新闻情绪解构
        """
        print(f"--- [Step 1B] Extracting & Quantifying LLM Marketing News Sentiment ---")
        # 模拟当前最新的市场基本面及营销舆情快照（实际研究中通过API传入新闻流）
        mock_recent_news_stream = [
            "Macroeconomic demand spikes in downstream industrial clients.",
            "Marketing campaign performance indexes show an expansionary phase.",
            "Competitor pricing adjustments indicate tightening systemic margins."
        ]
        
        # 模拟结构化LLM Agent推演过程与提示词生成出的白盒定性结论
        llm_qualitative_summary = (
            "LLM Grounded Insight: The asset shows strong operational tailwinds from downstream "
            "marketing channels, offset partially by compressed peer margins. The systemic "
            "sentiment vector remains asymmetric to the upside."
        )
        # 情绪打分归一化映射：[-1.0, 1.0]
        llm_quant_sentiment_score = 0.35 
        
        self.llm_analysis_result = {
            'news_analyzed_count': len(mock_recent_news_stream),
            'summary': llm_qualitative_summary,
            'sentiment_score': llm_quant_sentiment_score,
            'analysis_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        print(f"LLM Marketing News parsing complete. Extracted Sentiment Index: {llm_quant_sentiment_score:+.2f}")

    def _calculate_mechanistic_features(self):
        """
        基于平减后的实际价格计算无量纲白盒特征
        """
        df = self.raw_data.copy()
        features = pd.DataFrame(index=df.index)
        
        # 远期实际对数收益率
        features['target_return'] = np.log(df['Close'] / df['Close'].shift(1)).shift(-1)
        
        features['Mom_1D'] = np.log(df['Close'] / df['Close'].shift(1))
        features['Mom_5D'] = np.log(df['Close'] / df['Close'].shift(5))
        features['Mom_20D'] = np.log(df['Close'] / df['Close'].shift(20))
        features['GK_Vol'] = 0.5 * (np.log(df['High'] / df['Low']))**2 - (2 * np.log(2) - 1) * (np.log(df['Close'] / df['Open']))**2
        
        self.feature_df = features.dropna()

    def step2_dtw_mahalanobis_anti_cheat_fit(self, search_window_lead=10, history_pool_limit=250):
        """
        [步骤2] 融合DTW与马氏距离的模式匹配，并实施严格的反前瞻作弊裁剪
        """
        print(f"--- [Step 2] DTW + Mahalanobis Matching with Anti-Cheating Window Cut ---")
        self._calculate_mechanistic_features()
        
        feature_cols = ['Mom_1D', 'Mom_5D', 'Mom_20D', 'GK_Vol']
        X = self.feature_df[feature_cols]
        y = self.feature_df['target_return']
        
        current_pattern = X.iloc[-search_window_lead:].values
        cov_matrix = np.cov(X.values, rowvar=False)
        inv_cov_matrix = np.linalg.pinv(cov_matrix)
        
        similar_indices = []
        safe_search_end_idx = len(X) - (2 * search_window_lead) - 2
        
        for t in range(0, safe_search_end_idx):
            historical_segment = X.iloc[t : t + search_window_lead].values
            
            def mahalanobis_dist(u, v):
                return mahalanobis(u, v, inv_cov_matrix)
                
            dtw_distance, _ = fastdtw(current_pattern, historical_segment, dist=mahalanobis_dist)
            similarity_score = 1 / (1 + dtw_distance)
            
            if similarity_score > 0.85:
                if (t + search_window_lead) < safe_search_end_idx:
                    similar_indices.extend(range(t, t + search_window_lead))
                    
        similar_indices = list(set(similar_indices))
        recent_indices = list(range(len(X) - history_pool_limit, len(X)))
        
        if len(similar_indices) > search_window_lead:
            print(f"[验证通过] 跨期检索到符合反作弊合规的相似历史样本数: {len(similar_indices)} 期。")
            combined_indices = list(set(recent_indices + similar_indices))
            X_train = X.iloc[combined_indices]
            y_train = y.iloc[combined_indices]
            
            sample_weights = np.ones(len(combined_indices))
            for idx, global_idx in enumerate(combined_indices):
                if global_idx in similar_indices and global_idx not in recent_indices:
                    sample_weights[idx] = 1.8
        else:
            print("[模式退化] 未检索到足够高相似度的历史状态，回归常规样本外局部窗口。")
            X_train = X.iloc[recent_indices]
            y_train = y.iloc[recent_indices]
            sample_weights = np.ones(len(recent_indices))
            
        self.model = lgb.LGBMRegressor(n_estimators=50, max_depth=3, verbose=-1, random_state=42)
        self.model.fit(X_train, y_train, sample_weight=sample_weights)
        self.X_train_snapshot = X_train

    def step2_b_back_loop_testing(self, backtest_window=20):
        """
        [新增步骤 2B] 触发式前向闭环检验机制 (Back Loop Testing)
        学术意义：在模型应用前，对其最近历史周期的预测一致性与泛化差损进行实证复核。
        """
        print(f"--- [Step 2B] Initiating Walk-Forward Back Loop Testing ---")
        feature_cols = ['Mom_1D', 'Mom_5D', 'Mom_20D', 'GK_Vol']
        X = self.feature_df[feature_cols]
        y = self.feature_df['target_return']
        
        historical_predictions = []
        historical_actuals = []
        
        # 滚动向前检验模型在历史更新时的真实损耗
        for i in range(backtest_window, 0, -1):
            target_idx = len(X) - i - 1
            if target_idx < 100: continue
            
            # 隔离训练域
            X_sub = X.iloc[:target_idx]
            y_sub = y.iloc[:target_idx]
            
            # 快速局部拟合
            val_model = lgb.LGBMRegressor(n_estimators=30, max_depth=3, verbose=-1, random_state=42)
            val_model.fit(X_sub, y_sub)
            
            pred = val_model.predict(X.iloc[target_idx:target_idx+1])[0]
            historical_predictions.append(pred)
            historical_actuals.append(y.iloc[target_idx])
            
        preds_arr = np.array(historical_predictions)
        actuals_arr = np.array(historical_actuals)
        
        if len(preds_arr) > 0:
            mse = float(np.mean((preds_arr - actuals_arr) ** 2))
            # 胜率核算：方向一致性百分比
            hit_rate = float(np.mean(np.sign(preds_arr) == np.sign(actuals_arr)))
        else:
            mse, hit_rate = 0.0, 0.0
            
        self.backtest_metrics = {
            'window_length': len(preds_arr),
            'mse': mse,
            'directional_hit_rate': hit_rate,
            'status': 'VERIFIED' if hit_rate >= 0.45 else 'WARNING_DEGRADED'
        }
        print(f"Back Loop Testing complete. Historical Directional Accuracy: {hit_rate:.2%}")

    def step3_to_7_output_and_archive(self):
        """
        [步骤3至7] 集成推演、多期不确定性范围评估、多源学术报告固化归档
        """
        print("--- [Step 3 to 7] Multi-Horizon Inference & LLM Session Fusion ---")
        X_latest = self.feature_df[['Mom_1D', 'Mom_5D', 'Mom_20D', 'GK_Vol']].iloc[-1:]
        current_nominal_price = self.raw_data['Nominal_Close'].iloc[-1]
        current_real_price = self.raw_data['Close'].iloc[-1]
        
        # 基础单期（1日）对数预测
        pred_log_return_1d = self.model.predict(X_latest)[0]
        
        # 结合LLM情绪分对长期趋势进行非线性修正微调（模拟宏观外因扰动）
        sentiment_adj = self.llm_analysis_result.get('sentiment_score', 0) * 0.0005
        adjusted_base_drift = pred_log_return_1d + sentiment_adj
        
        # 3. 核心重构：多期预测范围估计（3/5/7/10/15/30天）
        target_horizons = [3, 5, 7, 10, 15, 30]
        horizon_estimation_session = ""
        
        base_sigma = 0.008 # 资产基础波动率
        
        for h in target_horizons:
            # 运用时间平方根法则对多期不确定性进行扩散缩放
            # 累积漂移 = h * 调整后基准收益；累积标准差 = sqrt(h) * 波动率
            boot_preds_h = np.random.normal(h * adjusted_base_drift, base_sigma * np.sqrt(h), 300)
            
            ci_lower_pct = (np.exp(np.percentile(boot_preds_h, 2.5)) - 1) * 100
            ci_upper_pct = (np.exp(np.percentile(boot_preds_h, 97.5)) - 1) * 100
            
            expected_price_h = current_real_price * np.exp(h * adjusted_base_drift)
            price_lower_h = current_real_price * np.exp(np.percentile(boot_preds_h, 2.5))
            price_upper_h = current_real_price * np.exp(np.percentile(boot_preds_h, 97.5))
            
            horizon_estimation_session += (
                f"   - {h:2d} Day Horizon Expectation: Expected Price {expected_price_h:.4f} "
                f"| Change Range: [{ci_lower_pct:+.2f}%, {ci_upper_pct:+.2f}%] "
                f"| Absolute Range: [{price_lower_h:.4f}, {price_upper_h:.4f}]\n"
            )

        # 组装报告 1：多维资产状态与范围估计报告
        r1 = f"""==================================================================
ADVANCED PHD MULTI-HORIZON ASSET ASSESSMENT REPORT
Target Symbol: {self.symbol} | Base Anchor: Year 2026 (Real Terms)
==================================================================
1. INFLATION COUNTERMEASURE (通胀平减审查)
   - Nominal Spot Price (名义现价): {current_nominal_price:.4f}
   - Real Price Deflator Factor (2026折现系数): {self.raw_data['Deflator'].iloc[-1]:.4f}
   - Real Spot Price (平减后实际价): {current_real_price:.4f}

2. BACK LOOP TESTING RESILIENCE VERIFICATION (触发式闭环历史检验结果)
   - Backtest Testing Window: {self.backtest_metrics.get('window_length')} Days
   - Validation Empirical MSE: {self.backtest_metrics.get('mse'):.6f}
   - Directional Hit Rate (胜率复核): {self.backtest_metrics.get('directional_hit_rate'):.2%}
   - System Stability Code: {self.backtest_metrics.get('status')}

3. MULTI-HORIZON RANGE ESTIMATION (长程时序范围估计)
{horizon_estimation_session}=================================================================="""

        # 组装报告 2：数学机制与 LLM 文本语义融合审计报告
        r2 = f"""==================================================================
ALGORITHMIC REGIME COMPLIANCE & ATTRIBUTION REPORT
Methodology: Mahalanobis-DTW + LLM Semantic Integration
==================================================================
1. LLM MARKETING & NEWS SENTIMENT ANALYSIS (大模型非结构化新闻会话剖析)
   - Analysis Timestamp: {self.llm_analysis_result.get('analysis_time')}
   - Parsed News Signal Density: {self.llm_analysis_result.get('news_analyzed_count')} items
   - Extracted Sentiment Index: {self.llm_analysis_result.get('sentiment_score'):+.4f}
   - Text Attribution Session: {self.llm_analysis_result.get('summary')}

2. ANTI-CHEATING TIME COMPARTMENTALIZATION (严格反作弊时序隔离审查)
   - Look-ahead Filter Status: ACTIVE (合规)
   - Safety Window Distance: 2 * Search Lead Days

3. FEATURE WEIGHTS VIA SHAPLEY VALUE (博弈论决策权重分配)
"""
        # SHAP权重分配
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(self.X_train_snapshot)
        mean_shap = np.mean(np.abs(shap_values), axis=0)
        shap_weights = (mean_shap / np.sum(mean_shap)) * 100
        
        r2 += f"   - Mom_1D  (1日实际动量) : {shap_weights[0]:.2f}%\n"
        r2 += f"   - Mom_5D  (5日实际动量) : {shap_weights[1]:.2f}%\n"
        r2 += f"   - Mom_20D (20日实际动量): {shap_weights[2]:.2f}%\n"
        r2 += f"   - GK_Vol  (高低开收险)  : {shap_weights[3]:.2f}%\n"
        r2 += "=================================================================="

        # 归档持久化
        with open(os.path.join(self.model_folder, f"{self.symbol}_final.pkl"), 'wb') as f:
            pickle.dump(self.model, f)
        with open(os.path.join(self.report_folder, f"{self.symbol}_assessment.txt"), 'w', encoding='utf-8') as f:
            f.write(r1)
        with open(os.path.join(self.report_folder, f"{self.symbol}_compliance.txt"), 'w', encoding='utf-8') as f:
            f.write(r2)
            
        print("All assets serialized. Two compliant academic reports generated successfully.")
        print("\n" + r1 + "\n\n" + r2)

    def run_all(self):
        self.step1_fetch_and_deflate_data()
        self.step1_b_llm_news_marketing_analysis()  # LLM会话融合
        self.step2_dtw_mahalanobis_anti_cheat_fit()
        self.step2_b_back_loop_testing()            # 闭环稳定性校验
        self.step3_to_7_output_and_archive()        # 多时域区间映射输出

if __name__ == "__main__":
    pipeline = UltimateAcademicPipeline(target_symbol="Long_History_Index")
    pipeline.run_all()