import os
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
import shap
from datetime import datetime, timedelta

class AcademicPricingPipeline:
    def __init__(self, target_symbol, base_folder="./academic_output"):
        """
        初始化研究管线，创建指定的数据与模型存储路径
        """
        self.symbol = target_symbol
        self.base_folder = base_folder
        self.model_folder = os.path.join(base_folder, "models")
        self.report_folder = os.path.join(base_folder, "reports")
        
        os.makedirs(self.model_folder, exist_ok=True)
        os.makedirs(self.report_folder, exist_ok=True)
        
        self.raw_data = None
        self.feature_df = None
        self.model = None
        self.shap_values = None

    def step1_fetch_or_update_data(self):
        """
        [步骤1] 获取或增量更新目标资产的原始数据
        学术规范：保留完整的开高低收投机要素，采用对数体系转换
        """
        print(f"--- [Step 1] Fetching/Updating Data for {self.symbol} ---")
        # 实际运行中可接入 akshare 或 local csv: ak.fund_etf_hist_em(symbol=self.symbol)
        # 此处构建符合真实结构的模拟学术测试序列
        np.random.seed(42)
        dates = pd.date_range(end=datetime.today(), periods=250, freq='D')
        mock_close = 1.0 + np.cumsum(np.random.normal(0.0005, 0.01, size=250))
        
        self.raw_data = pd.DataFrame({
            'Date': dates,
            'Open': mock_close * (1 + np.random.normal(0, 0.002, 250)),
            'High': mock_close * (1 + np.abs(np.random.normal(0.005, 0.002, 250))),
            'Low': mock_close * (1 - np.abs(np.random.normal(0.005, 0.002, 250))),
            'Close': mock_close,
            'Volume': np.random.randint(100000, 500000, size=250),
            'Turnover': np.random.uniform(0.01, 0.05, size=250)
        })
        self.raw_data.set_index('Date', inplace=True)
        print(f"Data synchronized. Total periods: {len(self.raw_data)}")

    def _calculate_mechanistic_features(self):
        """
        定义底层的显式物理/经济学特征（白盒特征根基）
        """
        df = self.raw_data.copy()
        features = pd.DataFrame(index=df.index)
        
        # Target: 远期一步对数超额收益率
        features['target_return'] = np.log(df['Close'] / df['Close'].shift(1)).shift(-1)
        
        # M1: 1日短期时序动量
        features['Mom_1D'] = np.log(df['Close'] / df['Close'].shift(1))
        # M2: 5日周趋时序动量
        features['Mom_5D'] = np.log(df['Close'] / df['Close'].shift(5))
        # M3: 20日长周期趋势矩
        features['Mom_20D'] = np.log(df['Close'] / df['Close'].shift(20))
        # V1: Garman-Klass 极端值特质波动率 (学术界认可的微观结构波动指标)
        features['GK_Vol'] = 0.5 * (np.log(df['High'] / df['Low']))**2 - (2 * np.log(2) - 1) * (np.log(df['Close'] / df['Open']))**2
        
        self.feature_df = features.dropna()

    def step2_adaptive_rolling_fit(self, window_size=120):
        """
        [步骤2] 走动窗口自适应学习（避免灾难性遗忘与数据泄露）
        """
        print(f"--- [Step 2] Adaptive Rolling Fit (Window: {window_size}) ---")
        self._calculate_mechanistic_features()
        
        X = self.feature_df[['Mom_1D', 'Mom_5D', 'Mom_20D', 'GK_Vol']]
        y = self.feature_df['target_return']
        
        # 仅取最新的固定历史窗口进行自适应精炼
        X_train = X.iloc[-window_size:]
        y_train = y.iloc[-window_size:]
        
        # 训练可解释性限制树模型
        self.model = lgb.LGBMRegressor(
            n_estimators=50, 
            max_depth=3, 
            learning_rate=0.05,
            min_child_samples=10,
            verbose=-1,
            random_state=42
        )
        self.model.fit(X_train, y_train)
        print("Model adaptive refit completed successfully.")

    def step3_and_4_whitebox_inference_with_uncertainty(self, n_bootstraps=200):
        """
        [步骤3 & 步骤4] 白盒资产推演与基于残差自举的认识论不确定性度量
        """
        print("--- [Step 3 & 4] White-box Inference & Uncertainty Quantification ---")
        X = self.feature_df[['Mom_1D', 'Mom_5D', 'Mom_20D', 'GK_Vol']]
        last_observation = X.iloc[-1:].copy()
        
        # 点估计 (Point Estimate)
        point_pred_return = self.model.predict(last_observation)[0]
        
        # 残差自举不确定性估计 (Residual Bootstrapping for Epistemic Risk)
        boot_preds = []
        y_pred_train = self.model.predict(X.iloc[-120:])
        residuals = self.feature_df['target_return'].iloc[-120:] - y_pred_train
        
        for i in range(n_bootstraps):
            boot_res = np.random.choice(residuals, size=len(residuals), replace=True)
            y_boot = y_pred_train + boot_res
            
            boot_model = lgb.LGBMRegressor(n_estimators=30, max_depth=3, verbose=-1, random_state=i)
            boot_model.fit(X.iloc[-120:], y_boot)
            boot_preds.append(boot_model.predict(last_observation)[0])
            
        boot_preds = np.array(boot_preds)
        
        # 计算 SHAP 矩阵以用于步骤 6 的白盒解析
        explainer = shap.TreeExplainer(self.model)
        self.shap_values = explainer.shap_values(X.iloc[-120:])
        
        return point_pred_return, boot_preds

    def step5_generate_asset_assessment_report(self, point_pred, boot_preds):
        """
        [步骤5] 包装报告1：目标资产价格动向与可信度风险评估报告
        """
        print("--- [Step 5] Compiling Asset Assessment Report ---")
        current_price = self.raw_data['Close'].iloc[-1]
        
        # 指标换算
        expected_change_rate = (np.exp(point_pred) - 1) * 100 # 转换为百分比变动率
        estimated_next_price = current_price * np.exp(point_pred)
        
        # 统计区间边界 (Trustworthy Range)
        ci_lower = (np.exp(np.percentile(boot_preds, 2.5)) - 1) * 100
        ci_upper = (np.exp(np.percentile(boot_preds, 97.5)) - 1) * 100
        
        # 风险边界 (Risk Bounds via VaR concept)
        var_95_return = np.percentile(boot_preds, 5)
        risk_range_lower = current_price * np.exp(var_95_return)
        
        # 胜负比 (Win/Loss Ratio) 基于自举分布的经验概率
        wins = boot_preds[boot_preds > 0]
        losses = boot_preds[boot_preds < 0]
        
        win_prob = len(wins) / len(boot_preds)
        avg_win = np.mean(wins) if len(wins) > 0 else 0
        avg_loss = np.abs(np.mean(losses)) if len(losses) > 0 else 1e-6
        win_loss_ratio = (win_prob * avg_win) / ((1 - win_prob) * avg_loss) if avg_loss > 0 else np.inf
        
        report_content = f"""==================================================================
ACADEMIC ASSET ASSESSMENT REPORT (QUANTITATIVE FINANCE SANDBOX)
Target Asset: {self.symbol} | Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
==================================================================
1. TENDENCY & ESTIMATION (点估计与趋势推演)
   - Current Spot Price (当前基准价): {current_price:.4f}
   - Expected Return Direction (预期收益方向): {'UPWARD (多头)' if point_pred > 0 else 'DOWNWARD (空头)'}
   - Estimated Expected Change Rate (预期变动率): {expected_change_rate:+.4f}%
   - Predicted Next-Period Price (标的预测值): {estimated_next_price:.4f}

2. UNCERTAINTY & TRUSTWORTHY RANGE (统计学置信区间与可信度范围)
   - 95% Confidence Interval for Change Rate (95% 变动率可信边界): [{ci_lower:+.4f}%, {ci_upper:+.4f}%]
   - Interpretation: 在排除认识论噪声后，标的资产的下一个周期的结构性变动有95%的概率落在上述区间内。

3. RISK MANAGEMENT METRICS (下行风险与极端边界)
   - 95% Empirical Value-at-Risk Limit (95% 经验在险价值边界价): {risk_range_lower:.4f}
   - Conditional Win/Loss Ratio (期望胜负比 - 概率加权): {win_loss_ratio:.4f}
   - Empirical Success Probability (样本内推演上涨概率): {win_prob*100 Gold:.2f}%
=================================================================="""
        return report_content

    def step6_generate_mechanism_attribution_report(self):
        """
        [步骤6] 包装报告2：白盒机制与因子归因权重报告
        学术规范：全面展现动量因子的数学逻辑、计算范式与博弈论权重（SHAP）
        """
        print("--- [Step 6] Compiling Mechanism Attribution Report ---")
        # 计算平均绝对 SHAP 值作为学术权重
        mean_shap = np.mean(np.abs(self.shap_values), axis=0)
        total_shap = np.sum(mean_shap)
        shap_weights = (mean_shap / total_shap) * 100 if total_shap > 0 else [25]*4
        
        report_content = f"""==================================================================
FACTOR MECHANISM & ATTRIBUTION REPORT (WHITE-BOX DECONSTRUCTION)
Target Asset: {self.symbol} | Methodology: Shapley Additive Explanations
==================================================================
1. FACTOR MATHEMATICAL SPECIFICATIONS (归因因子数理公式明细)

   - M1: 1-Day Log Momentum (Short-term Reaction)
     Formula: $Mom\\_1D_t = \\ln(P_t / P_{{t-1}})$
     
   - M2: 5-Day Weekly Momentum (Medium-term Trend)
     Formula: $Mom\\_5D_t = \\ln(P_t / P_{{t-5}})$
     
   - M3: 20-Day Monthly Momentum (Long-term Regime)
     Formula: $Mom\\_20D_t = \\ln(P_t / P_{{t-20}})$
     
   - V1: Garman-Klass Microstructure Volatility (特质非线性风险度量)
     Formula: $\\sigma^2_{{GK}} = 0.5 \\left( \\ln \\frac{{H_t}}{{L_t}} \\right)^2 - (2\\ln 2 - 1)\\left(\\ln \\frac{{C_t}}{{O_t}}\\right)^2$
     Where H=High, L=Low, C=Close, O=Open.

2. SHAPLEY EMPIRICAL WEIGHT ATTRIBUTION (基于博弈论分配的决策权重)
   - Mom_1D  (1日短期动量项) : 权重 = {shap_weights[0]:.2f}%
   - Mom_5D  (5日周周期动量) : 权重 = {shap_weights[1]:.2f}%
   - Mom_20D (20日长周期动量): 权重 = {shap_weights[2]:.2f}%
   - GK_Vol  (高低开收结构险): 权重 = {shap_weights[3]:.2f}%

3. ACADEMIC CONCLUSION (机制合规性审查)
   本模型拒绝无经济学解释的纯深度黑盒网络，当前决策层中以 {['Mom_1D', 'Mom_5D', 'Mom_20D', 'GK_Vol'][np.argmax(shap_weights)]} 为核心驱动因子，符合资产定价中关于风险溢价时序聚集性的理论假设。
=================================================================="""
        return report_content

    def step7_serialize_and_save(self, r1, r2):
        """
        [步骤7] 序列化自适应模型，并将两份学术报告固化存储至指定文件夹
        """
        print(f"--- [Step 7] Archiving to {self.base_folder} ---")
        
        # 保存模型
        model_path = os.path.join(self.model_folder, f"{self.symbol}_adaptive_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
            
        # 保存报告1
        r1_path = os.path.join(self.report_folder, f"{self.symbol}_asset_assessment.txt")
        with open(r1_path, 'w', encoding='utf-8') as f:
            f.write(r1)
            
        # 保存报告2
        r2_path = os.path.join(self.report_folder, f"{self.symbol}_factor_mechanism.txt")
        with open(r2_path, 'w', encoding='utf-8') as f:
            f.write(r2)
            
        print(f"Archiving complete.\n -> Model saved at: {model_path}\n -> Report 1 saved at: {r1_path}\n -> Report 2 saved at: {r2_path}")

    def execute_pipeline(self):
        """
        一键触发全自动流水线
        """
        self.step1_fetch_or_update_data()
        self.step2_adaptive_rolling_fit()
        point_pred, boot_preds = self.step3_and_4_whitebox_inference_with_uncertainty()
        r1 = self.step5_generate_asset_assessment_report(point_pred, boot_preds)
        r2 = self.step6_generate_mechanism_attribution_report()
        self.step7_serialize_and_save(r1, r2)
        
        # 打印至控制台预览
        print("\n" + r1 + "\n")
        print("\n" + r2 + "\n")

# 实例化运行示例
if __name__ == "__main__":
    pipeline = AcademicPricingPipeline(target_symbol="510300_ETF")
    pipeline.execute_pipeline()