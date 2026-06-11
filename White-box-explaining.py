import numpy as np
import pandas as pd
import shap
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler

def white_box_feature_engineering(df):
    """
    将原始ETF价格序列重构为符合资产定价学术规范的特征体系
    """
    processed_df = pd.DataFrame(index=df.index)
    processed_df['date'] = df['日期']
    
    # 1. 目标变量重构：对数收益率 (Log Return) 替代绝对价格
    processed_df['target_return'] = np.log(df['收盘'] / df['收盘'].shift(1)).shift(-1) 
    
    # 2. 特征一：经典时序动量因子 (Time-Series Momentum)
    processed_df['mom_1d'] = np.log(df['收盘'] / df['收盘'].shift(1))
    processed_df['mom_5d'] = np.log(df['收盘'] / df['收盘'].shift(5))
    
    # 3. 特征二：标准化特质波动率 (Idiosyncratic Volatility Proxy)
    # 使用 Garman-Klass 极端值波动率估计法，比单纯的最高/最低价更具学术认可度
    processed_df['gk_vol'] = 0.5 * (np.log(df['最高'] / df['最低']))**2 - (2 * np.log(2) - 1) * (np.log(df['收盘'] / df['开盘']))**2
    
    # 4. 特征三：市场情绪变动率 (Turnover Saliency)
    processed_df['turnover_shck'] = df['换手率'] / df['换手率'].rolling(5).mean()
    
    return processed_df.dropna()

def train_and_explain_model(X, y):
    """
    使用SHAP对机器学习模型进行白盒解构，论证因子对资产定价的边际贡献
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    # 训练可解释的轻量级树模型
    model = lgb.LGBMRegressor(n_estimators=50, max_depth=3, random_state=42)
    model.fit(X_scaled_df, y)
    
    # 构建白盒解释器
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled_df)
    
    print("【PhD白盒验证】已生成特征的学术可解释性矩阵（SHAP值）。")
    # 在Notebook中可以使用 shap.summary_plot(shap_values, X_scaled_df) 进行可视化导出
    return model, explainer, shap_values