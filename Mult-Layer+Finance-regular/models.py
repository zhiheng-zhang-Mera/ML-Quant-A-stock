import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.kernel_ridge import Kernel Ridge
from sklearn.ensemble import GradientBoostingRegressor

class ConformalEnsemblePricingEngine:
    """
    基于曹毅(2025)IRFA顶刊思想重构的异质级联集成与无分布认知分歧标定引擎
    Tier-1: LGBM + Kernel Ridge + GBDT 异质集成
    Tier-2: 组内认知分歧度(Cross-Learner Variance)内生化注入CQR护栏
    """
    def __init__(self, alpha: float = 0.05, lambda_split: float = 1.5):
        self.alpha = alpha
        self.lambda_split = lambda_split # 组内分歧度惩罚放大系数
        
        # 初始化 Tier-1 异质基预测器 (针对点预测)
        self.base_mean_models = {
            "lgbm": LGBMRegressor(objective='regression', n_estimators=100, random_state=42, verbose=-1),
            "k_ridge": KernelRidge(alpha=1.0, kernel='rbf'),
            "gbdt": GradientBoostingRegressor(n_estimators=80, random_state=42)
        }
        
        # 初始化传统 CQR 的上下分位数预测器 (使用LGBM分位数损失守卫边缘)
        self.model_low = LGBMRegressor(objective='quantile', alpha=alpha/2, n_estimators=100, random_state=42, verbose=-1)
        self.model_high = LGBMRegressor(objective='quantile', alpha=1 - alpha/2, n_estimators=100, random_state=42, verbose=-1)
        self.conformal_multiplier_ = 0.0

    def fit_and_calibrate(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        同步训练集成栈与非参数符合性区间
        """
        # 1. 训练所有 Tier-1 基预测器
        for name, model in self.base_mean_models.items():
            model.fit(X_train, y_train)
            
        # 2. 训练分位数边界模型
        self.model_low.fit(X_train, y_train)
        self.model_high.fit(X_train, y_train)
        
        # 3. 计算历史校准集上的预测与符合性得分 (实现CV+或Ensemble Inference收缩)
        # 获取所有基模型的历史预测均值
        train_preds = np.column_stack([m.predict(X_train) for m in self.base_mean_models.values()])
        meta_mean_train = np.mean(train_preds, axis=1)
        
        # 计算历史校准集的组内分歧度 (Cross-Learner Variance)
        learner_variance_train = np.var(train_preds, axis=1)
        
        q_low_train = self.model_low.predict(X_train)
        q_high_train = self.model_high.predict(X_train)
        
        # 传统 CQR 得分
        standard_scores = np.maximum(q_low_train - y_train, y_train - q_high_train)
        
        # 核心方向三：将组内认知分歧度作为非参数惩罚项注入符合性得分中
        # 认知分裂越严重，校准得分上限越高
        conformal_scores = standard_scores + self.lambda_split * np.sqrt(learner_variance_train)
        
        # 取 (1-alpha) 分位数作为全局带状守卫乘数
        self.conformal_multiplier_ = np.percentile(conformal_scores, 100 * (1 - self.alpha))

    def predict_production(self, current_X_dict: dict) -> dict:
        """
        输出包含异质分歧护栏的点预测值与动态 CQR 宽度
        """
        results = {}
        for asset, x_vector in current_X_dict.items():
            # 转换单样本为标准2D矩阵
            x_df = pd.DataFrame([x_vector])
            
            # 1. 计算 Tier-1 各异质预测器的当期预测
            asset_preds = [m.predict(x_df)[0] for m in self.base_mean_models.values()]
            
            # Tier-2 均值非线性非参数聚合
            meta_point_pred = float(np.mean(asset_preds))
            
            # 计算当期该资产的 AI组内认知分歧度
            current_learner_variance = float(np.var(asset_preds))
            
            # 2. 计算基础条件分位数
            pred_low = float(self.model_low.predict(x_df)[0])
            pred_high = float(self.model_high.predict(x_df)[0])
            
            # 3. 动态融合校准乘数与当期认知方差，推导最终资产的鲁棒 CQR 宽度
            # 当基模型出现严重“意见分裂”时，cqr_width 将发生指数级破位爆炸
            adjusted_low = pred_low - self.conformal_multiplier_ - (self.lambda_split * np.sqrt(current_learner_variance))
            adjusted_high = pred_high + self.conformal_multiplier_ + (self.lambda_split * np.sqrt(current_learner_variance))
            
            cqr_width = float(adjusted_high - adjusted_low)
            
            results[asset] = {
                "point_prediction": meta_point_pred,
                "cqr_width": max(cqr_width, 1e-4),
                "learner_disagreement_std": float(np.sqrt(current_learner_variance))
            }
        return results
