# data_processor.py
import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from config import PipelineConfig

logger = logging.getLogger("QuantPipeline.DataProcessor")

class DataProcessor:
    def __init__(self, config: PipelineConfig):
        print("\nInitializing DataProcessor with provided configuration...")
        print("初始化数据处理器...")
        self.config = config

    def apply_inflation_deflator(self, df: pd.DataFrame) -> pd.DataFrame:
        print("\nApplying inflation deflator to raw price data...")
        print("对原始价格数据应用通胀平减...")
        """
        根据交易日历的实际跨度进行跨期购买力平减，阻断名义价格的非平稳性噪声。
        """
        if df.empty:
            raise ValueError("Input dataframe is empty, deflation aborted.")
            
        df = df.copy()
        max_date = df.index.max()
        days_back = (max_date - df.index).days
        years_back = days_back / 365.25
        deflator = (1.0 + self.config.ANNUAL_INFLATION_RATE) ** years_back
        
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in df.columns:
                df[f'Real_{col}'] = df[col] * deflator
            else:
                raise KeyError(f"Required structural price column '{col}' missing from data.")
        return df

    def build_feature_space(self, raw_data_dict: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        加工白盒化多资产横截面特征面板。融合 CQR 需要的精确错时目标变量。
        返回多级索引(Date, Symbol)的特征矩阵 X 与目标收益率 y。
        """
        logger.info("\nStarting multi-asset feature matrix cross-sectional generation...")
        print("\nStarting multi-asset feature matrix cross-sectional generation...")
        print("开始生成多资产特征矩阵横截面...")
        feature_frames = []
        target_frames = []

        all_trading_days = pd.Index(
            sorted(
                set().union(*(df.sort_index().index for df in raw_data_dict.values()))
            )
        )

        for symbol in self.config.SYMBOLS:
            if symbol not in raw_data_dict:
                raise KeyError(f"Asset symbol '{symbol}' missing from raw input dictionary.")
            
            df_symbol = raw_data_dict[symbol].sort_index()
            df_real = self.apply_inflation_deflator(df_symbol)
            
            symbol_space = pd.DataFrame(index=df_real.index)
            
            # 目标变量：向前跨越一步的对数实际收益率
            prev_close = df_real['Real_Close'].shift(1)
            safe_prev_close = np.where(prev_close <= 0, 1e-8, prev_close)
            symbol_space['target_return'] = np.log(df_real['Real_Close'] / safe_prev_close).shift(-1)
            
            # 多周期时序动量特征
            symbol_space['Mom_1D'] = np.log(df_real['Real_Close'] / df_real['Real_Close'].shift(1))
            symbol_space['Mom_5D'] = np.log(df_real['Real_Close'] / df_real['Real_Close'].shift(5))
            symbol_space['Mom_20D'] = np.log(df_real['Real_Close'] / df_real['Real_Close'].shift(20))
            
            # Garman-Klass 极端值特质波动率
            hl_ratio = np.log(df_real['Real_High'] / np.where(df_real['Real_Low'] <= 0, 1e-5, df_real['Real_Low']))
            co_ratio = np.log(df_real['Real_Close'] / np.where(df_real['Real_Open'] <= 0, 1e-5, df_real['Real_Open']))
            symbol_space['GK_Vol'] = 0.5 * (hl_ratio ** 2) - (2 * np.log(2) - 1) * (co_ratio ** 2)
            
            # 换手率异常冲击
            rolling_turnover_mean = df_real['Turnover'].rolling(5).mean()
            symbol_space['Turnover_Shock'] = df_real['Turnover'] / np.where(rolling_turnover_mean <= 0, 1e-5, rolling_turnover_mean)
            
            
            # cleaned_symbol = symbol_space.dropna()# 清洗当前资产的异常值
            # 仅当特征列存在 NaN 时才删除，允许最后一天的标签为 NaN
            cleaned_symbol = symbol_space.dropna(subset=self.config.FEATURE_COLS)
            is_finite_mask = np.isfinite(cleaned_symbol[self.config.FEATURE_COLS]).all(axis=1)
            cleaned_symbol = cleaned_symbol[is_finite_mask]
            
            if len(cleaned_symbol) < self.config.MIN_REQUIRED_SAMPLES:
                raise RuntimeError(f"Asset {symbol} has insufficient valid sample rows ({len(cleaned_symbol)}).")
            
            # 转化为多级索引结构
            cleaned_symbol = cleaned_symbol.copy()
            cleaned_symbol['Symbol'] = symbol
            cleaned_symbol = cleaned_symbol.set_index('Symbol', append=True)
            
            feature_frames.append(cleaned_symbol[self.config.FEATURE_COLS])
            target_frames.append(cleaned_symbol['target_return'])
            
        X_panel = pd.concat(feature_frames, axis=0).sort_index()
        y_panel = pd.concat(target_frames, axis=0).sort_index()
        return X_panel, y_panel