# data_processor.py
import numpy as np
import pandas as pd
from config import PipelineConfig

class DataProcessor:
    def __init__(self, config: PipelineConfig):
        self.config = config
        
    def neutralize_inflation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies compounding purchasing power deflation to historical nominal prices."""
        df = df.sort_index()
        dates = df.index.get_level_values(0)
        max_date = dates.max()
        
        # Calculate days differential relative to maximum historical edge
        days_diff = (max_date - dates).days
        deflator = (1.0 + self.config.ANNUAL_INFLATION_RATE) ** (days_diff / 365.25)
        
        res_df = df.copy()
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in res_df.columns:
                res_df[col] = res_df[col] * deflator
        return res_df

    def build_feature_space(self, raw_data_dict: dict) -> tuple:
        """Generates white-box features across multi-asset cross-sectional data panels."""
        panel_list = []
        for symbol, df in raw_data_dict.items():
            deflated_df = self.neutralize_inflation(df)
            
            # Extract log returns and momentum features
            log_p = np.log(deflated_df['Close'])
            deflated_df['Mom_1D'] = log_p.diff(1)
            deflated_df['Mom_5D'] = log_p.diff(5)
            deflated_df['Mom_20D'] = log_p.diff(20)
            
            # Intraday Garman-Klass Volatility Metric
            log_hl = np.log(deflated_df['High'] / deflated_df['Low']) ** 2
            log_cc = np.log(deflated_df['Close'] / deflated_df['Open']) ** 2
            deflated_df['GK_Vol'] = np.sqrt(0.5 * log_hl - (2 * np.log(2) - 1) * log_cc)
            
            # Standardized Dynamic Volume Volume Shock
            turnover_roll = deflated_df['Turnover'].rolling(window=5).mean()
            deflated_df['Turnover_Shock'] = deflated_df['Turnover'] / (turnover_roll + 1e-8)
            
            # Target generation (Next-day forward return shifted backward)
            deflated_df['target_return'] = log_p.diff(1).shift(-1)
            deflated_df['Symbol'] = symbol
            panel_list.append(deflated_df)
            
        full_panel = pd.concat(panel_list).set_index(['Symbol'], append=True)
        full_panel = full_panel.reorder_levels(['Date', 'Symbol']).sort_index()
        full_panel = full_panel.dropna(subset=['Mom_1D', 'Mom_5D', 'Mom_20D', 'GK_Vol', 'Turnover_Shock'])
        
        feature_cols = ['Mom_1D', 'Mom_5D', 'Mom_20D', 'GK_Vol', 'Turnover_Shock']
        X_panel = full_panel[feature_cols]
        y_panel = full_panel['target_return']
        return X_panel, y_panel