# config.py
import os
from dataclasses import dataclass, field
from typing import List

@dataclass(frozen=True)
class PipelineConfig:
    # 多资产横截面标的池组 (Aligned with text simulation assets)
    SYMBOLS: List[str] = field(default_factory=lambda: [
        "510720_ETF", "159937_ETF", "399006_ETF", "600115_SH", "601088_SH", "601229_SH", "512480"
    ])
    
    # 统计不确定性与资产优化核心超参数
    ALPHA: float = 0.05  # CQR 覆盖率显著性水平 (1-alpha 为目标置信度)
    TAU: float = 0.02    # BL组合中机器学习观点的相对权重标度
    RISK_FREE_RATE: float = 0.0001  # 盘后无风险日收益率基准
    ANNUAL_INFLATION_RATE: float = 0.03  # 年度通胀率平减基准
    
    # 时序特征与滚动窗口参数
    ROLLING_WINDOW_SIZE: int = 120
    MIN_REQUIRED_SAMPLES: int = 150
    EMBARGO_PERIOD: int = 20  # 严格时序禁运期跨度，消除重叠特征带来的非直视数据泄露
    
    # 底层可解释显式特征名
    FEATURE_COLS: List[str] = field(default_factory=lambda: [
        'Mom_1D', 'Mom_5D', 'Mom_20D', 'GK_Vol', 'Turnover_Shock'
    ])
    
    # 生产文件持久化路径
    BASE_OUTPUT_FOLDER: str = "./production_output"
    MODEL_DIR: str = field(init=False)
    REPORT_DIR: str = field(init=False)

    def __post_init__(self):
        print("\nSaving configuration and setting up directories...")
        print("配置已保存，正在设置目录...")
        # 动态组装绝对或相对存储路径
        object.__setattr__(self, 'MODEL_DIR', os.path.join(self.BASE_OUTPUT_FOLDER, "models"))
        object.__setattr__(self, 'REPORT_DIR', os.path.join(self.BASE_OUTPUT_FOLDER, "reports"))