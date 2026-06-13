```python
# config.py
# [EN] This is the entry point of the configuration script, defining the main module name.
# [ZH] 这是配置脚本的入口，定义了主模块名称。

import os
# [EN] Imports Python's standard OS library to handle operating system dependent tasks like file path manipulation.
# [ZH] 导入 Python 标准库中的 os 模块，用于处理与操作系统相关的任务（如文件路径拼接与目录操作）。

from dataclasses import dataclass, field
# [EN] Imports 'dataclass' for automatic boilerplate generation and 'field' for advanced default value configurations.
# [ZH] 从 dataclasses 模块中导入 'dataclass' 装饰器（自动生成基础方法）和 'field' 函数（用于高级默认值配置）。

from typing import List
# [EN] Imports 'List' from the typing module to explicitly declare data types for better code clarity and IDE linting.
# [ZH] 从 typing 模块中导入 'List'，用于显式声明列表类型，提高代码可读性与 IDE 代码提示效果。

@dataclass(frozen=True)
# [EN] Decorator that turns the class into a data container. 'frozen=True' makes it immutable to prevent accidental run-time alterations.
# [ZH] 类装饰器，将该类转换为数据类。'frozen=True' 表示该类实例是不可变的，防止在程序运行时被意外篡改。
class PipelineConfig:
    # [EN] Defines the configuration blueprint class named PipelineConfig.
    # [ZH] 定义名为 PipelineConfig 的配置蓝图类。

    # 多资产横截面标的池组
    # [EN] Comment indicating the following variable defines the multi-asset cross-sectional asset investment universe.
    # [ZH] 注释提示：下方变量定义了多资产横截面选股/交易的标的目标池。
    SYMBOLS: List[str] = field(default_factory=lambda: [
        "510720_ETF", "159937_ETF", "399006_ETF","600115_SH", "601088_SH", "601229_SH"
    ])
    # [EN] Declares a string list of 6 asset tickers (ETFs and Shanghai equities). 'default_factory=lambda' safely initializes mutable lists in dataclasses.
    # [ZH] 声明一个包含 6 个资产代码（包含 ETF 和沪市股票）的字符串列表。使用 'default_factory=lambda' 是为了在数据类中安全地初始化可变列表。
    
    # 统计不确定性与资产优化核心超参数
    # [EN] Comment indicating core hyperparameters for handling statistical uncertainty and asset allocation optimization.
    # [ZH] 注释提示：下方为处理统计不确定性与资产配置优化模型的核心超参数。
    ALPHA: float = 0.05  
    # [EN] Significance level for Conformal Quantile Regression (CQR). The target confidence level is calculated as 1 - alpha = 0.95 (95%).
    # [ZH] 符合性分位数回归 (CQR) 的显著性水平。目标置信度计算为 1 - alpha = 0.95 (即 95% 置信区间)。
    
    TAU: float = 0.02    
    # [EN] The relative weight scale of the machine learning views within the Black-Litterman (BL) portfolio optimization framework.
    # [ZH] Black-Litterman (BL) 资产配置模型中，机器学习预测观点的相对权重标度（代表对 AI 预测的信任度系数）。
    
    RISK_FREE_RATE: float = 0.0001  
    # [EN] The baseline daily risk-free return rate (0.01% per day), utilized in calculating performance metrics like the Sharpe Ratio.
    # [ZH] 盘后无风险日收益率基准（每日 0.01%），用于计算夏普比率（Sharpe Ratio）等风险收益指标。
    
    ANNUAL_INFLATION_RATE: float = 0.03  
    # [EN] Annual inflation rate baseline (3% per year), used to deflate nominal returns into real purchasing power returns.
    # [ZH] 年度通胀率平减基准（每年 3%），用于将名义收益率平减转化为扣除通胀后的真实购买力收益率。
    
    # 时序特征与滚动窗口参数
    # [EN] Comment indicating time-series parameters and rolling window configurations.
    # [ZH] 注释提示：下方为时间序列特征提取与数据滚动窗口的相关参数。
    ROLLING_WINDOW_SIZE: int = 120
    # [EN] The size of the moving window (120 trading days, ~6 months) used to calculate rolling technical indicators and statistical metrics.
    # [ZH] 滚动窗口大小（120个交易日，约合大半年），用于计算移动平均、滚动波动率等动态时序指标。
    
    MIN_REQUIRED_SAMPLES: int = 150
    # [EN] Minimum historical sample data size required. Assets with fewer days are mathematically filtered out to ensure statistical validity.
    # [ZH] 模型计算所需的最小历史样本天数。少于 150 天的数据样本将被过滤，以确保协方差矩阵等数学计算的统计学有效性。
    
    EMBARGO_PERIOD: int = 20  
    # [EN] Strict 20-day time-series embargo buffer to eliminate temporal data overlap and avoid look-ahead bias/leakage between train and test datasets.
    # [ZH] 严格的 20 天时序禁运期跨度。用于消除因特征重叠带来的非直视数据泄露，阻断训练集与测试集之间的“信息作弊”。
    
    # 底层可解释显式特征名
    # [EN] Comment indicating human-interpretable explicit features.
    # [ZH] 注释提示：下方为底层具有明确金融逻辑、可解释的显式特征名称。
    FEATURE_COLS: List[str] = field(default_factory=lambda: [
        'Mom_1D', 'Mom_5D', 'Mom_20D', 'GK_Vol', 'Turnover_Shock'
    ])
    # [EN] List declaration of 5 explicit engineering features: 1/5/20-day price momentum, Garman-Klass volatility, and liquidity turnover shock.
    # [ZH] 声明包含 5 个显式特征名称的列表：1/5/20日价格动量、Garman-Klass高低价波动率、以及换手率流动性冲击指标。
    
    # 生产文件持久化路径
    # [EN] Comment indicating persistence folders for production pipelines.
    # [ZH] 注释提示：下方为生产环境文件持久化落盘的根存储路径。
    BASE_OUTPUT_FOLDER: str = "./production_output"
    # [EN] The root directory path string where all pipeline results will be stored.
    # [ZH] 存储所有系统输出文件的根目录路径字符串。
    
    MODEL_DIR: str = field(init=False)
    # [EN] Declares the model save path attribute, explicitly telling the dataclass constructor NOT to initialize it via default inputs (`init=False`).
    # [ZH] 声明模型存储路径属性，显式告知数据类构造函数在初始化时不要直接通过外部输入为其赋值 (`init=False`)。
    
    REPORT_DIR: str = field(init=False)
    # [EN] Declares the performance report save path attribute, also excluded from primary constructor initialization (`init=False`).
    # [ZH] 声明分析报告存储路径属性，同样在初始实例化构造时被排除 (`init=False`)。

    def __post_init__(self):
        # [EN] A special dataclass post-initialization magic method that automatically triggers immediately after __init__ concludes.
        # [ZH] 数据类特有的后初始化魔法方法，在构造函数 `__init__` 执行完毕后被自动触发执行。
        
        print("\\nSaving configuration and setting up directories...")
        # [EN] Prints status message to the console in English indicating the configuration is running.
        # [ZH] 向控制台打印英文状态信息，提示正在保存配置并建立目录。
        
        print("配置已保存，正在设置目录...")
        # [EN] Prints status message to the console in Simplified Chinese.
        # [ZH] 向控制台打印简体中文状态信息。
        
        # 动态组装绝对或相对存储路径
        # [EN] Comment indicating dynamic assembly of relative or absolute system storage paths.
        # [ZH] 注释提示：动态组装绝对或相对的系统磁盘存储路径。
        object.__setattr__(self, 'MODEL_DIR', os.path.join(self.BASE_OUTPUT_FOLDER, "models"))
        # [EN] Uses base object bypass to assign the computed path `./production_output/models` to the frozen/immutable attribute MODEL_DIR.
        # [ZH] 利用底层基类的 `__setattr__` 方法绕过 frozen(不可变) 限制，将拼接好的 `./production_output/models` 路径强行赋值给 MODEL_DIR 属性。
        
        object.__setattr__(self, 'REPORT_DIR', os.path.join(self.BASE_OUTPUT_FOLDER, "reports"))
        # [EN] Uses base object bypass to assign the computed path `./production_output/reports` to the frozen/immutable attribute REPORT_DIR.
        # [ZH] 同样利用基类方法绕过限制，将拼接好的 `./production_output/reports` 路径强行赋值给 REPORT_DIR 属性。

```
