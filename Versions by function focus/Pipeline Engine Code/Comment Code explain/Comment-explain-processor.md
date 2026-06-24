# Data_processor.py Line-by-Line Code Explanation / 逐行代码注释解析

Below is the fully annotated `data_processor.py` analytics script. Each line of code contains explicit inline structural comments mapping out the system logic in both English [EN] and Simplified Chinese [ZH].

以下是完整带注释的 `data_processor.py` 数据处理脚本。每行代码配有明确的中英文双语结构化注释，解析其底层逻辑。

```python
# data_processor.py
# [EN] Feature engineering script that handles data normalization, technical indicators, and panel parsing.
# [ZH] 特征工程脚本，负责数据归一化、技术指标计算以及跨截面面板组装。

import logging
# [EN] Main Python logging facility used to record workflow operations under tracking channels.
# [ZH] Python 标准日志模块，用于在指定的追踪频道下记录系统流水线工作流运行状态。

import numpy as np
# [EN] Numeric processing extension deployed to handle rapid vectorized mathematics, logarithmic parsing, and value checks.
# [ZH] 数值处理扩展库，用于处理高效的向量化数学运算、对数解析及数值边界替换。

import pandas as pd
# [EN] Core data structure engine driving DataFrame matrix alignment and time-series multi-indexing operations.
# [ZH] 核心数据结构引擎，驱动 DataFrame 矩阵对齐以及时间序列多级索引操作。

from typing import Dict, Tuple
# [EN] Static type hint definitions for explicitly annotating structural dictionaries and return object tuples.
# [ZH] 静态类型提示定义，用于显式声明字典容器与返回对象元组的结构类型。

from config import PipelineConfig
# [EN] Import path to consume the globally shared parameters blueprint dataclass definitions.
# [ZH] 导入全局共享的参数蓝图数据类定义，以获取超参数配置。

logger = logging.getLogger("QuantPipeline.DataProcessor")
# [EN] Sets up a distinct log tracking child workspace named explicitly under the main application bus registry.
# [ZH] 建立一个在主程序总线注册表下命名的、独立的日志追踪子工作空间。

class DataProcessor:
    # [EN] Declares the primary pipeline class wrapper driving all feature scaling and consolidation matrices.
    # [ZH] 声明负责驱动所有特征缩放与多资产矩阵合并的主管道类。

    def __init__(self, config: PipelineConfig):
        # [EN] Class constructor anchoring configuration settings onto the local processor instance namespace.
        # [ZH] 类构造函数，将外部配置实例挂载并锁定到本地处理器实例的命名空间中。
        
        print("\nInitializing DataProcessor with provided configuration...")
        # [EN] Outputs text in English signaling initialization onto the basic terminal screen.
        # [ZH] 向控制台终端屏幕输出英文提示，标志数据处理器开始初始化。
        
        print("初始化数据处理器...")
        # [EN] Outputs the synchronized system initialization state in Simplified Chinese text.
        # [ZH] 向终端屏幕输出同步的简体中文初始化状态提示。
        
        self.config = config
        # [EN] Locks down the read-only parameters class context object internally.
        # [ZH] 将只读的全局配置参数对象锁定到本地私有变量 `self.config` 中。

    def apply_inflation_deflator(self, df: pd.DataFrame) -> pd.DataFrame:
        # [EN] Method evaluating market price row columns to remove continuous currency purchasing power loss over time.
        # [ZH] 评估市场价格列的方法，用于消除货币购买力随时间推移而产生的持续通胀贬值效应。
        
        print("\nApplying inflation deflator to raw price data...")
        # [EN] Sends process update statement to the standard out screen stream in English.
        # [ZH] 向标准输出流发送英文进程更新声明，提示正在应用通胀平减。
        
        print("对原始价格数据应用通胀平减...")
        # [EN] Echoes process update statement to the screen stream in Simplified Chinese text.
        # [ZH] 向终端屏幕输出同步的简体中文进程更新状态提示。
        
        """
        根据交易日历的实际跨度进行跨期购买力平减，阻断名义价格的非平稳性噪声。
        """
        if df.empty:
            raise ValueError("Input dataframe is empty, deflation aborted.")
            # [EN] Guards against empty objects by raising a calculation halt error if rows are totally missing.
            # [ZH] 异常防御性边界：如果传入的矩阵行数为空，则抛出数值错误并强制中断计算。
            
        df = df.copy()
        # [EN] Explicitly clones the input container memory to strictly avoid setting-with-copy leakage bugs.
        # [ZH] 显式克隆输入的 DataFrame 内存块，严格避免触发“对切片副本赋值”的隐式内存泄漏 Bug。
        
        max_date = df.index.max()
        # [EN] Identifies the most recent timeline date index boundary present inside the asset time series.
        # [ZH] 提取出该资产序列中最新（最大）的交易日期时间戳索引边界。
        
        days_back = (max_date - df.index).days
        # [EN] Evaluates structural timeline distance arrays by subtracting indices from the maximum terminal date.
        # [ZH] 计算时间轴上每个历史交易日距离最新终点日期的绝对天数差值向量。
        
        years_back = days_back / 365.25
        # [EN] Converts crude row integer day counts into localized year fractions accounts for leap adjustments.
        # [ZH] 将原始整数天数除以 365.25，转化为包含闰年修正因子的精准年分数跨度。
        
        deflator = (1.0 + self.config.ANNUAL_INFLATION_RATE) ** years_back
        # [EN] Performs continuous compounded exponential smoothing curves using the annual rate hyperparameter.
        # [ZH] 利用年度通胀率超参数，计算出各历史时间点对应的复利平减指数曲线系数。
        
        for col in ['Open', 'High', 'Low', 'Close']:
            # [EN] Sequentially loops inside core structural quote columns to perform vector alterations.
            # [ZH] 顺序遍历开盘、最高、最低、收盘等核心行情价格列，以执行批量矩阵变换。
            
            if col in df.columns:
                df[f'Real_{col}'] = df[col] * deflator
                # [EN] Multiplies nominal price inputs by deflator values to create structural real valuation arrays.
                # [ZH] 将原始名义价格乘以平减系数，生成并追加代表真实价值的资产价格新列。
                
            else:
                raise KeyError(f"Required structural price column '{col}' missing from data.")
                # [EN] Throws a key parsing error if requisite standard price components are absent.
                # [ZH] 如果资产矩阵中缺失基础的价格必填列，则抛出键查找错误。
                
        return df
        # [EN] Hands back the augmented dataframe containing the calculated deflation-adjusted real price arrays.
        # [ZH] 向调用方返回包含全新计算得到的真实价格特征列的 DataFrame 矩阵。

    def build_feature_space(self, raw_data_dict: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        # [EN] Method parsing cross-sectional data frames to yield the global combined X panel matrix and label vector y.
        # [ZH] 构建特征空间的方法，加工生成全局融合的横截面特征面板 X 矩阵与目标标签 y 向量。
        
        """
        加工白盒化多资产横截面特征面板。融合 CQR 需要的精确错时目标变量。
        返回多级索引(Date, Symbol)的特征矩阵 X 与目标收益率 y。
        """
        logger.info("\nStarting multi-asset feature matrix cross-sectional generation...")
        # [EN] Logs formal matrix construction lifecycle events onto the information stream logs.
        # [ZH] 在信息级系统日志流中记录多资产特征矩阵横截面构建的正式启动事件。
        
        print("\nStarting multi-asset feature matrix cross-sectional generation...")
        # [EN] Duplicate notification sent directly to standard output terminal screen for developer monitoring.
        # [ZH] 向标准输出终端控制台屏幕输出相同的英文状态提示，方便开发者监控。
        
        print("开始生成多资产特征矩阵横截面...")
        # [EN] Simplified Chinese status update notification sent to standard terminal output screen.
        # [ZH] 向控制台输出同步的简体中文多资产特征生成状态提示。
        
        feature_frames = []
        # [EN] Array list designed to aggregate standalone calculated feature data frames across symbols.
        # [ZH] 内部列表容器，用于收集各个资产独立计算完成的特征子 DataFrame。
        
        target_frames = []
        # [EN] Array list designed to aggregate matched clean target return vectors across symbols.
        # [ZH] 内部列表容器，用于收集各个资产对应的、清洗干净的目标收益率标签向量。

        all_trading_days = pd.Index(
            sorted(
                set().union(*(df.sort_index().index for df in raw_data_dict.values()))
            )
        )
        # [EN] Combines and sorts disparate dates across inputs into a unique, comprehensive reference trading master index calendar.
        # [ZH] 对输入的所有资产时间戳求并集并进行正向排序，构建出一个唯一且完整的全局主交易日历索引。

        for symbol in self.config.SYMBOLS:
            # [EN] Initiates loop logic across the predefined string identifiers provided inside the config setup list.
            # [ZH] 开始遍历配置蓝图列表中预先定义好的证券代码字符串标识。
            
            if symbol not in raw_data_dict:
                raise KeyError(f"Asset symbol '{symbol}' missing from raw input dictionary.")
                # [EN] Guard rail raising a mapping exception if data dictionaries fail to contain the required asset identifier.
                # [ZH] 安全防御门槛：如果原始行情输入字典中缺少配置所要求的资产代号，则抛出键缺失异常。
            
            df_symbol = raw_data_dict[symbol].sort_index()
            # [EN] Extra defense step sorting the time-series chronological sequences to guarantee index safety.
            # [ZH] 对当前证券的行情数据按时间轴执行正向排序，确保时序计算不会错位。
            
            df_real = self.apply_inflation_deflator(df_symbol)
            # [EN] Routes raw prices through the inflation script to obtain stabilized real dollar evaluation streams.
            # [ZH] 将排序后的行情送入通胀平减方法中，获得平稳化后的真实价值价格流。
            
            symbol_space = pd.DataFrame(index=df_real.index)
            # [EN] Instantiates an asset-specific target matrix sharing the identical time index axis framework.
            # [ZH] 实例化一个属于当前资产的特征容器 DataFrame，共享相同的时间戳索引轴框架。
            
            # 目标变量：向前跨越一步的对数实际收益率
            prev_close = df_real['Real_Close'].shift(1)
            # [EN] Shifts the closing prices forward by 1 tick downward to retrieve the immediate previous base return reference level.
            # [ZH] 将真实收盘价序列向下平移 1 期，获取前一日的参考收盘价，用作对数收益率计算的分母。
            
            safe_prev_close = np.where(prev_close <= 0, 1e-8, prev_close)
            # [EN] Numeric replacement substituting zeros or negative anomalies with tight margins to prevent division-by-zero math infinity faults.
            # [ZH] 数值数值替换防御：若前一日收盘价小于等于0，则强转为 1e-8，杜绝分母为零引发的数学除法无穷大故障。
            
            symbol_space['target_return'] = np.log(df_real['Real_Close'] / safe_prev_close).shift(-1)
            # [EN] Computes standard log return vectors and shifts the row values backward (-1) to serve as a valid forward-peeking target placeholder.
            # [ZH] 计算标准的对数真实收益率序列，并通过向上平移 1 期 (`.shift(-1)`)，使其成为代表未来一期预测目标的因变量标签。
            
            # 多周期时序动量特征
            symbol_space['Mom_1D'] = np.log(df_real['Real_Close'] / df_real['Real_Close'].shift(1))
            # [EN] Evaluates the 1-day momentum metric representing log relative performance over the nearest trading day.
            # [ZH] 计算 1 日动量特征，即当前收盘价相对于前 1 日收盘价的对数相对收益表现。
            
            symbol_space['Mom_5D'] = np.log(df_real['Real_Close'] / df_real['Real_Close'].shift(5))
            # [EN] Evaluates the 5-day momentum metric tracking structural returns over a weekly horizon baseline.
            # [ZH] 计算 5 日动量特征，追踪资产在过去 5 个交易日（约一周周频）时序上的累积收益表现。
            
            symbol_space['Mom_20D'] = np.log(df_real['Real_Close'] / df_real['Real_Close'].shift(20))
            # [EN] Evaluates the 20-day momentum metric capturing log rate velocity adjustments over roughly one business month.
            # [ZH] 计算 20 日动量特征，捕捉资产在过去 20 个交易日（约一整个工作月）时序上的中线速度表现。
            
            # Garman-Klass 极端值特质波动率
            hl_ratio = np.log(df_real['Real_High'] / np.where(df_real['Real_Low'] <= 0, 1e-5, df_real['Real_Low']))
            # [EN] Compiles log ratio vectors tracking High against adjusted Low extremes to parse intraday trading range dispersion.
            # [ZH] 计算最高价相对于调整后最低价的对数比值，用以捕获日内极端价格变动幅宽。
            
            co_ratio = np.log(df_real['Real_Close'] / np.where(df_real['Real_Open'] <= 0, 1e-5, df_real['Real_Open']))
            # [EN] Compiles log ratio vectors tracking Close against adjusted Open bounds to capture overnight gap vs intraday trend segments.
            # [ZH] 计算收盘价相对于开盘价的对数比值，捕获日内从开盘到收盘的核心实体趋势段强度。
            
            symbol_space['GK_Vol'] = 0.5 * (hl_ratio ** 2) - (2 * np.log(2) - 1) * (co_ratio ** 2)
            # [EN] Applies the explicit Garman-Klass calculus formula to derive highly efficient daily idiosyncratic volatility estimates.
            # [ZH] 应用经典的 Garman-Klass 微积分公式，推演导出比单纯收盘价更高效、更具信息量的日内特质波动率估计值。
            
            # 换手率异常冲击
            rolling_turnover_mean = df_real['Turnover'].rolling(5).mean()
            # [EN] Slides an averaging aggregator window backward over 5 days to compute a smoothed baseline for regular turnover volume.
            # [ZH] 针对换手率序列计算 5 日滚动均值，以此作为常规市场流动性换手的平滑基准线。
            
            symbol_space['Turnover_Shock'] = df_real['Turnover'] / np.where(rolling_turnover_mean <= 0, 1e-5, rolling_turnover_mean)
            # [EN] Evaluates volume turnover shock by standardizing today's rate against the rolling mean proxy to pinpoint sudden liquidity spikes.
            # [ZH] 用当日换手率除以滚动均值基准线，计算出换手率异常冲击指标，用以精确定位突发性的流动性多空放量。
            
            # cleaned_symbol = symbol_space.dropna()
            cleaned_symbol = symbol_space.dropna(subset=self.config.FEATURE_COLS)
            # [EN] Drops history rows containing missing elements exclusively within predictive feature columns, safely preserving tomorrow's target label on the final live row.
            # [ZH] 仅针对预测特征列执行空值剔除，允许最新（最后一天）的因变量标签为 NaN，从而保留实盘所需的最新观测样本。
            
            is_finite_mask = np.isfinite(cleaned_symbol[self.config.FEATURE_COLS]).all(axis=1)
            # [EN] Constructs a boolean conditional masking array across rows to target and eliminate NaN, Inf, or -Inf float anomalies.
            # [ZH] 针对特征矩阵构建一重布尔条件掩码，用于检测并剔除任何包含 NaN、Inf 或 -Inf 等无法参与数学计算的异常浮点行。
            
            cleaned_symbol = cleaned_symbol[is_finite_mask]
            # [EN] Filters out index locations matching invalid rows via bitwise assignment vector logic.
            # [ZH] 运用位向量过滤逻辑，将不符合计算规范的异常行彻底从当前矩阵空间中剥离剔除。
            
            if len(cleaned_symbol) < self.config.MIN_REQUIRED_SAMPLES:
                raise RuntimeError(f"Asset {symbol} has insufficient valid sample rows ({len(cleaned_symbol)}).")
                # [EN] Validates row lengths against global configuration limits to prevent matrix rank deficiencies in downstream regression layers.
                # [ZH] 校核清洗后的有效样本总数，若低于配置下限则抛出运行时致命错误，防止后续回归层出现矩阵亏秩或欠拟合。
            
            # 转化为多级索引结构
            cleaned_symbol = cleaned_symbol.copy()
            # [EN] Re-clones memory states to prevent structural downstream warning flags during indexing expansions.
            # [ZH] 再次对清洗后的数据执行深度克隆，规避后续追加索引时可能触发的链式赋值警告。
            
            cleaned_symbol['Symbol'] = symbol
            # [EN] Appends a categorical label literal identifier row column denoting the ticker name.
            # [ZH] 追加一个专门存储当前资产代号字符串的常数值辅助列。
            
            cleaned_symbol = cleaned_symbol.set_index('Symbol', append=True)
            # [EN] Pushes the newly declared asset label column upward into the row index to build a multi-level index hierarchical matrix.
            # [ZH] 将新声明的 Symbol 列推入索引维度轴，与原日期轴并列，组装出标准的多级索引层次化面板结构。
            
            feature_frames.append(cleaned_symbol[self.config.FEATURE_COLS])
            # [EN] Extracts feature arrays using configuration filters and appends them to the storage array.
            # [ZH] 利用配置特征列过滤器提取特征矩阵，并将其追加到特征集存储容器中。
            
            target_frames.append(cleaned_symbol['target_return'])
            # [EN] Isolates matched label vectors and stores them inside the secondary target tracker repository list.
            # [ZH] 单独提取对齐的因变量标签收益率向量，并将其存入标签集存储容器中。
            
        X_panel = pd.concat(feature_frames, axis=0).sort_index()
        # [EN] Merges individual feature segments vertically along rows (axis=0) and enforces absolute chronological index sorting.
        # [ZH] 纵向（按行维度 axis=0）将所有资产的特征块拼合成庞大的全局面板矩阵，并执行绝对时间戳索引排序。
        
        y_panel = pd.concat(target_frames, axis=0).sort_index()
        # [EN] Merges individual matching prediction target columns vertically along rows, sorting rows identically to mirror the X feature matrix structure.
        # [ZH] 纵向拼接所有资产配对的预测因变量标签列，执行完全相同的多级索引排序以严格镜像对齐特征空间。
        
        return X_panel, y_panel
        # [EN] Returns the fully consolidated feature panel DataFrame and label vector Series back to the root application caller.
        # [ZH] 将整合就绪的全局特征面板 DataFrame 与对应的标签向量 Series 组成元组，正式返回给系统核心调用方。
