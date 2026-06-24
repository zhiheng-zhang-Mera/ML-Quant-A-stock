# Main-dual-lang.py Line-by-Line Code Explanation / 逐行代码注释解析

Below is the fully annotated `main-dual-lang.py` orchestration script. Each line of code contains explicit inline structural comments mapping out the system logic in both English [EN] and Simplified Chinese [ZH].

以下是完整带注释的 `main-dual-lang.py` 主程序调度脚本。每行代码配有明确的中英文双语结构化注释，解析其底层逻辑。

```python
# main-dual-lang.py
# [EN] Main execution orchestrator file handling multi-asset data pipelines and portfolio loops.
# [ZH] 主执行调度文件，负责管理多资产数据流水线和投资组合分配循环。

import sys
# [EN] Standard library to interact with system parameters and environment variables.
# [ZH] 标准系统库，用于与系统底层参数及运行环境进行交互。

import os
# [EN] Standard operating system interface library for handling path directories and platform utilities.
# [ZH] 标准操作系统接口库，用于处理路径文件夹目录及平台底层工具。

import json
# [EN] Data encoding module used to save and transmit structured dictionary states into JSON text strings.
# [ZH] 数据编码模块，用于将结构化的字典数据状态存储并转化为标准的 JSON 文本字符串。

import warnings
# [EN] Library to trap or silence compilation warnings that could clutter production output logs.
# [ZH] 异常警告捕获库，用于过滤或抑制可能污染生产环境输出日志的编译警告。

import logging
# [EN] Core Python logging engine framework to manage systematic text reporting across execution layers.
# [ZH] 核心日志引擎框架，用于系统化管理软件运行生命周期中各个执行层面的文本输出。

from datetime import datetime
# [EN] Object handling system time dates, formatting time strings, and measuring temporal deltas.
# [ZH] 专门处理系统日期时间、格式化时间字符串以及计算时间跨度的对象。

import numpy as np
# [EN] Matrix mathematics engine handling high-performance vector transformations and random simulations.
# [ZH] 矩阵数学计算引擎，用于高性能向量变换、多维数组运算及随机数模拟。

import pandas as pd
# [EN] Data analytics matrix framework optimized for time-series panels and multi-indexed structures.
# [ZH] 数据分析矩阵框架，专为时间序列面板数据与多重索引结构进行了深度优化。

warnings.filterwarnings('ignore', category=UserWarning)
# [EN] Suppresses basic user alerts and package usage notifications to preserve raw terminal readability.
# [ZH] 屏蔽基础用户警告和第三方库提示，以保持控制台终端的高可读性与整洁度。

from config import PipelineConfig
# [EN] Imports the immutable project configuration dataclass setup blueprint.
# [ZH] 导入不可变的项目配置数据类参数设置蓝图。

from data_processor import DataProcessor
# [EN] Imports the processing module responsible for computing raw features from asset ticker price dictionaries.
# [ZH] 导入数据处理模块，负责从原始多资产价格字典中清洗并计算技术指标矩阵。

from models import StatisticalAdaptiveEngine
# [EN] Imports the Conformal Quantile Regression engine that handles machine learning predictions and adaptive risk boundaries.
# [ZH] 导入符合性分位数回归引擎，负责机器学习预测及自适应风险边界度量。

from portfolio import BayesianExecutionBridge
# [EN] Imports the optimization layer that blends ML predictions into Black-Litterman and executes Mean-Variance optimization.
# [ZH] 导入投资组合优化层，负责将机器学习预测融入 Black-Litterman 模型并执行均值-方差优化。

def setup_production_logging(config: PipelineConfig):
    # [EN] Function declaring and structuring localized system logging engines for multi-device routing.
    # [ZH] 定义并结构化本地系统日志引擎的函数，支持向控制台与文件同时分流打印。
    
    log_format = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s')
    # [EN] Sets text formatting template: timestamp, message severity level, sub-module caller identifier, and text string.
    # [ZH] 设定日志文本格式模板：包含时间戳、日志严重等级、调用子模块标识、以及具体日志信息。
    
    logger = logging.getLogger("QuantPipeline")
    # [EN] Creates or retrieves a named localized logging singleton entity instance.
    # [ZH] 创建或获取一个名为 "QuantPipeline" 的特定命名空间本地日志单例对象。
    
    logger.setLevel(logging.INFO)
    # [EN] Configures the engine to process all events at or above the informational context threshold.
    # [ZH] 配置日志级别门槛，处理并记录所有等于或高于 INFO（信息级）状态的事件。
    
    ch = logging.StreamHandler()
    # [EN] Initializes a continuous stream handler routing log outputs straight onto the active interactive terminal.
    # [ZH] 初始化一个标准的控制台流处理器，将日志输出流直接导向当前的交互式终端界面。
    
    ch.setFormatter(log_format)
    # [EN] Links the defined format structure to the terminal display pipeline handler.
    # [ZH] 将之前定义的文本布局格式关联到该终端流处理器上。
    
    logger.addHandler(ch)
    # [EN] Registers the terminal stream component into the active core logging manager.
    # [ZH] 将控制台处理器注册注册到核心日志管理器中。
    
    os.makedirs(os.path.join(config.BASE_OUTPUT_FOLDER, "reports"), exist_ok=True)
    # [EN] Safely constructs output folder hierarchies on disk; ignores execution error if target folders already exist.
    # [ZH] 在磁盘上安全创建报告输出文件夹层级；如果目标文件夹已存在，则自动忽略错误。
    
    log_file = os.path.join(config.BASE_OUTPUT_FOLDER, "reports", "pipeline_execution.log")
    # [EN] Assembles path destination for persistent log storage on drive.
    # [ZH] 拼接出用于持久化落盘存储的日志文件目标物理绝对/相对路径。
    
    fh = logging.FileHandler(log_file, encoding='utf-8')
    # [EN] Opens a standard file append writer stream targeting the defined file path using UTF-8 text encoding.
    # [ZH] 初始化一个文件处理器，以 UTF-8 编码向指定的磁盘路径写出/追加文本日志。
    
    fh.setFormatter(log_format)
    # [EN] Sets the shared text layout standard onto the active disk writer component.
    # [ZH] 将统一的文本格式标准应用到该文件流处理器上。
    
    logger.addHandler(fh)
    # [EN] Registers the file writer device to the main tracking registry to split logs into text files.
    # [ZH] 将该文件处理器注册到主日志追踪体系中，实现日志向磁盘文本的分流落盘。
    
    return logger
    # [EN] Hands back the fully operational, multi-stream logging manager instance to the caller.
    # [ZH] 向调用方返回配置完毕、具备双向分流功能的标准日志管理器对象。

def run_production_pipeline(raw_multi_asset_data: dict) -> dict:
    """
    多资产生产调度总线核心生命周期逻辑
    """
    # [EN] Core pipeline function containing processing loops, mathematical models, and deployment configurations.
    # [ZH] 核心管道流函数，内部封装了数据加工循环、数学模型拟合、以及配置部署逻辑。
    
    config = PipelineConfig()
    # [EN] instantiates the immutable control configuration parameter object.
    # [ZH] 实例化不可变的基础控制配置参数对象。
    
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    # [EN] Creates the target binaries directory if missing from the operating system disk partition.
    # [ZH] 如果操作系统磁盘分区中缺失模型二进制文件夹，则自动创建该目标目录。
    
    os.makedirs(config.REPORT_DIR, exist_ok=True)
    # [EN] Creates the analytical storage folder structure safely on the environment platform.
    # [ZH] 在运行环境平台上安全创建分析报告存储文件夹结构。
    
    logger = setup_production_logging(config)
    # [EN] Triggers the logger engine initialization using the newly formed directory infrastructure parameters.
    # [ZH] 使用新构建的目录结构参数触发日志引擎初始化，获取全局记录器。
    
    logger.info(f"\n[INIT] ================ 激活多资产矩阵生产线 / STARTING MULTI-ASSET DISPATCH ================")
    # [EN] Logs the explicit initialization line signaling pipeline start.
    # [ZH] 记录清晰的初始化分隔线日志，标志着多资产调度流水线的正式启动。
    
    processor = DataProcessor(config)
    # [EN] Instantiates the domain technical feature engineering calculator component.
    # [ZH] 实例化技术特征工程指标计算组件。
    
    engine = StatisticalAdaptiveEngine(config)
    # [EN] Instantiates the conformal ML risk estimation and adaptive modeling pipeline object.
    # [ZH] 实例化符合性机器学习风险估计与自适应预测模型管道对象。
    
    bridge = BayesianExecutionBridge(config)
    # [EN] Instantiates the portfolio optimization and asset blending calculation engine layer.
    # [ZH] 实例化投资组合配置优化与资产融合计算引擎层对象。
    
    # 1. 加工多资产横截面面板数据
    X_panel, y_panel = processor.build_feature_space(raw_multi_asset_data)
    # [EN] Feeds raw storage dictionaries into the processor to parse multi-index rows of features and target returns.
    # [ZH] 将原始存储字典输入处理器，计算出多维复合索引的技术特征矩阵与目标收益率向量。
    
    # 提取交叉截面时间轴，并进行严格的【滚动时序禁运隔离】
    dates = X_panel.index.get_level_values(0).unique().sort_values()
    # [EN] Extracts all unique timestamps from the first index layer, sorting them to maintain logical time-series order.
    # [ZH] 从一重索引层中提取所有唯一的交易时间戳，并进行正向排序以维持时间序列的逻辑先后顺序。
    
    # 设定历史截断点 T-1 用于模拟生产环境
    train_end_idx = len(dates) - 1 - config.EMBARGO_PERIOD
    # [EN] Mathematical cutoff rule: Rewinds back by the exact embargo count from the terminal edge to prevent predictive overlap leak.
    # [ZH] 数学截断规则：从最新的终点数据向后倒退一个精确的禁运天数，阻断时序特征重叠带来的预测信息泄露。
    
    if train_end_idx < config.MIN_REQUIRED_SAMPLES:
        raise RuntimeError("Data length is insufficient after subtracting the Embargo period constraint.")
        # [EN] Throws a runtime halting exception if historical data length fails to meet mathematical requirements.
        # [ZH] 如果扣除禁运期后的样本长度达不到基础数学要求，则抛出运行时异常，强制中断程序。
        
    training_dates = dates[:train_end_idx]
    # [EN] Slices the chronologically safe date range assigned explicitly for training historical parameters.
    # [ZH] 切片截取安全的时间段，将其显式指定为可用于历史参数训练的日期向量。
    
    current_production_date = dates[-1]
    # [EN] Designates the terminal row entry as the modern live trade decision observation target date.
    # [ZH] 将最新的一行截面数据指定为当前模拟实盘决策的观察执行目标日。
    
    logger.info(f"\n[EMBARGO] 历史训练集阻断终点 / Train End: {training_dates[-1].strftime('%Y-%m-%d')}")
    # [EN] Formats and prints the final date edge assigned for machine learning fitting operations.
    # [ZH] 格式化并打印允许机器学习拟合操作的历史边界终点日期。
    
    logger.info(f"\n[EMBARGO] 盘后决策执行观测点 / Production Date: {current_production_date.strftime('%Y-%m-%d')}")
    # [EN] Formats and logs the precise execution snapshot target day.
    # [ZH] 格式化并记录本次盘后资产优化决策执行的精准截面基准观察日。
    
    # 2. 隔离抽取训练样本面板
    X_train = X_panel.loc[training_dates]
    # [EN] Uses location filters to pull out the features data frame strictly within safe training boundaries.
    # [ZH] 利用位置过滤器，严格提取处于安全训练日期边界内的特征矩阵子集。
    
    y_train = y_panel.loc[training_dates]
    # [EN] Extracts corresponding return vector columns matching identical historical data bounds.
    # [ZH] 提取与上述训练日期完全对应的目标收益率真实值标签向量。
    
    # 3. 激活符合性分位数回归引擎
    engine.fit_and_quantify(X_train, y_train)
    # [EN] Trains internal parameters on the training panel dataset to calibrate error metric boundaries.
    # [ZH] 在训练面板数据集上拟合内部参数，并计算出非参数化误差分布的校准边界值。
    
    # 4. 抽取当前生产观测日的最新特征向量
    current_X_dict = {}
    # [EN] Initializes an isolated lookup structure mapping active asset symbols to today's feature row vectors.
    # [ZH] 初始化一个隔离的查找字典，用于映射每个资产在今日最新的特征状态向量。
    
    historical_returns_builder = {}
    # [EN] Container dictionary designed to store continuous raw return columns for covariance matrix modeling.
    # [ZH] 容器字典，用于拉取存储连续的历史收益率序列，以备后续进行资产协方差矩阵建模。
    
    for symbol in config.SYMBOLS:
        # [EN] Begins asset-by-asset traversal looping across identifiers listed inside the configuration variables.
        # [ZH] 开始逐个资产遍历循环，对应配置变量中所列出的所有证券代码。
        
        current_X_dict[symbol] = X_panel.xs((current_production_date, symbol)).values
        # [EN] Uses cross-section slicing (.xs) to isolate the singular matrix row vector matching today's date and the current asset symbol.
        # [ZH] 使用多重索引交叉截面切片 (.xs) 提取出仅属于今日、且仅属于当前资产的单行特征数值向量。
        
        historical_returns_builder[symbol] = y_panel.xs(symbol, level='Symbol')
        # [EN] Isolates the complete history of returns across all dates for the specific asset to build covariance arrays.
        # [ZH] 切片抽取出该资产跨越所有历史日期的完整收益率纵向序列，用于组装风险矩阵。
        
    historical_returns_df = pd.DataFrame(historical_returns_builder).loc[:current_production_date].dropna()
    # [EN] Concat tracks into a unified DataFrame, limits timeline up to today, and strips out missing/NaN artifacts to avoid optimizer crashes.
    # [ZH] 将各资产收益序列合并为统一的 DataFrame，截取至今日，并滤除最后一天未发生的 NaN 空值，确保底层马科维茨优化器不会因无效值崩溃。

    # 5. 推演异方差感知不确定性空间极值
    predictions_dict = engine.predict_with_bounds(current_X_dict)
    # [EN] Runs inference through the calibrated model to generate asset return predictions, ceilings, and floor boundaries.
    # [ZH] 将今日的最新的特征输入已校准的模型中，推演计算出未来一期收益率的点预测值、非对称上下置信边界。
    
    # 6. 跨越贝叶斯网桥，执行矩阵级 BL 融合更新与马科维茨多资产 MVO 最优权重分配
    bl_returns, target_weights, asset_variances = bridge.compute_matrix_bl_and_optimize(
        predictions_dict, historical_returns_df
    )
    # [EN] Executes Black-Litterman mathematical blending of ML outputs and resolves quadratic equations to find optimal portfolio weights.
    # [ZH] 执行 Black-Litterman 数学矩阵计算融合机器学习观点，并通过求解二次型方程寻找效用最大化的资产配置目标权重。
    
    # 7. 组装 LLM-Ready 决策 Payload 快照
    llm_payload = {
        "metadata": {
            "execution_timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            # [EN] Records the exact computational system wall-clock execution time.
            # [ZH] 记录该算法在计算机系统运行时的精确现实钟表执行时间戳。
            
            "production_data_date": current_production_date.strftime('%Y-%m-%d'),
            # [EN] Standardizes the market state timestamp tracking index into an ISO string.
            # [ZH] 将市场行情观察日的索引时间戳标准化转换为标准的字符串日期。
            
            "tracked_assets": config.SYMBOLS
            # [EN] Passes the underlying universe token configuration down into the summary map.
            # [ZH] 将底层交易标的资产池代号配置存入元数据映射表中。
        },
        "cqr_uncertainty_metrics": {
            sym: {
                "point_prediction": float(predictions_dict[sym][0]),
                # [EN] Casts the model's base directional return vector projection to float primitive.
                # [ZH] 将机器学习模型预测的基础未来方向收益率强转为基础 float 类型。
                
                "conformal_floor": float(predictions_dict[sym][1]),
                # [EN] Statistical worst-case safe baseline return rate bound.
                # [ZH] 统计学最坏情况下的安全下限预测收益率边界值。
                
                "conformal_ceiling": float(predictions_dict[sym][2]),
                # [EN] Statistical best-case return rate limit projection.
                # [ZH] 统计学最好情况下的上限收益率极限预测边界值。
                
                "heteroskedastic_width": float(predictions_dict[sym][2] - predictions_dict[sym][1])
                # [EN] Subtracts floor from ceiling to gauge dynamic variance space width on specific asset conditions.
                # [ZH] 用上限减去下限，度量出该资产因今日状态引发的动态条件风险区间的整体幅宽（异方差宽度）。
            } for sym in config.SYMBOLS
        },
        "bayesian_portfolio_allocator": {
            "bl_posterior_expected_returns": [float(r) for r in bl_returns],
            # [EN] Maps combined post-inference Bayesian mathematical asset return outputs into a clean list array.
            # [ZH] 将经由贝叶斯融合更新后的后验复合预期收益率数组转为标准的浮点数列表。
            
            "optimal_mvo_weights": {sym: float(w) for sym, w in zip(config.SYMBOLS, target_weights)}
            # [EN] Pairs symbols with optimal Markowitz allocations to construct the explicit production target state dictionary.
            # [ZH] 将股票代码与优化出的马科维茨最佳仓位权重配对，构建出具体的实盘分配决策字典。
        }
    }
    
    json_out = os.path.join(config.REPORT_DIR, "multi_asset_llm_payload.json")
    # [EN] Configures path targeting output folder for serializing metadata records onto local disk memory.
    # [ZH] 拼接出用于存储大模型就绪格式文件的本地磁盘绝对/相对物理路径。
    
    with open(json_out, 'w', encoding='utf-8') as f:
        # [EN] Safely opens file handle writing interface under systemic UTF-8 compliance text structures.
        # [ZH] 安全打开物理文件写入句柄，指定系统级 UTF-8 编码规范写入文本流。
        
        json.dump(llm_payload, f, indent=4, ensure_ascii=False)
        # [EN] Serializes object into human-readable indent structured text block, suppressing character transformations.
        # [ZH] 将对象序列化写出为带有 4 格缩进的结构化文本，并禁用 ASCII 转义以原生支持中文字符。
        
    # 8. 呈现多资产中英双语核心决策看板
    print("\n" + "="*85)
    print(f"      📊 MULTI-ASSET CONFORMAL PORTFOLIO OPTIMIZATION DISPATCH TERMINAL")
    print(f"      数据观测日 / Market Snapshot Date: {llm_payload['metadata']['production_data_date']}")
    print("="*85)
    print(f" 1. 符合性分位数回归 (CQR) 异方差估计 / HETEROSKEDASTIC UNCERTAINTY QUANTIFICATION:")
    for sym in config.SYMBOLS:
        metrics = llm_payload['cqr_uncertainty_metrics'][sym]
        print(f"    ▶ 资产 [{sym}] -> 点预测: {metrics['point_prediction']:+.5f} | "
              f"符合性区间: [{metrics['conformal_floor']:+.4f}, {metrics['conformal_ceiling']:+.4f}] | "
              f"动态幅宽: {metrics['heteroskedastic_width']:.5f}")
    print("-"*85)
    print(f" 2. 贝叶斯后验与最优资产分配权重 / BAYESIAN POSTERIOR & MVO TARGET WEIGHTS:")
    for i, sym in enumerate(config.SYMBOLS):
        w_pct = llm_payload['bayesian_portfolio_allocator']['optimal_mvo_weights'][sym] * 100
        post_r = llm_payload['bayesian_portfolio_allocator']['bl_posterior_expected_returns'][i]
        print(f"    ▶ 资产 [{sym:11s}] -> BL后验预期收益: {post_r:+.6f} | ⚙️ 建议目标配资权重: {w_pct:6.2f}%")
    print("="*85 + "\n")
    # [EN] Block that displays formatting print logs directly into standard out terminal console environment for visual status validation.
    # [ZH] 控制台日志可视化打印输出块，将格式化的中英核心指标直接输出到标准终端中以供状态监控。
    
    return llm_payload
    # [EN] Returns the nested structured deployment object container to parent execution calls.
    # [ZH] 向父级调用函数返回封装完毕、可供外部继续读取的嵌套决策配置数据字典。

if __name__ == "__main__":
    # [EN] Native interpreter conditional trap checking if the runtime file is being launched as the main application thread.
    # [ZH] 原生脚本执行入口条件陷阱，判断当前文件是否作为主程序线程被独立直接启动运行。
    
    print("Starting main script...运行主管道...")
    # [EN] Prints boot initiation message straight to system output stream.
    # [ZH] 向系统标准流输出启动初始化提示信息。
    
    # 生成多资产同步时序仿真测试沙盒
    np.random.seed(42)
    # [EN] Freezes underlying numeric generation seeds to guarantee consistent multi-asset price vector values across development testing.
    # [ZH] 固定底层数学伪随机数发生器种子，确保开发测试阶段模拟出的多资产价格波动序列完全一致可复现。
    
    config = PipelineConfig()
    # [EN] Generates baseline settings descriptor mapping the targeted tracking symbols.
    # [ZH] 实例化全局配置描述符，获取系统指定的资产代码目标池。
    
    mock_dates = pd.date_range(end=datetime.today(), periods=250, freq='D')
    # [EN] Generates a continuous daily time vector going backward 250 trading intervals from today's system clock.
    # [ZH] 以当前系统时钟为基准向前倒退，生成包含连续 250 个每日频率项的时间序列轴向量。
    
    raw_multi_asset_data = {}
    # [EN] Initializes an empty lookup dictionary to anchor generated simulated market ticker states.
    # [ZH] 初始化一个空的数据字典，用于挂载后续生成的多资产模拟行情历史状态。
    
    for sym in config.SYMBOLS:
        # [EN] Loops inside test environment to mock up functional time-series structures per registered ticker asset.
        # [ZH] 在测试环境中循环遍历，为配置资产池中注册的每个标的代码虚构出行情时序。
        
        sim_close = 10.0 * np.exp(np.cumsum(np.random.normal(0.0003, 0.012, size=250)))
        # [EN] Formulates a standard Geometric Brownian Motion random walk path simulating 250 historical closing prices starting at $10.0.
        # [ZH] 使用几何布朗运动随机漫步公式，为目标资产模拟生成 250 期符合正态对数分布、从 10 元面值起步的收盘价序列。
        
        raw_multi_asset_data[sym] = pd.DataFrame({
            'Open': sim_close * (1 + np.random.normal(0, 0.001, 250)),
            'High': sim_close * (1 + np.abs(np.random.normal(0.002, 0.001, 250))),
            'Low': sim_close * (1 - np.abs(np.random.normal(0.002, 0.001, 250))),
            'Close': sim_close,
            'Volume': np.random.randint(200000, 800000, size=250),
            'Turnover': np.random.uniform(0.01, 0.06, size=250)
        }, index=mock_dates)
        # [EN] Constructs a multi-column dataframe with realistic noise variations for Open, High, Low, Volume and Turnover metrics.
        # [ZH] 组装包含开/高/低/收/量/率的多列模拟行情 DataFrame，并在其中加入符合金融统计特征的扰动噪声。
        
    payload = run_production_pipeline(raw_multi_asset_data)
    # [EN] Launches the core scheduling pipeline using the simulated dataset sandbox, returning the structural metrics file.
    # [ZH] 将模拟生成的测试沙盒数据集送入核心调度管道中运行，最终获得大模型就绪的决策负载字典。
