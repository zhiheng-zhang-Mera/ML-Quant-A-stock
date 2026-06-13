# Models.py Line-by-Line Code Explanation / 机器学习模型层逐行代码注释解析

Below is the fully annotated `models.py` analytics script. Each line of code contains explicit inline structural comments mapping out the system logic in both English [EN] and Simplified Chinese [ZH].

以下是完整带注释的 `models.py` 统计预测模型脚本。每行代码配有明确的中英文双语结构化注释，解析其底层逻辑。

```python
# models.py
# [EN] Predictive machine learning layer implementing Conformal Quantile Regression (CQR) via LightGBM for uncertainty quantification.
# [ZH] 预测机器学习层脚本，基于 LightGBM 实现了符合性分位数回归 (CQR)，用以量化条件异方差金融不确定性。

import logging
# [EN] Core Python tracking module deployed to stream script lifecycle notifications onto registration targets.
# [ZH] Python 核心日志追踪模块，用于向指定注册目标流式输出脚本生命周期状态。

import numpy as np
# [EN] Vector array mathematical extensions utilized to resolve array optimizations, maximizations, and quantiles.
# [ZH] 向量数组数学扩展库，用于求解数组的大矩阵优化、极值比对及分位数统计。

import pandas as pd
# [EN] Data analysis structure engine used to handle multi-indexed cross-sectional panel records.
# [ZH] 数据分析结构引擎，用于处理多级索引的跨截面时序面板记录。

import lightgbm as lgb
# [EN] High-performance Gradient Boosting Decision Tree (GBDT) framework used for multi-quantile training equations.
# [ZH] 高性能梯度提升决策树 (GBDT) 框架，用于多分位数回归方程的机器学习拟合训练。

from typing import Tuple, Dict
# [EN] Static typing placeholders utilized to flag explicit structured coordinate lookup parameters.
# [ZH] 静态类型占位提示符，用于显式声明字典查找容器与多元返回组的结构参数。

from config import PipelineConfig
# [EN] Import path to fetch structural immutable parameters from the control config blueprint.
# [ZH] 全局配置路径，用于获取控制基准蓝图中的系统不可变参数。

logger = logging.getLogger("QuantPipeline.Models")
# [EN] Sets up a dedicated localized logging sub-registry instance named for the mathematical execution modules.
# [ZH] 建立一个专门为数学预测执行模块命名的、独立的本地日志追踪子单例系统。

class StatisticalAdaptiveEngine:
    # [EN] Main wrapper class governing LightGBM model tracking arrays and Conformal Concurrence calibration structures.
    # [ZH] 驱动统计适应引擎的核心逻辑类，管理 LightGBM 模型追踪池以及符合性校准参数结构。

    def __init__(self, config: PipelineConfig):
        # [EN] Constructor method initiating local model storage memory addresses mapping asset parameters.
        # [ZH] 类构造函数，初始化处理器本地用于映射资产参数的模型存储内存空地址。
        
        print("\nInitializing StatisticalAdaptiveEngine with provided configuration...")
        # [EN] Outputs execution update statement directly to standard user terminal screen in English.
        # [ZH] 向标准用户终端控制台屏幕输出英文运行提示，表明正在初始化当前引擎。
        
        print("初始化统计适应引擎...")
        # [EN] Synchronizes corresponding system validation notification onto console outputs in Simplified Chinese.
        # [ZH] 向控制台输出同步的简体中文状态初始化提示。
        
        self.config = config
        # [EN] Locks down the read-only configuration context blueprint properties onto the local entity object space.
        # [ZH] 将只读的全局配置参数蓝图属性锁定绑定至本地实体空间变量 `self.config` 中。
        
        self.models: Dict[str, Dict[str, lgb.LGBMRegressor]] = {}
        # [EN] Initializes a nested storage lookup dictionary to map asset symbols to their three quantile booster pipelines.
        # [ZH] 初始化一个嵌套的存储查找字典，用以映射每个资产代号与其专属的三叉分流弱估计器模型实例。
        
        self.cqr_thresholds: Dict[str, float] = {}
        # [EN] Memory repository holding non-parametric scalar correction thresholds computed during sample calibrations.
        # [ZH] 内存仓库容器，用于存放历史样本校准阶段计算得出的非参数化标量修正阈值。

    def fit_and_quantify(self, X_panel: pd.DataFrame, y_panel: pd.Series):
        # [EN] Core calibration module training multi-quantile models and compiling risk deviation parameters.
        # [ZH] 核心校准拟合方法：训练多分位数回归决策树，并计算统计学意义上的非参数风险偏离误差阈值。
        
        print("\nFitting conformal quantile regression models and quantifying uncertainty...")
        # [EN] Logs processing phase changes onto standard out console windows in English.
        # [ZH] 向标准输出控制台窗口发送英文进程更改日志，提示正在拟合 CQR 模型。
        
        print("拟合符合性分位数回归模型并量化不确定性...")
        # [EN] Logs synchronized validation notices onto developer screens in Simplified Chinese text.
        # [ZH] 向开发控制台发送同步的简体中文拟合与量化更新提示。
        
        """
        执行符合性分位数回归（CQR）训练，显式捕获条件异方差性。
        """
        self.models.clear()
        # [EN] Resets the model locator registry to fully wipe any obsolete historical parameters before training.
        # [ZH] 强制清空模型定位注册表字典，彻底擦除上一轮残留的历史陈旧训练参数。
        
        self.cqr_thresholds.clear()
        # [EN] Flushes out past threshold arrays to prevent memory leak cross-contamination over modern computing loops.
        # [ZH] 强制清空过往的校准阈值数组，杜绝新一轮循环中出现内存交叉污染。
        
        w_size = self.config.ROLLING_WINDOW_SIZE
        # [EN] Retrieves day bounds from settings specifying exactly how many historical data points to slide over.
        # [ZH] 从配置中提取指定的滚动窗口天数，界定参与本次机器学习训练的历史样本长度。
        
        low_q = self.config.ALPHA / 2.0
        # [EN] Calculates target percentage boundaries for the lower tail risk estimator (e.g., 0.05 / 2 = 0.025 or 2.5%).
        # [ZH] 计算低分位数回归器的目标分位线点（例如 0.05 / 2 = 0.025，即 2.5% 分位边界）。
        
        high_q = 1.0 - (self.config.ALPHA / 2.0)
        # [EN] Calculates target percentage boundaries for the upper tail risk estimator (e.g., 1.0 - 0.025 = 0.975 or 97.5%).
        # [ZH] 计算高分位数回归器的目标分位线点（例如 1.0 - 0.025 = 0.975，即 97.5% 分位边界）。
        
        for symbol in self.config.SYMBOLS:
            # [EN] Commences multi-asset parsing loops across string tickers registered in system parameters.
            # [ZH] 开始跨多资产遍历循环，依次处理系统参数中注册的每个证券标的代码。
            
            X_symbol = X_panel.xs(symbol, level='Symbol')
            # [EN] Cross-sections the feature data frame matrix to extract historical tracking records for the asset.
            # [ZH] 运用截面交叉切片，剥离取出仅属于当前资产的历史特征数据子矩阵矩阵。
            
            y_symbol = y_panel.xs(symbol, level='Symbol')
            # [EN] Cross-sections the target labels vector to extract corresponding chronological returns.
            # [ZH] 运用截面交叉切片，剥离取出与上述特征完全对齐的纵向真实目标收益率向量。
            
            X_train, y_train = X_symbol.iloc[-w_size:], y_symbol.iloc[-w_size:]
            # [EN] Slices trailing array matrices based on the rolling window size parameter to isolate safe training sets.
            # [ZH] 按照滚动天数参数截取时间轴末端最新的一段切片，隔离出用于估计当前状态的专用训练集。
            
            # 1. 初始化 CQR 所需的三叉分流弱估计器
            model_low = lgb.LGBMRegressor(objective='quantile', alpha=low_q, n_estimators=40, max_depth=3, learning_rate=0.05, verbose=-1, random_state=42)
            # [EN] Builds the lower tail LightGBM regressor using pinball loss tracking targets below alpha_low.
            # [ZH] 构建下限 LightGBM 分位数回归器，指定分位数目标为 `low_q`，通过分球损失追踪下尾部极端边界。
            
            model_high = lgb.LGBMRegressor(objective='quantile', alpha=high_q, n_estimators=40, max_depth=3, learning_rate=0.05, verbose=-1, random_state=42)
            # [EN] Builds the upper tail LightGBM regressor optimized to predict boundary constraints above alpha_high.
            # [ZH] 构建上限 LightGBM 分位数回归器，指定分位数目标为 `high_q`，用于追踪上尾部极端边界。
            
            model_mean = lgb.LGBMRegressor(objective='regression', n_estimators=40, max_depth=3, learning_rate=0.05, verbose=-1, random_state=42)
            # [EN] Builds the baseline conditional expectation model minimizing standard L2 mean squared error deviations.
            # [ZH] 构建传统的条件均值回归器，指定常规均方误差最小化目标，用作资产未来基础期望收益的方向预测。
            
            model_low.fit(X_train, y_train)
            # [EN] Runs gradient boosting trees optimization loops to fit the lower boundary estimator.
            # [ZH] 执行决策树梯度提升迭代优化，拟合训练出下限分位数预测模型。
            
            model_high.fit(X_train, y_train)
            # [EN] Runs gradient boosting trees optimization loops to fit the upper boundary estimator.
            # [ZH] 执行决策树梯度提升迭代优化，拟合训练出上限分位数预测模型。
            
            model_mean.fit(X_train, y_train)
            # [EN] Runs gradient boosting trees optimization loops to fit the baseline conditional expectation estimator.
            # [ZH] 执行决策树梯度提升迭代优化，拟合训练出基础均值期望预测模型。
            
            # 2. 计算校准集上的异方差符合性非参数误差分数 E_i
            preds_low = model_low.predict(X_train)
            # [EN] Extracts lower-bound prediction array sequences over calibration inputs.
            # [ZH] 在训练校准集上提取下限回归器输出的历史预测序列数组。
            
            preds_high = model_high.predict(X_train)
            # [EN] Extracts upper-bound prediction array sequences over calibration inputs.
            # [ZH] 在训练校准集上提取上限回归器输出的历史预测序列数组。
            
            # E_i = max(q_low(X_i) - y_i, y_i - q_high(X_i))
            cqr_residuals = np.maximum(preds_low - y_train.values, y_train.values - preds_high)
            # [EN] Formulates the exact element-wise conformal error scores mapping prediction deviations beyond estimated quantiles.
            # [ZH] 采用向量化对位最大化公式，计算各时间点真实值跳出预测分位数区间的非参数化绝对偏离误差得分向量。
            
            q_error_threshold = np.quantile(cqr_residuals, 1.0 - self.config.ALPHA)
            # [EN] Resolves the high-order scalar percentile boundary across error scores acting as the conformal calibration gap.
            # [ZH] 对上述偏离误差数组求解 $1 - \alpha$ 分位数（如 95% 分位数），解出具有严格数学覆盖率保证的校准步进标度阈值。
            
            self.models[symbol] = {
                "low": model_low,
                "high": model_high,
                "mean": model_mean
            }
            # [EN] Stores the calibrated model bundle inside the entity class look-up workspace matching the unique symbol.
            # [ZH] 将拟合完毕的三台核心模型打包成字典，注册锁定到全局模型容器 `self.models` 的资产代号槽位中。
            
            self.cqr_thresholds[symbol] = q_error_threshold
            # [EN] Logs the computed scalar margin buffer into localized storage fields matching the unique ticker symbol.
            # [ZH] 将计算出的校准阈值标度永久存入本地 `self.cqr_thresholds` 映射字典中。
            
        print("\nHeteroskedastic Conformal Quantile Regression (CQR) calibration matrix complete.")
        # [EN] Signals completion of multi-asset statistical tuning loops onto basic standard display screens in English.
        # [ZH] 向标准控制台终端打印英文结束通知，提示异方差符合性分位数回归校准矩阵构建完成。
        
        print("异方差符合性分位数回归（CQR）校准矩阵完成。")
        # [EN] Echoes synchronized completion updates down to terminal monitors via Simplified Chinese strings.
        # [ZH] 向控制台输出同步的简体中文流水线完成声明。
        
        logger.info("\nHeteroskedastic Conformal Quantile Regression (CQR) calibration matrix complete.")
        # [EN] Records system milestone parameters into information layer persistent files.
        # [ZH] 将系统里程碑事件正式记入信息级持久化日志记录文件中。
        
        return self
        # [EN] Hands back self-referenced class instance memory coordinates to facilitate chain execution paradigms.
        # [ZH] 返回自身类实例对象的内存指针，以便于系统支持链式编程调用范式。

    def predict_with_bounds(self, current_X_dict: Dict[str, np.ndarray]) -> Dict[str, Tuple[float, float, float]]:
        # [EN] Method processing newest observation vectors to derive point expectations with conformal risk margins.
        # [ZH] 边界预测方法：传入今日最新的状态特征向量，输出具有鲁棒不确定性覆盖的预测极值三元组。
        
        print("\nPredicting with bounds...")
        # [EN] Sends inference block initiation string states straight onto text stdout screens in English.
        # [ZH] 向标准输出流发送英文日志，提示系统正式启动带不确定性边界的自适应推理预测。
        
        print("进行带边界的预测...")
        # [EN] Echoes matching inference execution block notification onto developer screens in Simplified Chinese.
        # [ZH] 向开发控制台发送同步的简体中文边界预测启动提示。
        
        """
        为所有资产的最新的单一观测截面输出条件点预测和高保真 CQR 符合性置信极值。
        """
        predictions = {}
        # [EN] Instantiates a fresh local collection map to anchor output result arrays.
        # [ZH] 初始化一个空预测结果字典，用以承载汇总各资产最终计算出的多元收益极值。
        
        for symbol in self.config.SYMBOLS:
            # [EN] Commences token asset mapping loop across identifiers declared inside configuration classes.
            # [ZH] 启动资产循环，顺序处理全局配置中所列出的各个交易证券代号。
            
            if symbol not in self.models:
                raise RuntimeError(f"Asset model for {symbol} has not been calibrated yet.")
                # [EN] Halts processing thread if a ticker is requested before its calibration routines have executed.
                # [ZH] 安全中断阈值：若当前资产代号在池中找不到对应的已校准模型，则抛出运行时异常并中断。
                
            x_inst = current_X_dict[symbol].reshape(1, -1)
            # [EN] Reshapes the 1D structural feature row array into a standard 2D matrix shape to satisfy scikit-predict rules.
            # [ZH] 提取今日单行一维特征数组，并通过 `.reshape(1, -1)` 将其重塑为标准的二维单行矩阵，满足模型推理格式。
            
            mods = self.models[symbol]
            # [EN] Fetches the local reference dictionary holding the three underlying fitted estimation engines.
            # [ZH] 从本地提取出当前资产专属的、包含三台拟合模型的内部组件字典。
            
            q_err = self.cqr_thresholds[symbol]
            # [EN] Retrieves the calibrated non-parametric statistical scale buffer parameter computed during training.
            # [ZH] 提取出该资产在历史校准期内沉淀算出的非参数化统计分位数修正标度 `q_err`。
            
            raw_mean = mods["mean"].predict(x_inst)[0]
            # [EN] Evaluates the base expectation tree to extract the directional point return prediction scalar.
            # [ZH] 调用均值期望树模型进行推理，获取代表未来基础一期方向性收益率的点预测标度。
            
            raw_low = mods["low"].predict(x_inst)[0]
            # [EN] Evaluates the lower tail model tree to compute raw uncalibrated bottom quantile returns.
            # [ZH] 调用低分位数树模型进行推理，获取未加校准修正的原始低分位数收益率估计值。
            
            raw_high = mods["high"].predict(x_inst)[0]
            # [EN] Evaluates the upper tail model tree to compute raw uncalibrated top quantile returns.
            # [ZH] 调用高分位数树模型进行推理，获取未加校准修正的原始高分位数收益率估计值。
            
            # 应用符合性单步带状平移守卫公式
            conformal_low = raw_low - q_err
            # [EN] Shifts the uncalibrated lower bound downward by the precise historical calibration scale to guarantee coverage safety.
            # [ZH] 运用符合性单步带状平移守卫公式，将原始下限减去校准步长，推演得出具有严格数学覆盖率保证的下限收益率边界。
            
            conformal_high = raw_high + q_err
            # [EN] Shifts the uncalibrated upper bound upward by the precise historical calibration scale to encompass uncertainty.
            # [ZH] 同样将原始上限加上校准步长，向外推展拓宽，推演得出具有严格数学覆盖率保证的上限收益率边界。
            
            predictions[symbol] = (raw_mean, conformal_low, conformal_high)
            # [EN] Packages variables into a standard tuple tuple and registers the row value inside the tracking lookup maps.
            # [ZH] 将计算出的（点预测预期值、符合性安全下限、符合性安全上限）打包成三元组，注册进结果映射字典。
            
        print("\nPrediction with bounds complete.")
        # [EN] Sends technical notification outlining batch model prediction endings down to standard terminal windows.
        # [ZH] 向标准输出控制台发送英文完成通告，提示所有证券标的的边界预测计算全面结束。
        
        print("带边界的预测完成。")
        # [EN] Sends synchronized technical process logging updates onto monitoring screens in Simplified Chinese text.
        # [ZH] 向系统屏幕输出同步的简体中文流水线阶段完成信息。
        
        return predictions
        # [EN] Transmits completed asset prediction dictionary blocks back up into parent execution loops.
        # [ZH] 将组装完毕、包含全资产高保真区间概率边界的数据字典正式返回给调度总线。
