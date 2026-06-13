# Portfolio.py Line-by-Line Code Explanation / 投资组合优化层逐行代码注释解析

Below is the fully annotated `portfolio.py` execution script. Each line of code contains explicit inline structural comments mapping out the system logic in both English [EN] and Simplified Chinese [ZH].

以下是完整带注释的 `portfolio.py` 投资组合优化模型脚本。每行代码配有明确的中英文双语结构化注释，解析其底层逻辑。

```python
# portfolio.py
# [EN] Portfolio optimization layer implementing Ledoit-Wolf robust shrinkage, Black-Litterman blending, and Markowitz optimization.
# [ZH] 投资组合优化层脚本，整合了 Ledoit-Wolf 鲁棒收缩协方差估计、Black-Litterman 贝叶斯融合以及马科维茨优化算法。

import logging
# [EN] Standard Python logging utility used to register internal pipeline events under modular namespaces.
# [ZH] Python 标准日志模块，用于在模块化命名空间下记录流水线内部运行状态。

import numpy as np
# [EN] Scientific computing extension used to execute array linear algebra, matrix inversions, and vector multiplications.
# [ZH] 科学计算扩展库，用于执行高性能数组线性代数、矩阵求逆以及向量乘法。

import pandas as pd
# [EN] Core data frame library deployed to handle historical asset price series slicing and time-series alignment.
# [ZH] 核心数据框结构库，用于处理历史资产价格序列的切片与时序对齐。

from scipy.optimize import minimize
# [EN] Scientific mathematical optimizer toolbox, specifically importing the non-linear multi-constraint optimization engine.
# [ZH] 科学数学优化器工具箱，专门引入非线性多约束规划求解引擎。

from typing import Dict, Tuple
# [EN] Type-hint variables used to explicitly label dictionary mappings and multi-variable return statement tuple definitions.
# [ZH] 类型提示变量，用于显式声明字典映射容器与多变量返回值元组的结构类型。

from config import PipelineConfig
# [EN] Import path to fetch structural immutable parameters and algorithmic threshold settings from config blueprints.
# [ZH] 导入全局配置路径，以获取系统不可变参数蓝图与算法阈值设置。

from sklearn.covariance import LedoitWolf
# [EN] Scikit-learn shrinkage estimation object used to optimize high-dimensional covariance matrices against noise artifacts.
# [ZH] Scikit-learn 机器学习库中的收缩估计器对象，用于在历史样本较短时优化高维协方差矩阵以对抗噪声。

logger = logging.getLogger("QuantPipeline.Portfolio")
# [EN] Initializes a unique localized logging child subsystem channel dedicated to portfolio allocations.
# [ZH] 初始化一个专属于投资组合管理子系统的独立本地日志追踪频道。

class BayesianExecutionBridge:
    # [EN] Main wrapper class driving the execution bridge blending ML forecasts with Markowitz asset allocation models.
    # [ZH] 封装贝叶斯执行桥的核心逻辑类，负责融合机器学习预测结果与马科维茨资产配置模型。

    def __init__(self, config: PipelineConfig):
        # [EN] Instantiates the management bridge and sets the global configuration settings context locally.
        # [ZH] 类构造函数，初始化配置管理桥并将全局参数上下文锁定在本地命名空间内。
        
        print("\nInitializing BayesianExecutionBridge with provided configuration...")
        # [EN] Prints process status message to terminal stdout window in English.
        # [ZH] 向控制台标准输出流打印英文状态信息，提示正在初始化当前模块。
        
        print("初始化数据桥...")
        # [EN] Prints corresponding synchronized status message to terminal screen in Simplified Chinese.
        # [ZH] 向控制台屏幕输出同步的简体中文状态初始化提示。
        
        self.config = config
        # [EN] Hardcodes the injected immutable control config dataclass parameters onto the private instance variable.
        # [ZH] 将只读的全局配置参数类实例永久绑定至本地私有变量 `self.config` 中。

    def compute_matrix_bl_and_optimize(
        self, 
        predictions_dict: Dict[str, Tuple[float, float, float]], 
        historical_returns_df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # [EN] Method consuming CQR prediction intervals and return histories to compute BL updates and return optimized weight vectors.
        # [ZH] 核心矩阵优化方法：传入 CQR 预测区间字典与历史收益率矩阵，求解 BL 后验分布并返回最优配资权重向量。
        
        """
        基于高维标准矩阵级黑-利特曼公式求解后验收益分布，并直接进行均值-方差最大夏普比率优化。
        """
        print("\nComputing matrix-level Black-Litterman posterior and optimizing portfolio weights...")
        # [EN] Sends progress logs to terminal standard out streams in English text.
        # [ZH] 向控制台输出英文日志，提示系统开始执行矩阵级 BL 后验计算与权重组合优化。
        
        print("计算矩阵级 Black-Litterman 后验并优化投资组合权重...")
        # [EN] Sends synchronized process update info onto display monitors in Simplified Chinese text.
        # [ZH] 向控制台输出同步的简体中文进程状态更新提示。
        
        symbols = self.config.SYMBOLS
        # [EN] Retrieves the core target asset list universe array from configuration settings.
        # [ZH] 从全局配置参数中拉取核心的目标证券标的池列表数组。
        
        n_assets = len(symbols)
        # [EN] Evaluates the absolute asset tracker count to establish linear dimension boundaries for tensor operations.
        # [ZH] 计算标的池资产总数，用以确立后续矩阵和张量代数运算的线性维度边界。
        
        w_size = self.config.ROLLING_WINDOW_SIZE
        # [EN] Slices the backward daily window size constraints to set statistical evaluation limits.
        # [ZH] 读取配置中的滚动时间窗口天数，用以限定参与统计评估的历史样本切片长度。
        
        # 1. 估计内生先验资产协方差矩阵 Sigma (NxN)
        hist_window = historical_returns_df[symbols].iloc[-w_size:]
        # [EN] Slices the bottom row indices matching the exact window limits for the tracked financial assets.
        # [ZH] 截取历史收益率矩阵中最后 `w_size` 行（即限定窗口期内）且仅属于目标资产池的数据切片。
        
        lw = LedoitWolf()
        # [EN] Instantiates the robust adaptive shrinkage model framework from sklearn.
        # [ZH] 实例化 sklearn 的 Ledoit-Wolf 自适应收缩模型对象，用于估计稳健的风险协方差。
        
        Sigma = lw.fit(hist_window.values).covariance_
        # [EN] Runs shrinkage estimation curves over rows of return variables to construct a non-singular robust covariance matrix.
        # [ZH] 在历史收益率数值矩阵上运行自适应收缩拟合，构建出一个能有效对抗随机噪声、非奇异的稳健协方差矩阵。
        
        Sigma += np.eye(n_assets) * 1e-8
        # [EN] Implements a localized ridge-penalty fallback item to provide an absolute numerical insurance shield against inversion error failures.
        # [ZH] 对角线强行叠加 1e-8 的微量岭惩罚项，作为双重数学保险，绝对杜绝奇异矩阵求逆时的数值崩溃。
        
        print("\nEstimating shrinkage prior asset covariance matrix Sigma via Ledoit-Wolf...")
        # [EN] Sends technical notification outlining covariance method execution down to standard outputs.
        # [ZH] 向标准输出流发送英文技术通告，提示正通过 Ledoit-Wolf 算法估计收缩风险协方差矩阵。
        
        print("利用 Ledoit-Wolf 估计收缩先验资产协方差矩阵 Sigma...")
        # [EN] Sends technical process logging updates onto developer screen monitors in Simplified Chinese.
        # [ZH] 向开发控制台发送同步的简体中文技术更新日志。
        
        # 2. 构造市场隐含均衡先验收益率向量 Pi (Nx1) - 采用经验滚动均值作为代理
        Pi = hist_window.mean().values.reshape(-1, 1)
        # [EN] Computes the mean return vector across window entries and reshapes the tensor array into an explicit Nx1 matrix column.
        # [ZH] 计算历史窗口内的资产收益率均值向量，并将其形状强转重塑为标准的 N 行 1 列的先验收益率输入矩阵。
        
        # 3. 提取 CQR 点预测作为 ML 观点向量 Q (Nx1)，并利用异方差幅宽构建不确定性协方差矩阵 Omega (NxN)
        Q = np.zeros((n_assets, 1))
        # [EN] Pre-allocates an all-zero tensor placeholder vector corresponding to rows of target assets to build ML views.
        # [ZH] 预先分配一个全零的、行数与资产数一致的 N 行 1 列列向量，用作存放机器学习观点的容器。
        
        Omega = np.zeros((n_assets, n_assets))
        # [EN] Pre-allocates an empty NxN square matrix template designed to hold the statistical variance bounds of ML error streams.
        # [ZH] 预先分配一个 N 行 N 列的全零方阵，用作存储机器学习预测误差不确定性的协方差方阵。
        
        for i, symbol in enumerate(symbols):
            # [EN] Iterates over the universe to index asset variables alongside their position trackers.
            # [ZH] 遍历证券代码列表，同时提取出当前标的资产的索引位置与字符串代号。
            
            p_mean, p_low, p_high = predictions_dict[symbol]
            # [EN] Unpacks the predictive tuple container containing point forecasts, conformal floors, and conformal ceilings.
            # [ZH] 解包当前资产在预测字典中对应的三元组，提取点预测值、符合性下限以及符合性上限。
            
            Q[i, 0] = p_mean
            # [EN] Assigns the point model calculation forecast into the respective slot inside the view vector matrix.
            # [ZH] 将机器学习模型的点预测预期收益率赋值到观点向量矩阵 $Q$ 的对应位置中。
            
            # 条件幅宽物理测度
            cqr_width = p_high - p_low
            # [EN] Evaluates the localized heteroskedastic return metric width derived from non-parametric regression profiles.
            # [ZH] 用上限减去下限，度量出基于非参数分位数回归推演出的条件异方差动态区间幅宽。
            
            # Omega_ii = width^2 * tau
            Omega[i, i] = (cqr_width ** 2) * self.config.TAU
            # [EN] Populates the active main diagonal coordinate cell using variance squares amplified by the baseline hyperparameter scalar scale.
            # [ZH] 填充对角线核心单元格：将区间幅宽进行平方转换为方差，并乘以相对权重标度参数 `TAU`。
            
            if Omega[i, i] <= 0:
                Omega[i, i] = 1e-7
                # [EN] Protection rule substituting any analytical arithmetic anomalies or exact zeros with safe floors to ensure invertibility.
                # [ZH] 零值防御边界：若计算出的误差方差小于等于 0，则强行赋予 1e-7 的底线正值，确保其可逆性。
                
        # ==============================================================================
        # 【已修复】以下核心贝叶斯矩阵计算与优化逻辑已移出 for 循环，确保 Omega 填充完毕后执行
        # ==============================================================================
        
        # 4. 执行标准高维 Black-Litterman 贝叶斯共轭更新封闭解
        inv_tau_Sigma = np.linalg.inv(self.config.TAU * Sigma)
        # [EN] Inverts the product tensor scaling prior asset variances against the global parameter tau factor.
        # [ZH] 对经由权重标度参数变焦放缩后的先验协方差矩阵（$\tau \Sigma$）执行标准的矩阵求逆变换。
        
        inv_Omega = np.linalg.inv(Omega)
        # [EN] Inverts the diagonal uncertainty covariance matrix to build the model prediction view precision matrix.
        # [ZH] 对不确定性对角矩阵进行求逆变换，构建出对模型预测观点的精度度量矩阵（$\Omega^{-1}$）。
        
        # 精度项叠加：(tau*Sigma)^-1 + P^T * Omega^-1 * P
        posterior_precision = np.linalg.inv(inv_tau_Sigma + inv_Omega)
        # [EN] Combines and inverts precision measurements to output the unified post-inference Bayesian consensus information array.
        # [ZH] 将市场先验精度与观点精度相加并执行求逆，解出统一的贝叶斯共轭更新后验协方差矩阵。
        
        # 后验期望收益率矩阵公式
        BL_returns = np.dot(posterior_precision, np.dot(inv_tau_Sigma, Pi) + np.dot(inv_Omega, Q)).flatten()
        # [EN] Computes the dot product equations combining historical priors and adaptive inputs, flattening the output vector into 1D space.
        # [ZH] 按照经典 BL 数学矩阵乘法公式融合多方数据信息源，并将最终输出的后验期望收益率多维矩阵展平成一维数组。
        
        # 5. 马科维茨均值-方差框架下的鲁棒最大夏普比率非线性二次规划优化器 (MVO)
        def negative_sharpe(weights):
            # [EN] Defines the localized runtime objective callback evaluating the negative Sharpe ratio parameter.
            # [ZH] 定义用于优化的本地回调目标函数，计算负的夏普比率（因为 scipy minimize 执行的是极小值寻优）。
            
            port_return = np.dot(weights, BL_returns)
            # [EN] Evaluates expected portfolio consensus return projections via vector dot products.
            # [ZH] 通过权重向量与后验收益向量的内积，计算出当前权重组合的预期总体收益。
            
            port_vol = np.sqrt(np.dot(weights.T, np.dot(Sigma, weights)))
            # [EN] Evaluates expected risk dispersion levels by mapping vectors across robust covariance structures.
            # [ZH] 将权重向量映射到稳健协方差方阵中，通过二次型矩阵乘法并开方计算出投资组合的预期总波动率。
            
            sharpe = (port_return - self.config.RISK_FREE_RATE) / (port_vol + 1e-8)
            # [EN] Standard formulation computing the excess risk return metric using small floor dividers to dodge zero faults.
            # [ZH] 标准夏普比率公式：超额收益除以组合波动率，分母加入 1e-8 杜绝数学除零故障。
            
            return -sharpe
            # [EN] Returns negative score mapping high Sharpe values down to optimization minima directions.
            # [ZH] 返回负的夏普值，将最大化夏普比率的数学问题巧妙转化为寻找极小值的寻优任务。

        # 现实交易约束：禁止任何形式的融券卖空 (Long-Only) 且全额满仓约束
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
        # [EN] Forces complete capital allocation constraints where total weights sum up to exactly 1.0 (100% full commitment).
        # [ZH] 强加线性等式约束限制：所有资产权重之和必须严格等于 1.0（即全额满仓，资金利用率 100%）。
        
        bounds = [(0.0, 1.0) for _ in range(n_assets)]
        # [EN] Establishes value arrays mapping each variable boundary index between 0.0 and 1.0 to enforce long-only allocation parameters.
        # [ZH] 为每个标的变量设立独立边界：限定权重在 [0.0, 1.0] 区间内，从底层机制上彻底禁止融券做空行为。
        
        initial_weights = np.array([1.0 / n_assets] * n_assets)
        # [EN] Generates an evenly weighted sample array vector to seed the start configuration coordinates.
        # [ZH] 构建均匀分配的等权重初始数组，用作非线性全局寻优搜索的起点坐标。
        
        optimization_result = minimize(
            negative_sharpe, 
            initial_weights, 
            method='SLSQP', 
            bounds=bounds, 
            constraints=constraints
        )
        # [EN] Launches the non-linear SLSQP optimization algorithm under explicit system equations limits.
        # [ZH] 调用系统优化核心，采用标准的 Sequential Least Squares Programming (SLSQP) 算法求解约束二次规划问题。
        
        optimal_weights = optimization_result.x if optimization_result.success else initial_weights
        # [EN] Filters optimization outcome flags; falls back to equal weighting defaults if numerical convergence fails.
        # [ZH] 校验求解收敛状态标志：若非线性寻优成功则输出最优解权重，若收敛失败则回退至等权重安全线。
        
        return BL_returns, optimal_weights, Sigma.diagonal()
        # [EN] Returns the consolidated analytical tuple array back to the orchestration runner namespace.
        # [ZH] 将后验收益率、最优决策分配权重、以及资产原特质方差（对角线数值）打包成三元组，正式返回给主管道。
