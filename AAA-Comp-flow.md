# Multi-Asset Conformal Portfolio Optimization Pipeline Workflow Report

# 多资产符合性投资组合优化流水线全局工作流报告

---

## Part 1: System-Wide Architecture & Complete Workflow

## 第一部分：全系统架构与完整工作流

This quantitative trading system utilizes advanced machine learning and robust Bayesian statistics to safely distribute capital across a multi-asset investment universe. Below is the end-to-end operational life cycle of how the five files orchestrate data and execute trades.

该量化交易系统利用先进的机器学习和稳健的贝叶斯统计学，在多资产投资组合中进行安全的资金分配。以下是这五个文件如何协同处理数据并执行交易的完整生命周期流程：

```
[config.py] ---------> Control Blueprint / 全局控制蓝图
                          |
[Market Data] --------> [data_processor.py] ---> Feature Panel Matrices / 特征面板矩阵
                          |
                   (Temporal Embargo) / 时序禁运隔离
                          |
                        [models.py] -----------> Heteroskedastic CQR Bounds / 动态异方差风险边界
                          |
                        [portfolio.py] --------> Bayesian BL & Markowitz MVO / 贝叶斯融合与仓位优化
                          |
[main-dual-lang.py] --> [LLM Payload JSON] ----> Structural Downstream Action / 大模型就绪决策报文

```

### 1. Unified Operational Workflow / 统一运行流步序

#### Step 1: Blueprint Bootstrapping (`config.py`) / 步骤 1：读取配置蓝图

* **What happens**: The system boots up and reads `config.py`. This script instantiates an immutable, read-only parameter schema (`PipelineConfig`) using Python's `@dataclass(frozen=True)`.
* **业务逻辑**：系统启动并读取 `config.py`。该脚本利用 Python 的 `@dataclass(frozen=True)` 实例化一个不可变、只读的全局参数架构（`PipelineConfig`）。
* **Calculated Metrics**: Sets up the target configuration variables: significance level $\alpha=0.05$ (95% target confidence), Black-Litterman scale factor $\tau=0.02$, rolling window size = 120 days, and the target asset symbols pool.
* **计算指标**：设定核心全局配置变量：显著性水平 $\alpha = 0.05$（即 95% 的目标置信度）、Black-Litterman 缩放因子 $\tau = 0.02$、滚动时间窗口大小 = 120 天，以及目标资产代码池。
* **Output**: A frozen global configuration instance imported by all other processing components.
* **输出成果**：一个冻结的全局配置实例，供后续所有数据处理与核心计算组件导入调用。

#### Step 2: Ingestion & Economic Neutralization (`data_processor.py`) / 步骤 2：数据加工与通胀平减

* **What happens**: `main-dual-lang.py` sends raw OHLCV market dictionary data straight into the `DataProcessor`. The data processor first strips out macroeconomic inflation noise by discounting nominal historical values against a compounding real purchasing power deflator.
* **业务逻辑**：主调度总线 `main-dual-lang.py` 将原始的 OHLCV（开高低收量）市场行情字典数据直接注入 `DataProcessor`。数据处理器首先通过使用复利形式的实际购买力平减指数对名义历史价格进行贴现，从而剥离宏观经济中的通货膨胀噪音。
* **Calculated Metrics / 计算指标**:

$$\mathrm{Deflator} = (1.0 + R_{\mathrm{annual\_inflation}})^{\frac{T_{\mathrm{max}} - T_{\mathrm{current}}}{365.25}}$$

$$\mathrm{Price}_{\mathrm{real}} = \mathrm{Price}_{\mathrm{nominal}} \times \mathrm{Deflator}$$

* **Output**: A clean, aligned, multi-indexed pandas DataFrame panel structured strictly by row coordinates `[Date, Symbol]`.
* **输出成果**：一个干净、对齐、严格按照 `[Date, Symbol]`（日期，代码）双重索引构建的 pandas DataFrame 面板数据坐标矩阵。

#### Step 3: Predictive Feature Engineering (`data_processor.py`) / 步骤 3：白盒特征矩阵构建

* **What happens**: The processor extracts five transparent, human-interpretable quantitative technical indicators across the real price arrays.
* **业务逻辑**：处理器基于平减后的实际价格序列，提取出五个高透明度、具备人类可解释性的白盒量化技术指标。
* **Calculated Metrics / 计算指标**:
* 1-day, 5-day, and 20-day log price momentum variables (`Mom_1D`, `Mom_5D`, `Mom_20D`).
* 1天、5天和20天的对数价格动量变量（`Mom_1D`, `Mom_5D`, `Mom_20D`）。
* Intraday Garman-Klass Volatility signatures (`GK_Vol`).
* 日内 Garman-Klass 日度波动率特征签名（`GK_Vol`）。
* Dynamic liquidity volume shocks (`Turnover_Shock`) standardizing turnover volumes against a 5-day rolling average window.
* 动态流动性成交量冲击指标（`Turnover_Shock`），通过将当前成交量对5日滚动均值窗口进行标准化得到。
* Next-day shifted log real forward return arrays as target labels (`target_return`).
* 次日错位对数实际远期收益率序列，用作预测目标标签（`target_return`）。


* **Output**: Two perfectly aligned matrix spaces: Feature Space Matrix `X_panel` and Target Label Vector `y_panel`.
* **输出成果**：两个完全对齐的矩阵空间：特征空间矩阵 `X_panel` 以及目标标签向量 `y_panel`。

#### Step 4: Strict Temporal Embargo Filtering (`main-dual-lang.py`) / 步骤 4：严格时序禁运隔离

* **What happens**: To guarantee that no information overlaps or leaks future details back into our historical data, the master bus splits the training and production dates apart. It rewinds the training timeline backward from the modern edge by a strict buffer zone.
* **业务逻辑**：为了确保没有任何信息重叠，并防止未来的数据泄露到历史训练集中，主调度总线将训练日期与生产日期进行了彻底的分离。系统从当前生产观测沿时间轴向后倒退一个严格的缓冲区。
* **Calculated Metrics / 计算指标**:

$$I_{\mathrm{train\_end}} = N_{\mathrm{total\_dates}} - 1 - P_{\mathrm{embargo}}$$

* **Output**: Completely isolated historical training datasets (`X_train`, `y_train`) and a singular cross-sectional feature array captured on the live production snapshot observation date (`current_X_dict`).
* **输出成果**：完全隔离的历史训练数据集（`X_train`, `y_train`）以及在当前生产实时快照观测日捕获的单截面特征数组（`current_X_dict`）。

#### Step 5: Adaptive Model Fitting & Statistical Calibration (`models.py`) / 步骤 5：自适应拟合与分位数校准

* **What happens**: The historical training subsets are sent into the `StatisticalAdaptiveEngine`. For each symbol in our asset pool, three distinct LightGBM decision tree regressions are fitted simultaneously.
* **业务逻辑**：历史训练子集被输入到 `StatisticalAdaptiveEngine`（统计自适应引擎）中。针对资产池中的每只资产，系统将同时拟合三个不同的 LightGBM 决策树回归模型。
* **Calculated Metrics**: Fits the conditional median mean, a lower tail pinball-loss model ($\alpha_{\mathrm{low}} = 0.025$), and an upper tail pinball-loss model ($\alpha_{\mathrm{high}} = 0.975$). It captures uncalibrated errors on the training set and computes an exact empirical margin penalty threshold:
* **计算指标**：分别拟合条件中位数、基于下尾部 Pinball 损失的模型（$\alpha_{\mathrm{low}} = 0.025$）以及基于上尾部 Pinball 损失的模型（$\alpha_{\mathrm{high}} = 0.975$）。随后，它捕获训练集上未校准的预测误差，并计算出一个精确的经验边际惩罚阈值：

$$E_i = \max\left(\hat{q}_{\mathrm{low}}(X_i) - y_i, \, y_i - \hat{q}_{\mathrm{high}}(X_i)\right)$$

$$Q_{\mathrm{error\_threshold}} = \mathrm{Percentile}_{95}\left(\{E_i\}\right)$$

* **Output**: Fully trained predictive models and static risk-scaling threshold constants stored per symbol.
* **输出成果**：每只资产专属的、已完成完全训练的预测模型组合以及静态风险缩放阈值常数。

#### Step 6: Heteroskedastic Inference & Risk Width Estimation (`models.py`) / 步骤 6：异方差推理与区间提取

* **What happens**: The system extracts today's latest feature row vector and feeds it into the calibrated model bundle to project forward performance parameters for the next trading session.
* **业务逻辑**：系统提取今日最新的特征行向量，并将其输入到校准后的模型群中，以预测下一交易日的远期表现参数。
* **Calculated Metrics**: Generates point estimates, applies the conformal error shift to establish distribution-free bounds, and calculates the exact conditional uncertainty spread width:
* **计算指标**：生成点预测值，应用符合性误差平移以建立无分布约束的资产风险安全边界，并计算出精确的条件不确定性价差幅宽：

$$\mathrm{Bounds}_{\mathrm{floor}} = \hat{\mathrm{y}}_{\mathrm{low}} - Q_{\mathrm{error\_threshold}}$$

$$\mathrm{Bounds}_{\mathrm{ceiling}} = \hat{\mathrm{y}}_{\mathrm{high}} + Q_{\mathrm{error\_threshold}}$$

$$W_{\mathrm{heteroskedastic}} = \mathrm{Bounds}_{\mathrm{ceiling}} - \mathrm{Bounds}_{\mathrm{floor}}$$

* **Output**: A structured multi-asset lookup dictionary (`predictions_dict`) containing the point returns and adaptive interval risks.
* **输出成果**：一个结构化的多资产查找字典（`predictions_dict`），其中包含各资产的点预测收益率及自适应区间风险幅宽。

#### Step 7: Bayesian Blending & Markowitz Allocation Solver (`portfolio.py`) / 步骤 7：贝叶斯矩阵融合与仓位优化

* **What happens**: The execution bus bridges the gap between machine learning and active portfolio risk constraints. It maps the CQR prediction widths directly into the Black-Litterman matrix equations alongside a robust Ledoit-Wolf shrinkage covariance matrix.
* **业务逻辑**：执行总线在机器学习与主动投资组合风险约束之间架起了桥梁。它将符合性分位数回归（CQR）的预测幅宽直接映射到 Black-Litterman 矩阵方程中，并结合了稳健的 Ledoit-Wolf 收缩协方差矩阵。
* **Calculated Metrics**: Generates an adaptive uncertainty diagonal covariance matrix $\Omega$ where $\Omega_{i,i} = (W_{\mathrm{heteroskedastic}, \, i})^2 \times \tau$. It solves the Bayesian conjugate equations to output adjusted posterior returns, then routes these into an SLSQP non-linear optimizer to solve Markowitz full-investment asset weights under a long-only zero-shorting mandate:
* **计算指标**：生成一个自适应不确定性对角协方差矩阵 $\Omega$，其中对角元素 $\Omega_{i,i} = (W_{\mathrm{heteroskedastic}, \, i})^2 \times \tau$。随后求解贝叶斯共轭方程以输出调整后的后验预期收益率，最后将这些收益率输入 SLSQP 非线性优化器，在仅做多、禁止放空的约束下，求解马科维茨全额投资资产权重：

$$\max_{w} \text{Sharpe Ratio} = \frac{w^T \cdot R_{\mathrm{BL}} - R_f}{\sqrt{w^T \cdot \Sigma_{\mathrm{robust}} \cdot w} + 10^{-8}}$$

$$\text{Subject to: } \sum w_i = 1.0, \quad w_i \in [0.0, 1.0]$$

* **Output**: Finalized multi-asset optimal portfolio target weight vectors.
* **输出成果**：最终确定的多资产最优投资组合目标权重向量。

#### Step 8: Production Log & Payload Serialization (`main-dual-lang.py`) / 步骤 8：日志保存与大模型JSON发布

* **What happens**: The orchestration bus packs all results, timestamps, risk interval analytics, and optimal weights into a unified JSON structure. It serializes this structure to disk (`multi_asset_llm_payload.json`) and prints a clean management dashboard.
* **业务逻辑**：编排总线将所有的计算结果、时间戳、风险区间分析指标以及最优资产权重打包进一个统一的 JSON 结构中。它将该结构序列化落盘至磁盘文件（`multi_asset_llm_payload.json`），并在终端打印出整洁的管理看板。
* **Output**: An LLM-Ready payload snapshot text file ready to be digested by automated upstream models or text alerting scripts.
* **输出成果**：一个“大模型就绪（LLM-Ready）”的决策数据报文快照文件，随时可供上游自动化模型或文本告警脚本解析吞吐。

---

## Part 2: Production Engineer User Guide

## 第二部分：生产环境工程师操作指南

### 1. Environment Preparation / 环境准备

Before executing the pipeline, ensure your runtime environment contains the mandatory dependencies. Install them via terminal:
在执行流水线管道前，确保您的运行环境包含以下强制依赖项。可在终端中通过以下命令安装：

```bash
pip install numpy pandas scikit-learn lightgbm scipy

```

### 2. Operational Execution Step / 生产环境运行步骤

1. Put all 5 files (`config.py`, `data_processor.py`, `models.py`, `portfolio.py`, `main-dual-lang.py`) into the same directory level.
将这 5 个核心代码文件放置在同一个文件目录层级。
2. To run the automated simulation test sandbox, execute the master bus from your terminal:
若要启动自动化仿真测试沙盒，请在您的终端中执行以下主脚本：

```bash
python main-dual-lang.py

```

### 3. File Directory Structural Changes / 磁盘文件目录变更

Upon initialization, the system will automatically create the following production output structure inside your working directory:
初始化启动后，系统将自动在您的当前工作目录下建立以下生产输出目录结构：

```
.
├── config.py
├── data_processor.py
├── models.py
├── portfolio.py
├── main-dual-lang.py
└── production_output/          <-- Automatically Generated / 自动生成
    ├── models/                 <-- Intended for model binary persistence / 用于存储二进制模型
    └── reports/                <-- Analytical outputs / 存放分析输出
        ├── pipeline_execution.log    <-- Running tracking text / 运行跟踪日志
        └── multi_asset_llm_payload.json <-- LLM Ready downstream payload / 决策数据报文

```

---

## Part 3: Result Reading & Dashboard Explanation Guide

## 第三部分：核心决策看板指标与数据结果解读指南

When the pipeline completes execution, it outputs an interactive dual-language terminal dashboard and drops a `multi_asset_llm_payload.json` file. Here is how a production manager or an upstream LLM agent should read and interpret these metrics.

当流水线执行完毕后，它会在控制台输出中英双语核心决策看板，并落盘 `multi_asset_llm_payload.json` 文件。以下是生产主管或上游大模型（LLM Agent）应该如何解读和应用这些核心数据指标的指南。

### 1. Terminal Dashboard Layout / 控制台看板输出示例

```text
=====================================================================================
      📊 MULTI-ASSET CONFORMAL PORTFOLIO OPTIMIZATION DISPATCH TERMINAL
      数据观测日 / Market Snapshot Date: 2026-06-13
=====================================================================================
 1. 符合性分位数回归 (CQR) 异方差估计 / HETEROSKEDASTIC UNCERTAINTY QUANTIFICATION:
    ▶ 资产 [510720_ETF] -> 点预测: +0.00124 | 符合性区间: [-0.0154, +0.0178] | 动态幅宽: 0.03320
    ▶ 资产 [159937_ETF] -> 点预测: -0.00045 | 符合性区间: [-0.0082, +0.0073] | 动态幅宽: 0.01550
    ...
-------------------------------------------------------------------------------------
 2. 贝叶斯后验与最优资产分配权重 / BAYESIAN POSTERIOR & MVO TARGET WEIGHTS:
    ▶ 资产 [510720_ETF  ] -> BL后验预期收益: +0.00085 | ⚙️ 建议目标配资权重:  15.20%
    ▶ 资产 [159937_ETF  ] -> BL后验预期收益: +0.00012 | ⚙️ 建议目标配资权重:  45.80%
=====================================================================================

```

### 2. Metrics Definition & Interpretation / 指标深度解读

#### Metric A: 点预测 (Point Prediction)

* **What it means**: The directional conditional expectation output by the LightGBM regression model for the next day's log return.
* **中文定义**：由 LightGBM 回归模型输出的、针对下一交易日对数收益率的方向性条件期望值。
* **Financial Interpretation**: A value of `+0.00124` implies the model predicts the asset's real price will increase by roughly `0.124%` tomorrow. A negative value represents an expected drop.
* **金融业务解读**：若指标显示为 `+0.00124`，意味着模型预测该资产的实际价格明天将上涨大约 `0.124%`。若为负值则代表预期下跌。

#### Metric B: 符合性区间 & 动态幅宽 (Conformal Interval & Heteroskedastic Width)

* **What it means**: This is the model-agnostic risk safety zone. The interval `[-0.0154, +0.0178]` guarantees a **95% probability** that tomorrow's true return will land within these bounds, based on the historical calibration threshold.
* **中文定义**：这是与模型无关的、极其稳健的风险安全带。基于历史校准阈值，该区间 `[-0.0154, +0.0178]` 能够严格确保明日的真实收益率有 **95% 的概率** 落在该界限内。
* **Risk Interpretation**: **The "Dynamic Width" measures true uncertainty.** If `510720_ETF` has a width of `0.03320` and `159937_ETF` has a width of `0.01550`, it means `510720_ETF` is experiencing a massive surge in conditional risk (heteroskedasticity). The system recognizes it is highly unpredictable right now.
* **风险业务解读**：**“动态幅宽”是衡量条件不确定性的核心标尺。** 如果 `510720_ETF` 的幅宽为 `0.03320`，而 `159937_ETF` 的幅宽仅为 `0.01550`，这表明 `510720_ETF` 当前正经历剧烈的条件风险飙升（异方差性）。系统已敏锐识别出该资产目前的走势极难预测。

#### Metric C: BL后验预期收益 (Black-Litterman Posterior Expected Returns)

* **What it means**: The mathematically adjusted target returns after running the Bayesian conjugate equation. It blends the historical market baseline with the machine learning views, shrinkage-weighted by the CQR width.
* **中文定义**：运行贝叶斯共轭方程后经数学层面动态调整的目标预期收益率。它将历史市场均衡基线与机器学习的主观观点相融合，并以 CQR 风险幅宽作为方差权重进行收缩调整。
* **Interpretation**: If a machine learning point prediction is aggressively high but its width is massive (high uncertainty), the Black-Litterman equation will discount it. It shrinks the returns back toward the market baseline, preventing the trading system from over-allocating capital to noisy AI predictions.
* **业务解读**：如果某个机器学习模型的点预测值极高，但其 CQR 幅宽巨大（表明不确定性极高），Black-Litterman 方程将会对其施加严厉的折扣惩罚。它会将该收益率向市场基线大幅收缩，从而有力阻止交易系统将资金过度分配给高噪声的 AI 预测。

#### Metric D: 建议目标配资权重 (Optimal MVO Target Weights)

* **What it means**: The final asset weight output generated by the non-linear Markowitz optimization program.
* **中文定义**：由非线性马科维茨（Markowitz）均值-方差优化求解器最终生成的资产配置权重。
* **Execution Rule**: These values represent your asset allocation percentages for the next trading day. For example, `45.80%` means if your fund has $1,000,000 in capital, you should allocate exactly $458,000 to hold that specific asset. It strictly enforces a long-only boundary, meaning no asset can have a negative weight (no short selling).
* **执行准则**：这些数值直接代表下一交易日该资产在组合中的资金配比。例如，`45.80%` 意味着如果您的基金账户拥有 $1,000,000 的可投资资金，您应该恰好分配 $458,000 去持有该特定资产。系统严格执行了“仅做多”的边界约束，这意味着任何资产的权重都不可能为负数（即禁止做空）。

### 3. Structural JSON Payload Structure Example / JSON 数据结构示例

This snippet shows the raw parameters exported inside `multi_asset_llm_payload.json`:
该代码片段展示了导出至 `multi_asset_llm_payload.json` 中的原始参数结构：

```json
{
    "metadata": {
        "execution_timestamp": "2026-06-13 13:27:05",
        "production_data_date": "2026-06-13",
        "tracked_assets": ["510720_ETF", "159937_ETF", "399006_ETF", "600115_SH", "601088_SH", "601229_SH"]
    },
    "cqr_uncertainty_metrics": {
        "510720_ETF": {
            "point_prediction": 0.00124,
            "conformal_floor": -0.0154,
            "conformal_ceiling": 0.0178,
            "heteroskedastic_width": 0.0332
        }
    },
    "bayesian_portfolio_allocator": {
        "bl_posterior_expected_returns": [0.00085, 0.00012, -0.0004, 0.0021, 0.0003, 0.0009],
        "optimal_mvo_weights": {
            "510720_ETF": 0.152,
            "159937_ETF": 0.458,
            "399006_ETF": 0.000,
            "600115_SH": 0.210,
            "601088_SH": 0.180,
            "601229_SH": 0.000
        }
    }
}

```

This payload is optimized for automated parsing. If an asset's weight is `0.000`, the system mandates completely liquidating or avoiding that specific market tracking vector for the session.

该核心数据报文已为自动化解析完成全面优化。如果某一资产的最终配资权重降至 `0.000`，交易系统将强制要求在当前交易日内对该特定市场目标执行全额平仓、或对其保持完全空仓隔离。
