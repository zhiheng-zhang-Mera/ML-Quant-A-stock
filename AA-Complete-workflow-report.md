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
* **Calculated Metrics**: Sets up the target configuration variables: significance level $\alpha=0.05$ (95% target confidence), Black-Litterman scale factor $\tau=0.02$, rolling window size $= 120$ days, and the target asset symbols pool.
* **Output**: A frozen global configuration instance imported by all other processing components.

#### Step 2: Ingestion & Economic Neutralization (`data_processor.py`) / 步骤 2：数据加工与通胀平减

* **What happens**: `main-dual-lang.py` sends raw OHLCV market dictionary data straight into the `DataProcessor`. The data processor first strips out macroeconomic inflation noise by discounting nominal historical values against a compounding real purchasing power deflator.
* **Calculated Metrics**:

$$\text{Deflator} = (1.0 + \text{ANNUAL\_INFLATION\_RATE})^{\frac{\text{Max Date} - \text{Current Date}}{365.25}} \text{}$$


$$\text{Real Price} = \text{Nominal Price} \times \text{Deflator} \text{}$$


* **Output**: A clean, aligned, multi-indexed pandas DataFrame panel structured strictly by row coordinates `[Date, Symbol]`.

#### Step 3: Predictive Feature Engineering (`data_processor.py`) / 步骤 3：白盒特征矩阵构建

* **What happens**: The processor extracts five transparent, human-interpretable quantitative technical indicators across the real price arrays.
* **Calculated Metrics**:
* 1-day, 5-day, and 20-day log price momentum variables (`Mom_1D`, `Mom_5D`, `Mom_20D`).
* Intraday Garman-Klass Volatility signatures (`GK_Vol`).
* Dynamic liquidity volume shocks (`Turnover_Shock`) standardizing turnover volumes against a 5-day rolling average window.
* Next-day shifted log real forward return arrays as target labels (`target_return`).


* **Output**: Two perfectly aligned matrix spaces: Feature Space Matrix `X_panel` and Target Label Vector `y_panel`.

#### Step 4: Strict Temporal Embargo Filtering (`main-dual-lang.py`) / 步骤 4：严格时序禁运隔离

* **What happens**: To guarantee that no information overlaps or leaks future details back into our historical data, the master bus splits the training and production dates apart. It rewinds the training timeline backward from the modern edge by a strict buffer zone.
* **Calculated Metrics**:

$$\text{Train End Index} = \text{Total Dates Count} - 1 - \text{EMBARGO\_PERIOD} \text{}$$


* **Output**: Completely isolated historical training datasets (`X_train`, `y_train`) and a singular cross-sectional feature array captured on the live production snapshot observation date (`current_X_dict`).

#### Step 5: Adaptive Model Fitting & Statistical Calibration (`models.py`) / 步骤 5：自适应拟合与分位数校准

* **What happens**: The historical training subsets are sent into the `StatisticalAdaptiveEngine`. For each symbol in our asset pool, three distinct LightGBM decision tree regressions are fitted simultaneously.
* **Calculated Metrics**: Fits the conditional median mean, a lower tail pinball-loss model ($\alpha_{\text{low}} = 0.025$), and an upper tail pinball-loss model ($\alpha_{\text{high}} = 0.975$). It captures uncalibrated errors on the training set and computes an exact empirical margin penalty threshold:

$$E_i = \max\left(\hat{q}_{\text{low}}(X_i) - y_i, \, y_i - \hat{q}_{\text{high}}(X_i)\right) \text{}$$


$$\text{Q Error Threshold} = \text{95th Percentile of } \{E_i\} \text{}$$


* **Output**: Fully trained predictive models and static risk-scaling threshold constants stored per symbol.

#### Step 6: Heteroskedastic Inference & Risk Width Estimation (`models.py`) / 步骤 6：异方差推理与区间提取

* **What happens**: The system extracts today's latest feature row vector and feeds it into the calibrated model bundle to project forward performance parameters for the next trading session.
* **Calculated Metrics**: Generates point estimates, applies the conformal error shift to establish distribution-free bounds, and calculates the exact conditional uncertainty spread width:

$$\text{Conformal Floor} = \text{Raw Prediction}_{\text{low}} - \text{Q Error Threshold} \text{}$$


$$\text{Conformal Ceiling} = \text{Raw Prediction}_{\text{high}} + \text{Q Error Threshold} \text{}$$


$$\text{Heteroskedastic Width} = \text{Conformal Ceiling} - \text{Conformal Floor} \text{}$$


* **Output**: A structured multi-asset lookup dictionary (`predictions_dict`) containing the point returns and adaptive interval risks.

#### Step 7: Bayesian Blending & Markowitz Allocation Solver (`portfolio.py`) / 步骤 7：贝叶斯矩阵融合与仓位优化

* **What happens**: The execution bus bridges the gap between machine learning and active portfolio risk constraints. It maps the CQR prediction widths directly into the Black-Litterman matrix equations alongside a robust Ledoit-Wolf shrinkage covariance matrix.
* **Calculated Metrics**: Generates an adaptive uncertainty diagonal covariance matrix $\Omega$ where $\Omega_{i,i} = (\text{CQR Width}_i)^2 \times \tau$. It solves the Bayesian conjugate equations to output adjusted posterior returns, then routes these into an SLSQP non-linear optimizer to solve Markowitz full-investment asset weights under a long-only zero-shorting mandate:

$$\max_{w} \text{Sharpe Ratio} = \frac{w^T \cdot \text{BL Returns} - R_f}{\sqrt{w^T \cdot \Sigma_{\text{robust}} \cdot w} + 10^{-8}} \text{}$$


$$\text{Subject to: } \sum w_i = 1.0, \quad w_i \in [0.0, 1.0] \text{}$$


* **Output**: Finalized multi-asset optimal portfolio target weight vectors.

#### Step 8: Production Log & Payload Serialization (`main-dual-lang.py`) / 步骤 8：日志保存与大模型JSON发布

* **What happens**: The orchestration bus packs all results, timestamps, risk interval analytics, and optimal weights into a unified JSON structure. It serializes this structure to disk (`multi_asset_llm_payload.json`) and prints a clean management dashboard.
* **Output**: An LLM-Ready payload snapshot text file ready to be digested by automated upstream models or text alerting scripts.

---

## Part 2: Production Engineer User Guide

## 第二部分：生产环境工程师操作指南

### 1. Environment Preparation / 环境准备

Before executing the pipeline, ensure your runtime environment contains the mandatory dependencies. Install them via terminal:
在执行管道前，确保您的运行环境包含以下强制依赖项。在终端中安装：

```bash
pip install numpy pandas scikit-learn lightgbm scipy

```

### 2. Operational Execution Step / 生产环境运行步骤

1. Put all 5 files (`config.py`, `data_processor.py`, `models.py`, `portfolio.py`, `main-dual-lang.py`) into the same directory level.
将这 5 个文件放至同一个文件目录层级。
2. To run the automated simulation test sandbox, execute the master bus from your terminal:
若要启动自动化仿真测试沙盒，请在终端执行主脚本：
```bash
python main-dual-lang.py

```



### 3. File Directory Structural Changes / 磁盘文件目录变更

Upon initialization, the system will automatically create the following production output structure inside your working directory:
初始化启动后，系统将自动在您的工作目录下建立以下生产输出结构：

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

当流水线执行完毕后，它会在控制台输出中英双语核心决策看板，并落盘 `multi_asset_llm_payload.json` 文件。以下是生产主管或上游大模型（LLM）应该如何解读这些核心数据指标的指南。

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
* **Financial Interpretation**: A value of `+0.00124` implies the model predicts the asset's real price will increase by roughly `0.124%` tomorrow. A negative value represents an expected drop.

#### Metric B: 符合性区间 & 动态幅宽 (Conformal Interval & Heteroskedastic Width)

* **What it means**: This is the model-agnostic risk safety zone. The interval `[-0.0154, +0.0178]` guarantees a **95% probability** that tomorrow's true return will land within these bounds, based on the historical calibration threshold.
* **Risk Interpretation**: **The "Dynamic Width" measures true uncertainty.** * If `510720_ETF` has a width of `0.03320` and `159937_ETF` has a width of `0.01550`, it means `510720_ETF` is experiencing a massive surge in conditional risk (heteroskedasticity). The system recognizes it is highly unpredictable right now.

#### Metric C: BL后验预期收益 (Black-Litterman Posterior Expected Returns)

* **What it means**: The mathematically adjusted target returns after running the Bayesian conjugate equation. It blends the historical market baseline with the machine learning views, shrinkage-weighted by the CQR width.
* **Interpretation**: If a machine learning point prediction is aggressively high but its width is massive (high uncertainty), the Black-Litterman equation will discount it. It shrinks the returns back toward the market baseline, preventing the trading system from over-allocating capital to noisy AI predictions.

#### Metric D: 建议目标配资权重 (Optimal MVO Target Weights)

* **What it means**: The final asset weight output generated by the non-linear Markowitz optimization program.
* **Execution Rule**: These values represent your asset allocation percentages for the next trading day. For example, `45.80%` means if your fund has $1,000,000 in capital, you should allocate exactly $458,000 to hold that specific asset. It strictly enforces a long-only boundary, meaning no asset can have a negative weight (no short selling).

### 3. Structural JSON Payload Structure Example / JSON 数据结构结构示例

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
