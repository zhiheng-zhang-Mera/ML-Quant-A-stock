## Part 1: Chinese/English Dual-Language Process Report

This script acts as the **Master Orchestration Pipeline Bus (核心调度总线)**. While `config.py` provides the blueprint settings, `main-dual-lang.py` executes the entire pipeline lifecycle—from raw financial data parsing, training safety handling, risk interval calculation, and portfolio weight mathematical optimization, to outputting an LLM-ready structured payload.

以下是系统核心运作流程的详细拆解：
Here is the step-by-step systemic workflow breakdown:

### Step 1: Logging Infrastructure Activation / 激活生产日志系统

* **What this step does / 此步骤的作用**: Initializes a localized logger instance that simultaneously outputs tracking info to the interactive CLI terminal and dumps structural text into a permanent persistence log file (`pipeline_execution.log`).
* **Metrics & Calculations / 使用指标与计算**: No financial calculations here. It explicitly sets the logging engine to filter for `logging.INFO` levels and formats entries via standard local timestamp formats.
* **What data is used for / 数据用途**: Acts as the system auditor. If the optimization framework fails midway, engineers can parse this text trail to locate exactly which sub-module broke.
* **What it returns / 返回值**: A configured standard Python `logging.Logger` instance.

### Step 2: Cross-Sectional Data Feature Matrix Engineering / 多资产横截面特征加工

* **What this step does / 此步骤的作用**: Consumes the dictionary of raw OHLCV market feeds and routes them into the `DataProcessor`. It structures multi-asset price and volume arrays into a continuous, indexed panel matrix.
* **Metrics & Calculations / 使用指标与计算**: Calculates independent feature columns ($X$) and target log return columns ($y$) mapped back to the assets and dates defined in the configuration.
* **What data is used for / 数据用途**: Converts unstructured, chaotic raw streaming market tickers into an ordered input state matrix required by advanced statistical estimators.
* **What it returns / 返回值**: Two multi-indexed pandas DataFrames representing the Feature Space Matrix (`X_panel`) and the Ground Truth Label Vector (`y_panel`).

### Step 3: Strict Temporal Embargo Implementation / 严格时序禁运隔离处理

* **What this step does / 此步骤的作用**: Identifies the terminal edge of available time-series entries and actively rewinds the timeline back by the set embargo distance. This prevents information overlapping or future data leaking backward into training subsets.
* **Metrics & Calculations / 使用指标与计算**:

$$\text{Train_End_Index} = \text{Total-Dates-Count} - 1 - \text{EMBARGO\_PERIOD}$$



It explicitly asserts that the sliced subset satisfies:

$$\text{Train End Index} \ge \text{MIN\_REQUIRED\_SAMPLES}$$


* **What data is used for / 数据用途**: Enforces mathematical safety boundaries. It tells the execution bus exactly which cross-sections are safe to train on and isolates the singular date designated for live production decision-making.
* **What it returns / 返回值**: Two sliced date indices arrays representing historical training coordinates (`training_dates`) and the target production observation date (`current_production_date`).

### Step 4: Machine Learning CQR Engine Calibration / 激活符合性分位数回归引擎

* **What this step does / 此步骤的作用**: Slices the historical panel arrays based on the embargoed date vector and feeds them directly into the `StatisticalAdaptiveEngine`.
* **Metrics & Calculations / 使用指标与计算**: Calibrates the regression engine under the set significance level $\alpha=0.05$. It builds non-parametric distribution metrics around errors.
* **What data is used for / 数据用途**: Trains the ML model to recognize feature patterns and establishes global non-parametric calibration thresholds to map predictive uncertainty intervals.
* **What it returns / 返回值**: Updates internal weights and calibration matrices inside the initialized `StatisticalAdaptiveEngine` instance.

### Step 5: Heteroskedastic Predictive Interval Extraction / 异方差预测不确定性极值推演

* **What this step does / 此步骤的作用**: Extracts the vector of the newest market features for all tracked assets on the live production date. It executes conditional predictive inferences for the forward trading day.
* **Metrics & Calculations / 使用指标与计算**: Evaluates the model to compute point asset return forecasts alongside exact conformal ceilings and floors. It explicitly gauges the dynamic variance spread:

$$\text{Heteroskedastic Width} = \text{Conformal Ceiling} - \text{Conformal Floor}$$


* **What data is used for / 数据用途**: Feeds predictive asset expected returns alongside objective, model-agnostic measurement widths into the downstream portfolio optimizer.
* **What it returns / 返回值**: A structured data lookup dictionary (`predictions_dict`) mapping tickers to their respective localized return bounds and point vector arrays.

### Step 6: Bayesian Black-Litterman Matrix Optimization / 贝叶斯网桥融合与马科维茨优化

* **What this step does / 此步骤的作用**: Computes a dynamic historical covariance structure using filtered historical asset log returns. It maps the predictive ranges derived from the CQR ML engine onto the market baseline model.
* **Metrics & Calculations / 使用指标与计算**: Runs asset covariance estimators and merges raw statistical priors with ML views via the Black-Litterman matrix equation framework. It routes the resulting adjusted posterior projections through a Markowitz Mean-Variance Optimization (MVO) controller.
* **What data is used for / 数据用途**: Formulates optimal target capital allocations across assets while considering asset risk parameters and covariance dependencies.
* **What it returns / 返回值**: A mathematical tuple consisting of BL Posterior Expected Returns, Optimal Target Allocation Portfolio Weights, and raw asset variance estimates.

### Step 7: LLM-Ready Output Payload Packaging / 组装大模型就绪决策快照

* **What this step does / 此步骤的作用**: Consolidates timestamps, metadata, statistical safety measurements, and resulting allocation vectors into a clean JSON structure, saving it directly to disk (`multi_asset_llm_payload.json`).
* **Metrics & Calculations / 使用指标与计算**: Transforms complex technical tensor and pandas arrays into raw float and text primitive parameters.
* **What data is used for / 数据用途**: Serves as an interface payload. It provides upstream Large Language Models (LLMs) or automated alert agents with structured JSON inputs to generate text-based summaries.
* **What it returns / 返回值**: A nested structural Python data dictionary (`llm_payload`).
