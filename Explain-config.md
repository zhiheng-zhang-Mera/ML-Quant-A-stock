以下是针对 `config.py` 代码流程与设计架构的中英文双语详细解析。本说明专为初学者或入门级计算机科学（CS）人员设计，将复杂的量化金融概念拆解为易于理解的步骤。

Here is the detailed Chinese/English dual-language explanation of the `config.py` code workflow and architecture. This breakdown is designed specifically for entry-level CS personnel, translating complex quantitative finance concepts into digestible steps.

---

## 核心定位 / Core Role

在深入具体步骤之前，我们需要明确：**这是一个配置文件（Configuration File）。** 它本身不处理海量数据、不训练机器学习模型，也不执行实际交易。相反，它扮演着整个量化交易与资产配置系统的“控制室”或“蓝图”角色。它定义了全局常量、超参数和文件路径，供系统中的其他脚本读取和遵循。

Before diving into the steps, we must clarify: **this is a Configuration File.** It does not process massive data, train machine learning models, or execute live trades itself. Instead, it acts as the **"Control Room" or "Blueprint"** for the entire quantitative trading and asset allocation pipeline. It defines global constants, hyperparameters, and file paths that all other scripts will read and follow.

---

## 预备步骤：构建配置容器 / Pre-requisites: Structuring the Configuration Container

### 1. 第一步是什么？ / What is the first step?

代码导入了必要的 Python 标准库，并使用 `@dataclass(frozen=True)` 装饰器初始化了一个名为 `PipelineConfig` 的数据类。
The code imports the necessary standard Python libraries and initializes a data class named `PipelineConfig` using the `@dataclass(frozen=True)` decorator.

### 2. 这个步骤是做什么的？ / What does this step do?

* **导入模块 (`os`, `dataclasses`, `typing`):** 引入用于处理系统文件路径的工具（`os`）、构建清晰数据结构的工具（`dataclass`）以及提高代码可读性的类型提示工具（`typing`）。
**Imports (`os`, `dataclasses`, `typing`):** It brings in tools for handling file paths (`os`), structuring clean data containers (`dataclass`), and code readability helpers (`typing`).
* **`@dataclass(frozen=True)`:** 这是一个非常关键的 Python 特性。数据类（dataclass）可以自动在后台生成存储数据的底层代码。而 `frozen=True`（冻结）使这个配置对象变成不可变（Immutable）的。这意味着一旦配置加载完成，任何其他代码都无法在运行中意外修改这些参数，从而避免了实盘交易或回测中的潜在 Bug。
**`@dataclass(frozen=True)`:** This is a crucial Python feature. A dataclass automatically generates background boilerplate code to store data cleanly. The `frozen=True` part makes the entire configuration object **immutable**. This means once the pipeline starts, no other piece of code can accidentally alter these settings, preventing catastrophic bugs during live trading or backtesting.

---

## 步骤 1：定义资产池（目标标的） / Step 1: Defining the Asset Pool (The Targets)

### 1. 对应代码是什么？ / What is this step?

定义 `SYMBOLS` 变量。
Defining the `SYMBOLS` variable.

### 2. 这个步骤是做什么的？ / What does this step do?

它精确指定了该量化系统被允许分析和交易的金融资产范围。
It specifies exactly which financial assets the quantitative system is allowed to analyze and trade.

### 3. 使用了什么指标？ / What metrics are used?

它列出了一个代表多资产横截面标的池的字符串列表：
It lists tickers/identifiers representing a multi-asset cross-sectional pool:

* **`_ETF`（交易所交易基金）：** 跟踪特定指数或板块的基金（例如 `510720_ETF`）。
**`_ETF` (Exchange Traded Funds):** Funds tracking broad markets or specific sectors (e.g., `510720_ETF`).
* **`_SH`（上海证券交易所）：** 在上交所上市的单只股票（例如 `600115_SH`）。
**`_SH` (Shanghai Stock Exchange):** Individual equities listed on the Shanghai stock market (e.g., `600115_SH`).

### 4. 计算了什么？计算出的数据用于什么？ / What are calculated and what are they used for?

* **计算内容：** 此处不进行任何数学计算，仅执行静态列表的初始化。
**Calculated:** Nothing is mathematically calculated here; it is a static list initialization.
* **数据用途：** 该列表会告诉“数据获取模块”应该下载哪些资产的历史价格，同时告诉“资产配置模块”应该在哪些资产之间分配资金。
**What it is used for:** This list tells the data-fetching module exactly which historical prices to download and tells the portfolio optimization module which assets to distribute capital across.

### 5. 这个步骤返回什么？ / What does this step return?

它返回一个包含 6 个特定资产标识符的 Python 字符串列表（`List[str]`）。
It returns a Python `List[str]` (a list of text strings) containing 6 specific asset identifiers.

---

## 步骤 2：设定核心数学与金融超参数 / Step 2: Setting Core Mathematical & Financial Hyperparameters

### 1. 对应代码是什么？ / What is this step?

定义风险、优化和经济基准参数：`ALPHA`、`TAU`、`RISK_FREE_RATE` 和 `ANNUAL_INFLATION_RATE`。
Defining risk, optimization, and economic benchmarks: `ALPHA`, `TAU`, `RISK_FREE_RATE`, and `ANNUAL_INFLATION_RATE`.

### 2. 这个步骤是做什么的？ / What does this step do?

它为风险管理模型和机器学习预测模型建立了数学边界和金融基线。
It establishes the mathematical boundaries and financial baselines for risk management and predictive machine learning models.

### 3. 使用并计算了什么指标？ / What metrics are used and calculated?

* **`ALPHA = 0.05`:** 用于 **符合性分位数回归（CQR, Conformal Quantile Regression）**。
**`ALPHA = 0.05`:** Used in **Conformal Quantile Regression (CQR)**.
* *实际含义与计算：* 它定义了统计学上的显著性水平。目标置信度计算为 $1 - \alpha$（即 $1 - 0.05 = 0.95$ 或 **95%**）。机器学习模型将使用它来生成资产回报预测的“不确定性区间”，确保真实价格在 95% 的概率下落在该预测区间内。
*What it means & Calculation:* It defines the statistical significance level. The target confidence level is calculated as $1 - \alpha$ (which equals $1 - 0.05 = 0.95$ or **95%**). The machine learning model will use this to generate uncertainty bands around asset return predictions, ensuring the true price falls within the predicted range 95% of the time.


* **`TAU = 0.02`:** **Black-Litterman (BL) 资产配置模型**中机器学习观点的权重标度。
**`TAU = 0.02`:** The weight scale for machine learning views in the **Black-Litterman (BL) portfolio optimization model**.
* *实际含义：* 在量化金融中，BL 模型将“市场均衡假设”与机器学习的“主观预测观点”相结合。`TAU` 代表你对机器学习模型的信任程度。较小的 `TAU`（0.02）意味着系统会审慎、保守地对待 AI 的预测，防止系统因激进的 AI 预测而过度集中投资。
*What it means:* In quantitative finance, the BL model blends standard market equilibrium assumptions with unique "views" (predictions) from a machine learning model. `TAU` represents how much you trust your machine learning model relative to the market. A smaller `TAU` (0.02) means you treat the machine learning predictions with cautious skepticism, preventing the system from over-investing based on aggressive AI forecasts.


* **`RISK_FREE_RATE = 0.0001`（日频 0.01%）:** 盘后无风险日收益率基准。
**`RISK_FREE_RATE = 0.0001` (0.01% daily):** The post-close risk-free return baseline.
* *实际含义与用途：* 这是指你在不承担任何风险的情况下（如持有国债）每天能获得的收益率。它被用来计算**夏普比率（Sharpe Ratio）**：
$$\text{Sharpe} = \frac{\text{资产收益率} - \text{无风险利率}}{\text{波动率}}$$


 如果某项资产的预期收益连这个日频基线都无法战胜，系统就会拒绝该资产。
*What it is used for:* The daily return you could get by doing absolutely nothing risky (like holding government bonds). It is used to calculate the **Sharpe Ratio**: 
$$\text{Sharpe} = \frac{\text{Asset Return} - \text{Risk Free Rate}}{\text{Volatility}}$$


 If an asset cannot beat this daily baseline, the system will reject it.


* **`ANNUAL_INFLATION_RATE = 0.03`（年频 3%）:** 年度通胀率平减基准。
**`ANNUAL_INFLATION_RATE = 0.03` (3% yearly):** The economic inflation baseline.
* *用途：* 用于计算“真实收益率”与“名义收益率”。系统会利用它对原始利润进行扣除平减，以观察交易策略是否真正跑赢了货币购买力的贬值。
*What it is used for:* Used to calculate "real" returns versus "nominal" returns, deflating raw profits to see if the trading strategy actually beats purchasing power decay.



### 4. 这个步骤返回什么？ / What does this step return?

返回 4 个浮点数（`float`），分别映射到各自的超参数变量中。
Four floating-point decimal numbers (`float`) mapped to their respective hyperparameter variables.

---

## 步骤 3：设定时序特征与滚动数据窗口 / Step 3: Setting Time-Series & Data-Validation Windows

### 1. 对应代码是什么？ / What is this step?

定义时间序列的数据边界参数：`ROLLING_WINDOW_SIZE`、`MIN_REQUIRED_SAMPLES` 和 `EMBARGO_PERIOD`。
Defining time-series data boundaries: `ROLLING_WINDOW_SIZE`, `MIN_REQUIRED_SAMPLES`, and `EMBARGO_PERIOD`.

### 2. 这个步骤是做什么的？ / What does this step do?

在处理金融时间序列数据时，“前瞻偏差”（不小心偷看了未来的数据）和“数据重叠”会导致模型失效。本步骤设定了严格的时间边界，以防止出现**数据泄露（Data Leakage）**。
When dealing with financial time-series data, look-ahead bias (accidentally peeking into the future) and data overlap can ruin models. This step sets strict temporal boundaries to prevent **data leakage**.

### 3. 使用了什么指标？ / What metrics are used?

时间跨度度量指标：**交易日天数（Trading Days）**。
Time durations measured in **trading days**.

### 4. 计算了什么？计算出的数据用于什么？ / What are calculated and what are they used for?

* **`ROLLING_WINDOW_SIZE = 120`:** 滚动窗口大小。系统将观察一个移动的 120 交易日窗口（约 6 个月），用于计算滚动统计量（如移动平均线或滚动波动率）。
**`ROLLING_WINDOW_SIZE = 120`:** The system will look at a moving window of 120 trading days (roughly 6 months) to calculate rolling statistics like moving averages or volatility.
* **`MIN_REQUIRED_SAMPLES = 150`:** 最小样本量。系统会检查某只资产是否至少拥有 150 天的历史数据。如果某只新股只有 100 天的数据，它将被过滤掉，因为计算协方差矩阵等数学公式需要最小的样本量来保证统计学有效性。
**`MIN_REQUIRED_SAMPLES = 150`:** The system checks if an asset has at least 150 days of historical data. If a stock is newly listed and has only 100 days of data, it is mathematically filtered out because calculations like covariance matrices require a minimum sample size to be statistically valid.
* **`EMBARGO_PERIOD = 20`:** 严格的时序禁运期跨度。
**`EMBARGO_PERIOD = 20`:** A 20-day strict time-series embargo.
* *用途：* 在金融机器学习中，特征往往存在重叠（例如，今天的“20日动量”特征与昨天的特征共享了 19 天的历史数据）。当把数据切分为“训练集”和“测试集”时，不能让测试集紧跟在训练集后面，否则模型会通过重叠的数据“作弊”。系统强制在训练和测试数据之间留出 20 天的“空白期”（禁运期），确保测试是在真正未见过的未来数据上进行的。
*What it is used for:* In financial ML, features often overlap (e.g., a 20-day momentum feature today shares 19 days of data with yesterday's feature). When splitting data into "training" and "testing" sets, you cannot put testing data right after training data, or the model will "cheat" by remembering the overlap. The system forces a 20-day "blank space" (embargo) between data sets to ensure testing is done on genuinely unseen future data.



### 5. 这个步骤返回什么？ / What does this step return?

返回 3 个整数（`int`），定义了数据处理循环中严格的天数限制。
Three integer values (`int`) defining strict day-counts for data processing loops.

---

## 步骤 4：指定可解释的底层特征 / Step 4: Specifying Explanatory Features

### 1. 对应代码是什么？ / What is this step?

定义 `FEATURE_COLS` 列表。
Defining the `FEATURE_COLS` list.

### 2. 这个步骤是做什么的？ / What does this step do?

它列出了机器学习模型在预测未来资产走势时需要观察的、具有明确人类可解释性的数据特征（自变量）。
It lists the explicit, human-interpretable data features (independent variables) that the machine learning model will look at to predict future asset movements.

### 3. 使用了什么指标？ / What metrics are used?

量化金融技术指标：
Quantitative finance technical indicators:

* **`Mom_1D`, `Mom_5D`, `Mom_20D`:** 动量指标，分别测量资产在过去 1 天、5 天和 20 天内的价格回报率。
**`Mom_1D`, `Mom_5D`, `Mom_20D`:** Momentum indicators measuring the asset's price return over the past 1 day, 5 days, and 20 days respectively.
* **`GK_Vol`:** Garman-Klass 波动率。一种专业的波动率计算指标，它利用历史的开盘价、最高价、最低价和收盘价（而不仅仅是收盘价）来更精准地计算市场波动。
**`GK_Vol`:** Garman-Klass Volatility, a specialized metric that calculates market volatility using historical Open, High, Low, and Close prices rather than just the Closing price.
* **`Turnover_Shock`:** 换手率冲击。一种基于成交量的指标，用于衡量交易活跃度（流动性冲击）的突增或骤减。
**`Turnover_Shock`:** A volume-based metric measuring sudden surges or drops in trading activity (liquidity shocks).

### 4. 计算了什么？计算出的数据用于什么？ / What are calculated and what are they used for?

* **计算内容：** 此处仅做声明，不进行计算。
**Calculated:** None here; this is a declaration.
* **数据用途：** “特征工程脚本”会读取这个列表，查看原始的价格和成交量数据，计算出这 5 个特定的数学指标，然后将它们作为矩阵输入给机器学习模型。
**What it is used for:** The feature engineering script reads this list, looks at raw price and volume data, calculates these 5 specific mathematical metrics, and feeds them into the ML model matrix.

### 5. 这个步骤返回什么？ / What does this step return?

返回一个包含 5 个不同特征名称的字符串列表（`List[str]`）。
A `List[str]` containing 5 distinct feature names.

---

## 步骤 5：构建生产文件输出基础设施（初始化路径） / Step 5: Setting up Output Infrastructures (Directories)

### 1. 对应代码是什么？ / What is this step?

定义 `BASE_OUTPUT_FOLDER` 并执行 `__post_init__` 函数。
Defining the `BASE_OUTPUT_FOLDER` and executing the `__post_init__` function.

### 2. 这个步骤是做什么的？ / What does this step do?

这是系统的自动化运行准备。当 Python 数据类（dataclass）完成所有变量的加载后，它会自动触发一个名为 `__post_init__` 的特殊函数。这一步动态地构建了硬盘上的文件路径，用于存放训练好的 AI 模型和最终的策略表现报告。
This is the operational setup. When a Python dataclass finishes loading its variables, it automatically runs a special function called `__post_init__`. This step dynamically constructs the file pathways where the final trained AI models and performance reports will be saved on the hard drive.

### 3. 使用了什么指标？ / What metrics are used?

文件系统路径字符串（`string`）。
File system paths (`string`).

### 4. 计算了什么？计算出的数据用于什么？ / What are calculated and what are they used for?

* **计算内容：** 系统调用 `os.path.join` 动态拼接文本路径。
**Calculated:** The system uses `os.path.join` to dynamically combine text paths.
* `MODEL_DIR` 计算并赋值为 `./production_output/models`
`MODEL_DIR` becomes `./production_output/models`
* `REPORT_DIR` 计算并赋值为 `./production_output/reports`
`REPORT_DIR` becomes `./production_output/reports`


* **数据用途：** 它强制让数据类暂时绕过其“只读（frozen）”状态（通过 `object.__setattr__`），锁定这些文件夹路径。当整个量化流水线运行时，其他模块会读取这两个字符串，从而精确地知道要把训练好的模型二进制文件（如 `.pkl`）和表现分析报告（如 PDF/HTML）保存在电脑的什么位置。
**What it is used for:** It forces the class to override its read-only status (`object.__setattr__`) just long enough to lock in these folder paths. When the actual trading pipeline runs, it will look at these strings to know exactly where to write the trained AI model binary files (`.pkl`) and where to drop the performance reports.
* **副作用（控制台输出）：** 它还会向终端控制台打印中英文双语的状态日志，提示工程师环境初始化已成功。
**Side-effect:** It prints status logs to the console terminal in both English and Simplified Chinese to notify the engineer that the environment initialization is successful.

### 5. 这个步骤返回什么？ / What does this step return?

它完成了 `PipelineConfig` 对象的实例化。此时，该对象已经装满了被保护、不可篡改的配置属性，随时可以被系统中的其他核心模块（如数据下载、模型训练、回测模块）导入和调用。
It completes the instantiation of the `PipelineConfig` object, fully packed with frozen configuration attributes ready to be imported and deployed by other core modules across the system.
