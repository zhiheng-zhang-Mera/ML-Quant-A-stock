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
    
    # !!-- 切换模型修改 LLM_API_URL 和 LLM_MODEL_NAME 即可 --!!
    # =========================================================================
    #  【抽象升级】LLM 文本分析层灵活配置中心 (解耦硬编码，完美适配任何主流大模型服务商)
    # =========================================================================
    # 选项 A (默认DeepSeek): "https://api.deepseek.com/v1" | "deepseek-chat"
    # 选项 B (全球主流OpenAI): "https://api.openai.com/v1" | "gpt-4o-mini"
    # 选项 C (本地离线私有云): "http://localhost:11434/v1" | "qwen2.5:7b-instruct" (Ollama / vLLM)
    LLM_API_URL: str = "https://api.deepseek.com/v1"        # 统一 API 端点基础路径
    LLM_MODEL_NAME: str = "deepseek-chat"                  # 目标大模型名称标识符
    LLM_TEMPERATURE: float = 0.1                            # 采样温度 (极低温度保障结构化 JSON 高确定性输出)
    LLM_TIMEOUT: float = 15.0                              # 请求超时熔断阈值(秒)，防范实盘盘后网络挂起
    LLM_API_KEY_ENV: str = "QUANT_LLM_API_KEY"             # 系统环境变量名，代码库中不出现明文密钥
    
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