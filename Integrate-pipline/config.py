# config.py
import os
from dataclasses import dataclass, field
from typing import List

@dataclass(frozen=True)
class PipelineConfig:
    # Asset Pool
    SYMBOLS: List[str] = field(default_factory=lambda: [
        "510720_ETF", "159937_ETF", "399006_ETF", "600115_SH", "601088_SH", "601229_SH"
    ])
    
    # Statistical Engine Parameters
    ALPHA: float = 0.05                 # 95% Target Confidence Level
    TAU: float = 0.02                   # Black-Litterman scale factor
    ROLLING_WINDOW: int = 120           # Lookback window for technical signals
    EMBARGO_PERIOD: int = 5             # Buffer period preventing forward look-ahead leakages
    MIN_REQUIRED_SAMPLES: int = 60      # Floor for execution data pooling
    ANNUAL_INFLATION_RATE: float = 0.03 # Macro deflator compression baseline
    
    # Advanced Finance/ML Parameters
    LAMBDA_KERNEL: float = 0.25         # Pricing Kernel contraction operator shrinkage parameter
    RHO_COHORT: float = 0.15            # Asset cluster group correlation factor for block-diagonal Omega
    N_COHORTS: int = 2                  # Number of cross-sectional stratification clusters
    
    # =========================================================================
    # 🎛️ 【核心升级】：大模型多路由调度总线开关 (LLM Gateway Router Switch)
    # 选项: "DEEPSEEK" (云端高精度) 或 "OLLAMA" (本地零开销脱机)
    LLM_PROVIDER: str = "OLLAMA"        
    
    # Option A: Cloud DeepSeek Coordinates
    DEEPSEEK_API_KEY_ENV: str = "DEEPSEEK_API_KEY"
    DEEPSEEK_MODEL_NAME: str = "deepseek-chat"
    DEEPSEEK_API_URL: str = "https://api.deepseek.com/v1"
    
    # Option B: Local Ollama Coordinates (OpenAI-Compatible Engine Endpoints)
    OLLAMA_MODEL_NAME: str = "qwen2.5:7b"  # 可自由替换为 "llama3" 或 "deepseek-r1:7b"
    OLLAMA_API_URL: str = "http://localhost:11434/v1"
    
    # Shared Inference Hyperparameters
    LLM_TEMPERATURE: float = 0.1
    LLM_TIMEOUT: float = 45.0           # 🚨 本地推理容易遭遇硬件Stall，通胀超时拉回防线
    # =========================================================================
    
    # IO Infrastructure Coordinates
    BASE_OUTPUT_FOLDER: str = "production_output"
    MODEL_DIR: str = os.path.join("production_output", "models")
    REPORT_DIR: str = os.path.join("production_output", "reports")