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
    
    # Decoupled LLM Gateway configurations
    LLM_API_KEY_ENV: str = "DEEPSEEK_API_KEY"
    LLM_MODEL_NAME: str = "deepseek-chat"
    LLM_API_URL: str = "https://api.deepseek.com/v1"
    LLM_TEMPERATURE: float = 0.1
    LLM_TIMEOUT: float = 15.0
    
    # IO Infrastructure Coordinates
    BASE_OUTPUT_FOLDER: str = "production_output"
    MODEL_DIR: str = os.path.join("production_output", "models")
    REPORT_DIR: str = os.path.join("production_output", "reports")