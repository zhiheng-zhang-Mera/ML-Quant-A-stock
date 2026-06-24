# main.py
import os
import json
import logging
import tempfile
import warnings
from datetime import datetime
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', category=UserWarning)

from config import PipelineConfig
from data_processor import DataProcessor
from models import StatisticalAdaptiveEngine
from stratification import CrossModalStratifier
from portfolio import BayesianExecutionBridge
from llm_analyst import LLMTextAnalyst

def setup_production_logging(config: PipelineConfig):
    log_format = logging.Formatter('[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s')
    logger = logging.getLogger("QuantPipeline")
    logger.setLevel(logging.INFO)
    
    ch = logging.StreamHandler()
    ch.setFormatter(log_format)
    logger.addHandler(ch)
    
    os.makedirs(os.path.join(config.BASE_OUTPUT_FOLDER, "reports"), exist_ok=True)
    log_file = os.path.join(config.BASE_OUTPUT_FOLDER, "reports", "pipeline_execution.log")
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setFormatter(log_format)
    logger.addHandler(fh)
    return logger

def run_production_pipeline(raw_multi_asset_data: dict) -> dict:
    config = PipelineConfig()
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.REPORT_DIR, exist_ok=True)
    
    logger = setup_production_logging(config)
    logger.info(f"\n[INIT] ================ UNIFIED CONFORMAL-LLM MATRIX RUNTIME ENGINE ================")
    
    processor = DataProcessor(config)
    engine = StatisticalAdaptiveEngine(config)
    stratifier = CrossModalStratifier(config)
    bridge = BayesianExecutionBridge(config)
    
    # 1. Formulate White-Box Feature Spaces
    X_panel, y_panel = processor.build_feature_space(raw_multi_asset_data)
    dates = X_panel.index.get_level_values(0).unique().sort_values()
    
    # 2. Strict Temporal Embargo Filtering (Look-Ahead Leakage Protection)
    train_end_idx = len(dates) - 1 - config.EMBARGO_PERIOD
    if train_end_idx < config.MIN_REQUIRED_SAMPLES:
        raise RuntimeError("Insufficient historical time series depth for robust calibration.")
        
    training_dates = dates[:train_end_idx]
    current_production_date = dates[-1]
    
    logger.info(f"[EMBARGO] Historical Training Cutoff Anchor Date: {training_dates[-1].strftime('%Y-%m-%d')}")
    logger.info(f"[EMBARGO] Post-Market Production Execution Horizon: {current_production_date.strftime('%Y-%m-%d')}")
    
    X_train = X_panel.loc[X_panel.index.isin(training_dates, level=0)]
    y_train = y_panel.loc[y_panel.index.isin(training_dates, level=0)]
    
    # 3. Execute Quantile Stacking Cascade Training
    engine.fit_and_quantify(X_train, y_train)
    
    # 4. Synchronize Multi-Asset Cross Sections
    current_X_dict = {}
    historical_returns_builder = {}
    for symbol in config.SYMBOLS:
        if (current_production_date, symbol) in X_panel.index:
            current_X_dict[symbol] = X_panel.xs((current_production_date, symbol)).values
        else:
            symbol_data = X_panel.xs(symbol, level='Symbol')
            valid_historical = symbol_data.index[symbol_data.index <= current_production_date]
            current_X_dict[symbol] = symbol_data.loc[valid_historical[-1]].values
                
        historical_returns_builder[symbol] = y_panel.xs(symbol, level='Symbol')
        
    # Truncate forward targets (t+1) up to today to secure complete look-ahead data blockage
    historical_returns_df = pd.DataFrame(historical_returns_builder).loc[:current_production_date].iloc[:-1].dropna()

    # 5. Extract Conformal Extrema Estimates
    predictions_dict = engine.predict_with_bounds(current_X_dict)
    ml_predictions_map = {sym: float(predictions_dict[sym][0]) for sym in config.SYMBOLS}
    cqr_widths_map = {sym: float(predictions_dict[sym][2] - predictions_dict[sym][1]) for sym in config.SYMBOLS}

    # 6. Capture NLP Real-Time Context Data
    mock_todays_corpus = [{"headline": "Global cross-border asset inflows expand structurally", "timestamp": "14:45:00"}]
    api_key = os.getenv(config.LLM_API_KEY_ENV, "sk-mock-key-for-conformal-pipeline")
    
    analyst = LLMTextAnalyst(api_key=api_key, config=config)
    llm_views = analyst.analyze_news_to_views(mock_todays_corpus, config.SYMBOLS)
    logger.info(f"[LLM-VIEW] Structured Context Information: {llm_views}")

    # 7. Multi-Modal Stratification & Construct Block-Diagonal View Variance 
    cohorts = stratifier.compute_asset_cohorts(cqr_widths_map, llm_views)
    omega_matrix = stratifier.generate_stratified_omega(cqr_widths_map, cohorts)
    logger.info(f"[STRATIFICATION] Cohort Mappings: {cohorts}")

    # 8. Compute Pricing Kernel Black-Litterman Matrix Blending & Allocation Optimizations
    bl_returns, target_weights = bridge.compute_matrix_bl_and_optimize(
        historical_returns_df=historical_returns_df,
        ml_point_predictions=ml_predictions_map,
        omega_matrix=omega_matrix,
        llm_views=llm_views
    )
    
    # 9. Formulate Unified Downstream Executable JSON Output Struct
    llm_payload = {
        "metadata": {
            "execution_timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "production_data_date": current_production_date.strftime('%Y-%m-%d'),
            "tracked_assets": config.SYMBOLS
        },
        "cqr_uncertainty_metrics": {
            sym: {
                "point_prediction": ml_predictions_map[sym],
                "conformal_floor": float(predictions_dict[sym][1]),
                "conformal_ceiling": float(predictions_dict[sym][2]),
                "heteroskedastic_width": cqr_widths_map[sym],
                "assigned_cohort": cohorts[sym]
            } for sym in config.SYMBOLS
        },
        "bayesian_portfolio_allocator": {
            "bl_posterior_expected_returns": [float(r) for r in bl_returns],
            "optimal_mvo_weights": {sym: float(w) for sym, w in zip(config.SYMBOLS, target_weights)}
        }
    }
    
    # Multi-Round safe OS-Level atomic output writing sequences
    timestamp_str = datetime.now().strftime('%H%M%S')
    dynamic_filename = f"multi_asset_llm_payload_{current_production_date.strftime('%Y%m%d')}_{timestamp_str}.json"
    
    for dest_file in [dynamic_filename, "multi_asset_llm_payload.json"]:
        output_path = os.path.join(config.REPORT_DIR, dest_file)
        dir_name = os.path.dirname(output_path)
        with tempfile.NamedTemporaryFile('w', dir=dir_name, delete=False, encoding='utf-8') as tf:
            json.dump(llm_payload, tf, indent=4, ensure_ascii=False)
            temp_name = tf.name
        os.replace(temp_name, output_path)
        
    # 10. Render Dual-Language Corporate Dispatch Console Dashboard
    print("\n" + "="*95)
    print(f"      📊 UNIFIED CONFORMAL PORTFOLIO OPTIMIZATION CORE PRODUCTION DISPATCH TERMINAL")
    print(f"      数据观测日 / Market Snapshot Date: {llm_payload['metadata']['production_data_date']}")
    print("="*95)
    print(f" 1. 符合性堆叠级联 (CQR) 异方差估计 / HETEROSKEDASTIC UNCERTAINTY QUANTIFICATION:")
    for sym in config.SYMBOLS:
        m = llm_payload['cqr_uncertainty_metrics'][sym]
        print(f"    ▶ 资产 [{sym}] -> 堆叠预测: {m['point_prediction']:+.5f} | "
              f"符合性安全带: [{m['conformal_floor']:+.4f}, {m['conformal_ceiling']:+.4f}] | "
              f"风险幅宽: {m['heteroskedastic_width']:.5f} | 队列集群: Group {m['assigned_cohort']}")
    print("-"*95)
    print(f" 2. 贝叶斯定价核后验与最优资产分配权重 / BAYESIAN POSTERIOR & MVO TARGET WEIGHTS:")
    for i, sym in enumerate(config.SYMBOLS):
        w_pct = llm_payload['bayesian_portfolio_allocator']['optimal_mvo_weights'][sym] * 100
        post_r = llm_payload['bayesian_portfolio_allocator']['bl_posterior_expected_returns'][i]
        print(f"    ▶ 资产 [{sym:11s}] -> BL后验预期收益: {post_r:+.6f} | ⚙️ 建议目标配资权重: {w_pct:6.2f}%")
    print("="*95 + "\n")
    
    return llm_payload

if __name__ == "__main__":
    np.random.seed(42)
    mock_dates = pd.date_range(end=datetime.today(), periods=200, freq='D')
    raw_multi_asset_data = {}
    
    for sym in ["510720_ETF", "159937_ETF", "399006_ETF", "600115_SH", "601088_SH", "601229_SH"]:
        sim_p = 15.0 * np.exp(np.cumsum(np.random.normal(0.0002, 0.015, size=200)))
        raw_multi_asset_data[sym] = pd.DataFrame({
            'Open': sim_p * (1 + np.random.normal(0, 0.001, 200)),
            'High': sim_p * (1 + np.abs(np.random.normal(0.003, 0.001, 200))),
            'Low': sim_p * (1 - np.abs(np.random.normal(0.003, 0.001, 200))),
            'Close': sim_p,
            'Volume': np.random.randint(100000, 900000, size=200),
            'Turnover': np.random.uniform(0.01, 0.08, size=200)
        }, index=mock_dates)
        
    payload = run_production_pipeline(raw_multi_asset_data)