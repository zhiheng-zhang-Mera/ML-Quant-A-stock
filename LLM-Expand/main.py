# main-dual-lang.py
import sys
import os
import json
import warnings
import logging
from datetime import datetime
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', category=UserWarning)

from config import PipelineConfig
from data_processor import DataProcessor
from models import StatisticalAdaptiveEngine
from portfolio import BayesianExecutionBridge

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
    """
    多资产生产调度总线核心生命周期逻辑
    """
    config = PipelineConfig()
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.REPORT_DIR, exist_ok=True)
    
    logger = setup_production_logging(config)
    logger.info(f"\n[INIT] ================ 激活多资产矩阵生产线 / STARTING MULTI-ASSET DISPATCH ================")
    
    processor = DataProcessor(config)
    engine = StatisticalAdaptiveEngine(config)
    bridge = BayesianExecutionBridge(config)
    
    # 1. 加工多资产横截面面板数据
    X_panel, y_panel = processor.build_feature_space(raw_multi_asset_data)
    
    # 提取交叉截面时间轴，并进行严格的【滚动时序禁运隔离】
    dates = X_panel.index.get_level_values(0).unique().sort_values()
    
    # 设定历史截断点 T-1 用于模拟生产环境
    train_end_idx = len(dates) - 1 - config.EMBARGO_PERIOD
    if train_end_idx < config.MIN_REQUIRED_SAMPLES:
        raise RuntimeError("Data length is insufficient after subtracting the Embargo period constraint.")
        
    training_dates = dates[:train_end_idx]
    # current_production_date = today = datetime.today() # 生产环境中应替换为实际观测日，当前模拟使用系统日期
    current_production_date = dates[-1]
    
    logger.info(f"\n[EMBARGO] 历史训练集阻断终点 / Train End: {training_dates[-1].strftime('%Y-%m-%d')}")
    logger.info(f"\n[EMBARGO] 盘后决策执行观测点 / Production Date: {current_production_date.strftime('%Y-%m-%d')}")
    
    # 2. 隔离抽取训练样本面板
    X_train = X_panel.loc[training_dates]
    y_train = y_panel.loc[training_dates]
    
    # 3. 激活符合性分位数回归引擎
    engine.fit_and_quantify(X_train, y_train)
    
    # 4. 抽取当前生产观测日的最新特征向量
    current_X_dict = {}
    historical_returns_builder = {}
    
    for symbol in config.SYMBOLS:
        current_X_dict[symbol] = X_panel.xs((current_production_date, symbol)).values
        # 提取各个资产的完整历史对数收益率用于协方差估计
        historical_returns_builder[symbol] = y_panel.xs(symbol, level='Symbol')
        
    # historical_returns_df = pd.DataFrame(historical_returns_builder).loc[:current_production_date]
    # 截取到今天，并滤除最后一天尚未发生的 NaN 收益率，确保马科维茨优化器正常运行
    historical_returns_df = pd.DataFrame(historical_returns_builder).loc[:current_production_date].dropna()

    # 步骤 5：运行机器学习模型，得到量价预测与 CQR 宽度边界
    ml_predictions, cqr_widths = models_engine.fit_and_predict(X_train, y_train, current_market_features)

    # ============ 新增 Layer 1.5: 盘后定时文本分析过滤 ============
    # 严格执行时间戳禁运，抓取当天15:00收盘前的新闻语料
    todays_embargoed_news = news_db.fetch_before_timestamp(current_date, cutoff_time="15:00:00")

    analyst = LLMTextAnalyst(api_key=config.LLM_API_KEY)
    llm_views = analyst.analyze_news_to_views(todays_embargoed_news, config.target_assets)
    # ============================================================

    # 步骤 6：将量价条件、CQR幅宽、以及动态LLM文本观点联合输入贝叶斯网桥
        posterior_returns, optimal_weights = compute_matrix_bl_and_optimize(
        historical_returns=historical_returns_slice,
        ml_point_predictions=ml_predictions,
        cqr_widths=cqr_widths,
        llm_views=llm_views, # 注入动态文本观点
        asset_symbols=config.target_assets,
        config=config
    )

    # 步骤 7：照常输出带有可解释性指标的 JSON Payload
    export_production_payload(posterior_returns, optimal_weights, current_date)
    
    json_out = os.path.join(config.REPORT_DIR, "multi_asset_llm_payload.json")
    with open(json_out, 'w', encoding='utf-8') as f:
        json.dump(llm_payload, f, indent=4, ensure_ascii=False)
        
    # 8. 呈现多资产中英双语核心决策看板
    print("\n" + "="*85)
    print(f"      📊 MULTI-ASSET CONFORMAL PORTFOLIO OPTIMIZATION DISPATCH TERMINAL")
    print(f"      数据观测日 / Market Snapshot Date: {llm_payload['metadata']['production_data_date']}")
    print("="*85)
    print(f" 1. 符合性分位数回归 (CQR) 异方差估计 / HETEROSKEDASTIC UNCERTAINTY QUANTIFICATION:")
    for sym in config.SYMBOLS:
        metrics = llm_payload['cqr_uncertainty_metrics'][sym]
        print(f"    ▶ 资产 [{sym}] -> 点预测: {metrics['point_prediction']:+.5f} | "
              f"符合性区间: [{metrics['conformal_floor']:+.4f}, {metrics['conformal_ceiling']:+.4f}] | "
              f"动态幅宽: {metrics['heteroskedastic_width']:.5f}")
    print("-"*85)
    print(f" 2. 贝叶斯后验与最优资产分配权重 / BAYESIAN POSTERIOR & MVO TARGET WEIGHTS:")
    for i, sym in enumerate(config.SYMBOLS):
        w_pct = llm_payload['bayesian_portfolio_allocator']['optimal_mvo_weights'][sym] * 100
        post_r = llm_payload['bayesian_portfolio_allocator']['bl_posterior_expected_returns'][i]
        print(f"    ▶ 资产 [{sym:11s}] -> BL后验预期收益: {post_r:+.6f} | ⚙️ 建议目标配资权重: {w_pct:6.2f}%")
    print("="*85 + "\n")
    
    return llm_payload

if __name__ == "__main__":
    print("Starting main script...运行主管道...")
    # 生成多资产同步时序仿真测试沙盒
    np.random.seed(42)
    config = PipelineConfig()
    mock_dates = pd.date_range(end=datetime.today(), periods=250, freq='D')
    
    raw_multi_asset_data = {}
    for sym in config.SYMBOLS:
        sim_close = 10.0 * np.exp(np.cumsum(np.random.normal(0.0003, 0.012, size=250)))
        raw_multi_asset_data[sym] = pd.DataFrame({
            'Open': sim_close * (1 + np.random.normal(0, 0.001, 250)),
            'High': sim_close * (1 + np.abs(np.random.normal(0.002, 0.001, 250))),
            'Low': sim_close * (1 - np.abs(np.random.normal(0.002, 0.001, 250))),
            'Close': sim_close,
            'Volume': np.random.randint(200000, 800000, size=250),
            'Turnover': np.random.uniform(0.01, 0.06, size=250)
        }, index=mock_dates)
        
    payload = run_production_pipeline(raw_multi_asset_data)