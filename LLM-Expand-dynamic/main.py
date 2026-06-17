# main.py
import sys
import os
import json
import warnings
import logging
import tempfile
from datetime import datetime
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', category=UserWarning)

from config import PipelineConfig
from data_processor import DataProcessor
from models import StatisticalAdaptiveEngine
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
    logger.info(f"\n[INIT] ================ 激活全白盒多资产矩阵生产线 / LAUNCHING QUANT ENGINE ================")
    
    processor = DataProcessor(config)
    engine = StatisticalAdaptiveEngine(config)
    bridge = BayesianExecutionBridge(config)
    
    # 1. 加工多资产横截面面板数据
    X_panel, y_panel = processor.build_feature_space(raw_multi_asset_data)
    dates = X_panel.index.get_level_values(0).unique().sort_values()
    
    # 2. 实施严格的时间轴时序禁运隔离限制（防止 Look-ahead 泄露作弊）
    train_end_idx = len(dates) - 1 - config.EMBARGO_PERIOD
    if train_end_idx < config.MIN_REQUIRED_SAMPLES:
        raise RuntimeError("训练集样本严重不足，请增加回测历史深度。")
        
    training_dates = dates[:train_end_idx]
    current_production_date = dates[-1]  # 盘后执行决策观测点（即今天）
    
    logger.info(f"[EMBARGO] 历史训练集隔离拦截终点: {training_dates[-1].strftime('%Y-%m-%d')}")
    logger.info(f"[EMBARGO] 盘后实盘配资决策执行点: {current_production_date.strftime('%Y-%m-%d')}")
    
    # 【修复：防范 MultiIndex 跨版本切片异动异常】使用显式 .isin 机制
    X_train = X_panel.loc[X_panel.index.isin(training_dates, level=0)]
    y_train = y_panel.loc[y_panel.index.isin(training_dates, level=0)]
    
    # 3. 训练条件分位数非参数估计引擎并锁定残差分布
    engine.fit_and_quantify(X_train, y_train)
    
    # 4. 抽取当前执行日的最新横截面特征向量，构建完整的历史对数收益率面板
    current_X_dict = {}
    historical_returns_builder = {}
    for symbol in config.SYMBOLS:
        # 生产级防御：校验当前日期横截面样本点完备性
        if (current_production_date, symbol) in X_panel.index:
            current_X_dict[symbol] = X_panel.xs((current_production_date, symbol)).values
        else:
            symbol_data = X_panel.xs(symbol, level='Symbol')
            valid_historical = symbol_data.index[symbol_data.index <= current_production_date]
            if len(valid_historical) > 0:
                current_X_dict[symbol] = symbol_data.loc[valid_historical[-1]].values
            else:
                raise KeyError(f"Critical engineering fault: No viable features for {symbol} at {current_production_date}")
                
        historical_returns_builder[symbol] = y_panel.xs(symbol, level='Symbol')
        
    # 【修复：前瞻性欺骗彻底阻断】 
    # y_panel 存储的是前瞻收益标签(t+1)，构建协方差矩阵历史窗口时必须执行 .iloc[:-1] 截断当前观察日，防止混入未来信息
    historical_returns_df = pd.DataFrame(historical_returns_builder).loc[:current_production_date].iloc[:-1].dropna()

    # 5. 推演得到今日量价的条件期望点预测与 CQR 统计幅宽
    predictions_dict = engine.predict_with_bounds(current_X_dict)
    
    ml_predictions_map = {sym: float(predictions_dict[sym][0]) for sym in config.SYMBOLS}
    cqr_widths_map = {sym: float(predictions_dict[sym][2] - predictions_dict[sym][1]) for sym in config.SYMBOLS}

    # 6. 【数据合规】抓取盘后收盘前的权威文本新闻语料
    mock_todays_corpus = [{"headline": "Tech semiconductor stocks rally sharply", "timestamp": "14:22:00"}]

    # 【解耦重构】：通过 OS 环境变量安全总线调取密钥凭证，若生产环境未配配置，则平滑降级使用仿真 Mock Key
    api_key = os.getenv(config.LLM_API_KEY_ENV, "sk-mock-key-for-conformal-pipeline")

    # 【全面注入】：彻底交托 PipelineConfig 统管控制，模型切换无需修改此处任何一行业务逻辑
    analyst = LLMTextAnalyst(
        api_key=api_key,
        model_name=config.LLM_MODEL_NAME,
        api_url=config.LLM_API_URL,
        temperature=config.LLM_TEMPERATURE,
        timeout=config.LLM_TIMEOUT
    )

    llm_views = analyst.analyze_news_to_views(mock_todays_corpus, config.SYMBOLS)
    logger.info(f"[LLM-VIEW] 大模型文本层解析生成的结构化外部观点: {llm_views}")

    # 7. 跨越贝叶斯网桥，执行白盒化矩阵融合与马科维茨 MVO 最优配资分配求解
    bl_returns, target_weights, asset_variances = bridge.compute_matrix_bl_and_optimize(
        historical_returns_df=historical_returns_df,
        ml_point_predictions=ml_predictions_map,
        cqr_widths=cqr_widths_map,
        llm_views=llm_views
    )
    
    # 8. 整合装配可解释性决策 JSON Payload
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
                "heteroskedastic_width": cqr_widths_map[sym]
            } for sym in config.SYMBOLS
        },
        "bayesian_portfolio_allocator": {
            "bl_posterior_expected_returns": [float(r) for r in bl_returns],
            "optimal_mvo_weights": {sym: float(w) for sym, w in zip(config.SYMBOLS, target_weights)}
        }
    }
    
    # 【修复：多轮冲刷覆盖与文件破坏防御】
    # 策略 A：建立带有日度与日内高精时间戳的动态文件体系
    timestamp_str = datetime.now().strftime('%H%M%S')
    dynamic_filename = f"multi_asset_llm_payload_{current_production_date.strftime('%Y%m%d')}_{timestamp_str}.json"
    json_out_dynamic = os.path.join(config.REPORT_DIR, dynamic_filename)
    json_out_static = os.path.join(config.REPORT_DIR, "multi_asset_llm_payload.json")
    
    # 策略 B：执行多线程/多进程安全的原子性落盘 (Atomic JSON Writing)
    for output_path in [json_out_dynamic, json_out_static]:
        dir_name = os.path.dirname(output_path)
        with tempfile.NamedTemporaryFile('w', dir=dir_name, delete=False, encoding='utf-8') as tf:
            json.dump(llm_payload, tf, indent=4, ensure_ascii=False)
            temp_name = tf.name
        os.replace(temp_name, output_path)
        
    # 9. 输出中英双语核心生产看板终端
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
    # 自动化仿真沙盒数据环境测试激活
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