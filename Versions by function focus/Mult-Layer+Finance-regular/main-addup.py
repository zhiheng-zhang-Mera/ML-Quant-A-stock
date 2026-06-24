# main.py 的 Step 5 局部组装示例：
# 初始化升级后的引擎 (增加分歧度放大系数 lambda_split)
engine = ConformalEnsemblePricingEngine(
    alpha=config.alpha, 
    lambda_split=1.5
)

# 喂入洗干净的数据进行异质级联拟合与复合标定
engine.fit_and_calibrate(X_train, y_train)

# 获取带有双重护栏的量价主动阿尔法预测
ml_view_results = engine.predict_production(current_X_dict)

# Step 6: 投喂给包含金融内核正则化的 BL 优化器
portfolio_output = compute_matrix_bl_and_optimize(
    historical_returns_df=historical_returns_df,
    ml_view_results=ml_view_results,
    llm_sentiment_scores=llm_sentiment_scores,
    pipeline_config=config
)
