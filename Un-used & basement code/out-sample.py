def rigid_rolling_oos_backtest(df_processed, window_size=60):
    """
    严格的样本外滚动窗口时间序列回测框架
    确保完全屏蔽前瞻信息，符合顶级金融期刊（如 Journal of Finance）的实证标准
    """
    X = df_processed[['mom_1d', 'mom_5d', 'gk_vol', 'turnover_shck']]
    y = df_processed['target_return']
    
    oos_predictions = []
    actual_returns = []
    dates = []
    
    # 从 window_size 开始滚动训练
    for t in range(window_size, len(df_processed)):
        X_train = X.iloc[t-window_size : t]
        y_train = y.iloc[t-window_size : t]
        
        X_test = X.iloc[t : t+1]
        
        # 训练当前窗口模型
        model = lgb.LGBMRegressor(n_estimators=30, max_depth=2, verbose=-1)
        model.fit(X_train, y_train)
        
        # 样本外单期预测
        pred_oos = model.predict(X_test)[0]
        
        oos_predictions.append(pred_oos)
        actual_returns.append(y.iloc[t])
        dates.append(df_processed['date'].iloc[t])
        
    results = pd.DataFrame({
        'Date': dates,
        'OOS_Pred_Return': oos_predictions,
        'Actual_Return': actual_returns
    })
    
    # 计算学术界标准的评价指标：样本外 R-square (OOS R²)
    denom = np.sum((results['Actual_Return'] - np.mean(y.iloc[:window_size]))**2)
    num = np.sum((results['Actual_Return'] - results['OOS_Pred_Return'])**2)
    oos_r_squared = 1 - (num / denom)
    
    print(f"【实证检验完毕】严格样本外 OOS R²: {oos_r_squared:.4f}")
    return results