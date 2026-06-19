# addup calling for main, should be integrated into main and deleted after final compilation
import sys
import os
# 确保增量文件夹内的模块可以被正确检索
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from portfolio_updated import compute_matrix_bl_stratified_optimize

def run_dss_pipeline_step(returns_df, prev_models_output, llm_layer_output):
    """
    在流水线的每个交易日迭代中调用的高层封装补丁
    :param returns_df: 滚动窗口内的资产历史收益率数据
    :param prev_models_output: 包含 base_omega_diag 和 cqr_widths 的前置多模型集成输出字典
    :param llm_layer_output: 大模型文本层解析出的原始观点得分字典
    """
    # 从前置 `mult-layer+finance-regular` 中承接变量
    base_omega_diag = prev_models_output['base_omega_diag']
    cqr_widths_dict = prev_models_output['cqr_widths']
    
    # 调用徐亦达教授研究增强版 BL 融合算法
    er_bl, Sigma, cohort_logs = compute_matrix_bl_stratified_optimize(
        returns_df=returns_df,
        base_omega_diag=base_omega_diag,
        ensemble_views_dict=llm_layer_output,
        cqr_widths_dict=cqr_widths_dict,
        lambda_kernel=0.15,
        n_cohorts=3,
        cohort_correlation=0.30
    )
    
    print(f">> [MF-DSS 激活成功] 今日资产分层结果快照: {cohort_logs}")
    return er_bl, Sigma
