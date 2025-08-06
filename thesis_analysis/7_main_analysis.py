# 7_main_analysis.py - Main analysis pipeline for ESG ETF thesis
# ============================================================

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import all modules
from config import *
from utils import print_section_header, export_to_latex
from data_loader import DataLoader
from portfolio_construction import PortfolioConstructor
from risk_metrics import RiskMetricsCalculator
from factor_models import FactorModelAnalysis
from statistical_tests import StatisticalTester, CrisisPeriodTester
from visualizations import ThesisVisualizer

def main():
    """
    Main analysis pipeline for ESG ETF thesis
    """
    print("="*60)
    print("ESG ETF THESIS ANALYSIS - COMPLETE PIPELINE")
    print("="*60)
    print(f"\nAnalysis Period: {START_DATE} to {END_DATE}")
    print(f"Output Directory: {RESULTS_PATH}")
    print("="*60)
    
    # ========================================
    # 1. DATA LOADING
    # ========================================
    loader = DataLoader()
    etf_returns, benchmark_returns, factor_data = loader.load_all_data()
    
    # Get summary statistics
    etf_stats, factor_stats = loader.get_summary_statistics()
    
    # ========================================
    # 2. PORTFOLIO CONSTRUCTION
    # ========================================
    constructor = PortfolioConstructor(etf_returns, benchmark_returns)
    portfolios = constructor.create_equal_weighted_portfolios()
    
    # Calculate rolling statistics
    rolling_stats = constructor.calculate_rolling_statistics(window=12)
    
    # Calculate tracking error
    tracking_error_results = constructor.calculate_tracking_error()
    
    # Get crisis performance
    crisis_performance = constructor.get_crisis_performance()
    
    # ========================================
    # 3. RISK METRICS CALCULATION
    # ========================================
    print_section_header("RISK METRICS CALCULATION")
    
    # Full period metrics
    risk_calculator = RiskMetricsCalculator(portfolios[['ETF_Portfolio', 'Benchmark_Portfolio']])
    full_period_metrics = risk_calculator.calculate_all_metrics()
    
    # Crisis period metrics
    crisis_metrics = {}
    for period in portfolios['Period'].unique():
        period_data = portfolios[portfolios['Period'] == period][['ETF_Portfolio', 'Benchmark_Portfolio']]
        if len(period_data) > 0:
            period_calculator = RiskMetricsCalculator(period_data)
            crisis_metrics[period] = period_calculator.calculate_all_metrics()
    
    # ========================================
    # 4. STATISTICAL TESTS
    # ========================================
    # Full period tests
    stat_tester = StatisticalTester(
        portfolios['ETF_Portfolio'],
        portfolios['Benchmark_Portfolio'],
        labels=('High-Sustainability ETF Portfolio', 'Benchmark Portfolio')
    )
    statistical_tests = stat_tester.run_all_tests()
    
    # Crisis period tests
    crisis_tester = CrisisPeriodTester(portfolios)
    crisis_test_results = crisis_tester.test_crisis_performance()
    
    # ========================================
    # 5. FACTOR MODEL ANALYSIS
    # ========================================
    factor_analysis = FactorModelAnalysis(
        portfolios[['ETF_Portfolio', 'Benchmark_Portfolio']], 
        factor_data
    )
    factor_results, factor_summary = factor_analysis.run_all_models()
    
    # Test alpha difference
    alpha_difference = factor_analysis.test_alpha_difference(model='Carhart')
    
    # ========================================
    # 6. VISUALIZATIONS
    # ========================================
    visualizer = ThesisVisualizer(save_figures=True)
    figures = visualizer.create_all_visualizations(
        portfolios,
        full_period_metrics,
        (factor_results, factor_summary),
        CRISIS_PERIODS
    )
    
    # ========================================
    # 7. EXPORT RESULTS
    # ========================================
    print_section_header("EXPORTING RESULTS")
    
    # Create Excel writer
    with pd.ExcelWriter(f'{RESULTS_PATH}thesis_results.xlsx', engine='openpyxl') as writer:
        
        # 1. Summary Statistics
        full_period_metrics.to_excel(writer, sheet_name='Full_Period_Metrics')
        
        # 2. Crisis Analysis
        crisis_performance.to_excel(writer, sheet_name='Crisis_Performance')
        
        # 3. Factor Models
        if factor_summary is not None:
            factor_summary.to_excel(writer, sheet_name='Factor_Models', index=False)
        
        # 4. Statistical Tests
        # Convert nested dict to DataFrame
        stat_test_df = pd.DataFrame()
        for test_type, tests in statistical_tests.items():
            for test_name, results in tests.items():
                row = {'Test_Type': test_type, 'Test': test_name}
                row.update(results)
                stat_test_df = pd.concat([stat_test_df, pd.DataFrame([row])], ignore_index=True)
        stat_test_df.to_excel(writer, sheet_name='Statistical_Tests', index=False)
        
        # 5. Tracking Error
        pd.DataFrame([tracking_error_results]).to_excel(writer, sheet_name='Tracking_Error', index=False)
        
        # 6. Portfolio Returns (for reference)
        portfolios.to_excel(writer, sheet_name='Portfolio_Returns')
        
        # 7. Rolling Statistics
        rolling_stats.to_excel(writer, sheet_name='Rolling_Statistics')
    
    print(f"   Results saved to: {RESULTS_PATH}thesis_results.xlsx")
    
    # Export key tables to LaTeX
    export_to_latex(
        full_period_metrics, 
        f'{RESULTS_PATH}full_period_metrics.tex',
        caption="Full Period Risk Metrics Comparison",
        label="tab:full_period_metrics"
    )
    
    export_to_latex(
        crisis_performance,
        f'{RESULTS_PATH}crisis_performance.tex',
        caption="Performance During Crisis Periods",
        label="tab:crisis_performance"
    )
    
    if factor_summary is not None:
        export_to_latex(
            factor_summary,
            f'{RESULTS_PATH}factor_models.tex',
            caption="Factor Model Results",
            label="tab:factor_models"
        )
    
    # ========================================
    # 8. GENERATE THESIS TEXT
    # ========================================
    print_section_header("KEY FINDINGS FOR THESIS")
    
    # Extract key metrics
    etf_metrics = full_period_metrics.loc['ETF_Portfolio']
    bench_metrics = full_period_metrics.loc['Benchmark_Portfolio']
    
    thesis_text = f"""
CHAPTER 5: EMPIRICAL RESULTS

5.1 Descriptive Statistics
--------------------------
The high-sustainability ETF portfolio generated an average monthly return of {etf_metrics['Mean_Return_Monthly']*100:.3f}% 
({etf_metrics['Mean_Return_Annual']*100:.2f}% annualized) compared to {bench_metrics['Mean_Return_Monthly']*100:.3f}% 
({bench_metrics['Mean_Return_Annual']*100:.2f}% annualized) for the benchmark portfolio over the sample period from 
January 2019 to February 2025.

The volatility of the high-sustainability portfolio was {etf_metrics['Volatility_Annual']:.2f}% (annualized) 
versus {bench_metrics['Volatility_Annual']:.2f}% for the benchmark portfolio, suggesting 
{'lower' if etf_metrics['Volatility_Annual'] < bench_metrics['Volatility_Annual'] else 'higher'} risk.

5.2 Risk-Adjusted Performance
-----------------------------
The Sharpe ratio for the high-sustainability ETF portfolio was {etf_metrics['Sharpe_Ratio']:.3f} compared to 
{bench_metrics['Sharpe_Ratio']:.3f} for the benchmark portfolio, indicating 
{'superior' if etf_metrics['Sharpe_Ratio'] > bench_metrics['Sharpe_Ratio'] else 'inferior'} risk-adjusted returns.

The Sortino ratio, which focuses on downside volatility, was {etf_metrics['Sortino_Ratio']:.3f} for the 
high-sustainability portfolio versus {bench_metrics['Sortino_Ratio']:.3f} for benchmarks.

5.3 Downside Risk Protection
----------------------------
Maximum drawdown for the high-sustainability ETF portfolio was {etf_metrics['Max_Drawdown_%']:.2f}% compared to 
{bench_metrics['Max_Drawdown_%']:.2f}% for the benchmark portfolio, suggesting 
{'better' if abs(etf_metrics['Max_Drawdown_%']) < abs(bench_metrics['Max_Drawdown_%']) else 'worse'} downside protection.

Value-at-Risk (95% confidence) was {etf_metrics['VaR_95_Monthly']*100:.2f}% monthly for high-sustainability ETFs 
versus {bench_metrics['VaR_95_Monthly']*100:.2f}% for benchmarks.

5.4 Statistical Significance
----------------------------
The difference in mean returns between portfolios was {'statistically significant' if statistical_tests['parametric']['t_test']['p_value'] < 0.05 else 'not statistically significant'} 
(p-value = {statistical_tests['parametric']['t_test']['p_value']:.4f}).

The difference in variances was {'statistically significant' if statistical_tests['variance']['f_test']['p_value'] < 0.05 else 'not statistically significant'} 
(p-value = {statistical_tests['variance']['f_test']['p_value']:.4f}).
"""
    
    # Add factor model results if available
    if factor_summary is not None and len(factor_summary) > 0:
        # Find Carhart results
        carhart_etf = factor_summary[(factor_summary['Model'] == 'Carhart') & 
                                    (factor_summary['Portfolio'] == 'ETF_Portfolio')]
        carhart_bench = factor_summary[(factor_summary['Model'] == 'Carhart') & 
                                      (factor_summary['Portfolio'] == 'Benchmark_Portfolio')]
        
        if len(carhart_etf) > 0 and len(carhart_bench) > 0:
            thesis_text += f"""
5.5 Factor Model Analysis
------------------------
Under the Carhart four-factor model, the high-sustainability ETF portfolio generated an alpha of 
{carhart_etf.iloc[0]['Alpha_Annual_%']:.2f}% annualized (p-value = {carhart_etf.iloc[0]['Alpha_p-value']:.4f}), 
while the benchmark portfolio alpha was {carhart_bench.iloc[0]['Alpha_Annual_%']:.2f}% annualized 
(p-value = {carhart_bench.iloc[0]['Alpha_p-value']:.4f}).

The adjusted R-squared was {carhart_etf.iloc[0]['Adj_R-squared']:.3f} for the ETF portfolio and 
{carhart_bench.iloc[0]['Adj_R-squared']:.3f} for the benchmark portfolio, indicating that the four-factor model 
explains {carhart_etf.iloc[0]['Adj_R-squared']*100:.1f}% and {carhart_bench.iloc[0]['Adj_R-squared']*100:.1f}% 
of return variation, respectively.
"""
    
    # Add crisis performance
    thesis_text += f"""
5.6 Crisis Period Performance
----------------------------"""
    
    for crisis in ['COVID-19', 'Inflation_Shock', 'Banking_Stress']:
        if crisis in crisis_performance.index:
            perf = crisis_performance.loc[crisis]
            thesis_text += f"""

{crisis.replace('_', ' ')}:
- ETF Portfolio: {perf['ETF_Cumulative']*100:.2f}% cumulative return
- Benchmark: {perf['Benchmark_Cumulative']*100:.2f}% cumulative return
- Relative Performance: {perf['Relative_Performance']*100:.2f}%"""
    
    print(thesis_text)
    
    # Save thesis text
    with open(f'{RESULTS_PATH}thesis_text.txt', 'w') as f:
        f.write(thesis_text)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print(f"\nResults saved to: {RESULTS_PATH}")
    print(f"Figures saved to: {FIGURES_PATH}")
    print(f"Thesis text saved to: {RESULTS_PATH}thesis_text.txt")
    print("\nNext steps:")
    print("1. Review the Excel results file")
    print("2. Check all generated figures")
    print("3. Use the thesis text as a starting point for your results chapter")
    print("4. Run additional analyses as needed")
    
    return {
        'portfolios': portfolios,
        'full_period_metrics': full_period_metrics,
        'crisis_metrics': crisis_metrics,
        'factor_results': factor_results,
        'statistical_tests': statistical_tests,
        'crisis_performance': crisis_performance,
        'tracking_error': tracking_error_results,
        'figures': figures
    }


if __name__ == "__main__":
    # Run the complete analysis
    results = main()