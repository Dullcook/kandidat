# 7_main_analysis.py - Main analysis pipeline for ESG ETF thesis
# ============================================================

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Import all modules
from config import *
from utils import print_section_header

# Import numbered modules using importlib
import importlib.util

# Import data_loader
spec = importlib.util.spec_from_file_location("data_loader", "1_data_loader.py")
data_loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(data_loader)
DataLoader = data_loader.DataLoader
load_all_data = data_loader.load_all_data

# Import portfolio_construction
spec = importlib.util.spec_from_file_location("portfolio_construction", "2_portfolio_construction.py")
portfolio_construction = importlib.util.module_from_spec(spec)
spec.loader.exec_module(portfolio_construction)
PortfolioConstructor = portfolio_construction.PortfolioConstructor
create_portfolios = portfolio_construction.create_portfolios

# Import risk_metrics
spec = importlib.util.spec_from_file_location("risk_metrics", "3_risk_metrics.py")
risk_metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(risk_metrics)
RiskMetricsCalculator = risk_metrics.RiskMetricsCalculator

# Import factor_models
spec = importlib.util.spec_from_file_location("factor_models", "4_factor_models.py")
factor_models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(factor_models)
FactorModelAnalysis = factor_models.FactorModelAnalysis

# Import statistical_tests
spec = importlib.util.spec_from_file_location("statistical_tests", "5_statistical_tests.py")
statistical_tests = importlib.util.module_from_spec(spec)
spec.loader.exec_module(statistical_tests)
StatisticalTester = statistical_tests.StatisticalTester

# Import visualizations
spec = importlib.util.spec_from_file_location("visualizations", "6_visualizations.py")
visualizations = importlib.util.module_from_spec(spec)
spec.loader.exec_module(visualizations)
ThesisVisualizer = visualizations.ThesisVisualizer

# Old main function removed - replaced with simplified analysis

def main_simplified():
    """Run simplified, defensible analysis"""
    print_section_header("SIMPLIFIED ANALYSIS - REVISED PLAN")
    
    try:
        # 1. Load data
        print("\n1. Loading data...")
        etf_returns, benchmark_returns, factor_data = load_all_data()
        
        # 2. Create portfolios
        print("\n2. Creating portfolios...")
        portfolios = create_portfolios(etf_returns, benchmark_returns)
        
        # 3. Portfolio composition analysis
        print("\n3. Analyzing portfolio composition...")
        portfolio_constructor = PortfolioConstructor(etf_returns, benchmark_returns)
        # Make sure portfolios are created first
        portfolio_constructor.create_equal_weighted_portfolios()
        etf_types = portfolio_constructor.analyze_portfolio_composition()
        
        # 4. Basic risk metrics only
        print("\n4. Calculating risk metrics...")
        # Filter to only numeric columns for risk calculation
        numeric_portfolios = portfolios[['ETF_Portfolio', 'Benchmark_Portfolio']]
        risk_calculator = RiskMetricsCalculator(numeric_portfolios)
        risk_metrics = risk_calculator.calculate_all_metrics()
        
        # 5. Simple statistical tests
        print("\n5. Running statistical tests...")
        etf_returns_series = portfolios['ETF_Portfolio'].dropna()
        bench_returns_series = portfolios['Benchmark_Portfolio'].dropna()
        
        stat_tester = StatisticalTester(etf_returns_series, bench_returns_series, 
                                      ('ESG ETF', 'Benchmark'))
        stat_tests = stat_tester.run_simple_tests()
        
        # 6. Factor models (CAPM and FF3 only)
        print("\n6. Running factor models...")
        factor_analyzer = FactorModelAnalysis(portfolios, factor_data)
        factor_results, factor_summary = factor_analyzer.run_all_models()
        
        # 7. Crisis analysis
        print("\n7. Analyzing crisis periods...")
        crisis_performance = portfolio_constructor.get_crisis_performance()
        if crisis_performance is None:
            print("   Warning: Crisis performance calculation failed")
            crisis_performance = pd.DataFrame()
        
        # 8. Create visualizations
        print("\n8. Creating visualizations...")
        try:
            visualizer = ThesisVisualizer(save_figures=True)
            
            # Create essential visualizations individually
            print("   Creating cumulative performance chart...")
            visualizer.plot_cumulative_performance(portfolios, CRISIS_PERIODS)
            
            print("   Creating risk-return comparison...")
            visualizer.plot_simple_risk_return(risk_metrics)
            
            print("   Creating crisis performance analysis...")
            visualizer.plot_crisis_analysis(portfolios)
            
            print("   Creating VaR comparison...")
            visualizer.plot_var_comparison(risk_metrics)
            
            print("   Creating factor model comparison...")
            visualizer.plot_factor_models(factor_results)
            
            print("   Creating portfolio composition table...")
            visualizer.plot_portfolio_composition(etf_types)
            
            # Add maximum drawdown visualization
            print("   Creating maximum drawdown visualization...")
            visualizer.plot_maximum_drawdown(portfolios)
            
            # Add risk ratios table
            print("   Creating risk ratios comparison table...")
            visualizer.plot_risk_ratios_table(risk_metrics, stat_tests)
            
            print("   ✓ All visualizations created successfully")
        except Exception as e:
            print(f"   ⚠️ Visualization error: {e}")
            print("   Continuing with analysis...")
        
        # 9. Print key results
        print_section_header("KEY RESULTS SUMMARY")
        
        print(f"\nPortfolio Performance:")
        print(f"  ESG ETF: {risk_metrics.loc['ETF_Portfolio', 'Mean_Return_Annual']*100:.2f}% annual return")
        print(f"  Benchmark: {risk_metrics.loc['Benchmark_Portfolio', 'Mean_Return_Annual']*100:.2f}% annual return")
        print(f"  Difference: {(risk_metrics.loc['ETF_Portfolio', 'Mean_Return_Annual'] - risk_metrics.loc['Benchmark_Portfolio', 'Mean_Return_Annual'])*100:.2f}%")
        
        print(f"\nRisk Metrics:")
        print(f"  ESG ETF Sharpe: {risk_metrics.loc['ETF_Portfolio', 'Sharpe_Ratio']:.3f}")
        print(f"  Benchmark Sharpe: {risk_metrics.loc['Benchmark_Portfolio', 'Sharpe_Ratio']:.3f}")
        print(f"  ESG ETF Max DD: {risk_metrics.loc['ETF_Portfolio', 'Max_Drawdown_%']:.2f}%")
        print(f"  Benchmark Max DD: {risk_metrics.loc['Benchmark_Portfolio', 'Max_Drawdown_%']:.2f}%")
        
        # Add VaR metrics
        print(f"\nValue at Risk (VaR):")
        print(f"  ESG ETF VaR 95%: {risk_metrics.loc['ETF_Portfolio', 'VaR_95_Monthly']*100:.2f}% monthly")
        print(f"  Benchmark VaR 95%: {risk_metrics.loc['Benchmark_Portfolio', 'VaR_95_Monthly']*100:.2f}% monthly")
        print(f"  ESG ETF VaR 99%: {risk_metrics.loc['ETF_Portfolio', 'VaR_99_Monthly']*100:.2f}% monthly")
        print(f"  Benchmark VaR 99%: {risk_metrics.loc['Benchmark_Portfolio', 'VaR_99_Monthly']*100:.2f}% monthly")
        
        print(f"\nStatistical Tests:")
        if stat_tests:
            print(f"  T-test p-value: {stat_tests['t_test']['p_value']:.4f}")
            print(f"  Portfolio correlation: {stat_tests['correlation']:.4f}")
        
        print(f"\nCrisis Performance:")
        for period in crisis_performance.index:
            if period != 'Normal':
                rel_perf = crisis_performance.loc[period, 'Relative_Performance']
                print(f"  {period}: ESG {'outperformed' if rel_perf > 0 else 'underperformed'} by {rel_perf*100:.2f}%")
        
        print(f"\nFactor Models:")
        if 'ETF_Portfolio' in factor_results and 'CAPM' in factor_results['ETF_Portfolio']:
            etf_alpha = factor_results['ETF_Portfolio']['CAPM']['alpha']
            print(f"  CAPM Alpha (ESG): {etf_alpha*12*100:.2f}% annualized")
        
        print_section_header("ANALYSIS COMPLETE")
        print("Simplified analysis completed successfully!")
        
        return {
            'portfolios': portfolios,
            'risk_metrics': risk_metrics,
            'stat_tests': stat_tests,
            'factor_results': factor_results,
            'crisis_performance': crisis_performance,
            'figures': 'All visualizations created successfully'
        }
        
    except Exception as e:
        print(f"Error in simplified analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Run the simplified analysis
    results = main_simplified()