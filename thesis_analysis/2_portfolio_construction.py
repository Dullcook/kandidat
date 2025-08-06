# 2_portfolio_construction.py - Portfolio construction functions
# ============================================================

import pandas as pd
import numpy as np
from config import ETF_BENCHMARK_MAP, CRISIS_PERIODS
from utils import print_section_header, print_subsection_header

class PortfolioConstructor:
    """Class to handle portfolio construction"""
    
    def __init__(self, etf_returns, benchmark_returns):
        self.etf_returns = etf_returns
        self.benchmark_returns = benchmark_returns
        self.portfolios = None
        
    def create_equal_weighted_portfolios(self):
        """Create equal-weighted portfolios"""
        print_section_header("PORTFOLIO CONSTRUCTION")
        print_subsection_header("Creating Equal-Weighted Portfolios")
        
        # Create ETF portfolio (equal-weighted average)
        etf_portfolio = self.etf_returns.mean(axis=1)
        etf_portfolio.name = 'High_Sustainability_ETF_Portfolio'
        
        # Create benchmark portfolio (equal-weighted average)
        # First, we need to match benchmarks to ETFs
        matched_benchmarks = []
        for etf in self.etf_returns.columns:
            if etf in ETF_BENCHMARK_MAP:
                benchmark = ETF_BENCHMARK_MAP[etf]
                if benchmark in self.benchmark_returns.columns:
                    matched_benchmarks.append(benchmark)
                else:
                    print(f"   Warning: Benchmark '{benchmark}' not found for ETF '{etf}'")
        
        # Use only matched benchmarks
        benchmark_portfolio = self.benchmark_returns[matched_benchmarks].mean(axis=1)
        benchmark_portfolio.name = 'Benchmark_Portfolio'
        
        print(f"   ETF Portfolio: {len(etf_portfolio)} observations")
        print(f"   Benchmark Portfolio: {len(benchmark_portfolio)} observations")
        print(f"   ETFs in portfolio: {len(self.etf_returns.columns)}")
        print(f"   Benchmarks matched: {len(matched_benchmarks)}")
        
        # Combine into single DataFrame
        self.portfolios = pd.DataFrame({
            'ETF_Portfolio': etf_portfolio,
            'Benchmark_Portfolio': benchmark_portfolio
        })
        
        # Add period labels
        self.portfolios = self._add_period_labels(self.portfolios)
        
        return self.portfolios
    
    def create_value_weighted_portfolios(self, weights_df=None):
        """
        Create value-weighted portfolios
        Note: Requires AUM or market cap data
        """
        print_subsection_header("Creating Value-Weighted Portfolios")
        
        if weights_df is None:
            print("   No weights provided, using equal weights")
            return self.create_equal_weighted_portfolios()
        
        # Implementation for value-weighted portfolios
        # Would require AUM data for each ETF
        pass
    
    def _add_period_labels(self, portfolios):
        """Add crisis period labels to portfolio data"""
        print_subsection_header("Identifying Crisis Periods")
        
        # Initialize with 'Normal' period
        portfolios['Period'] = 'Normal'
        portfolios['Crisis'] = False
        
        # Mark crisis periods
        for crisis_name, (start, end) in CRISIS_PERIODS.items():
            mask = (portfolios.index >= pd.to_datetime(start)) & \
                   (portfolios.index <= pd.to_datetime(end))
            portfolios.loc[mask, 'Period'] = crisis_name
            portfolios.loc[mask, 'Crisis'] = True
            
            crisis_obs = mask.sum()
            print(f"   {crisis_name}: {crisis_obs} months")
        
        # Summary
        normal_obs = (portfolios['Period'] == 'Normal').sum()
        crisis_obs = portfolios['Crisis'].sum()
        print(f"   Normal periods: {normal_obs} months")
        print(f"   Crisis periods: {crisis_obs} months total")
        
        return portfolios
    
    def calculate_rolling_statistics(self, window=12):
        """Calculate rolling statistics for portfolios"""
        print_subsection_header(f"Calculating {window}-Month Rolling Statistics")
        
        rolling_stats = pd.DataFrame(index=self.portfolios.index)
        
        for col in ['ETF_Portfolio', 'Benchmark_Portfolio']:
            # Rolling mean (annualized)
            rolling_stats[f'{col}_Rolling_Mean'] = \
                self.portfolios[col].rolling(window).mean() * 12
            
            # Rolling volatility (annualized)
            rolling_stats[f'{col}_Rolling_Vol'] = \
                self.portfolios[col].rolling(window).std() * np.sqrt(12)
            
            # Rolling Sharpe ratio
            rolling_stats[f'{col}_Rolling_Sharpe'] = \
                rolling_stats[f'{col}_Rolling_Mean'] / rolling_stats[f'{col}_Rolling_Vol']
        
        return rolling_stats
    
    def calculate_tracking_error(self):
        """Calculate tracking error of ETF portfolio vs benchmark"""
        print_subsection_header("Calculating Tracking Error")
        
        # Calculate difference in returns
        tracking_diff = self.portfolios['ETF_Portfolio'] - self.portfolios['Benchmark_Portfolio']
        
        # Tracking error (annualized)
        tracking_error = tracking_diff.std() * np.sqrt(12)
        
        # Information ratio
        info_ratio = (tracking_diff.mean() * 12) / tracking_error if tracking_error != 0 else np.nan
        
        results = {
            'Tracking_Error': tracking_error,
            'Information_Ratio': info_ratio,
            'Mean_Difference': tracking_diff.mean() * 12,  # Annualized
            'Hit_Rate': (tracking_diff > 0).mean()  # Percentage of months outperforming
        }
        
        print(f"   Tracking Error: {tracking_error:.2%} annualized")
        print(f"   Information Ratio: {info_ratio:.3f}")
        print(f"   Hit Rate: {results['Hit_Rate']:.1%}")
        
        return results
    
    def get_crisis_performance(self):
        """Extract performance during each crisis period"""
        crisis_performance = {}
        
        for period in self.portfolios['Period'].unique():
            period_data = self.portfolios[self.portfolios['Period'] == period]
            
            # Calculate cumulative returns
            etf_cum = (1 + period_data['ETF_Portfolio']).prod() - 1
            bench_cum = (1 + period_data['Benchmark_Portfolio']).prod() - 1
            
            crisis_performance[period] = {
                'ETF_Cumulative': etf_cum,
                'Benchmark_Cumulative': bench_cum,
                'Relative_Performance': etf_cum - bench_cum,
                'Months': len(period_data)
            }
        
        return pd.DataFrame(crisis_performance).T

# Standalone functions for importing
def create_portfolios(etf_returns, benchmark_returns):
    """Convenience function to create portfolios"""
    constructor = PortfolioConstructor(etf_returns, benchmark_returns)
    return constructor.create_equal_weighted_portfolios()