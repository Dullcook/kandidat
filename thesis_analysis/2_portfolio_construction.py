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
        etf_portfolio.name = 'ETF_Portfolio'
        
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
    
    def analyze_portfolio_composition(self):
        """Analyze what's in each portfolio"""
        print_subsection_header("Portfolio Composition Analysis")
        
        # Count ETF types
        etf_types = {
            'ESG-Focused': ['Xtrackers MSCI Europe ESG ETF 1C', 
                           'iShares MSCI Europe SRI ETF EUR Dist'],
            'Sector ETFs': [etf for etf in self.etf_returns.columns 
                           if etf not in ['Xtrackers MSCI Europe ESG ETF 1C', 
                                         'iShares MSCI Europe SRI ETF EUR Dist']]
        }
        
        # Calculate average returns by type
        esg_focused_returns = self.etf_returns[etf_types['ESG-Focused']].mean(axis=1)
        sector_returns = self.etf_returns[etf_types['Sector ETFs']].mean(axis=1)
        
        print(f"   ESG-Focused ETFs: {len(etf_types['ESG-Focused'])}")
        print(f"   Sector ETFs: {len(etf_types['Sector ETFs'])}")
        print(f"   Average return - ESG: {esg_focused_returns.mean()*12:.2%}")
        print(f"   Average return - Sector: {sector_returns.mean()*12:.2%}")
        
        # Calculate correlation between ESG and sector components
        correlation = esg_focused_returns.corr(sector_returns)
        print(f"   Correlation ESG vs Sector: {correlation:.3f}")
        
        return etf_types
    
    def analyze_period_details(self):
        """Deep dive into period performance"""
        print_subsection_header("Period Performance Analysis")
        
        for period in self.portfolios['Period'].unique():
            period_data = self.portfolios[self.portfolios['Period'] == period]
            
            # Monthly breakdown
            monthly_diff = period_data['ETF_Portfolio'] - period_data['Benchmark_Portfolio']
            
            print(f"\n{period}:")
            print(f"  Months outperforming: {(monthly_diff > 0).sum()}/{len(monthly_diff)}")
            print(f"  Average monthly difference: {monthly_diff.mean()*100:.3f}%")
            print(f"  Best month: {monthly_diff.max()*100:.2f}%")
            print(f"  Worst month: {monthly_diff.min()*100:.2f}%")
            
            # Calculate cumulative performance for the period
            etf_cum = (1 + period_data['ETF_Portfolio']).prod() - 1
            bench_cum = (1 + period_data['Benchmark_Portfolio']).prod() - 1
            print(f"  Cumulative ETF: {etf_cum*100:.2f}%")
            print(f"  Cumulative Benchmark: {bench_cum*100:.2f}%")
            print(f"  Relative: {(etf_cum - bench_cum)*100:.2f}%")
        
        return True
    
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