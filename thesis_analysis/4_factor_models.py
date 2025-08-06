# 4_factor_models.py - Factor model analysis (CAPM, FF3, Carhart)
# ==============================================================

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import jarque_bera, durbin_watson
from config import RISK_FREE_RATE
from utils import print_section_header, print_subsection_header

class FactorModelAnalysis:
    """Class to run factor model regressions"""
    
    def __init__(self, portfolio_returns, factor_data):
        self.portfolio_returns = portfolio_returns
        self.factor_data = factor_data
        self.results = {}
        
    def run_all_models(self):
        """Run CAPM, FF3, and Carhart models for all portfolios"""
        print_section_header("FACTOR MODEL ANALYSIS")
        
        # Check which portfolios we have
        if isinstance(self.portfolio_returns, pd.DataFrame):
            portfolios = [col for col in self.portfolio_returns.columns 
                         if col in ['ETF_Portfolio', 'Benchmark_Portfolio']]
        else:
            portfolios = ['Portfolio']
            self.portfolio_returns = self.portfolio_returns.to_frame('Portfolio')
        
        # Run models for each portfolio
        for portfolio in portfolios:
            print_subsection_header(f"Factor Models for {portfolio}")
            self.results[portfolio] = self._run_portfolio_models(portfolio)
        
        # Create comparison summary
        self.summary = self._create_summary()
        
        return self.results, self.summary
    
    def _run_portfolio_models(self, portfolio_name):
        """Run all factor models for a single portfolio"""
        
        # Prepare data
        analysis_data = self._prepare_regression_data(portfolio_name)
        
        if analysis_data is None or len(analysis_data) < 30:
            print(f"   Insufficient data for {portfolio_name}")
            return None
        
        results = {}
        
        # 1. CAPM Model
        results['CAPM'] = self._run_capm(analysis_data)
        
        # 2. Fama-French 3-Factor Model
        results['FF3'] = self._run_ff3(analysis_data)
        
        # 3. Carhart 4-Factor Model (if momentum available)
        if 'Mom' in analysis_data.columns:
            results['Carhart'] = self._run_carhart(analysis_data)
        else:
            print("   Momentum factor not available - skipping Carhart model")
        
        # Add diagnostics
        for model_name, model_results in results.items():
            if model_results is not None:
                self._add_diagnostics(model_results)
        
        return results
    
    def _prepare_regression_data(self, portfolio_name):
        """Prepare data for regression analysis"""
        
        # Get portfolio returns
        portfolio_ret = self.portfolio_returns[portfolio_name].dropna()
        
        # Merge with factor data
        data = pd.DataFrame({
            'Portfolio_Return': portfolio_ret,
            'Date': portfolio_ret.index
        }).set_index('Date')
        
        # Merge with factors
        data = pd.merge(data, self.factor_data, left_index=True, right_index=True, how='inner')
        
        # Calculate excess returns
        if 'RF' in data.columns:
            data['Excess_Return'] = data['Portfolio_Return'] - data['RF']
        else:
            # Use configured risk-free rate
            data['Excess_Return'] = data['Portfolio_Return'] - (RISK_FREE_RATE / 12)
        
        print(f"   Observations for regression: {len(data)}")
        
        return data
    
    def _run_capm(self, data):
        """Run CAPM regression"""
        print("   Running CAPM...")
        
        # Prepare variables
        y = data['Excess_Return']
        X = sm.add_constant(data['Mkt-RF'])
        
        # Run regression
        model = sm.OLS(y, X).fit()
        
        # Extract results
        results = {
            'model': model,
            'alpha': model.params['const'],
            'beta': model.params['Mkt-RF'],
            'alpha_tstat': model.tvalues['const'],
            'beta_tstat': model.tvalues['Mkt-RF'],
            'alpha_pvalue': model.pvalues['const'],
            'beta_pvalue': model.pvalues['Mkt-RF'],
            'r_squared': model.rsquared,
            'adj_r_squared': model.rsquared_adj,
            'observations': model.nobs
        }
        
        # Calculate Treynor ratio
        mean_excess = data['Excess_Return'].mean() * 12
        results['treynor_ratio'] = mean_excess / results['beta'] if results['beta'] != 0 else np.nan
        
        # Print summary
        print(f"      Alpha: {results['alpha']:.4f} ({results['alpha']*12*100:.2f}% annualized)")
        print(f"      Beta:  {results['beta']:.4f}")
        print(f"      R²:    {results['r_squared']:.4f}")
        
        return results
    
    def _run_ff3(self, data):
        """Run Fama-French 3-Factor regression"""
        print("   Running Fama-French 3-Factor Model...")
        
        # Check if we have all factors
        required_factors = ['Mkt-RF', 'SMB', 'HML']
        if not all(factor in data.columns for factor in required_factors):
            print("      Missing required factors for FF3")
            return None
        
        # Prepare variables
        y = data['Excess_Return']
        X = sm.add_constant(data[required_factors])
        
        # Run regression
        model = sm.OLS(y, X).fit()
        
        # Extract results
        results = {
            'model': model,
            'alpha': model.params['const'],
            'beta_mkt': model.params['Mkt-RF'],
            'beta_smb': model.params['SMB'],
            'beta_hml': model.params['HML'],
            'alpha_pvalue': model.pvalues['const'],
            'r_squared': model.rsquared,
            'adj_r_squared': model.rsquared_adj,
            'observations': model.nobs
        }
        
        # Store all coefficients and p-values
        for factor in required_factors:
            results[f'{factor}_coef'] = model.params[factor]
            results[f'{factor}_pvalue'] = model.pvalues[factor]
            results[f'{factor}_tstat'] = model.tvalues[factor]
        
        print(f"      Alpha: {results['alpha']:.4f} ({results['alpha']*12*100:.2f}% annualized)")
        print(f"      Market Beta: {results['beta_mkt']:.4f}")
        print(f"      SMB Beta: {results['beta_smb']:.4f}")
        print(f"      HML Beta: {results['beta_hml']:.4f}")
        print(f"      Adj R²: {results['adj_r_squared']:.4f}")
        
        return results
    
    def _run_carhart(self, data):
        """Run Carhart 4-Factor regression"""
        print("   Running Carhart 4-Factor Model...")
        
        # Check if we have all factors
        required_factors = ['Mkt-RF', 'SMB', 'HML', 'Mom']
        if not all(factor in data.columns for factor in required_factors):
            print("      Missing required factors for Carhart")
            return None
        
        # Prepare variables
        y = data['Excess_Return']
        X = sm.add_constant(data[required_factors])
        
        # Run regression
        model = sm.OLS(y, X).fit()
        
        # Extract results
        results = {
            'model': model,
            'alpha': model.params['const'],
            'beta_mkt': model.params['Mkt-RF'],
            'beta_smb': model.params['SMB'],
            'beta_hml': model.params['HML'],
            'beta_mom': model.params['Mom'],
            'alpha_pvalue': model.pvalues['const'],
            'r_squared': model.rsquared,
            'adj_r_squared': model.rsquared_adj,
            'observations': model.nobs
        }
        
        # Store all coefficients and p-values
        for factor in required_factors:
            results[f'{factor}_coef'] = model.params[factor]
            results[f'{factor}_pvalue'] = model.pvalues[factor]
            results[f'{factor}_tstat'] = model.tvalues[factor]
        
        print(f"      Alpha: {results['alpha']:.4f} ({results['alpha']*12*100:.2f}% annualized)")
        print(f"      Market Beta: {results['beta_mkt']:.4f}")
        print(f"      Mom Beta: {results['beta_mom']:.4f}")
        print(f"      Adj R²: {results['adj_r_squared']:.4f}")
        
        return results
    
    def _add_diagnostics(self, results):
        """Add regression diagnostics"""
        model = results['model']
        
        # Durbin-Watson test for autocorrelation
        results['durbin_watson'] = durbin_watson(model.resid)
        
        # Jarque-Bera test for normality
        jb_stat, jb_pvalue, _, _ = jarque_bera(model.resid)
        results['jarque_bera_stat'] = jb_stat
        results['jarque_bera_pvalue'] = jb_pvalue
        
        # Heteroskedasticity test (Breusch-Pagan)
        try:
            bp_stat, bp_pvalue, _, _ = het_breuschpagan(model.resid, model.model.exog)
            results['breusch_pagan_stat'] = bp_stat
            results['breusch_pagan_pvalue'] = bp_pvalue
        except:
            results['breusch_pagan_stat'] = np.nan
            results['breusch_pagan_pvalue'] = np.nan
    
    def _create_summary(self):
        """Create summary of all model results"""
        summary_data = []
        
        for portfolio, models in self.results.items():
            if models is None:
                continue
                
            for model_name, model_results in models.items():
                if model_results is None:
                    continue
                
                summary_data.append({
                    'Portfolio': portfolio,
                    'Model': model_name,
                    'Alpha': model_results['alpha'],
                    'Alpha_Annual_%': model_results['alpha'] * 12 * 100,
                    'Alpha_p-value': model_results['alpha_pvalue'],
                    'Market_Beta': model_results.get('beta', model_results.get('beta_mkt')),
                    'R-squared': model_results['r_squared'],
                    'Adj_R-squared': model_results['adj_r_squared'],
                    'Observations': model_results['observations']
                })
        
        return pd.DataFrame(summary_data)
    
    def test_alpha_difference(self, model='Carhart'):
        """Test if alpha difference between portfolios is significant"""
        print_subsection_header(f"Testing Alpha Difference ({model})")
        
        if 'ETF_Portfolio' not in self.results or 'Benchmark_Portfolio' not in self.results:
            print("   Both portfolios needed for comparison")
            return None
        
        etf_results = self.results['ETF_Portfolio'].get(model)
        bench_results = self.results['Benchmark_Portfolio'].get(model)
        
        if etf_results is None or bench_results is None:
            print(f"   {model} results not available for both portfolios")
            return None
        
        # Get alphas and standard errors
        etf_alpha = etf_results['alpha']
        bench_alpha = bench_results['alpha']
        etf_se = etf_results['model'].bse['const']
        bench_se = bench_results['model'].bse['const']
        
        # Calculate difference and test statistic
        alpha_diff = etf_alpha - bench_alpha
        se_diff = np.sqrt(etf_se**2 + bench_se**2)
        t_stat = alpha_diff / se_diff if se_diff != 0 else np.nan
        
        # Two-tailed p-value
        from scipy import stats
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=min(etf_results['observations'], 
                                                            bench_results['observations']) - 1))
        
        results = {
            'ETF_Alpha': etf_alpha,
            'Benchmark_Alpha': bench_alpha,
            'Alpha_Difference': alpha_diff,
            'Alpha_Diff_Annual_%': alpha_diff * 12 * 100,
            'SE_Difference': se_diff,
            'T_Statistic': t_stat,
            'P_Value': p_value,
            'Significant_5%': p_value < 0.05,
            'Significant_10%': p_value < 0.10
        }
        
        print(f"   Alpha Difference: {results['Alpha_Diff_Annual_%']:.2f}% annualized")
        print(f"   T-statistic: {results['T_Statistic']:.3f}")
        print(f"   P-value: {results['P_Value']:.4f}")
        print(f"   Significant at 5%: {'Yes' if results['Significant_5%'] else 'No'}")
        
        return results


# Standalone function for easy import
def run_factor_models(portfolio_returns, factor_data):
    """Run all factor models"""
    analysis = FactorModelAnalysis(portfolio_returns, factor_data)
    return analysis.run_all_models()