# 5_statistical_tests.py - Statistical testing functions
# ====================================================

import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox
from utils import print_section_header, print_subsection_header

class StatisticalTester:
    """Class to perform various statistical tests"""
    
    def __init__(self, data1, data2=None, labels=('Group 1', 'Group 2')):
        self.data1 = data1
        self.data2 = data2
        self.labels = labels
        self.results = {}
        
    def run_all_tests(self):
        """Run comprehensive statistical tests"""
        print_section_header("STATISTICAL TESTS")
        
        if self.data2 is not None:
            # Two-sample tests
            self._run_two_sample_tests()
        else:
            # Single sample tests
            self._run_single_sample_tests()
        
        return self.results
    
    def _run_two_sample_tests(self):
        """Run tests comparing two samples"""
        print_subsection_header(f"Comparing {self.labels[0]} vs {self.labels[1]}")
        
        # Clean data
        data1_clean = self.data1.dropna()
        data2_clean = self.data2.dropna()
        
        print(f"   Sample sizes: {len(data1_clean)} vs {len(data2_clean)}")
        
        # 1. Parametric Tests
        self.results['parametric'] = self._parametric_tests(data1_clean, data2_clean)
        
        # 2. Non-parametric Tests
        self.results['nonparametric'] = self._nonparametric_tests(data1_clean, data2_clean)
        
        # 3. Variance Tests
        self.results['variance'] = self._variance_tests(data1_clean, data2_clean)
        
        # 4. Distribution Tests
        self.results['distribution'] = self._distribution_tests(data1_clean, data2_clean)
        
        # Print summary
        self._print_test_summary()
    
    def _parametric_tests(self, data1, data2):
        """Run parametric statistical tests"""
        print("\n   Parametric Tests:")
        results = {}
        
        # T-test for means (independent samples)
        t_stat, p_value = stats.ttest_ind(data1, data2)
        results['t_test'] = {
            'statistic': t_stat,
            'p_value': p_value,
            'mean_diff': data1.mean() - data2.mean(),
            'significant_5%': p_value < 0.05
        }
        print(f"      T-test: t={t_stat:.4f}, p={p_value:.4f}")
        
        # Welch's t-test (unequal variances)
        t_stat_welch, p_value_welch = stats.ttest_ind(data1, data2, equal_var=False)
        results['welch_t_test'] = {
            'statistic': t_stat_welch,
            'p_value': p_value_welch,
            'significant_5%': p_value_welch < 0.05
        }
        print(f"      Welch's t-test: t={t_stat_welch:.4f}, p={p_value_welch:.4f}")
        
        return results
    
    def _nonparametric_tests(self, data1, data2):
        """Run non-parametric statistical tests"""
        print("\n   Non-parametric Tests:")
        results = {}
        
        # Mann-Whitney U test
        u_stat, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
        results['mann_whitney'] = {
            'statistic': u_stat,
            'p_value': p_value,
            'median_diff': np.median(data1) - np.median(data2),
            'significant_5%': p_value < 0.05
        }
        print(f"      Mann-Whitney U: U={u_stat:.4f}, p={p_value:.4f}")
        
        # Wilcoxon rank-sum test (equivalent to Mann-Whitney)
        w_stat, p_value_w = stats.ranksums(data1, data2)
        results['wilcoxon_ranksum'] = {
            'statistic': w_stat,
            'p_value': p_value_w,
            'significant_5%': p_value_w < 0.05
        }
        
        # Kruskal-Wallis test
        h_stat, p_value_kw = stats.kruskal(data1, data2)
        results['kruskal_wallis'] = {
            'statistic': h_stat,
            'p_value': p_value_kw,
            'significant_5%': p_value_kw < 0.05
        }
        print(f"      Kruskal-Wallis: H={h_stat:.4f}, p={p_value_kw:.4f}")
        
        return results
    
    def _variance_tests(self, data1, data2):
        """Test for equality of variances"""
        print("\n   Variance Tests:")
        results = {}
        
        # F-test for equality of variances
        var1 = np.var(data1, ddof=1)
        var2 = np.var(data2, ddof=1)
        f_stat = var1 / var2
        dfn = len(data1) - 1
        dfd = len(data2) - 1
        p_value = 2 * min(stats.f.cdf(f_stat, dfn, dfd), 1 - stats.f.cdf(f_stat, dfn, dfd))
        
        results['f_test'] = {
            'statistic': f_stat,
            'p_value': p_value,
            'var1': var1,
            'var2': var2,
            'significant_5%': p_value < 0.05
        }
        print(f"      F-test: F={f_stat:.4f}, p={p_value:.4f}")
        
        # Levene's test (more robust)
        lev_stat, p_value_lev = stats.levene(data1, data2)
        results['levene_test'] = {
            'statistic': lev_stat,
            'p_value': p_value_lev,
            'significant_5%': p_value_lev < 0.05
        }
        print(f"      Levene's test: W={lev_stat:.4f}, p={p_value_lev:.4f}")
        
        # Bartlett's test
        bart_stat, p_value_bart = stats.bartlett(data1, data2)
        results['bartlett_test'] = {
            'statistic': bart_stat,
            'p_value': p_value_bart,
            'significant_5%': p_value_bart < 0.05
        }
        
        return results
    
    def _distribution_tests(self, data1, data2):
        """Test for equality of distributions"""
        print("\n   Distribution Tests:")
        results = {}
        
        # Kolmogorov-Smirnov test
        ks_stat, p_value = stats.ks_2samp(data1, data2)
        results['kolmogorov_smirnov'] = {
            'statistic': ks_stat,
            'p_value': p_value,
            'significant_5%': p_value < 0.05
        }
        print(f"      Kolmogorov-Smirnov: D={ks_stat:.4f}, p={p_value:.4f}")
        
        # Anderson-Darling test
        try:
            ad_result = stats.anderson_ksamp([data1.values, data2.values])
            results['anderson_darling'] = {
                'statistic': ad_result.statistic,
                'p_value': ad_result.pvalue,
                'significant_5%': ad_result.pvalue < 0.05
            }
            print(f"      Anderson-Darling: A²={ad_result.statistic:.4f}, p={ad_result.pvalue:.4f}")
        except:
            print("      Anderson-Darling: Not available")
        
        return results
    
    def _run_single_sample_tests(self):
        """Run tests on a single sample"""
        print_subsection_header(f"Testing {self.labels[0]}")
        
        data_clean = self.data1.dropna()
        
        # Normality tests
        self.results['normality'] = self._normality_tests(data_clean)
        
        # Autocorrelation tests
        self.results['autocorrelation'] = self._autocorrelation_tests(data_clean)
        
        # Stationarity tests
        self.results['stationarity'] = self._stationarity_tests(data_clean)
    
    def _normality_tests(self, data):
        """Test for normality"""
        print("\n   Normality Tests:")
        results = {}
        
        # Shapiro-Wilk test
        if len(data) <= 5000:  # Shapiro-Wilk has sample size limit
            stat, p_value = stats.shapiro(data)
            results['shapiro_wilk'] = {
                'statistic': stat,
                'p_value': p_value,
                'normal_5%': p_value > 0.05
            }
            print(f"      Shapiro-Wilk: W={stat:.4f}, p={p_value:.4f}")
        
        # Jarque-Bera test
        jb_stat, p_value = stats.jarque_bera(data)
        results['jarque_bera'] = {
            'statistic': jb_stat,
            'p_value': p_value,
            'normal_5%': p_value > 0.05
        }
        print(f"      Jarque-Bera: JB={jb_stat:.4f}, p={p_value:.4f}")
        
        # D'Agostino and Pearson's test
        k2_stat, p_value = stats.normaltest(data)
        results['dagostino_pearson'] = {
            'statistic': k2_stat,
            'p_value': p_value,
            'normal_5%': p_value > 0.05
        }
        print(f"      D'Agostino-Pearson: K²={k2_stat:.4f}, p={p_value:.4f}")
        
        return results
    
    def _autocorrelation_tests(self, data):
        """Test for autocorrelation"""
        print("\n   Autocorrelation Tests:")
        results = {}
        
        # Ljung-Box test
        lb_result = acorr_ljungbox(data, lags=10, return_df=True)
        results['ljung_box'] = {
            'statistics': lb_result['lb_stat'].values,
            'p_values': lb_result['lb_pvalue'].values,
            'significant_any': (lb_result['lb_pvalue'] < 0.05).any()
        }
        print(f"      Ljung-Box (lag 1): Q={lb_result['lb_stat'].iloc[0]:.4f}, p={lb_result['lb_pvalue'].iloc[0]:.4f}")
        
        return results
    
    def _stationarity_tests(self, data):
        """Test for stationarity"""
        print("\n   Stationarity Tests:")
        results = {}
        
        try:
            from statsmodels.tsa.stattools import adfuller, kpss
            
            # Augmented Dickey-Fuller test
            adf_result = adfuller(data, autolag='AIC')
            results['adf'] = {
                'statistic': adf_result[0],
                'p_value': adf_result[1],
                'critical_values': adf_result[4],
                'stationary_5%': adf_result[1] < 0.05
            }
            print(f"      ADF test: t={adf_result[0]:.4f}, p={adf_result[1]:.4f}")
            
            # KPSS test
            kpss_result = kpss(data, regression='c', nlags='auto')
            results['kpss'] = {
                'statistic': kpss_result[0],
                'p_value': kpss_result[1],
                'critical_values': kpss_result[3],
                'stationary_5%': kpss_result[1] > 0.05  # Note: reversed for KPSS
            }
            print(f"      KPSS test: LM={kpss_result[0]:.4f}, p={kpss_result[1]:.4f}")
            
        except ImportError:
            print("      Stationarity tests require statsmodels.tsa")
        
        return results
    
    def _print_test_summary(self):
        """Print summary of test results"""
        print("\n" + "="*50)
        print("TEST SUMMARY")
        print("="*50)
        
        if 'parametric' in self.results:
            print("\nMean Difference Tests:")
            t_test = self.results['parametric']['t_test']
            print(f"   Mean difference: {t_test['mean_diff']:.4f}")
            print(f"   T-test p-value: {t_test['p_value']:.4f} {'*' if t_test['significant_5%'] else ''}")
            
            mw_test = self.results['nonparametric']['mann_whitney']
            print(f"   Mann-Whitney p-value: {mw_test['p_value']:.4f} {'*' if mw_test['significant_5%'] else ''}")
        
        if 'variance' in self.results:
            print("\nVariance Equality Tests:")
            f_test = self.results['variance']['f_test']
            print(f"   Variance ratio: {f_test['statistic']:.4f}")
            print(f"   F-test p-value: {f_test['p_value']:.4f} {'*' if f_test['significant_5%'] else ''}")
        
        if 'distribution' in self.results:
            print("\nDistribution Tests:")
            ks_test = self.results['distribution']['kolmogorov_smirnov']
            print(f"   KS test p-value: {ks_test['p_value']:.4f} {'*' if ks_test['significant_5%'] else ''}")
        
        print("\n* indicates significance at 5% level")


class CrisisPeriodTester:
    """Specialized tests for crisis vs normal periods"""
    
    def __init__(self, portfolio_data):
        self.portfolio_data = portfolio_data
        self.results = {}
        
    def test_crisis_performance(self):
        """Test performance differences across crisis periods"""
        print_section_header("CRISIS PERIOD ANALYSIS")
        
        # Separate crisis and normal periods
        normal_data = self.portfolio_data[self.portfolio_data['Crisis'] == False]
        crisis_data = self.portfolio_data[self.portfolio_data['Crisis'] == True]
        
        # Test each portfolio
        for portfolio in ['ETF_Portfolio', 'Benchmark_Portfolio']:
            print_subsection_header(f"Crisis Analysis: {portfolio}")
            
            normal_returns = normal_data[portfolio]
            crisis_returns = crisis_data[portfolio]
            
            # Run statistical tests
            tester = StatisticalTester(
                crisis_returns, 
                normal_returns,
                labels=('Crisis', 'Normal')
            )
            
            self.results[portfolio] = tester.run_all_tests()
            
            # Additional crisis-specific metrics
            self._calculate_crisis_metrics(portfolio, normal_returns, crisis_returns)
        
        # Compare portfolios during crisis
        self._compare_crisis_resilience()
        
        return self.results
    
    def _calculate_crisis_metrics(self, portfolio, normal_returns, crisis_returns):
        """Calculate crisis-specific performance metrics"""
        
        metrics = {
            'normal_mean': normal_returns.mean() * 100,
            'crisis_mean': crisis_returns.mean() * 100,
            'performance_diff': (crisis_returns.mean() - normal_returns.mean()) * 100,
            'normal_vol': normal_returns.std() * np.sqrt(12) * 100,
            'crisis_vol': crisis_returns.std() * np.sqrt(12) * 100,
            'vol_increase': (crisis_returns.std() / normal_returns.std() - 1) * 100,
            'worst_month': crisis_returns.min() * 100,
            'best_month': crisis_returns.max() * 100
        }
        
        print(f"\n   Crisis Impact:")
        print(f"      Normal period return: {metrics['normal_mean']:.2f}% monthly")
        print(f"      Crisis period return: {metrics['crisis_mean']:.2f}% monthly")
        print(f"      Performance impact: {metrics['performance_diff']:.2f}%")
        print(f"      Volatility increase: {metrics['vol_increase']:.1f}%")
        
        self.results[portfolio]['crisis_metrics'] = metrics
    
    def _compare_crisis_resilience(self):
        """Compare crisis resilience between portfolios"""
        print_subsection_header("Crisis Resilience Comparison")
        
        crisis_data = self.portfolio_data[self.portfolio_data['Crisis'] == True]
        
        # Test if ETF portfolio outperforms during crisis
        etf_crisis = crisis_data['ETF_Portfolio']
        bench_crisis = crisis_data['Benchmark_Portfolio']
        
        # Paired t-test (same time periods)
        diff = etf_crisis - bench_crisis
        t_stat, p_value = stats.ttest_1samp(diff, 0)
        
        results = {
            'mean_outperformance': diff.mean() * 100,
            'outperformance_vol': diff.std() * 100,
            't_statistic': t_stat,
            'p_value': p_value,
            'months_outperformed': (diff > 0).sum(),
            'total_months': len(diff),
            'hit_rate': (diff > 0).mean() * 100
        }
        
        print(f"   Mean outperformance during crisis: {results['mean_outperformance']:.2f}% monthly")
        print(f"   Hit rate: {results['hit_rate']:.1f}%")
        print(f"   Paired t-test: t={t_stat:.3f}, p={p_value:.4f}")
        
        self.results['crisis_resilience'] = results


# Standalone functions for easy import
def perform_statistical_tests(data1, data2, labels=('ETF Portfolio', 'Benchmark Portfolio')):
    """Run comprehensive statistical tests"""
    tester = StatisticalTester(data1, data2, labels)
    return tester.run_all_tests()

def test_crisis_performance(portfolio_data):
    """Test crisis period performance"""
    tester = CrisisPeriodTester(portfolio_data)
    return tester.test_crisis_performance()