# 5_statistical_tests.py - Simplified statistical testing functions
# ==============================================================

import pandas as pd
import numpy as np
from scipy import stats
from utils import print_section_header, print_subsection_header

class StatisticalTester:
    """Class to perform simplified statistical tests"""
    
    def __init__(self, data1, data2=None, labels=('Group 1', 'Group 2')):
        self.data1 = data1
        self.data2 = data2
        self.labels = labels
        self.results = {}
        
    def run_simple_tests(self):
        """Run only essential tests"""
        print_section_header("SIMPLIFIED STATISTICAL TESTS")
        if self.data2 is None:
            print("   Need two samples for comparison")
            return None
        data1_clean = self.data1.dropna()
        data2_clean = self.data2.dropna()
        print(f"   Sample sizes: {len(data1_clean)} vs {len(data2_clean)}")
        
        # T-test for mean difference
        t_stat, p_value = stats.ttest_ind(data1_clean, data2_clean)
        
        # F-test for variance difference (Levene's test is more robust)
        var1 = np.var(data1_clean, ddof=1)
        var2 = np.var(data2_clean, ddof=1)
        f_stat = var1 / var2
        df1 = len(data1_clean) - 1
        df2 = len(data2_clean) - 1
        # Two-tailed F-test p-value
        f_p_value = 2 * min(stats.f.cdf(f_stat, df1, df2), 1 - stats.f.cdf(f_stat, df1, df2))
        
        # Correlation
        correlation = np.corrcoef(data1_clean, data2_clean)[0,1]
        
        # Mann-Whitney U test (non-parametric)
        try:
            mw_stat, mw_p_value = stats.mannwhitneyu(data1_clean, data2_clean, alternative='two-sided')
        except:
            mw_stat, mw_p_value = np.nan, np.nan
        
        results = {
            't_test': {'statistic': t_stat, 'p_value': p_value},
            'f_test': {'statistic': f_stat, 'p_value': f_p_value},
            'correlation': correlation,
            'mann_whitney': {'statistic': mw_stat, 'p_value': mw_p_value}
        }
        
        print(f"   T-test: t={t_stat:.4f}, p={p_value:.4f}")
        print(f"   F-test (variance): F={f_stat:.4f}, p={f_p_value:.4f}")
        print(f"   Correlation: {correlation:.4f}")
        if not np.isnan(mw_stat):
            print(f"   Mann-Whitney: U={mw_stat:.4f}, p={mw_p_value:.4f}")
        else:
            print("   Mann-Whitney: Could not compute")
        
        return results
    
    def run_all_tests(self):
        """Alias for simplified tests - keeping for compatibility"""
        return self.run_simple_tests()


# Standalone function for importing
def run_simple_comparison_tests(data1, data2, labels=('Group 1', 'Group 2')):
    """Convenience function to run simple statistical tests"""
    tester = StatisticalTester(data1, data2, labels)
    return tester.run_simple_tests()