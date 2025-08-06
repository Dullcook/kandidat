# 1_data_loader.py - Data loading and preparation functions
# ========================================================

import pandas as pd
import numpy as np
from config import *
from utils import print_section_header, print_subsection_header, convert_to_decimal

class DataLoader:
    """Class to handle all data loading operations"""
    
    def __init__(self):
        self.etf_data = None
        self.benchmark_data = None
        self.factor_data = None
        
    def load_all_data(self):
        """Load all required data files"""
        print_section_header("DATA LOADING AND PREPARATION")
        
        # Load each dataset
        self.etf_data = self._load_etf_data()
        self.benchmark_data = self._load_benchmark_data()
        self.factor_data = self._load_factor_data()
        
        # Validate data
        self._validate_data()
        
        return self.etf_data, self.benchmark_data, self.factor_data
    
    def _load_etf_data(self):
        """Load and process ETF return data"""
        print_subsection_header("Loading ETF Data")
        
        # Load wide format data
        df = pd.read_excel(DATA_PATH + ETF_FILE, index_col=0)
        print(f"   Loaded {df.shape[0]} ETFs with {df.shape[1]} monthly observations")
        
        # List all ETFs
        print("   ETFs in dataset:")
        for i, etf in enumerate(df.index, 1):
            print(f"   {i}. {etf}")
        
        # Transpose to long format (dates as index)
        df_long = df.T
        df_long.index = pd.to_datetime(df_long.index)
        df_long = df_long.sort_index()
        
        # Convert to decimal if needed
        df_long = convert_to_decimal(df_long)
        
        # Filter to analysis period
        df_long = df_long.loc[START_DATE:END_DATE]
        
        print(f"   Final shape: {df_long.shape}")
        print(f"   Date range: {df_long.index[0].strftime('%Y-%m-%d')} to {df_long.index[-1].strftime('%Y-%m-%d')}")
        
        return df_long
    
    def _load_benchmark_data(self):
        """Load and process benchmark return data"""
        print_subsection_header("Loading Benchmark Data")
        
        # Load wide format data
        df = pd.read_excel(DATA_PATH + BENCHMARK_FILE, index_col=0)
        print(f"   Loaded {df.shape[0]} benchmarks with {df.shape[1]} monthly observations")
        
        # Transpose to long format
        df_long = df.T
        df_long.index = pd.to_datetime(df_long.index)
        df_long = df_long.sort_index()
        
        # Convert to decimal if needed
        df_long = convert_to_decimal(df_long)
        
        # Filter to analysis period
        df_long = df_long.loc[START_DATE:END_DATE]
        
        print(f"   Final shape: {df_long.shape}")
        
        # Verify benchmark mapping
        self._verify_benchmark_mapping(df_long.columns)
        
        return df_long
    
    def _load_factor_data(self):
        """Load and process factor data"""
        print_subsection_header("Loading Factor Data")
        
        # Load wide format data
        df = pd.read_excel(DATA_PATH + FACTOR_FILE, index_col=0)
        print(f"   Loaded factors: {list(df.index)}")
        
        # Transpose to long format
        df_long = df.T
        df_long.index = pd.to_datetime(df_long.index)
        df_long = df_long.sort_index()
        
        # Standardize column names
        df_long.columns = df_long.columns.str.strip()
        
        # Check for momentum factor
        mom_cols = ['Mom', 'MOM', 'WML']
        mom_found = False
        for col in mom_cols:
            if col in df_long.columns:
                if col != 'Mom':
                    df_long = df_long.rename(columns={col: 'Mom'})
                mom_found = True
                break
        
        if not mom_found:
            print("   WARNING: Momentum factor not found. Only 3-factor models will be available.")
        
        # Clean European number formats first
        df_long = convert_to_decimal(df_long, force=False)
        
        # Special handling for risk-free rate
        if 'RF' in df_long.columns:
            rf_mean = df_long['RF'].mean()
            if rf_mean > 0.05:  # If RF > 5% annually, it's likely wrong
                print(f"   WARNING: RF rate seems too high ({rf_mean:.4f}), adjusting...")
                # Assume it should be around 2% annually = 0.17% monthly
                df_long['RF'] = 0.02 / 12  # Set to 2% annually
        
        # Convert other factors to decimal if needed
        for col in df_long.columns:
            if col != 'RF' and abs(df_long[col].mean()) > 1:  # Likely percentage
                print(f"   Converting {col} from percentage to decimal")
                df_long[col] = df_long[col] / 100
        
        # Filter to analysis period
        df_long = df_long.loc[START_DATE:END_DATE]
        
        print(f"   Final shape: {df_long.shape}")
        print(f"   Available factors: {list(df_long.columns)}")
        
        return df_long
    
    def _verify_benchmark_mapping(self, benchmark_columns):
        """Verify that all ETF benchmarks are available"""
        print_subsection_header("Verifying ETF-Benchmark Mapping")
        
        missing_benchmarks = []
        for etf, benchmark in ETF_BENCHMARK_MAP.items():
            if benchmark not in benchmark_columns:
                missing_benchmarks.append((etf, benchmark))
        
        if missing_benchmarks:
            print("   WARNING: Missing benchmarks for:")
            for etf, benchmark in missing_benchmarks:
                print(f"   - {etf}: {benchmark}")
        else:
            print("   ✓ All benchmark mappings verified")
    
    def _validate_data(self):
        """Validate loaded data for completeness and consistency"""
        print_subsection_header("Data Validation")
        
        # Check for missing values
        etf_missing = self.etf_data.isnull().sum().sum()
        benchmark_missing = self.benchmark_data.isnull().sum().sum()
        factor_missing = self.factor_data.isnull().sum().sum()
        
        print(f"   Missing values - ETFs: {etf_missing}, Benchmarks: {benchmark_missing}, Factors: {factor_missing}")
        
        # Check date alignment
        etf_dates = set(self.etf_data.index)
        benchmark_dates = set(self.benchmark_data.index)
        factor_dates = set(self.factor_data.index)
        
        common_dates = etf_dates.intersection(benchmark_dates).intersection(factor_dates)
        print(f"   Common dates across all datasets: {len(common_dates)}")
        
        if len(common_dates) < len(etf_dates):
            print(f"   WARNING: Date mismatch. ETF dates: {len(etf_dates)}, Common: {len(common_dates)}")
    
    def get_summary_statistics(self):
        """Generate summary statistics for all loaded data"""
        print_subsection_header("Data Summary Statistics")
        
        # ETF statistics
        etf_stats = self.etf_data.describe()
        print("\nETF Returns Summary (in %):")
        print((etf_stats * 100).round(2).loc[['mean', 'std', 'min', 'max']])
        
        # Factor statistics
        factor_stats = self.factor_data.describe()
        print("\nFactor Summary (in decimal):")
        print(factor_stats.round(4).loc[['mean', 'std', 'min', 'max']])
        
        return etf_stats, factor_stats


# Standalone function for importing in other modules
def load_all_data():
    """Convenience function to load all data"""
    loader = DataLoader()
    return loader.load_all_data()