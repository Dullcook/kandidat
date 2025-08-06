# utils.py - Utility functions for ESG ETF analysis
# =================================================

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def check_data_format(df, data_type="returns"):
    """
    Check if data is in percentage or decimal format
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data to check
    data_type : str
        Type of data ('returns' or 'factors')
    
    Returns:
    --------
    bool : True if in percentage format, False if decimal
    """
    # Get a sample of non-zero values
    sample_values = df.values.flatten()
    # Convert to numeric and filter out NaN and zeros
    sample_values = pd.to_numeric(sample_values, errors='coerce')
    sample_values = sample_values[~np.isnan(sample_values) & (sample_values != 0)]
    
    if len(sample_values) == 0:
        return False
    
    # Check mean absolute value
    mean_abs = np.mean(np.abs(sample_values))
    
    # Returns/factors > 0.5 in absolute value are likely percentages
    return mean_abs > 0.5

def clean_european_numbers(value):
    """
    Clean European number format (comma decimals, non-breaking spaces)
    """
    if pd.isna(value) or isinstance(value, (int, float)):
        return value
    
    if isinstance(value, str):
        # Remove non-breaking spaces and regular spaces
        value = value.replace('\xa0', '').replace(' ', '')
        # Replace comma with dot for decimal
        value = value.replace(',', '.')
        try:
            return float(value)
        except ValueError:
            return np.nan
    
    return value

def convert_to_decimal(df, force=False):
    """
    Convert percentage data to decimal format if needed
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data to convert
    force : bool
        Force conversion regardless of detection
    
    Returns:
    --------
    pd.DataFrame : Converted data
    """
    # First clean European number formats
    print("   Cleaning European number formats...")
    df_cleaned = df.applymap(clean_european_numbers)
    
    # Convert to numeric
    df_numeric = df_cleaned.apply(pd.to_numeric, errors='coerce')
    
    if force or check_data_format(df_numeric):
        print("   Converting from percentage to decimal format...")
        return df_numeric / 100
    return df_numeric

def align_dates(*dataframes):
    """
    Align multiple dataframes to common date range
    
    Parameters:
    -----------
    *dataframes : pd.DataFrame
        Variable number of dataframes to align
    
    Returns:
    --------
    list : List of aligned dataframes
    """
    # Find common date range
    all_dates = [set(df.index) for df in dataframes]
    common_dates = set.intersection(*all_dates)
    common_dates = sorted(list(common_dates))
    
    print(f"   Aligning data to {len(common_dates)} common dates")
    print(f"   Date range: {common_dates[0].strftime('%Y-%m')} to {common_dates[-1].strftime('%Y-%m')}")
    
    # Align all dataframes
    aligned = []
    for df in dataframes:
        aligned.append(df.loc[common_dates])
    
    return aligned

def calculate_period_returns(returns, period_start, period_end):
    """
    Calculate cumulative return for a specific period
    
    Parameters:
    -----------
    returns : pd.Series or pd.DataFrame
        Return series
    period_start : str or datetime
        Start date
    period_end : str or datetime
        End date
    
    Returns:
    --------
    float or pd.Series : Cumulative return for the period
    """
    period_returns = returns.loc[period_start:period_end]
    return (1 + period_returns).prod() - 1

def create_summary_statistics(data, name=""):
    """
    Create comprehensive summary statistics
    
    Parameters:
    -----------
    data : pd.Series or pd.DataFrame
        Data to summarize
    name : str
        Name for the summary
    
    Returns:
    --------
    pd.DataFrame : Summary statistics
    """
    if isinstance(data, pd.DataFrame):
        return data.apply(lambda x: create_summary_statistics(x, x.name))
    
    stats = {
        'Count': len(data),
        'Mean': data.mean(),
        'Std Dev': data.std(),
        'Min': data.min(),
        '25%': data.quantile(0.25),
        'Median': data.median(),
        '75%': data.quantile(0.75),
        'Max': data.max(),
        'Skewness': data.skew(),
        'Kurtosis': data.kurtosis()
    }
    
    return pd.Series(stats, name=name)

def format_results_for_thesis(value, metric_type='return', decimals=4):
    """
    Format numerical results for thesis presentation
    
    Parameters:
    -----------
    value : float
        Value to format
    metric_type : str
        Type of metric ('return', 'ratio', 'percentage')
    decimals : int
        Number of decimal places
    
    Returns:
    --------
    str : Formatted string
    """
    if pd.isna(value):
        return "N/A"
    
    if metric_type == 'return' or metric_type == 'percentage':
        return f"{value:.{decimals-2}f}%"
    elif metric_type == 'ratio':
        return f"{value:.{decimals}f}"
    else:
        return f"{value:.{decimals}f}"

def export_to_latex(df, filename, caption="", label=""):
    """
    Export DataFrame to LaTeX format for thesis
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data to export
    filename : str
        Output filename
    caption : str
        Table caption
    label : str
        LaTeX label
    """
    latex_str = df.to_latex(
        caption=caption,
        label=label,
        escape=False,
        float_format=lambda x: f"{x:.4f}",
        bold_rows=True
    )
    
    with open(filename, 'w') as f:
        f.write(latex_str)
    
    print(f"   LaTeX table exported to {filename}")

def print_section_header(title):
    """Print formatted section header"""
    print("\n" + "="*60)
    print(title.center(60))
    print("="*60)

def print_subsection_header(title):
    """Print formatted subsection header"""
    print("\n" + "-"*40)
    print(title)
    print("-"*40)