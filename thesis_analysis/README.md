# ESG ETF Thesis Analysis

## Project Overview

This project analyzes the performance of high-sustainability European ETFs compared to their benchmark indices, with a focus on downside risk protection during market crises.

## Directory Structure

```
thesis_analysis/
│
├── config.py                  # Configuration settings and parameters
├── utils.py                   # Utility functions
├── 1_data_loader.py          # Data loading and preparation
├── 2_portfolio_construction.py # Portfolio creation functions
├── 3_risk_metrics.py         # Risk metrics calculations
├── 4_factor_models.py        # Factor model analysis (CAPM, FF3, Carhart)
├── 5_statistical_tests.py    # Statistical testing functions
├── 6_visualizations.py       # Visualization functions
├── 7_main_analysis.py        # Main analysis pipeline
│
├── requirements.txt          # Python package requirements
├── README.md                # This file
│
├── Data Files (Required - in parent directory):
│   ├── ESG_ETF_CLEANED.xlsx         # ETF returns (wide format)
│   ├── ESG_BENCHMARK_CLEANED.xlsx   # Benchmark returns (wide format)
│   └── Cleaned_Factor_data.xlsx     # Fama-French factors (wide format)
│
└── results/                  # Output directory (created automatically)
    ├── thesis_results.xlsx   # All numerical results
    ├── thesis_text.txt       # Generated thesis text
    ├── *.tex                 # LaTeX tables
    └── figures/              # All visualizations
        ├── cumulative_performance.png
        ├── risk_metrics_comparison.png
        ├── drawdown_analysis.png
        ├── factor_analysis.png
        ├── crisis_analysis.png
        ├── return_distributions.png
        └── rolling_statistics.png
```

## Installation

1. Clone or download this repository
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Data Requirements

### Input Data Format
All data files should be in **wide format** (dates as columns, assets/factors as rows):

1. **ESG_ETF_CLEANED.xlsx**: Monthly returns for 15 high-sustainability ETFs
2. **ESG_BENCHMARK_CLEANED.xlsx**: Monthly returns for corresponding benchmark indices
3. **Cleaned Factor data.xlsx**: Fama-French factors (Mkt-RF, SMB, HML, RF) and Momentum (Mom)

### Time Period
- Analysis period: January 2019 to February 2025
- Crisis periods automatically identified:
  - COVID-19: Feb-Apr 2020
  - Inflation Shock: Mar-Oct 2022
  - Banking Stress: Mar 2023

## Usage

### Option 1: Run Complete Analysis
```python
python 7_main_analysis.py
```

This will:
1. Load and validate all data
2. Construct equal-weighted portfolios
3. Calculate comprehensive risk metrics
4. Run statistical tests
5. Perform factor model analysis
6. Generate all visualizations
7. Export results to Excel and LaTeX
8. Create thesis text template

### Option 2: Run Individual Components
```python
# Load data only
from data_loader import DataLoader
loader = DataLoader()
etf_data, benchmark_data, factor_data = loader.load_all_data()

# Calculate risk metrics only
from risk_metrics import calculate_risk_metrics
metrics = calculate_risk_metrics(portfolio_returns)

# Run factor models only
from factor_models import run_factor_models
results = run_factor_models(portfolio_returns, factor_data)
```

## Key Analyses Performed

### 1. Portfolio Construction
- Equal-weighted portfolios of ETFs and benchmarks
- Crisis period identification
- Tracking error analysis

### 2. Risk Metrics
- Standard metrics: Mean return, volatility, Sharpe ratio
- Downside metrics: Maximum drawdown, Sortino ratio, downside deviation
- Tail risk: VaR, CVaR at 95% and 99% confidence levels
- Higher moments: Skewness, kurtosis

### 3. Factor Models
- CAPM (single-factor)
- Fama-French 3-factor model
- Carhart 4-factor model (if momentum data available)
- Alpha significance testing

### 4. Statistical Tests
- Parametric: t-tests, F-tests
- Non-parametric: Mann-Whitney, Kolmogorov-Smirnov
- Crisis vs. normal period comparisons

### 5. Visualizations
- Cumulative performance with crisis shading
- Risk-return scatter plots
- Drawdown analysis
- Factor loadings comparison
- Return distributions
- Rolling statistics

## Output Files

### Excel Results (`thesis_results.xlsx`)
Multiple sheets containing:
- Full period metrics
- Crisis performance analysis
- Factor model results
- Statistical test results
- Portfolio returns
- Rolling statistics

### Figures
High-resolution PNG files (300 DPI) suitable for thesis inclusion

### LaTeX Tables
Pre-formatted tables ready for thesis inclusion

### Thesis Text (`thesis_text.txt`)
Template text with key findings for your results chapter

## Customization

### Modify Analysis Parameters
Edit `config.py` to change:
- Analysis period
- Crisis period definitions
- Risk-free rate
- Confidence levels
- Plot styling

### Add New Analyses
Each module can be extended with new functions. Follow the existing pattern:
1. Add function to appropriate module
2. Import in main analysis
3. Include in results export

## Troubleshooting

### Common Issues

1. **"File not found" errors**
   - Ensure all data files are in the same directory as the Python files
   - Check file names match exactly (case-sensitive)

2. **Package import errors**
   - Run: `pip install -r requirements.txt`
   - Use Python 3.9 or higher

3. **Memory errors with large datasets**
   - Close other applications
   - Process data in chunks if needed

4. **Factor data issues**
   - Ensure momentum factor is included as a row
   - Check data is in percentage format (will be converted automatically)

## Support

For questions or issues:
1. Check data format matches examples
2. Review error messages for specific issues
3. Ensure all required packages are installed

## Citation

If using this code in academic work, please cite appropriately.

## License

This code is provided for academic research purposes.