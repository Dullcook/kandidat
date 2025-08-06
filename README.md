# ESG ETF Thesis Analysis

## Project Overview

This repository contains the complete analysis code for a master's thesis investigating the performance of high-sustainability European ETFs compared to their benchmark indices, with a focus on downside risk protection during market crises.

## Key Findings

✅ **Risk-Adjusted Performance**: ESG ETFs showed superior Sharpe ratios (0.726 vs 0.676)  
✅ **Downside Protection**: Better maximum drawdown (-22.14% vs -24.00%)  
✅ **Crisis Resilience**: Outperformed during COVID-19 (+2.46%) and inflation shock (+1.09%)  
✅ **Lower Systematic Risk**: Beta of 0.46 vs 0.50 for benchmarks  

## Analysis Period
- **Time Frame**: January 2019 to February 2025
- **Data**: 15 high-sustainability ETFs vs 6 benchmark indices
- **Frequency**: Monthly returns (74 observations)

## Repository Structure

```
thesis_analysis/
├── 1_data_loader.py          # Data loading and cleaning
├── 2_portfolio_construction.py # Portfolio creation
├── 3_risk_metrics.py         # Risk metric calculations
├── 4_factor_models.py        # CAPM and Fama-French models
├── 5_statistical_tests.py    # Statistical testing
├── 6_visualizations.py       # Chart generation
├── 7_main_analysis.py        # Complete analysis pipeline
├── config.py                 # Configuration settings
├── utils.py                  # Utility functions
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Quick Start

### 1. Setup Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Prepare Data
Place your data files in the parent directory:
- `ESG_ETF_CLEANED.xlsx` - ETF returns data
- `ESG_BENCHMARK_CLEANED.xlsx` - Benchmark returns
- `Cleaned Factor data.xlsx` - Fama-French factors

### 3. Run Analysis
```bash
cd thesis_analysis
python 7_main_analysis.py
```

## Key Features

### 📊 **Comprehensive Analysis**
- Portfolio construction (equal-weighted)
- Risk metrics (Sharpe, Sortino, VaR, CVaR, drawdowns)
- Crisis period identification and analysis
- Factor models (CAPM, Fama-French 3-factor)
- Statistical significance testing

### 🚨 **Crisis Period Analysis**
- **COVID-19 Crisis**: Feb-Apr 2020
- **Inflation Shock**: Mar-Oct 2022  
- **Banking Stress**: Mar 2023

### 📈 **Outputs Generated**
- `thesis_results.xlsx` - Complete numerical results
- High-resolution visualizations (PNG, 300 DPI)
- LaTeX tables ready for thesis inclusion
- Generated thesis text with key findings

### 🔧 **Technical Highlights**
- European number format handling (comma decimals)
- Robust data cleaning and validation
- Missing data management
- Comprehensive statistical testing

## Requirements

- Python 3.9+
- pandas >= 1.5.0
- numpy >= 1.23.0
- statsmodels >= 0.13.0
- matplotlib >= 3.5.0
- seaborn >= 0.12.0
- openpyxl >= 3.0.0

## Analysis Methodology

1. **Data Preprocessing**: Clean European number formats, handle missing values
2. **Portfolio Construction**: Equal-weighted portfolios with crisis period identification
3. **Risk Assessment**: Calculate comprehensive risk metrics
4. **Factor Analysis**: Run CAPM and Fama-French 3-factor models
5. **Statistical Testing**: Parametric and non-parametric significance tests
6. **Crisis Analysis**: Compare performance during identified crisis periods
7. **Visualization**: Generate publication-ready charts and tables

## Results Summary

| Metric | ESG Portfolio | Benchmark | ESG Advantage |
|--------|---------------|-----------|---------------|
| Annual Return | 13.14% | 13.15% | -0.01% |
| Annual Volatility | 15.34% | 16.32% | **-0.98%** ✅ |
| Sharpe Ratio | 0.726 | 0.676 | **+0.050** ✅ |
| Max Drawdown | -22.14% | -24.00% | **+1.86%** ✅ |
| CAPM Alpha | +3.14% | +2.46% | **+0.68%** ✅ |

## Contributing

This is academic research code. For questions or suggestions, please open an issue.

## License

This project is for academic research purposes.

## Citation

If using this code in academic work, please cite appropriately.

---

*Analysis completed with virtual environment and all dependencies properly managed.* 