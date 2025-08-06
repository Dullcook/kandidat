# config.py - Configuration settings for ESG ETF thesis analysis
# ==============================================================

import os
from datetime import datetime

# Create directories if they don't exist
os.makedirs('results', exist_ok=True)
os.makedirs('results/figures', exist_ok=True)

# File paths
DATA_PATH = '../'
RESULTS_PATH = './results/'
FIGURES_PATH = './results/figures/'

# Data file names
ETF_FILE = 'ESG_ETF_CLEANED.xlsx'
BENCHMARK_FILE = 'ESG_BENCHMARK_CLEANED.xlsx'
FACTOR_FILE = 'Cleaned Factor data.xlsx'

# Analysis period
START_DATE = '2019-01-01'
END_DATE = '2025-02-28'

# Crisis periods definition
CRISIS_PERIODS = {
    'COVID-19': ('2020-02-01', '2020-04-30'),
    'Inflation_Shock': ('2022-03-01', '2022-10-31'),
    'Banking_Stress': ('2023-03-01', '2023-03-31')
}

# ETF to Benchmark mapping
ETF_BENCHMARK_MAP = {
    'Xtrackers MSCI Europe ESG ETF 1C': 'MSCI Euro Low Carbon SRI Selection NR EUR',
    'SPDR® EURO STOXX 50 ETF': 'EURO STOXX 50 NR USD',
    'iShares MSCI Europe SRI ETF EUR Dist': 'MSCI EURO SRI SEL RED FOSSIL FUEL NR EUR',
    'SPDR® MSCI Europe Industrials ETF': 'MSCI Europe Industrials 35/20 Capped NR',
    'SPDR® MSCI Europe Financials ETF': 'MSCI Europe Financials 35/20 Capped NR',
    'iShares STOXX Europe 600 Fin Ser (DE)': 'STOXX Europe 600 Financial Svc NR EUR',
    'iShares STOXX Europe 600 Insurance (DE)': 'STOXX Europe 600 Insurance NR EUR',
    'iShares STOXX Europe 600 Utilities (DE)': 'STOXX Europe 600 Utilities NR EUR',
    'iShares STOXX Europe 600 IG & Ser (DE)': 'STOXX Europe 600 Indl Gd&Svcs NR EUR',
    'Xtrackers MSCI Europe HC Scr ETF 1C': 'MSCI Euro HlCr ESG Scrn 20-35 Sel NR EUR',
    'SPDR® MSCI Europe Health Care ETF': 'MSCI Europe Health Care 35/20 Capped NR',
    'SPDR® MSCI Europe Utilities ETF': 'MSCI Europe Utilities 35/20 Capped NR',
    'Xtrackers MSCI Europe CSer Scr ETF 1C': 'MSCI EUR Com Ser ESG Scr 20-35 Se NR EUR',
    'iShares STOXX Europe 600 Media (DE)': 'STOXX Europe 600 Media NR EUR',
    'Xtrackers MSCI Europe Indust Scr ETF 1C': 'MSCI Eur Ind ESG Scr 20-35 Sel NR EUR'
}

# Risk-free rate (annual)
RISK_FREE_RATE = 0.02  # 2% annual, will be converted to monthly

# Analysis parameters
ROLLING_WINDOW_MONTHS = 12
CONFIDENCE_LEVELS = [0.95, 0.99]
DECIMALS = 4  # Decimal places for results

# Plotting parameters
PLOT_STYLE = 'seaborn-v0_8-darkgrid'
PLOT_DPI = 300
FIGURE_SIZE = (12, 8)
COLORS = {
    'etf': '#2E86AB',
    'benchmark': '#A23B72',
    'covid': '#FF6B6B',
    'inflation': '#FFA500',
    'banking': '#9B59B6'
}