# 6_visualizations.py - Visualization functions for thesis
# ======================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec
from config import *
from utils import print_section_header

# Set style
plt.style.use(PLOT_STYLE)
sns.set_palette("husl")

class ThesisVisualizer:
    """Class to create all thesis visualizations"""
    
    def __init__(self, save_figures=True):
        self.save_figures = save_figures
        self.figures = {}
        
    def create_all_visualizations(self, portfolios, risk_metrics, factor_results, crisis_periods):
        """Create all visualizations for the thesis"""
        print_section_header("CREATING VISUALIZATIONS")
        
        # 1. Main performance plot
        self.figures['performance'] = self.plot_cumulative_performance(portfolios, crisis_periods)
        
        # 2. Risk metrics comparison
        self.figures['risk'] = self.plot_risk_metrics(risk_metrics)
        
        # 3. Drawdown analysis
        self.figures['drawdown'] = self.plot_drawdown_analysis(portfolios)
        
        # 4. Factor model results
        if factor_results:
            self.figures['factors'] = self.plot_factor_analysis(factor_results)
        
        # 5. Crisis period analysis
        self.figures['crisis'] = self.plot_crisis_analysis(portfolios)
        
        # 6. Return distribution
        self.figures['distribution'] = self.plot_return_distributions(portfolios)
        
        # 7. Rolling statistics
        self.figures['rolling'] = self.plot_rolling_statistics(portfolios)
        
        print(f"\nCreated {len(self.figures)} visualizations")
        
        return self.figures
    
    def plot_cumulative_performance(self, portfolios, crisis_periods):
        """Create main cumulative performance chart"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[3, 1])
        
        # Calculate cumulative returns
        cum_returns = (1 + portfolios[['ETF_Portfolio', 'Benchmark_Portfolio']]).cumprod()
        
        # Plot cumulative returns
        ax1.plot(cum_returns.index, cum_returns['ETF_Portfolio'], 
                label='High-Sustainability ETF Portfolio', linewidth=2.5, color=COLORS['etf'])
        ax1.plot(cum_returns.index, cum_returns['Benchmark_Portfolio'], 
                label='Benchmark Portfolio', linewidth=2.5, color=COLORS['benchmark'])
        
        # Add crisis period shading
        for crisis_name, (start, end) in crisis_periods.items():
            color = COLORS.get(crisis_name.lower().split('_')[0], 'gray')
            ax1.axvspan(pd.to_datetime(start), pd.to_datetime(end), 
                       alpha=0.2, color=color, label=crisis_name.replace('_', ' '))
        
        # Format main plot
        ax1.set_ylabel('Cumulative Return', fontsize=12)
        ax1.set_title('Cumulative Performance: High-Sustainability ETFs vs Benchmarks', 
                     fontsize=14, pad=20)
        ax1.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(cum_returns.index[0], cum_returns.index[-1])
        
        # Format y-axis as percentage
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y - 1)))
        
        # Plot relative performance (bottom panel)
        relative_perf = cum_returns['ETF_Portfolio'] / cum_returns['Benchmark_Portfolio'] - 1
        ax2.plot(relative_perf.index, relative_perf * 100, linewidth=2, color='darkgreen')
        ax2.fill_between(relative_perf.index, 0, relative_perf * 100, 
                        where=(relative_perf >= 0), color='green', alpha=0.3, label='Outperformance')
        ax2.fill_between(relative_perf.index, 0, relative_perf * 100, 
                        where=(relative_perf < 0), color='red', alpha=0.3, label='Underperformance')
        
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax2.set_ylabel('Relative Performance (%)', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper left', fontsize=10)
        ax2.set_xlim(cum_returns.index[0], cum_returns.index[-1])
        
        plt.tight_layout()
        
        if self.save_figures:
            plt.savefig(f'{FIGURES_PATH}cumulative_performance.png', dpi=PLOT_DPI, bbox_inches='tight')
        
        return fig
    
    def plot_risk_metrics(self, risk_metrics):
        """Create risk metrics comparison visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Prepare data
        etf_metrics = risk_metrics[risk_metrics.index.str.contains('ETF')].iloc[0]
        bench_metrics = risk_metrics[risk_metrics.index.str.contains('Benchmark')].iloc[0]
        
        # 1. Risk-Return Scatter
        ax = axes[0, 0]
        ax.scatter(bench_metrics['Volatility_Annual'], bench_metrics['Mean_Return_Annual'] * 100, 
                  s=200, color=COLORS['benchmark'], label='Benchmark', marker='o')
        ax.scatter(etf_metrics['Volatility_Annual'], etf_metrics['Mean_Return_Annual'] * 100, 
                  s=200, color=COLORS['etf'], label='ESG ETF', marker='s')
        
        # Add arrows showing movement
        ax.annotate('', xy=(etf_metrics['Volatility_Annual'], etf_metrics['Mean_Return_Annual'] * 100),
                   xytext=(bench_metrics['Volatility_Annual'], bench_metrics['Mean_Return_Annual'] * 100),
                   arrowprops=dict(arrowstyle='->', lw=2, color='gray', alpha=0.5))
        
        ax.set_xlabel('Annual Volatility (%)', fontsize=12)
        ax.set_ylabel('Annual Return (%)', fontsize=12)
        ax.set_title('Risk-Return Profile', fontsize=13)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Downside Risk Comparison
        ax = axes[0, 1]
        metrics_to_compare = ['Max_Drawdown_%', 'Downside_Dev_Annual', 'VaR_95_Monthly', 'CVaR_95_Monthly']
        x = np.arange(len(metrics_to_compare))
        width = 0.35
        
        etf_values = [abs(etf_metrics[m]) for m in metrics_to_compare]
        bench_values = [abs(bench_metrics[m]) for m in metrics_to_compare]
        
        ax.bar(x - width/2, etf_values, width, label='ESG ETF', color=COLORS['etf'])
        ax.bar(x + width/2, bench_values, width, label='Benchmark', color=COLORS['benchmark'])
        
        ax.set_ylabel('Risk Measure (%)', fontsize=12)
        ax.set_title('Downside Risk Metrics Comparison', fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(['Max DD', 'Down Dev', 'VaR 95%', 'CVaR 95%'], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # 3. Risk-Adjusted Returns
        ax = axes[1, 0]
        ratios = ['Sharpe_Ratio', 'Sortino_Ratio', 'Calmar_Ratio']
        x = np.arange(len(ratios))
        
        etf_ratios = [etf_metrics[r] for r in ratios]
        bench_ratios = [bench_metrics[r] for r in ratios]
        
        ax.bar(x - width/2, etf_ratios, width, label='ESG ETF', color=COLORS['etf'])
        ax.bar(x + width/2, bench_ratios, width, label='Benchmark', color=COLORS['benchmark'])
        
        ax.set_ylabel('Ratio Value', fontsize=12)
        ax.set_title('Risk-Adjusted Performance Ratios', fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(['Sharpe', 'Sortino', 'Calmar'])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        
        # 4. Summary Statistics
        ax = axes[1, 1]
        summary_text = f"""Portfolio Performance Summary
        
ESG ETF Portfolio:
  • Annual Return: {etf_metrics['Mean_Return_Annual']*100:.2f}%
  • Annual Volatility: {etf_metrics['Volatility_Annual']:.2f}%
  • Sharpe Ratio: {etf_metrics['Sharpe_Ratio']:.3f}
  • Max Drawdown: {etf_metrics['Max_Drawdown_%']:.2f}%
  
Benchmark Portfolio:
  • Annual Return: {bench_metrics['Mean_Return_Annual']*100:.2f}%
  • Annual Volatility: {bench_metrics['Volatility_Annual']:.2f}%
  • Sharpe Ratio: {bench_metrics['Sharpe_Ratio']:.3f}
  • Max Drawdown: {bench_metrics['Max_Drawdown_%']:.2f}%"""
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.axis('off')
        
        plt.suptitle('Risk Metrics Analysis', fontsize=16, y=0.98)
        plt.tight_layout()
        
        if self.save_figures:
            plt.savefig(f'{FIGURES_PATH}risk_metrics_comparison.png', dpi=PLOT_DPI, bbox_inches='tight')
        
        return fig
    
    def plot_drawdown_analysis(self, portfolios):
        """Create detailed drawdown analysis"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Calculate drawdowns
        for i, (col, color) in enumerate([('ETF_Portfolio', COLORS['etf']), 
                                          ('Benchmark_Portfolio', COLORS['benchmark'])]):
            cum_returns = (1 + portfolios[col]).cumprod()
            running_max = cum_returns.expanding().max()
            drawdown = (cum_returns - running_max) / running_max * 100
            
            # Plot drawdown
            ax = ax1 if i == 0 else ax2
            ax.fill_between(drawdown.index, 0, drawdown, color=color, alpha=0.7)
            ax.plot(drawdown.index, drawdown, color=color, linewidth=1.5)
            
            # Mark maximum drawdown
            max_dd_date = drawdown.idxmin()
            max_dd_value = drawdown.min()
            ax.plot(max_dd_date, max_dd_value, 'ro', markersize=8)
            ax.annotate(f'Max DD: {max_dd_value:.1f}%\n{max_dd_date.strftime("%Y-%m")}',
                       xy=(max_dd_date, max_dd_value), xytext=(10, 10),
                       textcoords='offset points', fontsize=10,
                       bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                       arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            
            # Format
            ax.set_ylabel(f'{"ESG ETF" if i == 0 else "Benchmark"}\nDrawdown (%)', fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
            
            # Add crisis shading
            for crisis_name, (start, end) in CRISIS_PERIODS.items():
                color_crisis = COLORS.get(crisis_name.lower().split('_')[0], 'gray')
                ax.axvspan(pd.to_datetime(start), pd.to_datetime(end), 
                          alpha=0.15, color=color_crisis)
        
        ax1.set_title('Drawdown Analysis: ESG ETF vs Benchmark', fontsize=14, pad=10)
        ax2.set_xlabel('Date', fontsize=12)
        
        plt.tight_layout()
        
        if self.save_figures:
            plt.savefig(f'{FIGURES_PATH}drawdown_analysis.png', dpi=PLOT_DPI, bbox_inches='tight')
        
        return fig
    
    def plot_factor_analysis(self, factor_results):
        """Create factor model visualization"""
        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(3, 2, figure=fig)
        
        # Extract results
        etf_results = factor_results[0].get('ETF_Portfolio', {})
        bench_results = factor_results[0].get('Benchmark_Portfolio', {})
        
        # 1. Alpha comparison across models
        ax1 = fig.add_subplot(gs[0, :])
        models = ['CAPM', 'FF3', 'Carhart']
        alphas_etf = []
        alphas_bench = []
        
        for model in models:
            if model in etf_results and etf_results[model]:
                alphas_etf.append(etf_results[model]['alpha'] * 12 * 100)
            else:
                alphas_etf.append(0)
                
            if model in bench_results and bench_results[model]:
                alphas_bench.append(bench_results[model]['alpha'] * 12 * 100)
            else:
                alphas_bench.append(0)
        
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, alphas_etf, width, label='ESG ETF Portfolio', color=COLORS['etf'])
        bars2 = ax1.bar(x + width/2, alphas_bench, width, label='Benchmark Portfolio', color=COLORS['benchmark'])
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.annotate(f'{height:.2f}%',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3 if height >= 0 else -15),
                           textcoords="offset points",
                           ha='center', va='bottom' if height >= 0 else 'top',
                           fontsize=10)
        
        ax1.set_ylabel('Annualized Alpha (%)', fontsize=12)
        ax1.set_title('Alpha Comparison Across Factor Models', fontsize=14)
        ax1.set_xticks(x)
        ax1.set_xticklabels(models)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        
        # 2. Factor loadings (Carhart model)
        ax2 = fig.add_subplot(gs[1, 0])
        if 'Carhart' in etf_results and etf_results['Carhart']:
            factors = ['Mkt-RF', 'SMB', 'HML', 'Mom']
            etf_loadings = [etf_results['Carhart'].get(f'{f}_coef', etf_results['Carhart'].get(f'beta_{f.lower()}', 0)) 
                           for f in factors]
            bench_loadings = [bench_results['Carhart'].get(f'{f}_coef', bench_results['Carhart'].get(f'beta_{f.lower()}', 0)) 
                             for f in factors]
            
            x = np.arange(len(factors))
            ax2.bar(x - width/2, etf_loadings, width, label='ESG ETF', color=COLORS['etf'])
            ax2.bar(x + width/2, bench_loadings, width, label='Benchmark', color=COLORS['benchmark'])
            
            ax2.set_ylabel('Factor Loading', fontsize=12)
            ax2.set_title('Factor Loadings (Carhart Model)', fontsize=13)
            ax2.set_xticks(x)
            ax2.set_xticklabels(['Market', 'Size', 'Value', 'Momentum'])
            ax2.legend()
            ax2.grid(True, alpha=0.3, axis='y')
            ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        
        # 3. R-squared comparison
        ax3 = fig.add_subplot(gs[1, 1])
        r2_etf = [etf_results.get(m, {}).get('r_squared', 0) for m in models]
        r2_bench = [bench_results.get(m, {}).get('r_squared', 0) for m in models]
        
        x = np.arange(len(models))
        ax3.bar(x - width/2, r2_etf, width, label='ESG ETF', color=COLORS['etf'])
        ax3.bar(x + width/2, r2_bench, width, label='Benchmark', color=COLORS['benchmark'])
        
        ax3.set_ylabel('R-squared', fontsize=12)
        ax3.set_title('Model Explanatory Power', fontsize=13)
        ax3.set_xticks(x)
        ax3.set_xticklabels(models)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.set_ylim(0, 1)
        
        # 4. Statistical significance summary
        ax4 = fig.add_subplot(gs[2, :])
        
        # Create significance table
        sig_data = []
        for model in models:
            if model in etf_results and etf_results[model]:
                sig_data.append({
                    'Model': model,
                    'Portfolio': 'ESG ETF',
                    'Alpha': f"{etf_results[model]['alpha']*12*100:.2f}%",
                    'p-value': f"{etf_results[model]['alpha_pvalue']:.4f}",
                    'Significant': '***' if etf_results[model]['alpha_pvalue'] < 0.01 else 
                                  '**' if etf_results[model]['alpha_pvalue'] < 0.05 else
                                  '*' if etf_results[model]['alpha_pvalue'] < 0.10 else 'No'
                })
            if model in bench_results and bench_results[model]:
                sig_data.append({
                    'Model': model,
                    'Portfolio': 'Benchmark',
                    'Alpha': f"{bench_results[model]['alpha']*12*100:.2f}%",
                    'p-value': f"{bench_results[model]['alpha_pvalue']:.4f}",
                    'Significant': '***' if bench_results[model]['alpha_pvalue'] < 0.01 else 
                                  '**' if bench_results[model]['alpha_pvalue'] < 0.05 else
                                  '*' if bench_results[model]['alpha_pvalue'] < 0.10 else 'No'
                })
        
        # Create table
        if sig_data:
            sig_df = pd.DataFrame(sig_data)
            table = ax4.table(cellText=sig_df.values,
                            colLabels=sig_df.columns,
                            cellLoc='center',
                            loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            
            # Style the table
            for i in range(len(sig_df.columns)):
                table[(0, i)].set_facecolor('#40466e')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            for i in range(1, len(sig_df) + 1):
                for j in range(len(sig_df.columns)):
                    if i % 2 == 0:
                        table[(i, j)].set_facecolor('#f0f0f0')
        
        ax4.axis('off')
        ax4.set_title('Statistical Significance Summary\n(*** p<0.01, ** p<0.05, * p<0.10)', 
                     fontsize=12, pad=20)
        
        plt.suptitle('Factor Model Analysis Results', fontsize=16, y=0.98)
        plt.tight_layout()
        
        if self.save_figures:
            plt.savefig(f'{FIGURES_PATH}factor_analysis.png', dpi=PLOT_DPI, bbox_inches='tight')
        
        return fig
    
    def plot_crisis_analysis(self, portfolios):
        """Create crisis period analysis visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Calculate performance by period
        period_performance = portfolios.groupby('Period').agg({
            'ETF_Portfolio': ['mean', 'std', 'count', ('cumulative', lambda x: (1 + x).prod() - 1)],
            'Benchmark_Portfolio': ['mean', 'std', 'count', ('cumulative', lambda x: (1 + x).prod() - 1)]
        })
        
        # 1. Average returns by period
        ax = axes[0, 0]
        periods = period_performance.index
        x = np.arange(len(periods))
        width = 0.35
        
        etf_means = period_performance[('ETF_Portfolio', 'mean')] * 100
        bench_means = period_performance[('Benchmark_Portfolio', 'mean')] * 100
        
        bars1 = ax.bar(x - width/2, etf_means, width, label='ESG ETF', color=COLORS['etf'])
        bars2 = ax.bar(x + width/2, bench_means, width, label='Benchmark', color=COLORS['benchmark'])
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}%',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3 if height >= 0 else -15),
                           textcoords="offset points",
                           ha='center', va='bottom' if height >= 0 else 'top',
                           fontsize=9)
        
        ax.set_ylabel('Average Monthly Return (%)', fontsize=12)
        ax.set_title('Average Returns by Period', fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(periods, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        
        # 2. Volatility by period
        ax = axes[0, 1]
        etf_vols = period_performance[('ETF_Portfolio', 'std')] * np.sqrt(12) * 100
        bench_vols = period_performance[('Benchmark_Portfolio', 'std')] * np.sqrt(12) * 100
        
        ax.bar(x - width/2, etf_vols, width, label='ESG ETF', color=COLORS['etf'])
        ax.bar(x + width/2, bench_vols, width, label='Benchmark', color=COLORS['benchmark'])
        
        ax.set_ylabel('Annualized Volatility (%)', fontsize=12)
        ax.set_title('Volatility by Period', fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(periods, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # 3. Cumulative returns by period
        ax = axes[1, 0]
        etf_cum = period_performance[('ETF_Portfolio', 'cumulative')] * 100
        bench_cum = period_performance[('Benchmark_Portfolio', 'cumulative')] * 100
        
        bars1 = ax.bar(x - width/2, etf_cum, width, label='ESG ETF', color=COLORS['etf'])
        bars2 = ax.bar(x + width/2, bench_cum, width, label='Benchmark', color=COLORS['benchmark'])
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}%',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3 if height >= 0 else -15),
                           textcoords="offset points",
                           ha='center', va='bottom' if height >= 0 else 'top',
                           fontsize=9)
        
        ax.set_ylabel('Cumulative Return (%)', fontsize=12)
        ax.set_title('Cumulative Returns by Period', fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels(periods, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        
        # 4. Relative performance summary
        ax = axes[1, 1]
        
        # Calculate outperformance
        outperformance = []
        for period in periods:
            etf = period_performance.loc[period, ('ETF_Portfolio', 'cumulative')]
            bench = period_performance.loc[period, ('Benchmark_Portfolio', 'cumulative')]
            outperf = (etf - bench) * 100
            months = period_performance.loc[period, ('ETF_Portfolio', 'count')]
            outperformance.append({
                'Period': period,
                'Outperformance': f"{outperf:.2f}%",
                'Months': int(months),
                'ETF Return': f"{etf*100:.2f}%",
                'Benchmark Return': f"{bench*100:.2f}%"
            })
        
        outperf_df = pd.DataFrame(outperformance)
        
        # Create table
        table = ax.table(cellText=outperf_df.values,
                        colLabels=outperf_df.columns,
                        cellLoc='center',
                        loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Style the table
        for i in range(len(outperf_df.columns)):
            table[(0, i)].set_facecolor('#40466e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        for i in range(1, len(outperf_df) + 1):
            for j in range(len(outperf_df.columns)):
                if 'Crisis' not in outperf_df.iloc[i-1]['Period'] and outperf_df.iloc[i-1]['Period'] != 'Normal':
                    table[(i, j)].set_facecolor('#ffcccc')
        
        ax.axis('off')
        ax.set_title('Performance Summary by Period', fontsize=13)
        
        plt.suptitle('Crisis Period Analysis', fontsize=16, y=0.98)
        plt.tight_layout()
        
        if self.save_figures:
            plt.savefig(f'{FIGURES_PATH}crisis_analysis.png', dpi=PLOT_DPI, bbox_inches='tight')
        
        return fig
    
    def plot_return_distributions(self, portfolios):
        """Create return distribution analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Histogram with KDE
        ax = axes[0, 0]
        portfolios['ETF_Portfolio'].hist(bins=30, alpha=0.6, label='ESG ETF', 
                                        color=COLORS['etf'], density=True, ax=ax)
        portfolios['Benchmark_Portfolio'].hist(bins=30, alpha=0.6, label='Benchmark', 
                                             color=COLORS['benchmark'], density=True, ax=ax)
        
        # Add KDE
        portfolios['ETF_Portfolio'].plot.density(ax=ax, color=COLORS['etf'], linewidth=2)
        portfolios['Benchmark_Portfolio'].plot.density(ax=ax, color=COLORS['benchmark'], linewidth=2)
        
        # Add normal distribution overlay
        from scipy import stats as sp_stats
        x = np.linspace(portfolios[['ETF_Portfolio', 'Benchmark_Portfolio']].min().min(),
                       portfolios[['ETF_Portfolio', 'Benchmark_Portfolio']].max().max(), 100)
        
        for col, color in [('ETF_Portfolio', COLORS['etf']), ('Benchmark_Portfolio', COLORS['benchmark'])]:
            mean = portfolios[col].mean()
            std = portfolios[col].std()
            ax.plot(x, sp_stats.norm.pdf(x, mean, std), '--', color=color, alpha=0.7, linewidth=1.5)
        
        ax.set_xlabel('Monthly Return', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Return Distribution Comparison', fontsize=13)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Q-Q Plot
        ax = axes[0, 1]
        for col, color, label in [('ETF_Portfolio', COLORS['etf'], 'ESG ETF'),
                                  ('Benchmark_Portfolio', COLORS['benchmark'], 'Benchmark')]:
            sp_stats.probplot(portfolios[col].dropna(), dist="norm", plot=ax)
            ax.get_lines()[-2].set_color(color)
            ax.get_lines()[-2].set_label(label)
            ax.get_lines()[-1].set_color('gray')
            ax.get_lines()[-1].set_alpha(0.5)
        
        ax.set_title('Q-Q Plot (Normal Distribution)', fontsize=13)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Box plot by period
        ax = axes[1, 0]
        data_to_plot = []
        labels_plot = []
        colors_plot = []
        
        for period in portfolios['Period'].unique():
            period_data = portfolios[portfolios['Period'] == period]
            data_to_plot.extend([period_data['ETF_Portfolio'] * 100, 
                               period_data['Benchmark_Portfolio'] * 100])
            labels_plot.extend([f'ETF\n{period}', f'Bench\n{period}'])
            colors_plot.extend([COLORS['etf'], COLORS['benchmark']])
        
        bp = ax.boxplot(data_to_plot, labels=labels_plot, patch_artist=True)
        
        for patch, color in zip(bp['boxes'], colors_plot):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Monthly Return (%)', fontsize=12)
        ax.set_title('Return Distribution by Period', fontsize=13)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        # 4. Distribution statistics
        ax = axes[1, 1]
        
        stats_data = []
        for col, label in [('ETF_Portfolio', 'ESG ETF'), ('Benchmark_Portfolio', 'Benchmark')]:
            returns = portfolios[col]
            stats_data.append({
                'Portfolio': label,
                'Mean': f"{returns.mean()*100:.3f}%",
                'Std Dev': f"{returns.std()*100:.3f}%",
                'Skewness': f"{returns.skew():.3f}",
                'Kurtosis': f"{returns.kurtosis():.3f}",
                'Min': f"{returns.min()*100:.2f}%",
                'Max': f"{returns.max()*100:.2f}%",
                'JB p-value': f"{sp_stats.jarque_bera(returns.dropna())[1]:.3f}"
            })
        
        stats_df = pd.DataFrame(stats_data)
        
        # Create table
        table = ax.table(cellText=stats_df.T.values,
                        rowLabels=stats_df.columns,
                        colLabels=stats_df['Portfolio'].values,
                        cellLoc='center',
                        loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2)
        
        # Style the table - simplified without error-prone indexing
        try:
            # Style header and row labels if possible
            for (i, j), cell in table.get_celld().items():
                if i == 0 or j == 0:  # Header row or first column
                    cell.set_facecolor('#40466e')
                    cell.set_text_props(weight='bold', color='white')
        except:
            # If styling fails, continue without styling
            pass
        
        ax.axis('off')
        ax.set_title('Distribution Statistics Summary', fontsize=13)
        
        plt.suptitle('Return Distribution Analysis', fontsize=16, y=0.98)
        plt.tight_layout()
        
        if self.save_figures:
            plt.savefig(f'{FIGURES_PATH}return_distributions.png', dpi=PLOT_DPI, bbox_inches='tight')
        
        return fig
    
    def plot_rolling_statistics(self, portfolios, window=12):
        """Create rolling statistics visualization"""
        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        
        # Calculate rolling statistics
        for col, color, label in [('ETF_Portfolio', COLORS['etf'], 'ESG ETF'),
                                  ('Benchmark_Portfolio', COLORS['benchmark'], 'Benchmark')]:
            
            # 1. Rolling returns (annualized)
            rolling_mean = portfolios[col].rolling(window).mean() * 12 * 100
            axes[0].plot(rolling_mean.index, rolling_mean, color=color, 
                        linewidth=2, label=label)
            
            # 2. Rolling volatility (annualized)
            rolling_vol = portfolios[col].rolling(window).std() * np.sqrt(12) * 100
            axes[1].plot(rolling_vol.index, rolling_vol, color=color, 
                        linewidth=2, label=label)
            
            # 3. Rolling Sharpe ratio
            rolling_sharpe = ((portfolios[col].rolling(window).mean() * 12 - RISK_FREE_RATE) / 
                            (portfolios[col].rolling(window).std() * np.sqrt(12)))
            axes[2].plot(rolling_sharpe.index, rolling_sharpe, color=color, 
                        linewidth=2, label=label)
        
        # Add crisis shading to all subplots
        for ax in axes:
            for crisis_name, (start, end) in CRISIS_PERIODS.items():
                color = COLORS.get(crisis_name.lower().split('_')[0], 'gray')
                ax.axvspan(pd.to_datetime(start), pd.to_datetime(end), 
                          alpha=0.15, color=color)
        
        # Format subplots
        axes[0].set_ylabel(f'{window}-Month\nRolling Return (%)', fontsize=11)
        axes[0].set_title(f'{window}-Month Rolling Statistics', fontsize=14, pad=10)
        axes[0].legend(loc='upper left')
        axes[0].grid(True, alpha=0.3)
        axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        
        axes[1].set_ylabel(f'{window}-Month\nRolling Volatility (%)', fontsize=11)
        axes[1].legend(loc='upper left')
        axes[1].grid(True, alpha=0.3)
        
        axes[2].set_ylabel(f'{window}-Month\nRolling Sharpe Ratio', fontsize=11)
        axes[2].set_xlabel('Date', fontsize=12)
        axes[2].legend(loc='upper left')
        axes[2].grid(True, alpha=0.3)
        axes[2].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        
        plt.tight_layout()
        
        if self.save_figures:
            plt.savefig(f'{FIGURES_PATH}rolling_statistics.png', dpi=PLOT_DPI, bbox_inches='tight')
        
        return fig


# Standalone function for easy import
def create_thesis_visualizations(portfolios, risk_metrics, factor_results, crisis_periods):
    """Create all thesis visualizations"""
    visualizer = ThesisVisualizer(save_figures=True)
    return visualizer.create_all_visualizations(portfolios, risk_metrics, factor_results, crisis_periods)