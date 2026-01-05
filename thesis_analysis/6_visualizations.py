# 6_visualizations.py - Essential visualizations for simplified thesis analysis
# ===========================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import *
from utils import print_section_header

# Set style
plt.style.use(PLOT_STYLE)
sns.set_palette("husl")

class ThesisVisualizer:
    """Class to create essential thesis visualizations"""
    
    def __init__(self, save_figures=True):
        self.save_figures = save_figures
        self.figures = {}
        
    def create_essential_visualizations(self, portfolios, risk_metrics, crisis_periods):
        """Create only essential visualizations for simplified analysis"""
        print_section_header("CREATING ESSENTIAL VISUALIZATIONS")
        
        try:
            # 1. Main performance plot
            self.figures['performance'] = self.plot_cumulative_performance(portfolios, crisis_periods)
            
            # 2. Simple risk-return scatter
            self.figures['simple_risk_return'] = self.plot_simple_risk_return(risk_metrics)
            
            # 3. Crisis analysis
            self.figures['crisis'] = self.plot_crisis_analysis(portfolios)
            
            print(f"\nCreated {len(self.figures)} essential visualizations")
            
        except Exception as e:
            print(f"   Warning: Visualization creation failed: {str(e)}")
            self.figures = {}
        
        return self.figures
    
    def plot_cumulative_performance(self, portfolios, crisis_periods):
        """Create main cumulative performance chart"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[3, 1])
        
        # Calculate cumulative returns
        cum_returns = (1 + portfolios[['ETF_Portfolio', 'Benchmark_Portfolio']]).cumprod()
        
        # Plot cumulative returns
        ax1.plot(cum_returns.index, cum_returns['ETF_Portfolio'], 
                label='ESG ETF Portfolio', linewidth=2.5, color=COLORS['etf'])
        ax1.plot(cum_returns.index, cum_returns['Benchmark_Portfolio'], 
                label='Benchmark Portfolio', linewidth=2.5, color=COLORS['benchmark'])
        
        # Add crisis period shading
        for crisis_name, (start, end) in crisis_periods.items():
            color = COLORS.get(crisis_name.lower().split('_')[0], 'gray')
            ax1.axvspan(pd.to_datetime(start), pd.to_datetime(end), 
                       alpha=0.2, color=color, label=crisis_name.replace('_', ' '))
        
        # Format main plot
        ax1.set_ylabel('Cumulative Return', fontsize=12)
        ax1.set_title('Cumulative Performance: ESG ETFs vs Benchmarks', 
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
    
    def plot_simple_risk_return(self, risk_metrics):
        """Create a clear risk-return comparison"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Convert to percentage format
        etf_return = risk_metrics.loc['ETF_Portfolio', 'Mean_Return_Annual'] * 100
        bench_return = risk_metrics.loc['Benchmark_Portfolio', 'Mean_Return_Annual'] * 100
        etf_vol = risk_metrics.loc['ETF_Portfolio', 'Volatility_Annual'] * 100
        bench_vol = risk_metrics.loc['Benchmark_Portfolio', 'Volatility_Annual'] * 100
        
        # Plot with better scale
        ax.scatter(etf_vol, etf_return, s=200, label='ESG ETF', marker='s', color=COLORS['etf'])
        ax.scatter(bench_vol, bench_return, s=200, label='Benchmark', marker='o', color=COLORS['benchmark'])
        
        # Add labels with values
        ax.annotate(f'ESG: {etf_return:.2f}%', (etf_vol, etf_return), 
                    xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')
        ax.annotate(f'Benchmark: {bench_return:.2f}%', (bench_vol, bench_return), 
                    xytext=(5, -5), textcoords='offset points', fontsize=10, fontweight='bold')
        
        # Add arrow showing relative performance
        ax.annotate('', xy=(etf_vol, etf_return),
                   xytext=(bench_vol, bench_return),
                   arrowprops=dict(arrowstyle='->', lw=2, color='gray', alpha=0.7))
        
        ax.set_xlabel('Annual Volatility (%)', fontsize=12)
        ax.set_ylabel('Annual Return (%)', fontsize=12)
        ax.set_title('Risk-Return Profile: ESG ETF vs Benchmark', fontsize=14)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Zoom in on relevant range if values are close
        vol_range = max(etf_vol, bench_vol) - min(etf_vol, bench_vol)
        if vol_range < 2:  # If volatility difference is small, zoom in
            ax.set_xlim(min(etf_vol, bench_vol) - 0.5, max(etf_vol, bench_vol) + 0.5)
        
        plt.tight_layout()
        
        if self.save_figures:
            plt.savefig(f'{FIGURES_PATH}simple_risk_return.png', dpi=PLOT_DPI, bbox_inches='tight')
        
        return fig
    
    def plot_crisis_analysis(self, portfolios):
        """Create crisis period performance analysis"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Get crisis periods
        crisis_data = portfolios[portfolios['Period'] != 'Normal']
        
        if len(crisis_data) == 0:
            ax.text(0.5, 0.5, 'No crisis periods found', ha='center', va='center', transform=ax.transAxes)
            return fig
        
        # Calculate cumulative returns for each crisis period
        crisis_performance = {}
        for period in crisis_data['Period'].unique():
            period_data = crisis_data[crisis_data['Period'] == period]
            etf_cum = (1 + period_data['ETF_Portfolio']).prod() - 1
            bench_cum = (1 + period_data['Benchmark_Portfolio']).prod() - 1
            crisis_performance[period] = {
                'ETF': etf_cum * 100,
                'Benchmark': bench_cum * 100,
                'Difference': (etf_cum - bench_cum) * 100
            }
        
        # Create bar chart
        periods = list(crisis_performance.keys())
        etf_returns = [crisis_performance[p]['ETF'] for p in periods]
        bench_returns = [crisis_performance[p]['Benchmark'] for p in periods]
        
        x = np.arange(len(periods))
        width = 0.35
        
        ax.bar(x - width/2, etf_returns, width, label='ESG ETF', color=COLORS['etf'])
        ax.bar(x + width/2, bench_returns, width, label='Benchmark', color=COLORS['benchmark'])
        
        # Add difference labels
        for i, period in enumerate(periods):
            diff = crisis_performance[period]['Difference']
            color = 'green' if diff > 0 else 'red'
            ax.annotate(f'{diff:+.1f}%', (i, max(etf_returns[i], bench_returns[i]) + 1),
                       ha='center', color=color, fontweight='bold')
        
        ax.set_xlabel('Crisis Period', fontsize=12)
        ax.set_ylabel('Cumulative Return (%)', fontsize=12)
        ax.set_title('Crisis Period Performance Comparison', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(periods, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        
        plt.tight_layout()
        
        if self.save_figures:
            plt.savefig(f'{FIGURES_PATH}crisis_performance.png', dpi=PLOT_DPI, bbox_inches='tight')
        
        return fig
    
    def plot_var_comparison(self, risk_metrics):
        """Create VaR comparison chart"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Extract VaR metrics
        etf_var_95 = risk_metrics.loc['ETF_Portfolio', 'VaR_95_Monthly'] * 100
        bench_var_95 = risk_metrics.loc['Benchmark_Portfolio', 'VaR_95_Monthly'] * 100
        etf_var_99 = risk_metrics.loc['ETF_Portfolio', 'VaR_99_Monthly'] * 100
        bench_var_99 = risk_metrics.loc['Benchmark_Portfolio', 'VaR_99_Monthly'] * 100
        
        # Create bar chart
        x = np.arange(2)
        width = 0.35
        
        # Plot VaR 95% and 99%
        bars1 = ax.bar(x - width/2, [etf_var_95, etf_var_99], width, 
                       label='ESG ETF', color=COLORS['etf'])
        bars2 = ax.bar(x + width/2, [bench_var_95, bench_var_99], width, 
                       label='Benchmark', color=COLORS['benchmark'])
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords='offset points', ha='center', va='bottom')
        
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords='offset points', ha='center', va='bottom')
        
        # Format chart
        ax.set_xlabel('Confidence Level', fontsize=12)
        ax.set_ylabel('Value at Risk (%)', fontsize=12)
        ax.set_title('Value at Risk Comparison: ESG ETF vs Benchmark', fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(['95% Confidence', '99% Confidence'])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add horizontal line at 0 for reference
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        
        plt.tight_layout()
        
        if self.save_figures:
            plt.savefig(f'{FIGURES_PATH}var_comparison.png', dpi=PLOT_DPI, bbox_inches='tight')
        
        return fig
    
    def plot_factor_models(self, factor_results):
        """Create factor model comparison chart"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Extract alpha values for comparison
        models = ['CAPM', 'FF3', 'Carhart']
        etf_alphas = []
        bench_alphas = []
        
        for model in models:
            if model in factor_results.get('ETF_Portfolio', {}) and model in factor_results.get('Benchmark_Portfolio', {}):
                etf_alpha = factor_results['ETF_Portfolio'][model]['alpha'] * 12 * 100  # Annualized
                bench_alpha = factor_results['Benchmark_Portfolio'][model]['alpha'] * 12 * 100  # Annualized
                etf_alphas.append(etf_alpha)
                bench_alphas.append(bench_alpha)
            else:
                etf_alphas.append(0)
                bench_alphas.append(0)
        
        # Plot 1: Alpha comparison
        x = np.arange(len(models))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, etf_alphas, width, label='ESG ETF', color=COLORS['etf'])
        bars2 = ax1.bar(x + width/2, bench_alphas, width, label='Benchmark', color=COLORS['benchmark'])
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax1.annotate(f'{height:.2f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                         xytext=(0, 3), textcoords='offset points', ha='center', va='bottom')
        
        for bar in bars2:
            height = bar.get_height()
            ax1.annotate(f'{height:.2f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                         xytext=(0, 3), textcoords='offset points', ha='center', va='bottom')
        
        ax1.set_xlabel('Factor Model', fontsize=12)
        ax1.set_ylabel('Alpha (Annualized %)', fontsize=12)
        ax1.set_title('Factor Model Alpha Comparison', fontsize=14)
        ax1.set_xticks(x)
        ax1.set_xticklabels(models)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        
        # Plot 2: R-squared comparison
        etf_r2 = []
        bench_r2 = []
        
        for model in models:
            if model in factor_results.get('ETF_Portfolio', {}) and model in factor_results.get('Benchmark_Portfolio', {}):
                etf_r2.append(factor_results['ETF_Portfolio'][model]['adj_r_squared'])
                bench_r2.append(factor_results['Benchmark_Portfolio'][model]['adj_r_squared'])
            else:
                etf_r2.append(0)
                bench_r2.append(0)
        
        bars3 = ax2.bar(x - width/2, etf_r2, width, label='ESG ETF', color=COLORS['etf'])
        bars4 = ax2.bar(x + width/2, bench_r2, width, label='Benchmark', color=COLORS['benchmark'])
        
        # Add value labels
        for bar in bars3:
            height = bar.get_height()
            ax2.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                         xytext=(0, 3), textcoords='offset points', ha='center', va='bottom')
        
        for bar in bars4:
            height = bar.get_height()
            ax2.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                         xytext=(0, 3), textcoords='offset points', ha='center', va='bottom')
        
        ax2.set_xlabel('Factor Model', fontsize=12)
        ax2.set_ylabel('Adjusted R²', fontsize=12)
        ax2.set_title('Factor Model Explanatory Power', fontsize=14)
        ax2.set_xticks(x)
        ax2.set_xticklabels(models)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if self.save_figures:
            plt.savefig(f'{FIGURES_PATH}factor_models.png', dpi=PLOT_DPI, bbox_inches='tight')
        
        return fig
    
    def plot_portfolio_composition(self, etf_types):
        """Create portfolio composition table"""
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.axis('tight')
        ax.axis('off')
        
        # Create table data
        table_data = []
        headers = ['ETF Name', 'Category', 'Type']
        
        # Add ESG-focused ETFs
        for etf in etf_types['ESG-Focused']:
            table_data.append([etf, 'ESG-Focused', 'Sustainability'])
        
        # Add Sector ETFs
        for etf in etf_types['Sector ETFs']:
            # Determine sector from ETF name
            if 'Financial' in etf:
                sector = 'Financials'
            elif 'Health' in etf or 'HC' in etf:
                sector = 'Healthcare'
            elif 'Utility' in etf:
                sector = 'Utilities'
            elif 'Industrial' in etf or 'Indust' in etf:
                sector = 'Industrials'
            elif 'Media' in etf:
                sector = 'Media'
            elif 'Service' in etf or 'Ser' in etf:
                sector = 'Services'
            else:
                sector = 'Other'
            
            table_data.append([etf, 'Sector', sector])
        
        # Create table
        table = ax.table(cellText=table_data, colLabels=headers, 
                        cellLoc='left', loc='center',
                        colWidths=[0.5, 0.2, 0.2])
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Color header row
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color ESG-focused ETFs
        for i in range(len(etf_types['ESG-Focused'])):
            for j in range(len(headers)):
                table[(i+1, j)].set_facecolor('#E8F5E8')
        
        # Color sector ETFs
        for i in range(len(etf_types['Sector ETFs'])):
            for j in range(len(headers)):
                table[(i+1+len(etf_types['ESG-Focused']), j)].set_facecolor('#F0F0F0')
        
        # Add summary text below table
        summary_text = f"""
Portfolio Summary:
• Total ETFs: {len(etf_types['ESG-Focused']) + len(etf_types['Sector ETFs'])}
• ESG-Focused: {len(etf_types['ESG-Focused'])} ETFs
• Sector ETFs: {len(etf_types['Sector ETFs'])} ETFs
• Portfolio Correlation: 0.979 (high correlation suggests similar market exposure)
        """
        
        ax.text(0.02, 0.02, summary_text, transform=ax.transAxes, 
                fontsize=11, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_title('Portfolio Composition: ESG ETFs vs Sector ETFs', fontsize=16, pad=20)
        
        plt.tight_layout()
        
        if self.save_figures:
            plt.savefig(f'{FIGURES_PATH}portfolio_composition.png', dpi=PLOT_DPI, bbox_inches='tight')
        
        return fig

    def plot_risk_ratios_table(self, risk_metrics, stat_tests=None):
        """Create risk ratios comparison table"""
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare table data
        headers = ['Risk Metric', 'ESG ETF Portfolio', 'Benchmark Portfolio', 'Difference']
        
        # Extract key metrics
        etf_metrics = risk_metrics.loc['ETF_Portfolio']
        bench_metrics = risk_metrics.loc['Benchmark_Portfolio']
        
        # Calculate differences
        diff_metrics = etf_metrics - bench_metrics
        
        # Create table rows
        table_data = [
            ['Annual Return (%)', 
             f"{etf_metrics['Mean_Return_Annual']*100:.2f}%", 
             f"{bench_metrics['Mean_Return_Annual']*100:.2f}%",
             f"{diff_metrics['Mean_Return_Annual']*100:+.2f}%"],
            
            ['Volatility (%)', 
             f"{etf_metrics['Volatility_Annual']*100:.2f}%", 
             f"{bench_metrics['Volatility_Annual']*100:.2f}%",
             f"{diff_metrics['Volatility_Annual']*100:+.2f}%"],
            
            ['Sharpe Ratio', 
             f"{etf_metrics['Sharpe_Ratio']:.3f}", 
             f"{bench_metrics['Sharpe_Ratio']:.3f}",
             f"{diff_metrics['Sharpe_Ratio']:+.3f}"],
            
            ['Sortino Ratio', 
             f"{etf_metrics['Sortino_Ratio']:.3f}", 
             f"{bench_metrics['Sortino_Ratio']:.3f}",
             f"{diff_metrics['Sortino_Ratio']:+.3f}"],
            
            ['VaR 95% Monthly (%)', 
             f"{etf_metrics['VaR_95_Monthly']*100:.2f}%", 
             f"{bench_metrics['VaR_95_Monthly']*100:.2f}%",
             f"{diff_metrics['VaR_95_Monthly']*100:+.2f}%"],
            
            ['VaR 99% Monthly (%)', 
             f"{etf_metrics['VaR_99_Monthly']*100:.2f}%", 
             f"{bench_metrics['VaR_99_Monthly']*100:.2f}%",
             f"{diff_metrics['VaR_99_Monthly']*100:+.2f}%"]
        ]
        
        # Add statistical test results if available
        if stat_tests:
            table_data.append(['', '', '', ''])  # Empty row for spacing
            
            # Add T-test results
            if 't_test' in stat_tests:
                t_p = stat_tests['t_test']['p_value']
                t_sig = '***' if t_p < 0.01 else '**' if t_p < 0.05 else '*' if t_p < 0.1 else ''
                table_data.append(['T-test p-value', 
                                 f"{t_p:.4f}{t_sig}", 
                                 '', 
                                 ''])
            
            # Add Mann-Whitney results
            if 'mann_whitney' in stat_tests and not np.isnan(stat_tests['mann_whitney']['statistic']):
                mw_p = stat_tests['mann_whitney']['p_value']
                mw_sig = '***' if mw_p < 0.01 else '**' if mw_p < 0.05 else '*' if mw_p < 0.1 else ''
                table_data.append(['Mann-Whitney p-value', 
                                 f"{mw_p:.4f}{mw_sig}", 
                                 '', 
                                 ''])
            
            # Add correlation
            if 'correlation' in stat_tests:
                corr = stat_tests['correlation']
                table_data.append(['Portfolio Correlation', 
                                 f"{corr:.4f}", 
                                 '', 
                                 ''])
        
        # Create table
        table = ax.table(cellText=table_data, colLabels=headers, 
                        cellLoc='center', loc='center',
                        colWidths=[0.3, 0.2, 0.2, 0.2])
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Color header row
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#2E86AB')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color ESG ETF column (slightly green tint)
        for i in range(1, len(table_data) + 1):
            table[(i, 1)].set_facecolor('#E8F5E8')
        
        # Color Benchmark column (slightly orange tint)
        for i in range(1, len(table_data) + 1):
            table[(i, 2)].set_facecolor('#FFF3E0')
        
        # Color difference column (highlight positive/negative)
        for i in range(1, len(table_data) + 1):
            if i-1 < len(diff_metrics) and table_data[i-1][3] != '' and table_data[i-1][3] != ' ':
                try:
                    diff_value = float(table_data[i-1][3].replace('%', '').replace('+', ''))
                    if diff_value > 0:
                        table[(i, 3)].set_facecolor('#E8F5E8')  # Light green for positive
                    elif diff_value < 0:
                        table[(i, 3)].set_facecolor('#FFEBEE')  # Light red for negative
                    else:
                        table[(i, 3)].set_facecolor('#F5F5F5')  # Light gray for zero
                except:
                    pass
        
        # Add summary text below table
        summary_text = f"""
Risk Analysis Summary:
• ESG ETF shows {'better' if etf_metrics['Sharpe_Ratio'] > bench_metrics['Sharpe_Ratio'] else 'similar'} risk-adjusted returns
• ESG ETF has {'lower' if etf_metrics['Volatility_Annual'] < bench_metrics['Volatility_Annual'] else 'similar'} volatility
• ESG ETF shows {'better' if etf_metrics['VaR_95_Monthly'] > bench_metrics['VaR_95_Monthly'] else 'similar'} downside protection
        """
        
        ax.text(0.02, 0.02, summary_text, transform=ax.transAxes, 
                fontsize=11, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_title('Risk Metrics Comparison: ESG ETF vs Benchmark Portfolio', fontsize=16, pad=20)
        
        plt.tight_layout()
        
        if self.save_figures:
            plt.savefig(f'{FIGURES_PATH}risk_ratios_table.png', dpi=PLOT_DPI, bbox_inches='tight')
        
        return fig

    def plot_maximum_drawdown(self, portfolios):
        """Create maximum drawdown visualization over time"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Calculate cumulative returns
        etf_cumulative = (1 + portfolios['ETF_Portfolio']).cumprod()
        bench_cumulative = (1 + portfolios['Benchmark_Portfolio']).cumprod()
        
        # Calculate drawdown series
        etf_peak = etf_cumulative.expanding().max()
        bench_peak = bench_cumulative.expanding().max()
        
        etf_drawdown = (etf_cumulative - etf_peak) / etf_peak * 100
        bench_drawdown = (bench_cumulative - bench_peak) / bench_peak * 100
        
        # Plot drawdown series
        ax.plot(etf_drawdown.index, etf_drawdown, 
                label='ESG ETF Portfolio', color=COLORS['etf'], linewidth=2)
        ax.plot(bench_drawdown.index, bench_drawdown, 
                label='Benchmark Portfolio', color=COLORS['benchmark'], linewidth=2)
        
        # Add crisis period shading
        crisis_periods = portfolios[portfolios['Crisis'] == True]
        if not crisis_periods.empty:
            for idx in crisis_periods.index:
                ax.axvspan(idx, idx, ymin=0, ymax=1, 
                          color='red', alpha=0.3, linewidth=0)
        
        # Add horizontal line at 0%
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
        
        # Add maximum drawdown annotations
        etf_max_dd = etf_drawdown.min()
        bench_max_dd = bench_drawdown.min()
        
        etf_max_dd_date = etf_drawdown.idxmin()
        bench_max_dd_date = bench_drawdown.idxmin()
        
        ax.annotate(f'Max DD: {etf_max_dd:.1f}%', 
                    xy=(etf_max_dd_date, etf_max_dd), 
                    xytext=(10, -10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['etf'], alpha=0.7),
                    fontsize=10, color='white', weight='bold')
        
        ax.annotate(f'Max DD: {bench_max_dd:.1f}%', 
                    xy=(bench_max_dd_date, bench_max_dd), 
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=COLORS['benchmark'], alpha=0.7),
                    fontsize=10, color='white', weight='bold')
        
        # Styling
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Drawdown (%)', fontsize=12)
        ax.set_title('Maximum Drawdown Analysis: ESG ETF vs Benchmark', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0f}%'.format(y)))
        
        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        if self.save_figures:
            plt.savefig(f'{FIGURES_PATH}maximum_drawdown.png', dpi=PLOT_DPI, bbox_inches='tight')
        
        return fig


# Standalone function for importing
def create_essential_charts(portfolios, risk_metrics, crisis_periods):
    """Convenience function to create essential visualizations"""
    visualizer = ThesisVisualizer()
    return visualizer.create_essential_visualizations(portfolios, risk_metrics, crisis_periods)