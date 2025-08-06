# 3_risk_metrics.py - Downside risk metrics calculations
# =====================================================

import pandas as pd
import numpy as np
from scipy import stats
from config import RISK_FREE_RATE, CONFIDENCE_LEVELS
from utils import print_subsection_header

class RiskMetricsCalculator:
    """Class to calculate various risk metrics"""
    
    def __init__(self, returns, rf_rate=RISK_FREE_RATE/12):  # Monthly risk-free rate
        self.returns = returns
        self.rf_rate = rf_rate
        
    def calculate_all_metrics(self):
        """Calculate all risk metrics"""
        
        # Handle both Series and DataFrame inputs
        if isinstance(self.returns, pd.DataFrame):
            results = {}
            for col in self.returns.columns:
                results[col] = self._calculate_single_series(self.returns[col])
            return pd.DataFrame(results).T
        else:
            return self._calculate_single_series(self.returns)
    
    def _calculate_single_series(self, returns):
        """Calculate metrics for a single return series"""
        returns_clean = returns.dropna()
        
        if len(returns_clean) == 0:
            return self._empty_metrics()
        
        # Basic statistics
        metrics = {
            'Observations': len(returns_clean),
            'Mean_Return_Monthly': returns_clean.mean(),
            'Mean_Return_Annual': returns_clean.mean() * 12,
            'Volatility_Monthly': returns_clean.std(),
            'Volatility_Annual': returns_clean.std() * np.sqrt(12),
            'Skewness': returns_clean.skew(),
            'Kurtosis': returns_clean.kurtosis(),
            'Min_Return': returns_clean.min(),
            'Max_Return': returns_clean.max()
        }
        
        # Risk-adjusted returns
        excess_returns = returns_clean - self.rf_rate
        metrics['Sharpe_Ratio'] = (metrics['Mean_Return_Annual'] - RISK_FREE_RATE) / \
                                  metrics['Volatility_Annual'] if metrics['Volatility_Annual'] != 0 else np.nan
        
        # Downside risk metrics
        downside_metrics = self._calculate_downside_metrics(returns_clean)
        metrics.update(downside_metrics)
        
        # Maximum drawdown
        dd_metrics = self._calculate_drawdown_metrics(returns_clean)
        metrics.update(dd_metrics)
        
        # VaR and CVaR
        var_metrics = self._calculate_var_cvar(returns_clean)
        metrics.update(var_metrics)
        
        return pd.Series(metrics)
    
    def _calculate_downside_metrics(self, returns):
        """Calculate downside-focused risk metrics"""
        
        # Downside deviation (below 0)
        downside_returns = returns[returns < 0]
        downside_dev_monthly = np.sqrt(np.mean(downside_returns**2)) if len(downside_returns) > 0 else 0
        downside_dev_annual = downside_dev_monthly * np.sqrt(12)
        
        # Semi-deviation (below mean)
        mean_return = returns.mean()
        below_mean = returns[returns < mean_return]
        semi_dev_monthly = below_mean.std() if len(below_mean) > 0 else 0
        semi_dev_annual = semi_dev_monthly * np.sqrt(12)
        
        # Sortino ratio
        excess_return_annual = (returns.mean() - self.rf_rate) * 12
        sortino_ratio = excess_return_annual / downside_dev_annual if downside_dev_annual != 0 else np.nan
        
        # Omega ratio (probability of gains vs losses)
        threshold = 0
        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns <= threshold]
        omega_ratio = gains.sum() / losses.sum() if losses.sum() != 0 else np.nan
        
        # Gain/Loss ratio
        gain_loss_ratio = gains.mean() / abs(losses.mean()) if len(losses) > 0 and losses.mean() != 0 else np.nan
        
        return {
            'Downside_Dev_Monthly': downside_dev_monthly,
            'Downside_Dev_Annual': downside_dev_annual,
            'Semi_Dev_Monthly': semi_dev_monthly,
            'Semi_Dev_Annual': semi_dev_annual,
            'Sortino_Ratio': sortino_ratio,
            'Omega_Ratio': omega_ratio,
            'Gain_Loss_Ratio': gain_loss_ratio,
            'Positive_Months_%': (returns > 0).mean() * 100
        }
    
    def _calculate_drawdown_metrics(self, returns):
        """Calculate drawdown-related metrics"""
        
        # Calculate cumulative returns
        cum_returns = (1 + returns).cumprod()
        
        # Calculate drawdown series
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        
        # Maximum drawdown
        max_drawdown = drawdown.min()
        
        # Maximum drawdown duration
        dd_start = None
        dd_periods = []
        
        for i, dd in enumerate(drawdown):
            if dd < 0 and dd_start is None:
                dd_start = i
            elif dd == 0 and dd_start is not None:
                dd_periods.append(i - dd_start)
                dd_start = None
        
        max_dd_duration = max(dd_periods) if dd_periods else 0
        avg_dd_duration = np.mean(dd_periods) if dd_periods else 0
        
        # Calmar ratio
        annual_return = returns.mean() * 12
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else np.nan
        
        # Recovery metrics
        if max_drawdown < 0:
            max_dd_idx = drawdown.idxmin()
            recovery_start = max_dd_idx
            recovery_end = None
            
            for date in drawdown.index[drawdown.index > max_dd_idx]:
                if drawdown[date] >= -0.01:  # Within 1% of peak
                    recovery_end = date
                    break
            
            if recovery_end:
                recovery_time = (recovery_end - recovery_start).days / 30  # In months
            else:
                recovery_time = np.nan
        else:
            recovery_time = 0
        
        return {
            'Max_Drawdown': max_drawdown,
            'Max_Drawdown_%': max_drawdown * 100,
            'Calmar_Ratio': calmar_ratio,
            'Max_DD_Duration_Months': max_dd_duration,
            'Avg_DD_Duration_Months': avg_dd_duration,
            'Recovery_Time_Months': recovery_time,
            'Current_Drawdown_%': drawdown.iloc[-1] * 100
        }
    
    def _calculate_var_cvar(self, returns):
        """Calculate Value at Risk and Conditional Value at Risk"""
        
        metrics = {}
        
        for confidence in CONFIDENCE_LEVELS:
            # VaR (Value at Risk)
            var_percentile = (1 - confidence) * 100
            var_monthly = np.percentile(returns, var_percentile)
            var_annual = var_monthly * np.sqrt(12)  # Assuming normal distribution
            
            # CVaR (Conditional Value at Risk / Expected Shortfall)
            cvar_returns = returns[returns <= var_monthly]
            cvar_monthly = cvar_returns.mean() if len(cvar_returns) > 0 else var_monthly
            cvar_annual = cvar_monthly * np.sqrt(12)
            
            confidence_pct = int(confidence * 100)
            metrics[f'VaR_{confidence_pct}_Monthly'] = var_monthly
            metrics[f'VaR_{confidence_pct}_Annual'] = var_annual
            metrics[f'CVaR_{confidence_pct}_Monthly'] = cvar_monthly
            metrics[f'CVaR_{confidence_pct}_Annual'] = cvar_annual
        
        return metrics
    
    def _empty_metrics(self):
        """Return empty metrics structure"""
        metrics = {
            'Observations': 0,
            'Mean_Return_Monthly': np.nan,
            'Mean_Return_Annual': np.nan,
            'Volatility_Monthly': np.nan,
            'Volatility_Annual': np.nan,
            'Sharpe_Ratio': np.nan,
            'Sortino_Ratio': np.nan,
            'Calmar_Ratio': np.nan,
            'Max_Drawdown_%': np.nan,
            'VaR_95_Monthly': np.nan,
            'CVaR_95_Monthly': np.nan
        }
        return pd.Series(metrics)
    
    def calculate_rolling_metrics(self, window=12):
        """Calculate rolling risk metrics"""
        
        if isinstance(self.returns, pd.Series):
            returns = self.returns.to_frame()
        else:
            returns = self.returns
        
        rolling_metrics = {}
        
        for col in returns.columns:
            # Rolling volatility
            rolling_metrics[f'{col}_Rolling_Vol'] = returns[col].rolling(window).std() * np.sqrt(12)
            
            # Rolling Sharpe
            rolling_mean = returns[col].rolling(window).mean() * 12
            rolling_metrics[f'{col}_Rolling_Sharpe'] = \
                (rolling_mean - RISK_FREE_RATE) / rolling_metrics[f'{col}_Rolling_Vol']
            
            # Rolling maximum drawdown
            rolling_dd = []
            for i in range(window-1, len(returns)):
                window_returns = returns[col].iloc[i-window+1:i+1]
                cum_ret = (1 + window_returns).cumprod()
                running_max = cum_ret.expanding().max()
                dd = ((cum_ret - running_max) / running_max).min()
                rolling_dd.append(dd)
            
            rolling_metrics[f'{col}_Rolling_MaxDD'] = pd.Series(
                rolling_dd, 
                index=returns.index[window-1:]
            )
        
        return pd.DataFrame(rolling_metrics)


# Standalone function for easy import
def calculate_risk_metrics(returns, rf_rate=RISK_FREE_RATE/12):
    """Calculate all risk metrics for given returns"""
    calculator = RiskMetricsCalculator(returns, rf_rate)
    return calculator.calculate_all_metrics()