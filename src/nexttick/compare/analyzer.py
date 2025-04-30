"""Modul für den Vergleich von TA- und ML-basierten Strategien."""

from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
from loguru import logger
from sklearn.metrics import mean_squared_error, mean_absolute_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..config import PROCESSED_DATA_DIR


class StrategyComparator:
    """Klasse für den Vergleich von Trading-Strategien."""
    
    def __init__(self, initial_capital: float = 100000.0):
        """Initialisiere den Comparator.
        
        Args:
            initial_capital: Anfangskapital für Backtesting
        """
        self.initial_capital = initial_capital
        
    def calculate_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Berechne Performance-Metriken für beide Strategien.
        
        Args:
            df: DataFrame mit Signalen und Returns
            
        Returns:
            Dict mit Metriken
        """
        metrics = {}
        
        # Berechne Returns für beide Strategien
        if 'returns' not in df.columns:
            df['returns'] = df['Close'].pct_change()
            
        # Stelle sicher, dass beide Signal-Spalten existieren
        for signal_col in ['TA_Signal', 'ML_Signal']:
            if signal_col not in df.columns:
                df[signal_col] = 0
                
        df['TA_Strategy_Return'] = df['returns'] * df['TA_Signal'].shift(1).fillna(0)
        df['ML_Strategy_Return'] = df['returns'] * df['ML_Signal'].shift(1).fillna(0)
        
        # Kumulierte Returns (mit Basis 1.0)
        df['TA_Cumulative_Return'] = (1 + df['TA_Strategy_Return']).cumprod()
        df['ML_Cumulative_Return'] = (1 + df['ML_Strategy_Return']).cumprod()
        
        # Drawdowns
        df['TA_Drawdown'] = df['TA_Cumulative_Return'] / df['TA_Cumulative_Return'].expanding().max() - 1
        df['ML_Drawdown'] = df['ML_Cumulative_Return'] / df['ML_Cumulative_Return'].expanding().max() - 1
        
        # Performance-Metriken
        for strategy in ['TA', 'ML']:
            returns = df[f'{strategy}_Strategy_Return'].dropna()
            cum_returns = df[f'{strategy}_Cumulative_Return'].dropna()
            
            # Gesamtrendite
            total_return = (cum_returns.iloc[-1] - 1) * 100 if len(cum_returns) > 0 else 0
            metrics[f'{strategy}_Total_Return'] = total_return
            
            # Anzahl der Trades
            if len(df) > 1:  # Sicherstellen, dass wir mindestens zwei Datenpunkte haben
                trades = df[f'{strategy}_Signal'].diff().abs().sum() / 2
                trades = max(1, int(trades))  # Mindestens 1 Trade 
            else:
                trades = 0
            metrics[f'{strategy}_Number_of_Trades'] = int(trades)
            
            # Win Rate
            winning_trades = returns[returns > 0]
            total_trades = returns[returns != 0]
            
            if len(total_trades) > 0:
                win_rate = (len(winning_trades) / len(total_trades) * 100)
            else:
                win_rate = 0
            metrics[f'{strategy}_Win_Rate'] = win_rate
            
            # Average Profit/Loss
            if len(winning_trades) > 0:
                metrics[f'{strategy}_Avg_Profit'] = winning_trades.mean() * 100
            else:
                metrics[f'{strategy}_Avg_Profit'] = 0
                
            losing_trades = returns[returns < 0]
            if len(losing_trades) > 0:
                metrics[f'{strategy}_Avg_Loss'] = losing_trades.mean() * 100
            else:
                metrics[f'{strategy}_Avg_Loss'] = 0
            
            # Profit Factor
            gross_profit = winning_trades.sum() if len(winning_trades) > 0 else 0
            gross_loss = abs(losing_trades.sum()) if len(losing_trades) > 0 else 1  # Vermeide Division durch 0
            profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else 0
            metrics[f'{strategy}_Profit_Factor'] = profit_factor
            
            # Maximum Drawdown
            max_drawdown = df[f'{strategy}_Drawdown'].min() * 100 if len(df) > 0 else 0
            metrics[f'{strategy}_Max_Drawdown'] = max_drawdown
            
            # Volatilität
            vol = returns.std() * np.sqrt(252) * 100 if len(returns) > 1 else 0
            metrics[f'{strategy}_Volatility'] = vol
            
            # Sharpe Ratio (Rf = 0 für Einfachheit)
            if vol != 0 and vol > 0:
                sharpe = (total_return / 100) / (vol / 100)
            else:
                sharpe = 0
            metrics[f'{strategy}_Sharpe_Ratio'] = sharpe
            
        return metrics
        
    def create_comparison_plots(self, df: pd.DataFrame) -> go.Figure:
        """Erstelle Vergleichsplots für beide Strategien.
        
        Args:
            df: DataFrame mit Signalen und Returns
            
        Returns:
            Plotly Figure-Objekt
        """
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Kumulierte Returns', 'Drawdowns'),
            vertical_spacing=0.15
        )
        
        # Plot kumulierte Returns
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['TA_Cumulative_Return'],
                name='TA Strategy',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['ML_Cumulative_Return'],
                name='ML Strategy',
                line=dict(color='red')
            ),
            row=1, col=1
        )
        
        # Berechne und plotte Drawdowns
        for strategy in ['TA', 'ML']:
            cum_returns = df[f'{strategy}_Cumulative_Return']
            rolling_max = cum_returns.expanding().max()
            drawdowns = (cum_returns - rolling_max) / rolling_max * 100
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=drawdowns,
                    name=f'{strategy} Drawdown',
                    line=dict(color='blue' if strategy == 'TA' else 'red')
                ),
                row=2, col=1
            )
        
        # Update Layout
        fig.update_layout(
            title='Strategie-Vergleich',
            height=800,
            showlegend=True,
            template='plotly_white'
        )
        
        fig.update_yaxes(title_text='Kumulierter Return', row=1, col=1)
        fig.update_yaxes(title_text='Drawdown (%)', row=2, col=1)
        
        return fig
        
    def save_results(self, df: pd.DataFrame, metrics: Dict[str, float],
                    symbol: str) -> None:
        """Speichere die Vergleichsergebnisse.
        
        Args:
            df: DataFrame mit Signalen und Returns
            metrics: Berechnete Metriken
            symbol: Symbol des Instruments
        """
        try:
            # Speichere Ergebnisse als CSV
            results_df = pd.DataFrame(metrics, index=[0])
            results_path = PROCESSED_DATA_DIR / f"{symbol}_comparison_results.csv"
            results_df.to_csv(results_path)
            logger.info(f"Vergleichsergebnisse gespeichert in {results_path}")
            
            # Speichere Plot als HTML
            fig = self.create_comparison_plots(df)
            plot_path = PROCESSED_DATA_DIR / f"{symbol}_comparison_plot.html"
            fig.write_html(str(plot_path))
            logger.info(f"Vergleichsplot gespeichert in {plot_path}")
            
        except Exception as e:
            logger.error(f"Fehler beim Speichern der Ergebnisse: {str(e)}")
            raise 