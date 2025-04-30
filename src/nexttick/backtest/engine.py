"""Modul für das Backtesting von Trading-Strategien."""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from loguru import logger
from dataclasses import dataclass
from datetime import datetime

from ..config import PROCESSED_DATA_DIR


@dataclass
class Trade:
    """Datenklasse für einzelne Trades."""
    
    entry_date: datetime
    entry_price: float
    position_size: float
    direction: int  # 1 für Long, -1 für Short
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    
    def close_trade(self, exit_date: datetime, exit_price: float) -> None:
        """Schließe den Trade.
        
        Args:
            exit_date: Datum des Trade-Exits
            exit_price: Exit-Preis
        """
        self.exit_date = exit_date
        self.exit_price = exit_price
        self.pnl = (exit_price - self.entry_price) * self.position_size * self.direction


class BacktestEngine:
    """Engine für das Backtesting von Trading-Strategien."""
    
    def __init__(self, initial_capital: float = 100000.0, commission: float = 0.001,
                 slippage: float = 0.001):
        """Initialisiere die Backtest-Engine.
        
        Args:
            initial_capital: Anfangskapital
            commission: Kommission pro Trade (%)
            slippage: Slippage pro Trade (%)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.reset()
        
    def reset(self) -> None:
        """Setze die Engine zurück."""
        self.capital = self.initial_capital
        self.equity = []
        self.trades: List[Trade] = []
        self.current_position = 0
        self.current_trade: Optional[Trade] = None
        
    def calculate_position_size(self, price: float) -> float:
        """Berechne die Positionsgröße basierend auf verfügbarem Kapital.
        
        Args:
            price: Aktueller Preis
            
        Returns:
            Positionsgröße in Einheiten
        """
        # Verwende 2% Risiko pro Trade
        risk_amount = self.capital * 0.02
        position_size = risk_amount / price
        return position_size
        
    def execute_trade(self, date: datetime, price: float, direction: int) -> None:
        """Führe einen Trade aus.
        
        Args:
            date: Handelsdatum
            price: Handelspreis
            direction: Handelsrichtung (1 für Long, -1 für Short)
        """
        # Berücksichtige Slippage
        adjusted_price = price * (1 + self.slippage * direction)
        
        # Berechne Positionsgröße
        position_size = self.calculate_position_size(adjusted_price)
        
        # Berücksichtige Kommission
        commission_cost = position_size * adjusted_price * self.commission
        self.capital -= commission_cost
        
        # Erstelle neuen Trade
        self.current_trade = Trade(
            entry_date=date,
            entry_price=adjusted_price,
            position_size=position_size,
            direction=direction
        )
        
        self.current_position = direction
        
    def close_position(self, date: datetime, price: float) -> None:
        """Schließe die aktuelle Position.
        
        Args:
            date: Handelsdatum
            price: Handelspreis
        """
        if self.current_trade and self.current_position != 0:
            # Berücksichtige Slippage
            adjusted_price = price * (1 - self.slippage * self.current_position)
            
            # Schließe Trade
            self.current_trade.close_trade(date, adjusted_price)
            
            # Aktualisiere Kapital
            self.capital += self.current_trade.pnl
            commission_cost = (
                self.current_trade.position_size * 
                adjusted_price * 
                self.commission
            )
            self.capital -= commission_cost
            
            # Füge Trade zur Historie hinzu
            self.trades.append(self.current_trade)
            
            # Reset Position
            self.current_position = 0
            self.current_trade = None
            
    def run_backtest(self, df: pd.DataFrame, signal_column: str) -> Dict[str, float]:
        """Führe den Backtest durch.
        
        Args:
            df: DataFrame mit Preisdaten und Signalen
            signal_column: Name der Signal-Spalte
            
        Returns:
            Dict mit Performance-Metriken
        """
        self.reset()
        
        # Berechne Returns
        df['returns'] = df['Close'].pct_change()
        
        for date, row in df.iterrows():
            # Schließe bestehende Position wenn Signal Null oder entgegengesetzt
            if (self.current_position != 0 and 
                (row[signal_column] == 0 or 
                 row[signal_column] * self.current_position < 0)):
                self.close_position(date, row['Close'])
            
            # Öffne neue Position
            if row[signal_column] != 0 and self.current_position == 0:
                self.execute_trade(date, row['Close'], row[signal_column])
            
            # Tracke Equity
            self.equity.append(self.capital)
            
        # Schließe offene Position am Ende
        if self.current_position != 0:
            self.close_position(df.index[-1], df['Close'].iloc[-1])
            
        # Berechne Performance-Metriken
        return self.calculate_metrics()
        
    def calculate_metrics(self) -> Dict[str, float]:
        """Berechne Performance-Metriken.
        
        Returns:
            Dict mit Metriken
        """
        metrics = {}
        
        # Konvertiere Equity zu Series
        equity_series = pd.Series(self.equity)
        
        # Gesamtrendite
        total_return = ((self.capital - self.initial_capital) / 
                       self.initial_capital * 100)
        metrics['Total_Return'] = total_return
        
        # Trades
        metrics['Number_of_Trades'] = len(self.trades)
        
        if self.trades:
            # Gewinnende Trades
            profitable_trades = len([t for t in self.trades if t.pnl > 0])
            metrics['Win_Rate'] = (profitable_trades / len(self.trades) * 100)
            
            # Durchschnittlicher Gewinn/Verlust
            pnls = [t.pnl for t in self.trades]
            profitable_pnls = [p for p in pnls if p > 0]
            losing_pnls = [p for p in pnls if p < 0]
            
            if profitable_pnls:
                metrics['Avg_Profit'] = np.mean(profitable_pnls)
            else:
                metrics['Avg_Profit'] = 0.0
                
            if losing_pnls:
                metrics['Avg_Loss'] = abs(np.mean(losing_pnls))
            else:
                metrics['Avg_Loss'] = 0.0
                
            # Profit Factor - vermeidet Division durch Null
            gross_profit = sum(profitable_pnls) if profitable_pnls else 0.0
            gross_loss = abs(sum(losing_pnls)) if losing_pnls else 1.0  # Vermeide Division durch 0
            metrics['Profit_Factor'] = gross_profit / gross_loss if gross_loss > 0 else 0.0
        else:
            # Wenn keine Trades vorhanden sind, setze Standardwerte
            metrics['Win_Rate'] = 0.0
            metrics['Avg_Profit'] = 0.0
            metrics['Avg_Loss'] = 0.0
            metrics['Profit_Factor'] = 0.0
            
        # Drawdown - mit Null-Prüfung
        if len(equity_series) > 1:
            rolling_max = equity_series.expanding().max()
            drawdowns = (equity_series - rolling_max) / rolling_max * 100
            metrics['Max_Drawdown'] = drawdowns.min()
        else:
            metrics['Max_Drawdown'] = 0.0
        
        # Volatilität - mit Null-Prüfung
        if len(equity_series) > 1:
            returns = equity_series.pct_change().dropna()
            if len(returns) > 0:
                metrics['Volatility'] = returns.std() * np.sqrt(252) * 100
                
                # Sharpe Ratio (Rf = 0 für Einfachheit)
                if returns.std() > 0:
                    sharpe = np.sqrt(252) * returns.mean() / returns.std()
                    metrics['Sharpe_Ratio'] = sharpe
                else:
                    metrics['Sharpe_Ratio'] = 0.0
            else:
                metrics['Volatility'] = 0.0
                metrics['Sharpe_Ratio'] = 0.0
        else:
            metrics['Volatility'] = 0.0
            metrics['Sharpe_Ratio'] = 0.0
        
        return metrics
        
    def save_results(self, metrics: Dict[str, float], symbol: str,
                    strategy: str) -> None:
        """Speichere die Backtest-Ergebnisse.
        
        Args:
            metrics: Performance-Metriken
            symbol: Symbol des Instruments
            strategy: Name der Strategie
        """
        try:
            # Speichere Metriken
            results_df = pd.DataFrame(metrics, index=[0])
            results_path = (PROCESSED_DATA_DIR / 
                          f"{symbol}_{strategy}_backtest_results.csv")
            results_df.to_csv(results_path)
            logger.info(f"Backtest-Ergebnisse gespeichert in {results_path}")
            
            # Speichere Trades
            trades_data = []
            for trade in self.trades:
                trades_data.append({
                    'Entry_Date': trade.entry_date,
                    'Entry_Price': trade.entry_price,
                    'Position_Size': trade.position_size,
                    'Direction': trade.direction,
                    'Exit_Date': trade.exit_date,
                    'Exit_Price': trade.exit_price,
                    'PnL': trade.pnl
                })
            
            trades_df = pd.DataFrame(trades_data)
            trades_path = (PROCESSED_DATA_DIR / 
                         f"{symbol}_{strategy}_trades.csv")
            trades_df.to_csv(trades_path)
            logger.info(f"Trade-Historie gespeichert in {trades_path}")
            
        except Exception as e:
            logger.error(f"Fehler beim Speichern der Ergebnisse: {str(e)}")
            raise 