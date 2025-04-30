"""Modul für technische Indikatoren und deren Berechnung."""

from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from loguru import logger
import ta

from ..config import TA_CONFIG, TA_FEATURES_DIR


class TechnicalAnalysis:
    """Klasse für die Berechnung technischer Indikatoren."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialisiere die TA-Klasse.
        
        Args:
            config: Optionale Konfiguration für die Indikatoren
        """
        self.config = config or TA_CONFIG
        
    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Berechne alle technischen Indikatoren.
        
        Args:
            df: DataFrame mit OHLCV-Daten
            
        Returns:
            DataFrame mit allen Indikatoren
        """
        df = df.copy()
        
        try:
            # Moving Averages
            for period in self.config['sma_periods']:
                df[f'SMA_{period}'] = ta.trend.sma_indicator(df['Close'], window=period)
                
            for period in self.config['ema_periods']:
                df[f'EMA_{period}'] = ta.trend.ema_indicator(df['Close'], window=period)
            
            # RSI
            df['RSI'] = ta.momentum.rsi(df['Close'], window=self.config['rsi_period'])
            
            # MACD
            macd = ta.trend.MACD(
                df['Close'],
                window_slow=self.config['macd_slow'],
                window_fast=self.config['macd_fast'],
                window_sign=self.config['macd_signal']
            )
            df['MACD'] = macd.macd()
            df['MACD_Signal'] = macd.macd_signal()
            df['MACD_Hist'] = macd.macd_diff()
            
            # Bollinger Bands
            bollinger = ta.volatility.BollingerBands(
                df['Close'],
                window=self.config['bollinger_period'],
                window_dev=self.config['bollinger_std']
            )
            df['BB_Upper'] = bollinger.bollinger_hband()
            df['BB_Middle'] = bollinger.bollinger_mavg()
            df['BB_Lower'] = bollinger.bollinger_lband()
            
            # Volatilität
            df['ATR'] = ta.volatility.average_true_range(
                df['High'], df['Low'], df['Close']
            )
            
            # Volumen
            df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
            
            return df
            
        except Exception as e:
            logger.error(f"Fehler bei der Berechnung der Indikatoren: {str(e)}")
            raise
            
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generiere Handelssignale basierend auf den Indikatoren.
        
        Args:
            df: DataFrame mit technischen Indikatoren
            
        Returns:
            DataFrame mit Handelssignalen
        """
        df = df.copy()
        
        try:
            # RSI Signale
            df['RSI_Signal'] = 0
            df.loc[df['RSI'] < 30, 'RSI_Signal'] = 1  # Überverkauft
            df.loc[df['RSI'] > 70, 'RSI_Signal'] = -1  # Überkauft
            
            # MACD Signale
            df['MACD_Signal_Line'] = 0
            df.loc[df['MACD'] > df['MACD_Signal'], 'MACD_Signal_Line'] = 1
            df.loc[df['MACD'] < df['MACD_Signal'], 'MACD_Signal_Line'] = -1
            
            # Bollinger Band Signale
            df['BB_Signal'] = 0
            df.loc[df['Close'] < df['BB_Lower'], 'BB_Signal'] = 1
            df.loc[df['Close'] > df['BB_Upper'], 'BB_Signal'] = -1
            
            # Kombiniertes Signal
            df['TA_Signal'] = (
                df['RSI_Signal'] + 
                df['MACD_Signal_Line'] + 
                df['BB_Signal']
            ) / 3
            
            return df
            
        except Exception as e:
            logger.error(f"Fehler bei der Generierung der Signale: {str(e)}")
            raise
            
    def save_features(self, df: pd.DataFrame, symbol: str) -> None:
        """Speichere die berechneten Features.
        
        Args:
            df: DataFrame mit technischen Indikatoren
            symbol: Symbol des Instruments
        """
        try:
            output_path = TA_FEATURES_DIR / f"{symbol}_ta_features.parquet"
            df.to_parquet(output_path)
            logger.info(f"TA-Features gespeichert in {output_path}")
            
        except Exception as e:
            logger.error(f"Fehler beim Speichern der Features: {str(e)}")
            raise 