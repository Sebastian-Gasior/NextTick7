"""Modul für die Beschaffung und Verarbeitung von Marktdaten."""

from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any

import pandas as pd
import yfinance as yf
from loguru import logger
import numpy as np

from ..config import RAW_DATA_DIR, PROCESSED_DATA_DIR


class MarketDataLoader:
    """Klasse für das Laden und Verarbeiten von Marktdaten."""

    def __init__(self, symbols: List[str], start_date: Optional[datetime] = None,
                 end_date: Optional[datetime] = None):
        """Initialisiere den MarketDataLoader.

        Args:
            symbols: Liste der zu ladenden Symbole
            start_date: Startdatum für die Daten
            end_date: Enddatum für die Daten
        """
        self.symbols = symbols
        self.start_date = start_date or (datetime.now() - timedelta(days=365*5))
        self.end_date = end_date or datetime.now()
        
        # Stelle sicher, dass die Daten in der richtigen Zeitzone sind
        if self.start_date.tzinfo is None:
            self.start_date = pd.Timestamp(self.start_date).tz_localize('America/New_York')
        if self.end_date.tzinfo is None:
            self.end_date = pd.Timestamp(self.end_date).tz_localize('America/New_York')
        
    def download_data(self) -> Dict[str, pd.DataFrame]:
        """Lade Marktdaten von Yahoo Finance.

        Returns:
            Dict mit Symbol als Schlüssel und DataFrame als Wert
        """
        logger.info(f"Lade Daten für {len(self.symbols)} Symbole")
        data_dict = {}
        
        for symbol in self.symbols:
            try:
                ticker = yf.Ticker(symbol)
                # Konvertiere Datum zu string im Format YYYY-MM-DD
                start_str = self.start_date.strftime('%Y-%m-%d')
                end_str = self.end_date.strftime('%Y-%m-%d')
                
                df = ticker.history(start=start_str, end=end_str)
                
                if df.empty:
                    logger.warning(f"Keine Daten gefunden für {symbol}")
                    continue
                    
                # Stelle sicher, dass der Index die richtige Zeitzone hat
                if df.index.tz is None:
                    df.index = df.index.tz_localize('America/New_York')
                
                # Speichere Rohdaten
                raw_path = RAW_DATA_DIR / f"{symbol}_raw.parquet"
                df.to_parquet(raw_path)
                logger.info(f"Rohdaten gespeichert in {raw_path}")
                
                data_dict[symbol] = df
                
            except Exception as e:
                logger.error(f"Fehler beim Laden von {symbol}: {str(e)}")
                continue
                
        return data_dict
    
    def process_data(self, data_dict: Dict[str, pd.DataFrame]) -> None:
        """Verarbeite die Rohdaten.

        Args:
            data_dict: Dict mit Symbol als Schlüssel und DataFrame als Wert
        """
        for symbol, df in data_dict.items():
            try:
                # Stelle sicher, dass der Index die richtige Zeitzone hat
                if df.index.tz is None:
                    df.index = df.index.tz_localize('America/New_York')
                else:
                    df.index = df.index.tz_convert('America/New_York')
                
                # Speichere verarbeitete Daten
                processed_path = PROCESSED_DATA_DIR / f"{symbol}_processed.parquet"
                df.to_parquet(processed_path)
                logger.info(f"Verarbeitete Daten gespeichert in {processed_path}")
                
            except Exception as e:
                logger.error(f"Fehler bei der Verarbeitung von {symbol}: {str(e)}")
                continue 