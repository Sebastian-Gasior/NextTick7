"""Hauptmodul für die NextTick Trading-Anwendung."""

import click
from loguru import logger
from typing import List
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from .data.market_data import MarketDataLoader
from .ta.indicators import TechnicalAnalysis
from .ml.lstm_model import LSTMPredictor, ModelTrainer
from .compare.analyzer import StrategyComparator
from .backtest.engine import BacktestEngine
from .dashboard.app import TradingDashboard
from .config import TA_FEATURES_DIR, get_symbols


@click.group()
def cli():
    """NextTick Trading System CLI."""
    pass


@cli.command()
@click.argument('symbols', nargs=-1)
@click.option('--days', default=365*5, help='Anzahl der Tage für historische Daten')
def download(symbols: List[str], days: int):
    """Lade historische Marktdaten für die angegebenen Symbole."""
    if not symbols:
        symbols = get_symbols()
    start_date = datetime.now() - timedelta(days=days)
    loader = MarketDataLoader(list(symbols), start_date=start_date)
    
    try:
        data_dict = loader.download_data()
        loader.process_data(data_dict)
        logger.info(f"Daten erfolgreich geladen für {len(data_dict)} Symbole")
    except Exception as e:
        logger.error(f"Fehler beim Laden der Daten: {str(e)}")
        raise click.Abort()


@cli.command()
@click.argument('symbols', nargs=-1)
def analyze(symbols: List[str]):
    """Führe technische Analyse für die angegebenen Symbole durch."""
    if not symbols:
        symbols = get_symbols()
    ta = TechnicalAnalysis()
    
    try:
        for symbol in symbols:
            # Lade verarbeitete Daten
            loader = MarketDataLoader([symbol])
            data_dict = loader.download_data()
            df = list(data_dict.values())[0]
            
            # Berechne Indikatoren
            df_with_ta = ta.calculate_all(df)
            df_with_signals = ta.generate_signals(df_with_ta)
            ta.save_features(df_with_signals, symbol)
            
            logger.info(f"TA erfolgreich durchgeführt für {symbol}")
    except Exception as e:
        logger.error(f"Fehler bei der technischen Analyse: {str(e)}")
        raise click.Abort()


@cli.command()
@click.argument('symbols', nargs=-1)
@click.option('--epochs', default=100, help='Anzahl der Trainings-Epochen')
def train(symbols: List[str], epochs: int):
    """Trainiere das LSTM-Modell für die angegebenen Symbole."""
    if not symbols:
        symbols = get_symbols()
    trainer = ModelTrainer()
    
    try:
        for symbol in symbols:
            # Lade Feature-Daten
            features_path = TA_FEATURES_DIR / f"{symbol}_ta_features.parquet"
            df = pd.read_parquet(features_path)
            
            # Bereite Sequenzen vor
            X, y = trainer.prepare_sequences(df[['Close']])
            
            # Erstelle und trainiere Modell
            model = LSTMPredictor(input_dim=X.shape[-1])
            trained_model = trainer.train(X, y, model)
            
            # Generiere Vorhersagen für die gesamten Trainingsdaten
            predictions = trainer.predict(trained_model, X)
            
            # Erstelle ein vorhersage-DataFrame, das genau so lang ist wie das Original
            full_predictions = np.zeros(len(df))
            # Setze die Vorhersagen in die zweite Hälfte des Arrays (typischerweise sind es ~50% der Daten)
            start_idx = max(0, len(df) - len(predictions))
            full_predictions[start_idx:] = predictions
            
            # Erstelle ML_Prediction- und ML_Signal-Spalten
            df['ML_Prediction'] = full_predictions
            
            # Generiere Signale: 1 wenn die Vorhersage höher als der aktuelle Preis ist (kaufen), -1 wenn niedriger (verkaufen)
            # Benutze einen besseren Ansatz für die Signalgenerierung:
            # 1 = Long (Preis wird steigen)
            # -1 = Short (Preis wird fallen)
            price_change_prediction = np.zeros(len(df))
            
            # Berechne die prozentuale Änderung zwischen Vorhersage und aktuellem Preis
            # Nur dort, wo wir Vorhersagen haben (full_predictions > 0)
            has_prediction = full_predictions > 0
            price_change_prediction[has_prediction] = (full_predictions[has_prediction] - df['Close'][has_prediction]) / df['Close'][has_prediction] * 100
            
            # Generiere Signale basierend auf der vorhergesagten Preisänderung
            # Verwende einen Schwellenwert von 0.5% für Signale
            df['ML_Signal'] = np.where(price_change_prediction > 0.5, 1,
                                     np.where(price_change_prediction < -0.5, -1, 0))
            
            # Speichere die aktualisierten Features
            df.to_parquet(features_path)
            logger.info(f"ML-Features aktualisiert für {symbol}")
            
            # Speichere Modell
            trainer.save_model(trained_model, symbol)
            
            logger.info(f"Modell erfolgreich trainiert für {symbol}")
    except Exception as e:
        logger.error(f"Fehler beim Training: {str(e)}")
        raise click.Abort()


@cli.command()
@click.argument('symbols', nargs=-1)
def backtest(symbols: List[str]):
    """Führe Backtesting für beide Strategien durch."""
    if not symbols:
        symbols = get_symbols()
    engine = BacktestEngine()
    comparator = StrategyComparator()
    
    try:
        for symbol in symbols:
            # Lade TA-Features
            features_path = TA_FEATURES_DIR / f"{symbol}_ta_features.parquet"
            df = pd.read_parquet(features_path)
            
            # Führe Backtests durch
            ta_metrics = engine.run_backtest(df, 'TA_Signal')
            engine.save_results(ta_metrics, symbol, 'TA')
            
            ml_metrics = engine.run_backtest(df, 'ML_Signal')
            engine.save_results(ml_metrics, symbol, 'ML')
            
            # Vergleiche Strategien
            metrics = comparator.calculate_metrics(df)
            comparator.save_results(df, metrics, symbol)
            
            logger.info(f"Backtesting erfolgreich durchgeführt für {symbol}")
    except Exception as e:
        logger.error(f"Fehler beim Backtesting: {str(e)}")
        raise click.Abort()


@cli.command()
@click.option('--port', default=8050, help='Port für den Dashboard-Server')
def dashboard(port: int):
    """Starte das Trading-Dashboard."""
    try:
        dashboard = TradingDashboard(port=port)
        logger.info(f"Dashboard gestartet auf Port {port}")
        dashboard.run()
    except Exception as e:
        logger.error(f"Fehler beim Starten des Dashboards: {str(e)}")
        raise click.Abort()


if __name__ == '__main__':
    cli() 