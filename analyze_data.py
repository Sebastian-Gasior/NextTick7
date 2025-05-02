import pandas as pd
import numpy as np
from src.nexttick.config import get_symbols

# Lade die Daten
symbols = get_symbols()

for symbol in symbols:
    # Lade TA-Features
    df = pd.read_parquet(f'data/ta_features/{symbol}_ta_features.parquet')
    
    # Analysiere ML-Signal
    print(f"\n{symbol} ML-Signal Analyse:")
    ml_counts = df['ML_Signal'].value_counts()
    print(f"ML-Signal Verteilung:\n{ml_counts}")
    print(f"Anzahl von Nullen: {(df.ML_Signal == 0).sum()}")
    print(f"Anzahl von Einsen: {(df.ML_Signal == 1).sum()}")
    print(f"Anzahl von Minus-Einsen: {(df.ML_Signal == -1).sum()}")
    print(f"Gesamtlänge des DataFrame: {len(df)}")
    
    # Überprüfe, ob die ML_Prediction-Werte vernünftig sind
    if 'ML_Prediction' in df.columns:
        non_zero_preds = df[df['ML_Prediction'] > 0]['ML_Prediction']
        if len(non_zero_preds) > 0:
            print(f"ML_Prediction (nicht Null): Min={non_zero_preds.min():.2f}, Max={non_zero_preds.max():.2f}, Mean={non_zero_preds.mean():.2f}")
        else:
            print("Keine ML_Prediction-Werte größer als 0")
    else:
        print("Keine ML_Prediction-Spalte gefunden") 