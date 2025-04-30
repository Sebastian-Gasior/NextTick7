# NextTick Trading System

Ein modernes Trading-System mit technischer Analyse und LSTM-basierter Preisvorhersage.

## Features

- **Datenbeschaffung**: Automatisiertes Laden von Marktdaten via yfinance
- **Technische Analyse**: Berechnung wichtiger Indikatoren (SMA, EMA, RSI, MACD, Bollinger Bands)
- **Machine Learning**: LSTM-Modell für Preisvorhersagen mit PyTorch
- **Backtesting**: Robuste Engine für Strategievergleiche
- **Visualisierung**: Interaktives Dashboard mit Plotly/Dash
- **Performance-Metriken**: Umfassende Analyse von Rendite, Risiko und Drawdown

## Installation

1. Python-Umgebung erstellen (Python 3.9+ erforderlich):
```bash
python -m venv .venv
```

2. Umgebung aktivieren:
```bash
# Windows
.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate
```

3. Abhängigkeiten installieren:
```bash
uv pip install -r requirements.txt
```

4. Pre-commit Hooks installieren:
```bash
pre-commit install
```

## Verwendung

### Daten laden

```bash
python -m nexttick.main download AAPL MSFT GOOGL --days 365
```

### Technische Analyse durchführen

```bash
python -m nexttick.main analyze AAPL MSFT GOOGL
```

### LSTM-Modell trainieren

```bash
python -m nexttick.main train AAPL MSFT GOOGL --epochs 100
```

### Backtesting durchführen

```bash
python -m nexttick.main backtest AAPL MSFT GOOGL
```

### Dashboard starten

```bash
python -m nexttick.main dashboard --port 8050
```

## Projektstruktur

```
nexttick/
├── data/               # Rohdaten und verarbeitete Daten
├── src/
│   └── nexttick/
│       ├── data/      # Datenbeschaffung
│       ├── ta/        # Technische Analyse
│       ├── ml/        # Machine Learning
│       ├── compare/   # Strategievergleich
│       ├── backtest/  # Backtesting-Engine
│       └── dashboard/ # Web-Interface
├── tests/             # Unit- und Integrationstests
├── docs/             # Dokumentation
└── notebooks/        # Jupyter Notebooks
```

## Entwicklung

### Tests ausführen

```bash
pytest tests/ -v
```

### Code formatieren

```bash
black src/ tests/
isort src/ tests/
```

### Typ-Checks

```bash
mypy src/
```

## Konfiguration

Die Konfiguration erfolgt über die `config.py`. Wichtige Parameter:

- Trading-Parameter (Lookback-Periode, Train/Test-Split, etc.)
- Technische Indikatoren (Perioden für MA, RSI, etc.)
- LSTM-Modell (Batch-Size, Learning Rate, etc.)
- Backtesting (Kommission, Slippage, etc.)

## Performance-Metriken

Das System berechnet folgende Metriken:

- Gesamtrendite und annualisierte Rendite
- Sharpe Ratio und Volatilität
- Maximum Drawdown
- Win Rate und Profit Factor
- Trade-Statistiken (Anzahl, Durchschnittsgewinn/-verlust)

## Lizenz

MIT

## Beitragen

1. Fork erstellen
2. Feature Branch erstellen (`git checkout -b feature/AmazingFeature`)
3. Änderungen committen (`git commit -m 'Add some AmazingFeature'`)
4. Branch pushen (`git push origin feature/AmazingFeature`)
5. Pull Request erstellen