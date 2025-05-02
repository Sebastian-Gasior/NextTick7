# Projektstruktur von NextTick

## Hauptverzeichnis

- **README.md**: Enthält eine Übersicht und Anleitung zum Projekt.
- **requirements.txt**: Listet alle Python-Abhängigkeiten auf, die für das Projekt benötigt werden.
- **pyproject.toml**: Konfigurationsdatei für das Python-Projekt (z.B. für Build-Tools).
- **.gitignore**: Definiert, welche Dateien und Ordner von Git ignoriert werden sollen.
- **.pre-commit-config.yaml**: Konfiguration für Pre-Commit-Hooks zur Codequalität.
- **.gitattributes**: Git-Attributdatei, z.B. für Zeilenendungen.
- **analyze_data.py**: Eigenständiges Skript zur Datenanalyse (genauer Inhalt siehe Datei).

### Wichtige Ordner

- **src/**: Enthält den gesamten Quellcode des Projekts.
  - **nexttick/**: Hauptpaket mit allen Modulen der Trading-Anwendung.
    - **main.py**: Einstiegspunkt und CLI für das Projekt. Hier werden alle Hauptfunktionen (Daten laden, Analyse, Training, Backtest) als Befehle bereitgestellt.
    - **config.py**: Zentrale Konfigurationsdatei für Pfade, Parameter und Einstellungen.
    - **__init__.py**: Initialisiert das Paket und konfiguriert das Logging.
    - **dashboard/**: Enthält das Dashboard zur Visualisierung und Analyse der Trading-Strategien.
      - **app.py**: Implementiert das interaktive Dashboard (z.B. mit Plotly Dash), inkl. Visualisierung von Preisen, Signalen und Metriken.
    - **backtest/**: Backtesting-Engine für die Auswertung von Handelsstrategien.
      - **engine.py**: Führt Backtests durch, berechnet Metriken und verwaltet Trades.
    - **compare/**: Vergleich von Strategien und Auswertung der Ergebnisse.
      - **analyzer.py**: Berechnet und speichert Vergleichsmetriken und Plots für verschiedene Strategien.
    - **ml/**: Maschinelles Lernen, insbesondere LSTM-Modelle für Preisprognosen.
      - **lstm_model.py**: Enthält das LSTM-Modell, Trainingslogik und Datenvorbereitung.
    - **ta/**: Technische Analyse (TA) von Finanzdaten.
      - **indicators.py**: Berechnet technische Indikatoren (z.B. SMA, EMA, RSI, MACD, Bollinger Bands) und generiert Handelssignale.
    - **data/**: Datenbeschaffung und -verarbeitung.
      - **market_data.py**: Lädt und verarbeitet Marktdaten (z.B. von Yahoo Finance).

- **models/**: Gespeicherte Modelle (z.B. trainierte LSTM-Modelle für verschiedene Aktien).
- **logs/**: Logdateien des Systems.
- **data/**: Enthält alle Daten, unterteilt in:
  - **raw/**: Rohdaten (z.B. direkt von Yahoo Finance geladen).
  - **processed/**: Vorverarbeitete Daten.
  - **ta_features/**: Daten mit berechneten technischen Indikatoren.
- **notebooks/**: (Derzeit leer) – Hier können Jupyter-Notebooks für Analysen oder Experimente abgelegt werden.
- **docs/**: (Derzeit leer) – Platz für zusätzliche Dokumentation.
- **tests/**: (Derzeit leer) – Platz für Unit-Tests und Testskripte.
- **Bilder/** und **project-images/**: (Inhalt nicht analysiert) – Vermutlich für Bilder und Visualisierungen.

## Ausführliche Ordner- und Dateiübersicht

```
nexttick/
├── analyze_data.py         # Eigenständiges Skript zur Analyse der ML-Signale und Predictions (z.B. Verteilung, Plausibilität)
├── requirements.txt        # Listet alle benötigten Python-Abhängigkeiten auf
├── README.md               # Projektbeschreibung, Installations- und Nutzungshinweise
├── pyproject.toml          # Build- und Tool-Konfiguration für das Python-Projekt
├── .gitignore              # Definiert, welche Dateien/Ordner von Git ignoriert werden
├── .pre-commit-config.yaml # Konfiguration für Pre-Commit-Hooks (Codequalität)
├── .gitattributes          # Git-Attribute, z.B. für Zeilenenden
├── project-images/         # (Optional) Bilder für Dokumentation oder Präsentation
├── Bilder/                 # (Optional) Weitere Bilder/Visualisierungen
├── models/                 # Gespeicherte, trainierte ML-Modelle (z.B. LSTM für verschiedene Aktien)
├── logs/                   # Logdateien des Systems
├── data/                   # Alle Daten (siehe Unterstruktur)
│   ├── raw/                # Rohdaten, direkt von z.B. Yahoo Finance geladen
│   ├── processed/          # Vorverarbeitete, bereinigte Daten
│   └── ta_features/        # Daten mit berechneten technischen Indikatoren und ML-Features
├── notebooks/              # (Derzeit leer) Platz für Jupyter-Notebooks
├── docs/                   # (Derzeit leer) Platz für zusätzliche Dokumentation
├── tests/                  # (Derzeit leer) Platz für Unit- und Integrationstests
└── src/
    └── nexttick/
        ├── __init__.py         # Initialisiert das Paket, konfiguriert Logging
        ├── main.py             # Hauptmodul: CLI für alle Kernfunktionen (Download, Analyse, Training, Backtest, Dashboard)
        ├── config.py           # Zentrale Konfiguration: Pfade, Parameter für Trading, TA, ML usw.
        ├── data/
        │   └── market_data.py  # Lädt und verarbeitet Marktdaten (z.B. von Yahoo Finance)
        ├── ta/
        │   └── indicators.py   # Berechnet technische Indikatoren (SMA, EMA, RSI, MACD, Bollinger Bands, ATR, OBV) und generiert Handelssignale
        ├── ml/
        │   └── lstm_model.py   # Enthält das LSTM-Modell (PyTorch), Trainingslogik, Datenvorbereitung und Modell-Handling
        ├── compare/
        │   └── analyzer.py     # Vergleicht Strategien (TA vs. ML), berechnet und speichert Performance-Metriken und Plots
        ├── backtest/
        │   └── engine.py       # Backtesting-Engine: Simuliert Trades, berechnet Kennzahlen wie Rendite, Drawdown, Sharpe Ratio
        └── dashboard/
            └── app.py          # Interaktives Dashboard (Plotly Dash): Visualisierung von Preisen, Signalen, Metriken, Strategie-Vergleich
```

### Hauptzusammenhänge und Datenfluss

- **main.py** ist der Einstiegspunkt (CLI):
  - Ruft je nach Befehl die Module für Datenbeschaffung (`market_data.py`), technische Analyse (`indicators.py`), ML-Training (`lstm_model.py`), Backtesting (`engine.py`), Strategievergleich (`analyzer.py`) und das Dashboard (`app.py`) auf.
- **config.py** stellt zentrale Pfade und Parameter bereit, die von allen Modulen genutzt werden.
- **market_data.py** lädt Rohdaten und speichert sie in `data/raw/` und `data/processed/`.
- **indicators.py** berechnet technische Indikatoren und speichert die Ergebnisse in `data/ta_features/`.
- **lstm_model.py** trainiert ML-Modelle auf Basis der Features und speichert Modelle in `models/`.
- **engine.py** führt Backtests auf Basis der Signale durch und berechnet Performance-Kennzahlen.
- **analyzer.py** vergleicht die Strategien und speichert Metriken/Plots in `data/processed/`.
- **app.py** visualisiert alle Ergebnisse und Metriken im Web-Dashboard.
- **analyze_data.py** dient zur explorativen Analyse der ML-Signale und Predictions.

Jede Datei ist ausführlich im Quellcode kommentiert und modular aufgebaut, sodass Erweiterungen (z.B. neue Indikatoren, Modelle oder Visualisierungen) einfach möglich sind. 