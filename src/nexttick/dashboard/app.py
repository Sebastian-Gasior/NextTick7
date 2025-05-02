"""Dashboard für die Visualisierung von Trading-Ergebnissen."""

from typing import Dict, List
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from loguru import logger

from ..config import PROCESSED_DATA_DIR, TA_FEATURES_DIR


class TradingDashboard:
    """Dashboard für Trading-Analysen und Vergleiche."""
    
    def __init__(self, port: int = 8050):
        """Initialisiere das Dashboard.
        
        Args:
            port: Port für den Dash-Server
        """
        self.port = port
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.BOOTSTRAP],
            title="NextTick Trading Dashboard"
        )
        
        # Finde verfügbare Symbole
        files = list(PROCESSED_DATA_DIR.glob("*_processed.parquet"))
        self.available_symbols = sorted(set(f.stem.split("_")[0] for f in files))
        
        # Setze Standardwerte
        self.default_symbol = self.available_symbols[0] if self.available_symbols else None
        if self.default_symbol:
            df = pd.read_parquet(PROCESSED_DATA_DIR / f"{self.default_symbol}_processed.parquet")
            self.default_start_date = df.index.min().strftime('%Y-%m-%d')
            self.default_end_date = df.index.max().strftime('%Y-%m-%d')
        else:
            self.default_start_date = None
            self.default_end_date = None
        
        self.setup_layout()
        self.setup_callbacks()
        
    def setup_layout(self) -> None:
        """Erstelle das Dashboard-Layout."""
        self.app.layout = dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("NextTick Trading Dashboard",
                           className="text-center mb-4")
                ])
            ]),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Einstellungen"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Label("Symbol:"),
                                    dcc.Dropdown(
                                        id="symbol-dropdown",
                                        options=[{"label": s, "value": s} for s in self.available_symbols],
                                        value=self.default_symbol
                                    )
                                ]),
                                dbc.Col([
                                    html.Label("Zeitraum:"),
                                    dcc.DatePickerRange(
                                        id="date-range",
                                        start_date=self.default_start_date,
                                        end_date=self.default_end_date
                                    )
                                ])
                            ])
                        ])
                    ])
                ])
            ], className="mb-4"),
            
            dbc.Row([
                dbc.Col([
                    dbc.Tabs([
                        dbc.Tab([
                            dcc.Graph(id="price-chart")
                        ], label="Preise & Indikatoren"),
                        
                        dbc.Tab([
                            dcc.Graph(id="ta-signals")
                        ], label="TA Signale"),
                        
                        dbc.Tab([
                            dcc.Graph(id="ml-predictions")
                        ], label="ML Vorhersagen"),
                        
                        dbc.Tab([
                            dcc.Graph(id="strategy-comparison")
                        ], label="Strategie-Vergleich")
                    ])
                ])
            ]),
            
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Performance-Metriken"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Div(id="ta-metrics")
                                ], width=6),
                                dbc.Col([
                                    html.Div(id="ml-metrics")
                                ], width=6)
                            ])
                        ])
                    ])
                ])
            ], className="mt-4")
            
        ], fluid=True)
        
    def setup_callbacks(self) -> None:
        """Registriere die Dashboard-Callbacks."""
        
        @self.app.callback(
            Output("symbol-dropdown", "options"),
            Input("symbol-dropdown", "value")
        )
        def update_symbols(value):
            """Aktualisiere die verfügbaren Symbole."""
            try:
                # Finde alle verfügbaren Symbole in processed_data
                files = list(PROCESSED_DATA_DIR.glob("*_processed.parquet"))
                symbols = [f.stem.split("_")[0] for f in files]
                return [{"label": s, "value": s} for s in sorted(set(symbols))]
            except Exception as e:
                logger.error(f"Fehler beim Laden der Symbole: {str(e)}")
                return []
        
        @self.app.callback(
            [Output("price-chart", "figure"),
             Output("ta-signals", "figure"),
             Output("ml-predictions", "figure"),
             Output("strategy-comparison", "figure"),
             Output("ta-metrics", "children"),
             Output("ml-metrics", "children")],
            [Input("symbol-dropdown", "value"),
             Input("date-range", "start_date"),
             Input("date-range", "end_date")]
        )
        def update_charts(symbol, start_date, end_date):
            """Aktualisiere alle Charts und Metriken."""
            if not symbol:
                return self.create_empty_figures()
                
            try:
                # Lade alle verfügbaren Daten
                df = pd.read_parquet(PROCESSED_DATA_DIR / f"{symbol}_processed.parquet")
                ta_df = pd.read_parquet(TA_FEATURES_DIR / f"{symbol}_ta_features.parquet")
                
                # Konvertiere start_date und end_date zu Pandas Timestamp mit der gleichen Zeitzone
                if start_date:
                    start_date = pd.Timestamp(start_date).tz_localize('America/New_York')
                if end_date:
                    end_date = pd.Timestamp(end_date).tz_localize('America/New_York')
                
                # Stelle sicher, dass die Index-Zeitzone korrekt ist
                if df.index.tz is None:
                    df.index = df.index.tz_localize('America/New_York')
                if ta_df.index.tz is None:
                    ta_df.index = ta_df.index.tz_localize('America/New_York')
                
                # Wenn start_date und end_date nicht gesetzt sind, verwende den gesamten verfügbaren Zeitraum
                if not start_date:
                    start_date = df.index.min()
                if not end_date:
                    end_date = df.index.max()
                
                # Konvertiere alle Zeitstempel in die gleiche Zeitzone
                df.index = df.index.tz_convert('America/New_York')
                ta_df.index = ta_df.index.tz_convert('America/New_York')
                start_date = start_date.tz_convert('America/New_York')
                end_date = end_date.tz_convert('America/New_York')
                
                # Filtere die Daten nach dem gewählten Zeitraum
                df = df[(df.index >= start_date) & (df.index <= end_date)]
                ta_df = ta_df[(ta_df.index >= start_date) & (ta_df.index <= end_date)]
                
                # Stelle sicher, dass beide DataFrames die gleichen Indizes haben
                common_idx = df.index.intersection(ta_df.index)
                df = df.loc[common_idx]
                ta_df = ta_df.loc[common_idx]
                
                # Berechne Returns und Drawdowns für beide Strategien
                df['returns'] = df['Close'].pct_change()
                
                # TA Strategie
                df['TA_Strategy_Return'] = df['returns'] * ta_df['TA_Signal'].shift(1).fillna(0)
                df['TA_Cumulative_Return'] = (1 + df['TA_Strategy_Return']).cumprod()
                df['TA_Peak'] = df['TA_Cumulative_Return'].expanding().max()
                df['TA_Drawdown'] = (df['TA_Cumulative_Return'] - df['TA_Peak']) / df['TA_Peak']
                
                # ML Strategie
                if 'ML_Signal' in ta_df.columns:
                    # Jetzt nutzen wir das ML-Signal aus dem TA-Features DataFrame
                    df['ML_Strategy_Return'] = df['returns'] * ta_df['ML_Signal'].shift(1).fillna(0)
                    df['ML_Cumulative_Return'] = (1 + df['ML_Strategy_Return']).cumprod()
                    df['ML_Peak'] = df['ML_Cumulative_Return'].expanding().max()
                    df['ML_Drawdown'] = (df['ML_Cumulative_Return'] - df['ML_Peak']) / df['ML_Peak']
                else:
                    # Füge leere Spalten hinzu, damit das Dashboard nicht abstürzt
                    df['ML_Strategy_Return'] = 0
                    df['ML_Cumulative_Return'] = 1  # Start mit 1.0
                    df['ML_Drawdown'] = 0
                
                # Erstelle Charts
                price_fig = self.create_price_chart(df)
                ta_fig = self.create_ta_chart(ta_df)
                ml_fig = self.create_ml_chart(df, ta_df)
                comparison_fig = self.create_comparison_chart(df)
                
                # Lade Metriken
                ta_metrics = self.load_metrics(symbol, "TA")
                ml_metrics = self.load_metrics(symbol, "ML")
                
                return (
                    price_fig, ta_fig, ml_fig, comparison_fig,
                    self.format_metrics(ta_metrics),
                    self.format_metrics(ml_metrics)
                )
                
            except FileNotFoundError as e:
                logger.error(f"Datei nicht gefunden: {str(e)}")
                return self.create_empty_figures()
            except Exception as e:
                logger.error(f"Fehler beim Aktualisieren der Charts: {str(e)}")
                return self.create_empty_figures()
                
    def create_price_chart(self, df: pd.DataFrame) -> go.Figure:
        """Erstelle den Preis-Chart.
        
        Args:
            df: DataFrame mit Preisdaten
            
        Returns:
            Plotly Figure
        """
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3]
        )
        
        # Candlestick
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name="OHLC"
            ),
            row=1, col=1
        )
        
        # Volumen
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['Volume'],
                name="Volume"
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title="Preis & Volumen",
            xaxis_rangeslider_visible=False,
            height=800
        )
        
        return fig
        
    def create_ta_chart(self, df: pd.DataFrame) -> go.Figure:
        """Erstelle den TA-Chart.
        
        Args:
            df: DataFrame mit TA-Daten
            
        Returns:
            Plotly Figure
        """
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1
        )
        
        # Preis und Indikatoren
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['Close'],
                name="Close",
                line=dict(color='black')
            ),
            row=1, col=1
        )
        
        # Moving Averages
        for col in df.columns:
            if col.startswith('SMA_') or col.startswith('EMA_'):
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df[col],
                        name=col
                    ),
                    row=1, col=1
                )
        
        # RSI
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['RSI'],
                name="RSI"
            ),
            row=2, col=1
        )
        
        # RSI Linien
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        fig.update_layout(
            title="Technische Analyse",
            xaxis_rangeslider_visible=False,
            height=800
        )
        
        return fig
        
    def create_ml_chart(self, df: pd.DataFrame, ta_df: pd.DataFrame = None) -> go.Figure:
        """Erstelle den ML-Vorhersage-Chart.
        
        Args:
            df: DataFrame mit Preisdaten
            ta_df: DataFrame mit TA-Features und ML-Vorhersagen
            
        Returns:
            Plotly Figure
        """
        fig = go.Figure()
        
        # Tatsächliche Preise
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['Close'],
                name="Actual",
                line=dict(color='black')
            )
        )
        
        # Vorhersagen (falls vorhanden)
        if ta_df is not None and 'ML_Prediction' in ta_df.columns:
            pred_df = ta_df[ta_df['ML_Prediction'] > 0]
            if not pred_df.empty:
                fig.add_trace(
                    go.Scatter(
                        x=pred_df.index,
                        y=pred_df['ML_Prediction'],
                        name="Prediction",
                        line=dict(color='red', dash='dash')
                    )
                )
            
        fig.update_layout(
            title="ML Vorhersagen",
            xaxis_rangeslider_visible=False,
            height=800
        )
        
        return fig
        
    def create_comparison_chart(self, df: pd.DataFrame) -> go.Figure:
        """Erstelle den Strategie-Vergleichs-Chart.
        
        Args:
            df: DataFrame mit Strategie-Daten
            
        Returns:
            Plotly Figure
        """
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.2
        )
        
        # Kumulierte Returns
        if 'TA_Cumulative_Return' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['TA_Cumulative_Return'],
                    name="TA Strategy"
                ),
                row=1, col=1
            )
            
        if 'ML_Cumulative_Return' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['ML_Cumulative_Return'],
                    name="ML Strategy"
                ),
                row=1, col=1
            )
            
        # Drawdowns
        if 'TA_Drawdown' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['TA_Drawdown'],
                    name="TA Drawdown",
                    line=dict(color='blue')
                ),
                row=2, col=1
            )
            
        if 'ML_Drawdown' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['ML_Drawdown'],
                    name="ML Drawdown",
                    line=dict(color='red')
                ),
                row=2, col=1
            )
            
        fig.update_layout(
            title="Strategie-Vergleich",
            xaxis_rangeslider_visible=False,
            height=800
        )
        
        return fig
        
    def create_empty_figures(self) -> tuple:
        """Erstelle leere Figures und Metriken.
        
        Returns:
            Tuple mit leeren Figures und Metriken
        """
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="Keine Daten verfügbar",
            xaxis_visible=False,
            yaxis_visible=False
        )
        
        return (
            empty_fig, empty_fig, empty_fig, empty_fig,
            "Keine TA-Metriken verfügbar",
            "Keine ML-Metriken verfügbar"
        )
        
    def load_metrics(self, symbol: str, strategy: str) -> Dict:
        """Lade Performance-Metriken.
        
        Args:
            symbol: Symbol des Instruments
            strategy: Name der Strategie
            
        Returns:
            Dict mit Metriken
        """
        try:
            metrics_path = (PROCESSED_DATA_DIR / 
                          f"{symbol}_{strategy}_backtest_results.csv")
            if metrics_path.exists():
                return pd.read_csv(metrics_path, index_col=0).to_dict('records')[0]
            return {}
        except Exception as e:
            logger.error(f"Fehler beim Laden der Metriken: {str(e)}")
            return {}
            
    def format_metrics(self, metrics: Dict) -> html.Div:
        """Formatiere Metriken für die Anzeige.
        
        Args:
            metrics: Dict mit Metriken
            
        Returns:
            Formatiertes HTML-Div
        """
        if not metrics:
            return html.Div("Keine Metriken verfügbar")
            
        return html.Div([
            html.Table([
                html.Tr([
                    html.Td(k.replace('_', ' ')),
                    html.Td(f"{v:.2f}" if isinstance(v, float) else str(v))
                ]) for k, v in metrics.items()
            ])
        ])
        
    def run(self) -> None:
        """Starte den Dashboard-Server."""
        try:
            import webbrowser
            url = f"http://127.0.0.1:{self.port}"
            webbrowser.open(url)
            self.app.run(port=self.port)
        except Exception as e:
            logger.error(f"Fehler beim Starten des Dashboards: {str(e)}")
            raise