"""LSTM-Modell für Preisvorhersagen."""

from typing import Tuple, Optional, Dict, Any, List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from loguru import logger

from ..config import TRADING_CONFIG, MODELS_DIR


class TimeSeriesDataset(Dataset):
    """Dataset für Zeitreihen-Daten."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """Initialisiere das Dataset.
        
        Args:
            X: Feature-Matrix
            y: Zielvariable
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self) -> int:
        return len(self.X)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class LSTMPredictor(nn.Module):
    """LSTM-Modell für Preisvorhersagen."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2,
                 dropout: float = 0.2):
        """Initialisiere das LSTM-Modell.
        
        Args:
            input_dim: Anzahl der Input-Features
            hidden_dim: Größe der Hidden-Layer
            num_layers: Anzahl der LSTM-Layer
            dropout: Dropout-Rate
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward-Pass durch das Modell.
        
        Args:
            x: Input-Tensor der Form (batch_size, seq_len, input_dim)
            
        Returns:
            Vorhersagen der Form (batch_size, 1)
        """
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out


class ModelTrainer:
    """Klasse für das Training des LSTM-Modells."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialisiere den Trainer.
        
        Args:
            config: Optionale Konfiguration für das Training
        """
        self.config = config or TRADING_CONFIG
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = MinMaxScaler()
        
    def prepare_sequences(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Bereite Sequenzen für das Training vor.
        
        Args:
            data: DataFrame mit Features
            
        Returns:
            X: Feature-Sequenzen
            y: Zielvariablen
        """
        # Skaliere die Daten
        scaled_data = self.scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(len(scaled_data) - self.config['lookback_period']):
            X.append(scaled_data[i:(i + self.config['lookback_period'])])
            y.append(scaled_data[i + self.config['lookback_period'], 0])  # Close-Preis
            
        return np.array(X), np.array(y)
        
    def train(self, X: np.ndarray, y: np.ndarray, model: LSTMPredictor) -> LSTMPredictor:
        """Trainiere das Modell.
        
        Args:
            X: Feature-Sequenzen
            y: Zielvariablen
            model: LSTM-Modell
            
        Returns:
            Trainiertes Modell
        """
        model = model.to(self.device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config['learning_rate'])
        
        # Erstelle DataLoader
        dataset = TimeSeriesDataset(X, y)
        train_loader = DataLoader(
            dataset,
            batch_size=self.config['batch_size'],
            shuffle=True
        )
        
        # Training Loop
        model.train()
        for epoch in range(self.config['epochs']):
            total_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Forward pass
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                
                # Backward pass und optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            avg_loss = total_loss / len(train_loader)
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch [{epoch+1}/{self.config['epochs']}], Loss: {avg_loss:.4f}")
                
        return model
        
    def save_model(self, model: LSTMPredictor, symbol: str) -> None:
        """Speichere das trainierte Modell.
        
        Args:
            model: Trainiertes LSTM-Modell
            symbol: Symbol des Instruments
        """
        try:
            model_path = MODELS_DIR / f"{symbol}_lstm.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'scaler': self.scaler
            }, model_path)
            logger.info(f"Modell gespeichert in {model_path}")
            
        except Exception as e:
            logger.error(f"Fehler beim Speichern des Modells: {str(e)}")
            raise
            
    def load_model(self, symbol: str) -> Tuple[LSTMPredictor, MinMaxScaler]:
        """Lade ein trainiertes Modell.
        
        Args:
            symbol: Symbol des Instruments
            
        Returns:
            Modell und Scaler
        """
        try:
            model_path = MODELS_DIR / f"{symbol}_lstm.pth"
            checkpoint = torch.load(model_path)
            
            input_dim = self.config['lookback_period']
            model = LSTMPredictor(input_dim)
            model.load_state_dict(checkpoint['model_state_dict'])
            scaler = checkpoint['scaler']
            
            return model, scaler
            
        except Exception as e:
            logger.error(f"Fehler beim Laden des Modells: {str(e)}")
            raise

    def predict(self, model: LSTMPredictor, X: np.ndarray) -> np.ndarray:
        """Generiere Vorhersagen mit dem trainierten Modell.
        
        Args:
            model: Trainiertes LSTM-Modell
            X: Eingabedaten
            
        Returns:
            Array mit Vorhersagen
        """
        model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            scaled_predictions = model(X_tensor).numpy().flatten()
            
            # Rücktransformation in den originalen Wertebereich
            # Erstelle eine temporäre Matrix für die Reskalierung
            temp = np.zeros((len(scaled_predictions), 1))
            temp[:, 0] = scaled_predictions
            
            # Anwenden der inversen Transformation
            original_predictions = self.scaler.inverse_transform(temp)[:, 0]
            
            return original_predictions 