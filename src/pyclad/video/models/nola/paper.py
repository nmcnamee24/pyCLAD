"""Paper-faithful k-DNN and decision-RNN model for continual NOLA VAD."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

from pyclad.video.data.matrix import VideoStrategySchema
from pyclad.video.models.base import VideoAnomalyModel
from pyclad.video.prediction_results import VideoPredictionResults


class NolaPaperModel(VideoAnomalyModel):
    """NOLA's k-DNN regression and two-step decision-RNN architecture.

    The WACV paper describes a three-hidden-layer k-DNN with 20 neurons per
    layer, followed by a single-layer LSTM that consumes two kNN-distance time
    steps. Synthetic anomalous distances are sampled between the 95th
    percentile of nominal distances and twice that value.

    The model accepts ordinary two-dimensional feature matrices, so pyCLAD's
    existing cumulative, MSTE, and replay strategies remain unchanged.
    """

    def __init__(
        self,
        feature_dim: int,
        *,
        strategy_schema: Optional[VideoStrategySchema] = None,
        neighbors: int = 5,
        hidden_dim: int = 20,
        decision_hidden_dim: int = 20,
        kdnn_epochs: int = 20,
        decision_epochs: int = 10,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        seed: int = 42,
        device: str = "cpu",
        threshold: float = 0.5,
    ):
        if (
            min(
                feature_dim,
                neighbors,
                hidden_dim,
                decision_hidden_dim,
                kdnn_epochs,
                decision_epochs,
                batch_size,
            )
            <= 0
        ):
            raise ValueError("NOLA paper-model dimensions and training counts must be positive")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between zero and one")

        import torch
        from torch import nn

        self._torch = torch
        self._nn = nn
        self.feature_dim = int(feature_dim)
        self.strategy_schema = strategy_schema or VideoStrategySchema(feature_dim=self.feature_dim)
        if self.strategy_schema.feature_dim != self.feature_dim:
            raise ValueError("strategy_schema feature_dim must match feature_dim")
        self.neighbors = int(neighbors)
        self.hidden_dim = int(hidden_dim)
        self.decision_hidden_dim = int(decision_hidden_dim)
        self.kdnn_epochs = int(kdnn_epochs)
        self.decision_epochs = int(decision_epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.seed = int(seed)
        self.device = torch.device(device)
        self.threshold = float(threshold)

        self._scaler: Optional[MinMaxScaler] = None
        self._kdnn = self._build_kdnn().to(self.device)
        self._decision_rnn = self._build_decision_rnn().to(self.device)
        self._fit_calls = 0
        self._fit_rows = 0
        self._nominal_distance_p95 = 0.0
        self._kdnn_training_mse = float("nan")

    def _build_kdnn(self):
        nn = self._nn
        return nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1),
            nn.Softplus(),
        )

    def _build_decision_rnn(self):
        nn = self._nn

        class DecisionRnn(nn.Module):
            def __init__(self, hidden_dim: int):
                super().__init__()
                self.recurrent = nn.LSTM(
                    input_size=1,
                    hidden_size=hidden_dim,
                    num_layers=1,
                    batch_first=True,
                )
                self.output = nn.Linear(hidden_dim, 1)

            def forward(self, distances):
                sequence, _ = self.recurrent(distances)
                return self.output(sequence[:, -1]).reshape(-1)

        return DecisionRnn(self.decision_hidden_dim)

    def fit(self, data: np.ndarray):
        features = self._features(data)
        if not len(features):
            raise ValueError("NOLA paper model cannot fit an empty feature matrix")
        if not np.isfinite(features).all():
            raise ValueError("NOLA paper features must contain only finite values")

        self._scaler = MinMaxScaler().fit(features)
        normalized = self._scaler.transform(features).astype(np.float32, copy=False)
        exact_distances = self._exact_knn_distances(normalized).astype(np.float32, copy=False)
        self._train_kdnn(normalized, exact_distances)

        predicted_distances = self._predict_distances_normalized(normalized)
        self._kdnn_training_mse = float(np.mean(np.square(predicted_distances - exact_distances)))
        self._nominal_distance_p95 = float(np.quantile(exact_distances, 0.95))
        self._train_decision_rnn(predicted_distances)

        self._fit_calls += 1
        self._fit_rows = len(features)
        return self

    def predict(self, data: np.ndarray) -> VideoPredictionResults:
        distances = self.raw_distance_scores(data)
        sequences = self._distance_sequences(distances)
        torch = self._torch
        self._decision_rnn.eval()
        probabilities = []
        with torch.no_grad():
            for offset in range(0, len(sequences), self.batch_size):
                batch = torch.from_numpy(sequences[offset : offset + self.batch_size]).to(self.device)
                logits = self._decision_rnn(batch)
                probabilities.append(torch.sigmoid(logits).cpu().numpy())
        anomaly_scores = (
            np.concatenate(probabilities).astype(np.float64, copy=False)
            if probabilities
            else np.empty(0, dtype=np.float64)
        )
        return VideoPredictionResults(
            y_pred=(anomaly_scores >= self.threshold).astype(np.int64),
            anomaly_scores=anomaly_scores,
            window_scores=anomaly_scores.copy(),
        )

    def raw_distance_scores(self, data: np.ndarray) -> np.ndarray:
        if self._scaler is None or self._fit_calls == 0:
            raise RuntimeError("NOLA paper model must be fitted before scoring")
        features = self._features(data, allow_features_only=True)
        normalized = self._scaler.transform(features).astype(np.float32, copy=False)
        return self._predict_distances_normalized(normalized).astype(np.float64, copy=False)

    def _exact_knn_distances(self, normalized: np.ndarray) -> np.ndarray:
        if len(normalized) == 1:
            return np.zeros(1, dtype=np.float64)
        effective = min(self.neighbors + 1, len(normalized))
        distances = (
            NearestNeighbors(n_neighbors=effective)
            .fit(normalized)
            .kneighbors(
                normalized,
                return_distance=True,
            )[0]
        )
        distances = distances[:, 1:] if distances.shape[1] > 1 else distances
        return distances.sum(axis=1)

    def _train_kdnn(self, features: np.ndarray, targets: np.ndarray) -> None:
        torch = self._torch
        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(features),
            torch.from_numpy(targets[:, None]),
        )
        generator = torch.Generator().manual_seed(self.seed + self._fit_calls)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
        )
        optimizer = torch.optim.Adam(self._kdnn.parameters(), lr=self.learning_rate)
        criterion = self._nn.MSELoss()
        self._kdnn.train()
        for _ in range(self.kdnn_epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                prediction = self._kdnn(batch_x.to(self.device))
                loss = criterion(prediction, batch_y.to(self.device))
                loss.backward()
                optimizer.step()

    def _predict_distances_normalized(self, normalized: np.ndarray) -> np.ndarray:
        torch = self._torch
        self._kdnn.eval()
        predictions = []
        with torch.no_grad():
            for offset in range(0, len(normalized), self.batch_size):
                batch = torch.from_numpy(normalized[offset : offset + self.batch_size]).to(self.device)
                predictions.append(self._kdnn(batch).reshape(-1).cpu().numpy())
        return np.concatenate(predictions) if predictions else np.empty(0, dtype=np.float32)

    def _train_decision_rnn(self, nominal_distances: np.ndarray) -> None:
        torch = self._torch
        nominal_sequences = self._distance_sequences(nominal_distances)
        rng = np.random.default_rng(self.seed + self._fit_calls)
        lower = max(self._nominal_distance_p95, np.finfo(np.float32).eps)
        synthetic_sequences = nominal_sequences.copy()
        synthetic_sequences[:, -1, 0] = rng.uniform(lower, 2.0 * lower, len(synthetic_sequences))
        features = np.concatenate([nominal_sequences, synthetic_sequences]).astype(np.float32, copy=False)
        targets = np.concatenate(
            [
                np.zeros(len(nominal_sequences), dtype=np.float32),
                np.ones(len(synthetic_sequences), dtype=np.float32),
            ]
        )
        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(features),
            torch.from_numpy(targets),
        )
        generator = torch.Generator().manual_seed(self.seed + 10_000 + self._fit_calls)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
        )
        optimizer = torch.optim.Adam(self._decision_rnn.parameters(), lr=self.learning_rate)
        criterion = self._nn.BCEWithLogitsLoss()
        self._decision_rnn.train()
        for _ in range(self.decision_epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                logits = self._decision_rnn(batch_x.to(self.device))
                loss = criterion(logits, batch_y.to(self.device))
                loss.backward()
                optimizer.step()

    @staticmethod
    def _distance_sequences(distances: np.ndarray) -> np.ndarray:
        values = np.asarray(distances, dtype=np.float32).reshape(-1)
        if not len(values):
            return np.empty((0, 2, 1), dtype=np.float32)
        previous = np.concatenate(([values[0]], values[:-1]))
        return np.stack([previous, values], axis=1)[:, :, None]

    def _features(self, data: np.ndarray, *, allow_features_only: bool = False) -> np.ndarray:
        matrix = np.asarray(data, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError(f"NOLA paper data must be two-dimensional, got {matrix.shape}")
        if matrix.shape[1] == self.strategy_schema.matrix_width:
            return self.strategy_schema.features(matrix)
        if allow_features_only and matrix.shape[1] == self.feature_dim:
            return matrix
        if not self.strategy_schema.target_names and matrix.shape[1] == self.feature_dim:
            return matrix
        raise ValueError(
            f"NOLA paper data width must be {self.strategy_schema.matrix_width}"
            + (f" or {self.feature_dim}" if allow_features_only else "")
            + f", got {matrix.shape[1]}"
        )

    def name(self) -> str:
        return "NOLA-Paper"

    def additional_info(self) -> Dict[str, Any]:
        return {
            "neighbors": self.neighbors,
            "feature_dim": self.feature_dim,
            "kdnn_hidden_layers": [self.hidden_dim, self.hidden_dim, self.hidden_dim],
            "decision_rnn_hidden_dim": self.decision_hidden_dim,
            "decision_rnn_input_steps": 2,
            "kdnn_epochs": self.kdnn_epochs,
            "decision_epochs": self.decision_epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "nominal_distance_p95": self._nominal_distance_p95,
            "kdnn_training_mse": self._kdnn_training_mse,
            "fit_rows": self._fit_rows,
            "fit_calls": self._fit_calls,
            "device": str(self.device),
        }
