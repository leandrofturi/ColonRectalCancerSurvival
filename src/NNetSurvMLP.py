import warnings
warnings.filterwarnings("ignore")

import os
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, KFold
from .load_data import (
    X_train, X_test,
    y_train, y_test,
    event_train, event_test, 
) 

import sys
sys.path.insert(0,'./') # including the path to deep-tasks folder
sys.path.insert(0,'./nnet-survival') # including the path to nnet-survival folder
from nnet_survival_pytorch import make_surv_array, surv_likelihood

# Diretório para salvar os resultados deste modelo
OUTPUT_DIR = "data/resultados/NNetSurvMLP"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class NNetSurvMLP(nn.Module):
    """
    Criação do modelo MLP em pytorch
    """
    def __init__(self, in_features, hidden=(64, 64), K=100, dropout=0.1):
        super().__init__()
        layers, prev = [], in_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            prev = h
        layers += [nn.Linear(prev, K)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class NNetSurvEstimator(BaseEstimator):
    """
    Wrapper sklearn-friendly para NNetSurvEstimator:
    """
    def __init__(
        self,
        in_features,
        K=20,
        hidden=(64, 64),
        dropout=0.1,
        lr=1e-3,
        weight_decay=1e-5,
        batch_size=256,
        epochs=50,
        val_size=0.2,
        patience=10,
        device=None,
        seed=42,
    ):
        self.in_features = in_features
        self.K = K
        self.hidden = hidden
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.val_size = val_size
        self.patience = patience
        self.device = device
        self.seed = seed

    # --- helpers ---
    def _split_y(self, y):
        y = np.asarray(y)
        if y.ndim == 1 and y.dtype.names is not None:
            times = y['time'].astype(float)
            events = y['event'].astype(int)
        else:
            assert y.ndim == 2 and y.shape[1] == 2, "y deve ter shape (n, 2): [time, event]."
            times = y[:, 0].astype(float)
            events = y[:, 1].astype(int)
        return times, events

    def _compute_breaks(self, times, K):
        qs = np.linspace(0, 1, K + 1)
        br = np.quantile(times, qs)
        br = np.unique(br)
        if br.shape[0] < 2:
            br = np.array([times.min(), times.max() + 1e-6])
        K_eff = br.shape[0] - 1
        return br, K_eff

    def _make_loaders(self, X, Y, batch_size, val_size, seed):
        n = X.shape[0]
        rng = np.random.RandomState(seed)
        idx = np.arange(n)
        rng.shuffle(idx)
        n_val = int(round(val_size * n))
        val_idx = idx[:n_val]
        tr_idx = idx[n_val:] if n_val < n else idx

        Xtr = torch.as_tensor(X[tr_idx], dtype=torch.float32)
        Ytr = torch.as_tensor(Y[tr_idx], dtype=torch.float32)
        Xva = torch.as_tensor(X[val_idx], dtype=torch.float32) if n_val > 0 else None
        Yva = torch.as_tensor(Y[val_idx], dtype=torch.float32) if n_val > 0 else None

        train_loader = DataLoader(TensorDataset(Xtr, Ytr), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(Xva, Yva), batch_size=max(512, batch_size), shuffle=False) if n_val > 0 else None
        return train_loader, val_loader

    def _p_surv(self, logits):
        return torch.sigmoid(logits).clamp(1e-7, 1 - 1e-7)

    # --- sklearn API ---
    def fit(self, X, y):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device_ = torch.device(device)

        times, events = self._split_y(y)
        self.breaks_, self.K_eff_ = self._compute_breaks(times, self.K)
        Y = make_surv_array(times, events, self.breaks_)

        train_loader, val_loader = self._make_loaders(X, Y, self.batch_size, self.val_size, self.seed)

        self.model_ = NNetSurvMLP(
            in_features=self.in_features,
            hidden=self.hidden,
            K=self.K_eff_,
            dropout=self.dropout
        ).to(self.device_)

        self.loss_fn_ = surv_likelihood(int(self.K_eff_))
        opt = torch.optim.Adam(self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_val = float("inf")
        best_state = None
        bad = 0

        for ep in range(1, self.epochs + 1):
            self.model_.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device_), yb.to(self.device_)
                opt.zero_grad()
                logits = self.model_(xb)
                p_surv = self._p_surv(logits)
                loss = self.loss_fn_(p_surv, yb)
                loss.backward()
                opt.step()

            cur_val = 0.0
            n = 0
            if val_loader is not None:
                self.model_.eval()
                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb, yb = xb.to(self.device_), yb.to(self.device_)
                        logits = self.model_(xb)
                        p_surv = self._p_surv(logits)
                        l = self.loss_fn_(p_surv, yb).item()
                        cur_val += l * xb.size(0)
                        n += xb.size(0)
                cur_val = cur_val / max(n, 1)

                if cur_val + 1e-8 < best_val:
                    best_val = cur_val
                    best_state = {k: v.detach().cpu().clone() for k, v in self.model_.state_dict().items()}
                    bad = 0
                else:
                    bad += 1
                    if bad >= self.patience:
                        break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        return self

    def score(self, X, y):
        times, events = self._split_y(y)
        Y = make_surv_array(times, events, self.breaks_)
        loader = DataLoader(
            TensorDataset(torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(Y, dtype=torch.float32)),
            batch_size=max(512, self.batch_size),
            shuffle=False
        )
        self.model_.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(self.device_), yb.to(self.device_)
                logits = self.model_(xb)
                p_surv = self._p_surv(logits)
                l = self.loss_fn_(p_surv, yb).item()
                tot += l * xb.size(0)
                n += xb.size(0)
        return -tot / max(n, 1)

    def predict_survival_function(self, X):
        self.model_.eval()
        Xte = torch.as_tensor(X, dtype=torch.float32).to(self.device_)
        with torch.no_grad():
            logits = self.model_(Xte)
            p_surv_int = self._p_surv(logits).cpu().numpy()
        S = np.cumprod(p_surv_int, axis=1)
        return S

################################################################################

if __name__ == "__main__":
    Y_train_nn = np.c_[y_train, event_train]
    Y_test_nn = np.c_[y_test, event_test]

    # Instancia o estimador
    est = NNetSurvEstimator(
        in_features=X_train.shape[1],
        device="cuda" if torch.cuda.is_available() else "cpu",
        epochs=80,
        patience=12,
        val_size=0.2,
        seed=123
    )
    
    # Grade de hiperparâmetros para testar
    param_grid = {
        "K": [20, 30],
        "hidden": [(64, 64), (128, 64)],
        "dropout": [0.1, 0.2],
        "lr": [1e-3, 5e-4],
        "weight_decay": [1e-5, 1e-4],
        "batch_size": [128, 256],
    }

    # Hiper para testes internos
    # param_grid = {
    #     "K": [20],
    #     "hidden": [(64, 64)],
    #     "dropout": [0.1],
    #     "lr": [1e-3],
    #     "weight_decay": [1e-5],
    #     "batch_size": [128],
    # }
    
    # Configura a validação cruzada
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    gcv = GridSearchCV(
        estimator=est,
        param_grid=param_grid,
        scoring=None,
        cv=cv,
        n_jobs=1,
        refit=True,
        verbose=10
    )
    
    print("Iniciando a busca de hiperparâmetros para NNetSurvMLP...")
    gcv.fit(X_train, Y_train_nn)

    print("\nBusca finalizada!")
    print("Melhor score (neg-loss na CV):", gcv.best_score_)
    print("Melhores parâmetros:", gcv.best_params_)

    # 1. Salvar os resultados da validação cruzada em CSV
    cv_results = pd.DataFrame(gcv.cv_results_)
    cv_summary = cv_results[["params", "mean_test_score", "std_test_score", "rank_test_score"]]
    cv_summary = cv_summary.sort_values(by="rank_test_score", ascending=True)
    cv_summary.to_csv(f"{OUTPUT_DIR}/cv_resultados.csv", index=False)
    print(f"\nResultados da validação cruzada salvos em: {OUTPUT_DIR}/cv_resultados.csv")

    # 2. Salvar o melhor modelo
    best_model = gcv.best_estimator_
    joblib.dump(best_model, f"{OUTPUT_DIR}/melhor_modelo.pkl")
    print(f"Melhor modelo salvo em: {OUTPUT_DIR}/melhor_modelo.pkl")

    # 3. Salvar as métricas de avaliação em TXT
    # Extrai os scores de cada fold para o melhor conjunto de parâmetros
    best_index = cv_results["rank_test_score"].idxmin()
    fold_scores = []
    for i in range(cv.get_n_splits()):
        fold_scores.append(cv_results.loc[best_index, f"split{i}_test_score"])

    with open(f"{OUTPUT_DIR}/metricas_avaliacao.txt", "w", encoding="utf-8") as f:
        f.write("Hiperparâmetros ótimos:\n")
        f.write(str(gcv.best_params_))
        f.write("\n\n--- Métricas (baseadas na Negative Log-Likelihood) ---\n")
        f.write(f"Score no conjunto de treino completo: {best_model.score(X_train, Y_train_nn)}\n")
        f.write(f"Score no conjunto de teste: {best_model.score(X_test, Y_test_nn)}\n")
        f.write("\n--- Validação Cruzada ---\n")
        f.write(f"Melhor score médio na CV: {gcv.best_score_}\n")
        f.write(f"Scores do melhor modelo em cada fold: {fold_scores}\n")
    
    print(f"Métricas de avaliação salvas em: {OUTPUT_DIR}/metricas_avaliacao.txt")