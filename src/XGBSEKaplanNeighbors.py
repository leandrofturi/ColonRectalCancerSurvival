import warnings
warnings.filterwarnings("ignore")

import os
from typing import Any, Dict, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GridSearchCV, KFold
from xgbse import XGBSEKaplanNeighbors
from xgbse.converters import convert_to_structured
from xgbse.metrics import concordance_index
from .load_data import (
    X_train, X_test,
    y_train, y_test,
    event_train, event_test, 
) 


OUTPUT_DIR = "data/resultados/XGBSEKaplanNeighbors"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class BaseEstimatorWrapper(BaseEstimator, RegressorMixin):
    """
    Wrapper sklearn-friendly para XGBSEKaplanNeighbors:
      - suporte a grid de xgb_params__*
      - suporte a grid de parâmetros específicos do KaplanNeighbors
      - time_bins como hiperparâmetro
    """
    def __init__(self, time_bins=None, **kwargs):
        self.xgb_params: Dict[str, Any] = {}
        self.est_params: Dict[str, Any] = {}
        self.time_bins = time_bins

        for k, v in kwargs.items():
            if k == "time_bins":
                self.time_bins = v
            elif k.startswith("xgb_params__"):
                self.xgb_params[k.replace("xgb_params__", "")] = v
            else:
                self.est_params[k] = v
        self.model: Optional[XGBSEKaplanNeighbors] = None

    def fit(self, X, y):
        self.model = XGBSEKaplanNeighbors(xgb_params=self.xgb_params, **self.est_params)
        self.model.fit(X, y, time_bins=self.time_bins)
        return self

    def predict(self, X):
        """
        Retorna a curva de sobrevivência (n_amostras x n_time_bins).
        """
        return self.model.predict(X, time_bins=self.time_bins)

    def score(self, X, y):
        """
        C-index score (quanto maior, melhor).
        """
        pred = self.predict(X)
        return float(concordance_index(y, pred))

    def get_params(self, deep=True):
        params = {}
        params.update({f"xgb_params__{k}": v for k, v in self.xgb_params.items()})
        params.update(self.est_params)
        return params

    def set_params(self, **params):
        for k, v in params.items():
            if k == "time_bins":
                self.time_bins = v
            elif k.startswith("xgb_params__"):
                self.xgb_params[k.replace("xgb_params__", "")] = v
            else:
                self.est_params[k] = v
        return self


################################################################################


if __name__ == "__main__":
    Y_train = convert_to_structured(pd.Series(y_train), pd.Series(event_train))
    Y_test = convert_to_structured(pd.Series(y_test), pd.Series(event_test))

    # Time bins
    __times = np.unique(np.concatenate([y_train, y_test]))
    __n = (len(__times) // 3) * 3 # minimo 3 eventos por bin
    __bins = __times[:__n].reshape(-1, 3)
    TIME_BINS = []
    for i in range(0, len(__bins) - 1):
        val = __bins[i][2] + (__bins[i+1][0] - __bins[i][2])/2
        TIME_BINS.append(val)
    TIME_BINS = np.array(TIME_BINS)


    param_grid = {
        "xgb_params__n_estimators": [50, 250],
        "xgb_params__max_depth": [3, 5, 7],
        "xgb_params__learning_rate": [0.05, 0.1],
        "xgb_params__objective": ["survival:aft", "survival:cox"],
        "xgb_params__eval_metric": ["mlogloss", "merror"],
        "n_neighbors": [50, 100, 200],
    }
    
    est = BaseEstimatorWrapper(time_bins=TIME_BINS)

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    gcv = GridSearchCV(
        estimator=est,
        param_grid=param_grid,
        scoring=None, # est.score (C-index)
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=1
    )

    gcv.fit(X_train, Y_train)

    print("Melhor C-index (CV):", gcv.best_score_)
    print("Melhores parâmetros:", gcv.best_params_)

    # Resultados da validação cruzada
    cv_results = pd.DataFrame(gcv.cv_results_)
    cv_summary = cv_results[["params", "mean_test_score", "std_test_score"]]
    cv_summary = cv_summary.sort_values(by="mean_test_score", ascending=False)
    cv_summary.to_csv(f"{OUTPUT_DIR}/cv_resultados.csv", index=False)


    # Resultados fold-a-fold do melhor modelo
    best_index = cv_results["rank_test_score"].idxmin()
    fold_scores = []
    for i in range(5):
        fold_scores.append(cv_results.loc[best_index, f"split{i}_test_score"])

    best_model = gcv.best_estimator_

    # Salvar modelo
    best_model = gcv.best_estimator_
    joblib.dump(best_model, f"{OUTPUT_DIR}/melhor_modelo.pkl")

    # Métricas de avaliação
    with open(f"{OUTPUT_DIR}/metricas_avaliacao.txt", "w", encoding="utf-8") as f:
        f.write("Hiperparâmetros ótimos:\n")
        f.write(str(gcv.best_params_))
        f.write(f"\nC-index treinamento: {best_model.score(X_train, Y_train)}\n")
        f.write(f"C-index teste: {best_model.score(X_test, Y_test)}\n")
        f.write(f"\nMédias de C-index para melhor fold: {fold_scores}\n")
