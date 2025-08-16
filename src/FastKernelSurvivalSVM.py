import warnings
warnings.filterwarnings("ignore")

import os
from typing import Any, Dict, Optional
import joblib
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GridSearchCV, KFold

from sksurv.metrics import concordance_index_censored
from sksurv.kernels import clinical_kernel
from sksurv.svm import FastKernelSurvivalSVM
from sksurv.util import Surv

from .load_data import (
    X_train, X_test,
    y_train, y_test,
    event_train, event_test,
)

OUTPUT_DIR = "data/resultados/FastKernelSurvivalSVM"
os.makedirs(OUTPUT_DIR, exist_ok=True)


class KernelSurvivalSVMWrapper(BaseEstimator, RegressorMixin):
    """
    Wrapper sklearn-friendly para FastKernelSurvivalSVM usando clinical_kernel.
    """
    def __init__(self, **kwargs):
        self.est_params: Dict[str, Any] = kwargs
        self.model: Optional[FastKernelSurvivalSVM] = None
        self.X_train_categorical = None

    def fit(self, X, y):
        import pandas as pd
        self.X_train_categorical = pd.DataFrame(X).astype("category")
        kernel_matrix = clinical_kernel(self.X_train_categorical)
        self.model = FastKernelSurvivalSVM(**self.est_params)
        self.model.fit(kernel_matrix, y)
        return self

    def predict(self, X):
        import pandas as pd
        X_categorical = pd.DataFrame(X).astype("category")
        kernel_matrix = clinical_kernel(X_categorical, self.X_train_categorical)
        return self.model.predict(kernel_matrix)


    def score(self, X, y):
        prediction = self.predict(X)
        result = concordance_index_censored(y["falha"], y["time_years"], prediction)
        return float(result[0])

    def get_params(self, deep=True):
        return dict(self.est_params)

    def set_params(self, **params):
        for k, v in params.items():
            self.est_params[k] = v
        return self


if __name__ == "__main__":
    # Criação do objeto Y no formato esperado
    Y_train = Surv.from_arrays(event_train, y_train, name_event="falha", name_time="time_years")
    Y_test = Surv.from_arrays(event_test, y_test, name_event="falha", name_time="time_years")

    # Grade de hiperparâmetros para busca
    param_grid = {
        "kernel": ["precomputed"], 
        "alpha": [0.01, 0.1, 1.0],
        "max_iter": [50, 100, 200]
    }

    est = KernelSurvivalSVMWrapper()

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    gcv = GridSearchCV(
        estimator=est,
        param_grid=param_grid,
        scoring=None,  # usa o score() definido no wrapper (C-index)
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=10
    )

    # Ajuste do modelo com busca em grade
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
    for i in range(cv.get_n_splits()):
        fold_scores.append(cv_results.loc[best_index, f"split{i}_test_score"])
        
    # Melhor modelo
    best_model = gcv.best_estimator_
    joblib.dump(best_model, f"{OUTPUT_DIR}/melhor_modelo.pkl")

    # Métricas de avaliação
    with open(f"{OUTPUT_DIR}/metricas_avaliacao.txt", "w", encoding="utf-8") as f:
        f.write("Hiperparâmetros ótimos:\n")
        f.write(str(gcv.best_params_))
        f.write(f"\nC-index treinamento: {best_model.score(X_train, Y_train)}\n")
        f.write(f"C-index teste: {best_model.score(X_test, Y_test)}\n")
        f.write(f"\nMédias de C-index para melhor fold: {fold_scores}\n")
