import warnings
warnings.filterwarnings("ignore")

import os
from typing import Optional
import joblib
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GridSearchCV, KFold
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.util import Surv
from .load_data import (
    X_train, X_test,
    y_train, y_test,
    event_train, event_test, 
) 


OUTPUT_DIR = "data/resultados/GradientBoostingSurvivalAnalysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class BaseEstimatorWrapper(BaseEstimator, RegressorMixin):
    """
    Wrapper sklearn-friendly para GradientBoostingSurvivalAnalysis:
    """
    def __init__(self, **kwargs):
        self.est_params = {}
        for k, v in kwargs.items():
            self.est_params[k] = v
        self.model: Optional[GradientBoostingSurvivalAnalysis] = None

    def fit(self, X, y):
        self.model = GradientBoostingSurvivalAnalysis(**self.est_params)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """
        Retorna a curva de sobrevivência (n_amostras x times).
        """
        return self.model.predict_survival_function(X, return_array=True)

    def score(self, X, y):
        """
        C-index score (quanto maior, melhor).
        """
        return float(self.model.score(X, y))

    def get_params(self, deep=True):
        params = {}
        params.update(self.est_params)
        return params

    def set_params(self, **params):
        for k, v in params.items():
            self.est_params[k] = v
        return self


################################################################################


if __name__ == "__main__":
    Y_train = Surv.from_arrays(event_train, y_train, name_event="falha", name_time="time_years")
    Y_test = Surv.from_arrays(event_test, y_test, name_event="falha", name_time="time_years")

    param_grid = {
        "loss": ["coxph"], 
        "learning_rate": [0.05, 0.1],
        "n_estimators": [200, 400],
        "subsample": [0.6, 1.0], 
        "max_depth": [2, 3],
        "min_samples_split": [2, 6],
        "min_samples_leaf": [1, 3],
        "max_features": ["sqrt", None],
        "dropout_rate": [0.0], 
        "ccp_alpha": [0.0],
    }
    
    est = BaseEstimatorWrapper()

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    gcv = GridSearchCV(
        estimator=est,
        param_grid=param_grid,
        scoring=None, # est.score (C-index)
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=10
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
