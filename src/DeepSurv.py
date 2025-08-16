import warnings
warnings.filterwarnings("ignore")

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GridSearchCV, KFold
import torchtuples as tt
from pycox.models import CoxPH
from pycox.evaluation import EvalSurv
from .load_data import (
    X_train, X_test,
    y_train, y_test,
    event_train, event_test, 
) 


OUTPUT_DIR = "data/resultados/DeepSurv"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class BaseEstimatorWrapper(BaseEstimator, RegressorMixin):
    """
    Wrapper sklearn-friendly para DeepSurv:
    """
    def __init__(self, **kwargs):
        self.est_params = {}
        for k, v in kwargs.items():
            self.est_params[k] = v
        self.model = None

    def fit(self, X, y):
        _y = y[:, 0].astype(np.float32)
        _event = y[:, 1].astype(np.float32)
        Y = (_y, _event)

        in_features = X.shape[1]
        out_features = 1
        batch_size = 256
        epochs = 512
        callbacks = [tt.callbacks.EarlyStopping()]

        net = tt.practical.MLPVanilla(in_features=in_features, 
                                      out_features=out_features, 
                                      **self.est_params)
        self.model = CoxPH(net, tt.optim.Adam)
        self.model.optimizer.set_lr(0.01)
        self.model.fit(X, Y, 
                       callbacks=callbacks, batch_size=batch_size, epochs=epochs, verbose=False)
        return self

    def predict(self, X):
        """
        Retorna a curva de sobrevivência (n_amostras x times).
        """
        _ = self.model.compute_baseline_hazards()
        surv = self.model.predict_surv_df(X)
        return surv

    def score(self, X, y):
        """
        C-index score (quanto maior, melhor).
        """
        surv = self.predict(X)
        ev = EvalSurv(surv, y[:, 0].astype(np.float32), y[:, 1].astype(np.float32), censor_surv='km')
        return float(ev.concordance_td())

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
    Y_train = np.column_stack([y_train, event_train])
    Y_test = np.column_stack([y_test, event_test])

    param_grid = {
        "num_nodes": [[32, 32], [64, 32], [128, 64, 32]],
        "batch_norm": [True, False],
        "dropout": [None, 0.1],
        "output_bias": [True, False],
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
