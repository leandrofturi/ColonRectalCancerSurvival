import warnings
warnings.filterwarnings("ignore")

# https://github.com/RyanWangZf/SurvTRACE/blob/main/experiment_seer.ipynb

import os
import dill
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.model_selection import train_test_split
import torchtuples as tt
from .SurvTRACE.survtrace.model import SurvTraceSingle
from .SurvTRACE.survtrace import Evaluator
from .SurvTRACE.survtrace import Trainer
from .SurvTRACE.survtrace import STConfig
from .SurvTRACE.survtrace.utils import LabelTransform
from .load_data import (
    X_train, X_test,
    y_train, y_test,
    event_train, event_test, 
) 


OUTPUT_DIR = "data/resultados/SurvTRACE"
os.makedirs(OUTPUT_DIR, exist_ok=True)


np.Inf = np.inf

class BaseEstimatorWrapper(BaseEstimator, RegressorMixin):
    """
    Wrapper sklearn-friendly para SurvTRACE:
    """
    def __init__(self, **kwargs):
        self.est_params = {}
        self.model_params = {}
        self.STConfig = None

        for k, v in kwargs.items():
            if k.startswith("model__STConfig"):
                self.STConfig = v
            elif k.startswith("model__"):
                self.model_params[k.replace("model__", "")] = v
            else:
                self.est_params[k] = v
        
        for k,v in self.est_params.items():
            self.STConfig[k] = v
        
        self.model = None
        self.df_train = None

    def fit(self, X, y):
        batch_size = 21
        epochs = 256
        
        self.df_train = pd.concat([X, y], axis=1)
        __x_train, __x_test, __y_train, __y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        self.model = SurvTraceSingle(self.STConfig)
        self.trainer = Trainer(self.model)
        self.trainer.fit(
            (__x_train, __y_train),
            (__x_test, __y_test),
            batch_size = batch_size,
            val_batch_size = int(batch_size/3),
            epochs = epochs,
            **self.model_params)
        
        return self

    def predict(self, X):
        """
        Retorna a curva de sobrevivência (n_amostras x times).
        """
        surv = self.model.predict_surv_df(X)
        return surv

    def score(self, X, y):
        """
        C-index score (quanto maior, melhor).
        """
        # evaluating

        df = pd.concat([self.df_train, pd.concat([X, y], axis=1)], ignore_index=True)

        evaluator = Evaluator(df, self.df_train.index)
        eval = evaluator.eval(self.model, (X, y))
        m = float(np.mean([v for k,v in eval.items() if 'ipcw' in k]))
        return m

    def get_params(self, deep=True):
        params = {}
        params["model__STConfig"] = self.STConfig
        params.update({f"model__{k}": v for k, v in self.model_params.items()})
        params.update(self.est_params)
        return params

    def set_params(self, **params):
        for k, v in params.items():
            if k == "model__STConfig":
                self.STConfig = v
            elif k.startswith("model__"):
                self.model_params[k.replace("model__", "")] = v
            else:
                self.est_params[k] = v
        return self


################################################################################


if __name__ == "__main__":

    STConfig['data'] = 'load'

    df_train, df_test = pd.DataFrame(X_train), pd.DataFrame(X_test)

    horizons = [.25, .5, .75] # the discrete intervals are cut at 0%, 25%, 50%, 75%, 100%
    time_years = np.concatenate((
        y_train[event_train == 1],
        y_test[event_test == 1]
    ))
    times = np.quantile(time_years, horizons).tolist()

    time_years = np.concatenate((
        y_train,
        y_test
    ))
    cuts = np.unique([time_years.min()] + times + [time_years.max()])

    labtrans = LabelTransform(cuts=np.array(cuts))
    labtrans.fit(y_train, event_train.astype(int))

    y_lab_train = labtrans.transform(y_train, event_train.astype(int))
    y_lab_test = labtrans.transform(y_test, event_test.astype(int))

    df_y_train = pd.DataFrame({'duration': y_lab_train[0], 'event': y_lab_train[1], 'proportion': y_lab_train[2]})
    df_y_test = pd.DataFrame({'duration': y_lab_test[0], 'event': y_lab_test[1], 'proportion': y_lab_test[2]})

    STConfig['num_categorical_feature'] = 0
    STConfig['num_numerical_feature'] = X_train.shape[1]
    STConfig['num_feature'] = X_train.shape[1]

    STConfig['labtrans'] = labtrans
    STConfig['vocab_size'] = 0
    STConfig['duration_index'] = labtrans.cuts
    STConfig['out_feature'] = len(cuts) - 1

    param_grid = {
        "num_hidden_layers": [2, 4],
        "hidden_size": [16, 32],
        "intermediate_size": [64],
        "num_attention_heads": [2, 8],
        "model__learning_rate": [1e-3, 5e-4],
        "model__weight_decay": [1e-5, 1e-4],
        "model__STConfig": [STConfig],
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

    gcv.fit(df_train, df_y_train)

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

    # Remover callbacks/histórico do objeto pycox/torchtuples
    m = getattr(best_model, "model", None)
    if m is not None:
        for attr in ["callbacks", "log", "cb", "_callbacks"]:
            if hasattr(m, attr):
                try:
                    setattr(m, attr, [])
                except Exception:
                    pass

    # Salvar modelo
    best_model = gcv.best_estimator_
    
    with open(f"{OUTPUT_DIR}/melhor_modelo.pkl", "wb") as f:
        dill.dump(best_model, f)

    # Métricas de avaliação
    with open(f"{OUTPUT_DIR}/metricas_avaliacao.txt", "w", encoding="utf-8") as f:
        f.write("Hiperparâmetros ótimos:\n")
        f.write(str(gcv.best_params_))
        f.write(f"\nC-index treinamento: {best_model.score(df_train, df_y_train)}\n")
        f.write(f"C-index teste: {best_model.score(df_test, df_y_test)}\n")
        f.write(f"\nMédias de C-index para melhor fold: {fold_scores}\n")
