import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
from xgbse import XGBSEKaplanNeighbors
from xgbse.converters import convert_to_structured
from sksurv.metrics import concordance_index_censored
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.base import BaseEstimator


class XGBSEKaplanWrapper(BaseEstimator):
    def __init__(self, **kwargs):
        self.xgb_params = {}
        self.time_bins = None
        for k, v in kwargs.items():
            if k == "time_bins":
                self.time_bins = v
            elif k.startswith("xgb_params__"):
                self.xgb_params[k.replace("xgb_params__", "")] = v
        self.kn_model = XGBSEKaplanNeighbors(xgb_params=self.xgb_params)

    def fit(self, X, y):
        self.kn_model = XGBSEKaplanNeighbors(xgb_params=self.xgb_params)
        self.kn_model.fit(X, y, time_bins=self.time_bins)
        return self

    def predict(self, X):
        return self.kn_model.predict(X, time_bins=self.time_bins)

    def get_params(self, deep=True):
        return {f"xgb_params__{k}": v for k, v in self.xgb_params.items()}

    def set_params(self, **params):
        for k, v in params.items():
            if k.startswith("xgb_params__"):
                self.xgb_params[k.replace("xgb_params__", "")] = v
        return self


def process_data_file(data_file_csv, base_path, prefix):
    # Criar diretório para salvar resultados
    output_dir = base_path / prefix
    os.makedirs(output_dir, exist_ok=True)

    # Read the CSV file
    data = pd.read_csv(data_file_csv)

    # dados de saida
    y = data["time_years"]
    
    # transforma os dados em numeros
    y_encoded = convert_to_structured(pd.Series(data['falha']), data["time_years"])

    # dados de entrada filtrados
    X = data.drop(columns=["time_years_cat", "time_years", "falha"])
    # transforma os dados em numeros
    X_encoded = pd.get_dummies(X)

    # 3. Divisão treino/teste
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.25, random_state=42)

    # bins de 0.25 em 0.25 anos, até o tempo máximo observado
    time_max = data['time_years'].max()
    time_bins = np.arange(0, time_max + 0.25, 0.25)

    # Modelo base
    mc = XGBSEKaplanWrapper(time_bins=time_bins, enable_categorical=True)

    # Grade de hiperparâmetros para conjunto de dados multilabel
    param_grid = {
        'time_bins': [time_bins],
        'xgb_params__random_state': [42],
        'xgb_params__n_estimators': [50, 100, 200, 500],
        'xgb_params__max_depth': [2, 4, 6],
        'xgb_params__learning_rate': [0.01, 0.05, 0.1],
        # 'xgb_params__objective': ['multi:softmax'],  # softmax = classes
        'xgb_params__objective': ['survival:aft', 'survival:cox'],
        'xgb_params__eval_metric': ['mlogloss', 'merror']  # mlogloss = log-loss, merror = erro de classificação
    }

    n_folds = 2

    # scorer de C‑index
    def cindex_scorer(estimator, X_, y_):
        # estimator.predict retorna tempo medio de sobrevivencia;
        # virar "risco" com sinal invertido
        pred = estimator.predict(X_)
        # retorna tupla (cindex, concordante, discordante, ...) -> index e [0]
        return concordance_index_censored(y_['event'], y_['time'], (1-pred).mean(axis=1))[0]
    
    # Grid Search com validação cruzada
    grid_search = GridSearchCV(
        estimator=mc,
        param_grid=param_grid,
        scoring=cindex_scorer,
        cv=n_folds,
        verbose=1,
        n_jobs=-1
    )

    # Executa a busca
    grid_search.fit(X_train, y_train)

    # Resultados da validação cruzada
    cv_results = pd.DataFrame(grid_search.cv_results_)
    cv_summary = cv_results[['params', 'mean_test_score', 'std_test_score']]
    cv_summary = cv_summary.sort_values(by='mean_test_score', ascending=False)
    cv_summary.to_csv(f"{output_dir}/cv_resultados.csv", index=False)

    # Obtém os resultados fold-a-fold do melhor modelo para realizar o teste de friedman
    best_index = cv_results['rank_test_score'].idxmin()
    fold_scores = []
    for i in range(n_folds):
        fold_scores.append(cv_results.loc[best_index, f'split{i}_test_score'])

    # Salvar modelo
    best_model = grid_search.best_estimator_
    joblib.dump(best_model, f"{output_dir}/melhor_modelo.pkl")

    # Previsões
    y_score_train = best_model.predict(X_train)
    # tempo esperado = soma de S(t) * delta t
    delta_t = np.diff(np.insert(time_bins, 0, 0))  # delta t entre pontos
    y_train_pred = np.sum(y_score_train * delta_t, axis=1)  # shape: (n_individuos,)
    
    y_score_test = best_model.predict(X_test)
    # tempo esperado = soma de S(t) * delta t
    delta_t = np.diff(np.insert(time_bins, 0, 0))  # delta t entre pontos
    y_pred_test = np.sum(y_score_test * delta_t, axis=1)  # shape: (n_individuos,)
    
    # Métricas - Treino
    y_event = np.array([y[0] for y in y_train])
    y_time = np.array([y[1] for y in y_train])
    c_train = concordance_index_censored(y_event, y_time, -y_train_pred)[0]
    
    # Métricas - Teste
    y_event = np.array([y[0] for y in y_test])
    y_time = np.array([y[1] for y in y_test])
    c_test = concordance_index_censored(y_event,  y_time, -y_pred_test)[0]

    # Salvar métricas em TXT
    with open(f"{output_dir}/metricas_avaliacao.txt", "w", encoding='utf-8') as f:
        f.write("Hiperparâmetros ótimos:\n")
        f.write(str(grid_search.best_params_))
        f.write("\n\nMétricas - Treinamento:\n")
        f.write(f"C‑index: {c_train:.4f}\n")
        f.write("\nMétricas - Teste:\n")
        f.write(f"C‑index:  {c_test:.4f}\n")
        f.write(f"\n\nMédias de C-index para melhor fold: {fold_scores}\n")

    print("Gráficos e métricas salvos em:", output_dir)
    print("Modelo salvo em 'melhor_modelo.pkl'")

data_dir = Path("data/resultados")
data_file = Path("data/dataset_fit.csv")
process_data_file(data_file, data_dir, "_xgbse_regressor")
