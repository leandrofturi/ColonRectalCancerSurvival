import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
import numpy as np
import matplotlib.pyplot as plt
import joblib


def process_data_file(data_file_csv, base_path, prefix):
    # Criar diretório para salvar resultados
    output_dir = base_path / prefix
    os.makedirs(output_dir, exist_ok=True)

    # Read the CSV file
    data = pd.read_csv(data_file_csv)

    # dados de saida
    y = data["time_years"]
    
    # transforma os dados em numeros
    y_encoded = Surv.from_arrays(data['falha'], y, name_event="falha", name_time="time_years")

    # dados de entrada filtrados
    X = data.drop(columns=["time_years_cat", "time_years", "falha"])
    # transforma os dados em numeros
    X_encoded = pd.get_dummies(X)

    # 3. Divisão treino/teste
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.25, random_state=42)

    # Modelo base
    mc = RandomSurvivalForest(random_state=42)

    # Grade de hiperparâmetros para conjunto de dados multilabel
    param_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    param_grid = {
        'n_estimators': [100],
        'max_depth': [None],
        'min_samples_split': [2],
        'min_samples_leaf': [1],
        'bootstrap': [True]
    }

    n_folds = 2

    # scorer de C‑index
    def cindex_scorer(estimator, X_, y_):
        # estimator.predict retorna tempo medio de sobrevivencia;
        # virar "risco" com sinal invertido
        pred = estimator.predict(X_)
        # retorna tupla (cindex, concordante, discordante, ...) -> index e [0]
        return concordance_index_censored(y_['event'], y_['time'], -pred)[0]
    
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
    y_train_pred = best_model.predict(X_train)
    
    y_pred_test = best_model.predict(X_test)
    
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
process_data_file(data_file, data_dir, "_randomsurvivalforest_regressor")
