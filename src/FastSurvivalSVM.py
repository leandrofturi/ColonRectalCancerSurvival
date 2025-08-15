import warnings
warnings.filterwarnings("ignore")

import os
import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold

from sksurv.svm import FastSurvivalSVM
from sksurv.util import Surv

# Supondo que os dados já foram carregados e pré-processados
# (usando as mesmas variáveis dos scripts anteriores)
from load_data import (
    X_train, X_test,
    y_train, y_test,
    event_train, event_test, 
)

# 1. Diretório de saída para os resultados deste modelo
OUTPUT_DIR = "data/resultados/FastSurvivalSVM"
os.makedirs(OUTPUT_DIR, exist_ok=True)


if __name__ == "__main__":
    # Preparando os dados de alvo no formato que sksurv espera
    Y_train = Surv.from_arrays(event_train, y_train, name_event="falha", name_time="time_years")
    Y_test = Surv.from_arrays(event_test, y_test, name_event="falha", name_time="time_years")

    # 2. Instanciando o modelo base com parâmetros fixos
    estimator = FastSurvivalSVM(max_iter=1000, random_state=42)

    # 3. Definindo a grade de hiperparâmetros
    param_grid = {
        'alpha': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1],
    }

    # 4. Configurando a validação cruzada e o GridSearchCV
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    gcv = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=None,  
        cv=cv,
        n_jobs=-1,      
        refit=True,     # Treina o melhor modelo com todos os dados no final
        verbose=3       # Mostra o progresso do treinamento
    )

    print("Iniciando a busca de hiperparâmetros para FastSurvivalSVM...")
    gcv.fit(X_train, Y_train)

    print("\nBusca finalizada!")
    print(f"Melhor C-index (CV): {gcv.best_score_:.4f}")
    print("Melhores parâmetros:", gcv.best_params_)

    # 5. Salvando os resultados da validação cruzada em CSV
    cv_results = pd.DataFrame(gcv.cv_results_)
    cv_summary = cv_results[["params", "mean_test_score", "std_test_score", "rank_test_score"]]
    cv_summary = cv_summary.sort_values(by="rank_test_score", ascending=True)
    cv_summary.to_csv(f"{OUTPUT_DIR}/cv_resultados.csv", index=False)
    print(f"\nResultados da validação cruzada salvos em: {OUTPUT_DIR}/cv_resultados.csv")

    # 6. Salvando o melhor modelo encontrado
    best_model = gcv.best_estimator_
    joblib.dump(best_model, f"{OUTPUT_DIR}/melhor_modelo.pkl")
    print(f"Melhor modelo salvo em: {OUTPUT_DIR}/melhor_modelo.pkl")

    # 7. Salvando um resumo das métricas de avaliação em TXT
    best_index = cv_results["rank_test_score"].idxmin()
    fold_scores = []
    for i in range(cv.get_n_splits()):
        fold_scores.append(cv_results.loc[best_index, f"split{i}_test_score"])

    with open(f"{OUTPUT_DIR}/metricas_avaliacao.txt", "w", encoding="utf-8") as f:
        f.write("Hiperparâmetros ótimos:\n")
        f.write(str(gcv.best_params_))
        f.write("\n\n--- Métricas (C-index) ---\n")
        f.write(f"C-index no conjunto de treino completo: {best_model.score(X_train, Y_train):.4f}\n")
        f.write(f"C-index no conjunto de teste: {best_model.score(X_test, Y_test):.4f}\n")
        f.write("\n--- Validação Cruzada ---\n")
        f.write(f"Melhor C-index médio na CV: {gcv.best_score_:.4f}\n")
        f.write(f"Scores do melhor modelo em cada fold (CV): {[round(s, 4) for s in fold_scores]}\n")

    print(f"Métricas de avaliação salvas em: {OUTPUT_DIR}/metricas_avaliacao.txt")