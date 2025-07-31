import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import label_binarize
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score, recall_score, \
    precision_score, classification_report
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import LabelEncoder


def process_data_file(data_file_csv, base_path, prefix, roc_axes):
    # Criar diretório para salvar resultados
    output_dir = base_path / prefix
    os.makedirs(output_dir, exist_ok=True)

    # Read the CSV file
    data = pd.read_csv(data_file_csv)

    # dados de saida
    y = data["time_years"]
    y_cat = data["time_years_cat"]
    
    # transforma os dados em numeros
    le = LabelEncoder()
    y_encoded_cat = le.fit_transform(y_cat)
    y_encoded = Surv.from_arrays(data['falha'], y, name_event="falha", name_time="time_years")

    # dados de entrada filtrados
    X = data.drop(columns=["time_years_cat", "time_years", "falha"])
    # transforma os dados em numeros
    X_encoded = pd.get_dummies(X)

    # 3. Divisão treino/teste
    X_train, X_test, _y_train, _y_test = train_test_split(X_encoded, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded_cat )
    y_train = np.array([y[1] for y in _y_train])
    y_test = np.array([y[1] for y in _y_test])

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

    _BINS = np.array([1, 3, 5])
    def discretize(y):
        return np.digitize(y, _BINS, right=True)

    def discrete_accuracy_scorer(estimator, X_, y_):
        pred = estimator.predict(X_)
        y_pred_cat = discretize(pred)
        y_true_cat = discretize(y_)
        return accuracy_score(y_true_cat, y_pred_cat) 
       
    # Grid Search com validação cruzada
    grid_search = GridSearchCV(
        estimator=mc,
        param_grid=param_grid,
        scoring=discrete_accuracy_scorer,
        cv=n_folds,
        verbose=1,
        n_jobs=-1
    )

    # Executa a busca
    grid_search.fit(X_train, _y_train)

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
    y_train_pred = discretize(y_score_train)
    
    y_score_test = best_model.predict(X_test)
    y_pred_test = discretize(y_score_test)
    
    # Métricas - Treino
    y_train_cat = discretize(y_train)
    acc_train = accuracy_score(y_train_cat, y_train_pred)
    f1_train = f1_score(y_train_cat, y_train_pred, average='macro')
    recall_train = recall_score(y_train_cat, y_train_pred, average='macro')
    precision_train = precision_score(y_train_cat, y_train_pred, average='macro')

    # Métricas - Teste
    y_test_cat = discretize(y_test)
    acc_test = accuracy_score(y_test_cat, y_pred_test)
    f1_test = f1_score(y_test_cat, y_pred_test, average='macro')
    recall_test = recall_score(y_test_cat, y_pred_test, average='macro')
    precision_test = precision_score(y_test_cat, y_pred_test, average='macro')

    # Salvar métricas em TXT
    with open(f"{output_dir}/metricas_avaliacao.txt", "w", encoding='utf-8') as f:
        f.write("Hiperparâmetros ótimos:\n")
        f.write(str(grid_search.best_params_))
        f.write("\n\nMétricas - Treinamento:\n")
        f.write(f"Acurácia:  {acc_train:.4f}\n")
        f.write(f"F1-score:  {f1_train:.4f}\n")
        f.write(f"Recall:    {recall_train:.4f}\n")
        f.write(f"Precision: {precision_train:.4f}\n")
        f.write("\nMétricas - Teste:\n")
        f.write(f"Acurácia:  {acc_test:.4f}\n")
        f.write(f"F1-score:  {f1_test:.4f}\n")
        f.write(f"Recall:    {recall_test:.4f}\n")
        f.write(f"Precision: {precision_test:.4f}\n")
        f.write(f"\n\nMédias de Acurácia para melhor fold: {fold_scores}\n")

    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred_test)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)

    plt.figure(figsize=(6, 5))
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title("Matriz de Confusão")
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/matriz_confusao.png")
    plt.close()

    plt.figure(figsize=(10, 6))

    # Curva ROC por classe
    plt.figure(figsize=(10, 6))
    y_test_bin = label_binarize(y_test, classes=np.arange(len(le.classes_)))
    fpr, tpr, roc_auc = {}, {}, {}
    n_classes = len(le.classes_)
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score_test[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], label=f'Classe {le.inverse_transform([i])[0]} (AUC = {roc_auc[i]:.2f})')
  
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('FPR (Falsos Positivos)')
    plt.ylabel('TPR (Verdadeiros Positivos)')
    plt.title('Curvas ROC')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/curva_roc.png")
    plt.close()

    print("Gráficos e métricas salvos em:", output_dir)
    print("Modelo salvo em 'melhor_modelo.pkl'")

roc_fig, roc_axes = plt.subplots()
data_dir = Path("data/resultados")
data_file = Path("data/dataset_fit.csv")
process_data_file(data_file, data_dir, "_randomsurvivalforest", roc_axes)
roc_axes.plot([0, 1], [0, 1], 'k--')
roc_axes.set_xlabel('FPR (Falsos Positivos)')
roc_axes.set_ylabel('TPR (Verdadeiros Positivos)')
roc_axes.set_title(f'Curva ROC')
roc_axes.legend(loc='lower right')
roc_axes.grid(True)
roc_fig.tight_layout()
roc_fig.savefig(f"{data_dir}/curva_roc_135_integrados_randomsurvivalforest.png")
plt.close(roc_fig)
