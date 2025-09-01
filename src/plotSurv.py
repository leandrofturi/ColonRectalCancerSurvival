import warnings
warnings.filterwarnings("ignore")

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc

def _to_structured(y):
            """
            Converte y no formato (n,2) [time, event] para structured array
            com dtype [('event', bool), ('time', float)]
            """
            y = np.asarray(y)
            if y.dtype.names is not None:
                # já é structured
                return y
            elif y.ndim == 2 and y.shape[1] == 2:
                times = y[:, 0].astype(float)
                events = y[:, 1].astype(bool)
                return np.array(list(zip(events, times)),
                                dtype=[('event', bool), ('time', float)])
            else:
                raise ValueError(f"Formato inesperado de y: {y.shape}, {y.dtype}")

def evaluate_model(model, X_train, Y_train, X_test, Y_test, OUTPUT_DIR, model_name="modelo"):
    """
    Avalia um modelo de sobrevivência:
      - C-index
      - AUC(t) em percentis (25, 50, 75)
      - Plot da curva de sobrevivência média com ±1 std

    Parâmetros
    ----------
    model : fitted estimator (precisa ter predict ou predict_survival_function)
    X_train, Y_train : dados de treino
    X_test, Y_test : dados de teste
    OUTPUT_DIR : str -> diretório de saída
    model_name : str -> nome do modelo para salvar arquivos

    Retorna
    -------
    dict com métricas {"c_index": float, "aucs": dict, "mean_auc": float, "plot_path": str}
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # C-index

    try:
        # pred_risk = model.predict(X_test)
        # c_index = concordance_index_censored(
        #     Y_test["falha"], Y_test["time_years"], pred_risk
        # )[0]
        c_index = model.score_c_index(X_test, Y_test)
    except Exception:
        c_index = model.score(X_test, Y_test)

    # AUC(t)
    # risco predito: soma das curvas de falha ou vetor de risco
    try:
        surv_df = model.predict(X_test)  # XGBSE retorna DataFrame
        risk_scores = -np.sum(surv_df.to_numpy(), axis=1)
    except Exception:
        surv = model.predict_survival_function(X_test)
        surv = np.asarray(surv)
        risk_scores = -np.sum(surv, axis=1)
        surv_df = pd.DataFrame(surv)

    # percentis de tempo
    times_all = np.concatenate([X_train[:,0] if isinstance(X_train, np.ndarray) else np.array([]),
                                X_test[:,0] if isinstance(X_test, np.ndarray) else np.array([])])
    if len(times_all) == 0:
        times_all = np.concatenate([Y_train["time_years"], Y_test["time_years"]])
    times_auc = np.percentile(times_all, [25, 50, 75])

    try:
        # antes do cumulative_dynamic_auc
        Y_train_struct = _to_structured(Y_train)
        Y_test_struct  = _to_structured(Y_test)

        aucs, mean_auc = cumulative_dynamic_auc(
            Y_train_struct, Y_test_struct, risk_scores, times_auc
        )
    except Exception:
        aucs, mean_auc = cumulative_dynamic_auc(
            Y_train, Y_test, risk_scores, times_auc
        )

    auc_dict = {float(t): float(a) for t, a in zip(times_auc, aucs)}


    # Plot curva de sobrevivência
    time_bins = surv_df.columns.astype(float)

    # curva média
    mean_surv = surv_df.mean(axis=0)  # média entre indivíduos em cada tempo
    std_surv  = surv_df.std(axis=0)    # desvio-padrão entre indivíduos

    plt.figure(figsize=(16,10))
    plt.rcParams['legend.fontsize'] = 22
    plt.rcParams.update({'font.size': 25})

    plt.step(time_bins, mean_surv, where="post", color="red", label="Média dos individuos")
    plt.fill_between(time_bins,
                mean_surv - std_surv,
                mean_surv + std_surv,
                step="post", alpha=0.1, color="red", label="±1 std")

    plt.xlabel("Tempo")
    plt.ylabel("Probabilidade de Sobrevivência")
    plt.title(f"Curva de sobrevivência $S(t|x)$ - {model_name}")
    plt.legend()
    # plt.grid(True)

    plot_dir = os.path.join(OUTPUT_DIR, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, f"surv_{model_name}.pdf")
    plt.savefig(plot_path, bbox_inches="tight")
    plt.close()

    metrics_file = os.path.join(OUTPUT_DIR, f"metricas_{model_name}_test.txt")
    with open(metrics_file, "w", encoding="utf-8") as f:
        f.write(f"Modelo: {model_name}\n")
        f.write(f"C-index teste: {c_index:.4f}\n")
        f.write("AUC(t):\n")
        for t, auc in auc_dict.items():
            f.write(f"  t={t:.1f} -> AUC={auc:.3f}\n")
        f.write(f"AUC médio: {mean_auc:.3f}\n")
        if plot_path:
            f.write(f"\nCurva de sobrevivência salva em: {plot_path}\n")

    return {
        "c_index": c_index,
        "aucs": auc_dict,
        "mean_auc": mean_auc,
        "plot_path": plot_path
    }