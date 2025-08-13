"""
This is a straightforward example of how to use the A-TOPSIS algorithm with this package.
This example uses two approaches:
1: using the decision matrices inside a csv files
2: using the matrices in a list of lists

Important: in this example we use the avg_cost_ben="cost" because the performance metric is the error (the lower, the
better). If it was the accuracy, for example, we'd use it as "benefit", since for this metric the higher, the better.
You must double-check it before running the method.
"""

from decision_making import ATOPSIS
from scipy.stats import friedmanchisquare


########################################################################################################################
# Cenario 1
########################################################################################################################

# realizar teste de friedman para verficar se existe diferença estatística entre os modelos
mlp_folds = [0.8432, 0.8420, 0.8452]
xgb_folds = [0.8439, 0.8423, 0.8452]
random_forest_folds = [0.8453, 0.8417, 0.8428]
stat, p = friedmanchisquare(mlp_folds, xgb_folds, random_forest_folds)

print(f'Estatística Cenário 1: {stat:.4f}, p-valor: {p:.4f}')

# se existe diferença, prosseguimos com atopsis para verificar qual é melhor
if p < 0.05:
    avg_mat = [
        [0.8435],
        [0.8438],
        [0.8433]
    ]
    std_mat = [
        [0.0013],
        [0.0012],
        [0.0011]
    ]

    print("-" * 50)
    print("- Cenário 1:")
    print("-" * 50)
    atop = ATOPSIS(avg_mat, std_mat, avg_cost_ben="benefit", std_cost_ben="cost")
    atop.get_ranking(True)
    atop.plot_ranking(alg_names=["MLP", "XGboost", "Randon Forest"])
    print("-" * 50)
    print("")


########################################################################################################################
# Cenario 2
########################################################################################################################

# realizar teste de friedman para verficar se existe diferença estatística entre os modelos
mlp_folds = [0.7803, 0.7697, 0.7591]
xgb_folds = [0.7811, 0.7695, 0.7642]
random_forest_folds = [0.7783, 0.7708, 0.7639]
stat, p = friedmanchisquare(mlp_folds, xgb_folds, random_forest_folds)

print(f'Estatística Cenário 2: {stat:.4f}, p-valor: {p:.4f}')

# se existe diferença, prosseguimos com atopsis para verificar qual é melhor
if p < 0.05:
    avg_mat = [
        [0.7697],
        [0.7716],
        [0.7710]
    ]
    std_mat = [
        [0.0086],
        [0.0070],
        [0.0058]
    ]

    print("-" * 50)
    print("- Cenário 1:")
    print("-" * 50)
    atop = ATOPSIS(avg_mat, std_mat, avg_cost_ben="benefit", std_cost_ben="cost")
    atop.get_ranking(True)
    atop.plot_ranking(alg_names=["MLP", "XGboost", "Randon Forest"])
    print("-" * 50)
    print("")

########################################################################################################################
# Cenario 3
########################################################################################################################

# realizar teste de friedman para verficar se existe diferença estatística entre os modelos
mlp_folds = [0.7618, 0.7742, 0.7622]
xgb_folds = [0.7660, 0.7745, 0.7637]
random_forest_folds = [0.7633, 0.7695, 0.7668]
stat, p = friedmanchisquare(mlp_folds, xgb_folds, random_forest_folds)

print(f'Estatística Cenário 3: {stat:.4f}, p-valor: {p:.4f}')

# se existe diferença, prosseguimos com atopsis para verificar qual é melhor
if p < 0.05:
    avg_mat = [
        [0.7660],
        [0.7681],
        [0.7666]
    ]
    std_mat = [
        [0.0057],
        [0.0046],
        [0.0025]
    ]

    print("-" * 50)
    print("- Cenário 1:")
    print("-" * 50)
    atop = ATOPSIS(avg_mat, std_mat, avg_cost_ben="benefit", std_cost_ben="cost")
    atop.get_ranking(True)
    atop.plot_ranking(alg_names=["MLP", "XGboost", "Randon Forest"])
    print("-" * 50)
    print("")

########################################################################################################################
# Cenario 4
########################################################################################################################

# realizar teste de friedman para verficar se existe diferença estatística entre os modelos
mlp_folds = [0.6019, 0.6058, 0.6013]
xgb_folds = [0.6050, 0.6040, 0.6100]
random_forest_folds = [0.6050, 0.6035, 0.6052]
stat, p = friedmanchisquare(mlp_folds, xgb_folds, random_forest_folds)

print(f'Estatística Cenário 4: {stat:.4f}, p-valor: {p:.4f}')

# se existe diferença, prosseguimos com atopsis para verificar qual é melhor
if p < 0.05:
    avg_mat = [
        [0.6030],
        [0.6064],
        [0.6046]
    ]
    std_mat = [
        [0.0019],
        [0.0026],
        [0.0007]
    ]

    print("-" * 50)
    print("- Cenário 1:")
    print("-" * 50)
    atop = ATOPSIS(avg_mat, std_mat, avg_cost_ben="benefit", std_cost_ben="cost")
    atop.get_ranking(True)
    atop.plot_ranking(alg_names=["MLP", "XGboost", "Randon Forest"])
    print("-" * 50)
    print("")


