import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


__data_file_csv = "data/dataset_fit.csv"

__data = pd.read_csv(__data_file_csv)
__data = __data.sort_values(by="time_years")

__data["time_years"] = __data["time_years"].replace(0, 0.0001)

# Divisão treino/teste
__data_train, __data_test = train_test_split(__data, test_size=0.25, random_state=42)

# Tempo
y_train = __data_train["time_years"].values.astype(np.float32)
y_test = __data_test["time_years"].values.astype(np.float32)

# Falha
event_train = __data_train["falha"].values.astype(np.float32)
event_test = __data_test["falha"].values.astype(np.float32)

# Covariáveis
onehot = OneHotEncoder()

__X_train_raw = __data_train.drop(columns=["time_years_cat", "time_years", "falha"])
X_train = onehot.fit_transform(__X_train_raw).toarray().astype(np.float32)

__X_test_raw = __data_test.drop(columns=["time_years_cat", "time_years", "falha"])
X_test = onehot.fit_transform(__X_test_raw).toarray().astype(np.float32)
