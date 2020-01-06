import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import shap

MAXLAGS = 14

def make_poly(data):
    columns = [str(i) for i in range(data.shape[1])]
    data = pd.DataFrame(data, columns=columns)
    data_2 = data.apply(lambda x: x**2)
    data_2.columns = [c + '_2' for c in data.columns]
    data_3 = data.apply(lambda x: x**3)
    data_3.columns = [c + '_3' for c in data.columns]

    poly = pd.concat((data, data_2, data_3), axis=1)
    return poly

def make_lagged(poly):
    lagged = []
    maxlags = MAXLAGS
    for i in range(maxlags):
        lag = i + 1
        part = pd.DataFrame(poly.values[lag: lag - maxlags - 1])
        part.columns = [str(c) + '_l' + str(lag) for c in poly.columns]
        lagged.append(part)
    return pd.concat(lagged, axis=1)

def rf_method(data, maxlags=1, correct_pvalues=True):

    # Input data is of shape (time, variables)
    T, N = data.shape

    N = data.shape[1]
    val_matrix = np.zeros((N, N))
    poly = make_poly(data)
    lagged = make_lagged(poly)

    for target in range(N):

        target = str(target)
        cols = [c for c in lagged.columns if not c.startswith(target)]

        model = RandomForestRegressor(n_jobs=11, max_depth=5, n_estimators=50)
        model.fit(lagged[cols].values,
                      data[:, int(target)][MAXLAGS + 1:])

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(lagged[cols].values)
        for coef, name in zip(np.sum(shap_values, axis=0), cols):
            if coef != 0:
                val_matrix[int(target)][int(name.split('_')[0])] += np.abs(coef)

    val_matrix = np.nan_to_num(val_matrix)

    return val_matrix.T, None, None


def final_method(data, maxlags=1, correct_pvalues=True):
    val_matrix1, _, _ = rf_method(data, maxlags)
    # lag_matrix2, _, _ = another_method(data, maxlags, ...)
    return val_matrix1, None, None
    # HOW TO BLEND:
    # return (val_matrix1 / np.median(val_matrix1)) + \
    #       (val_matrix2 / np.median(val_matrix2)), None, None