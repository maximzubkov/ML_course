import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scikitplot.metrics import plot_roc
from sklearn.model_selection import train_test_split
from scripts.proj1_helpers import create_csv_submission, load_csv_data
from implementations import ridge_regression
from sklearn.metrics import roc_auc_score
from tqdm import tqdm 
from implementations import build_k_indices, build_poly, calculate_mse, sigmoid
from metrics import accuracy, f1, roc_auc
from itertools import combinations

seed = 9
np.random.seed(seed)

with open("data/test.csv", "r") as f:
    columns = f.readline().split(",")[2:]

y_test, X_test, ids_test = load_csv_data('data/test.csv')
y_dev, X_dev, ids_dev = load_csv_data('data/train.csv')

ratio = 0.8
num_row = X_dev.shape[0]
indices = np.random.permutation(num_row)
    
index_split = int(np.floor(ratio * num_row))
index_train = indices[: index_split]
index_val = indices[index_split:]

# split
y_train, X_train, ids_train = y_dev[index_train], X_dev[index_train], ids_dev[index_train]
y_val, X_val, ids_val = y_dev[index_val], X_dev[index_val], ids_dev[index_val]

X_train = np.delete(X_train, [15, 18, 20], axis=1)
X_val = np.delete(X_val, [15, 18, 20], axis=1)
X_test = np.delete(X_test, [15, 18, 20], axis=1)

X_train_for_corr = np.where(X_train < -998.0, np.nan, X_train)
corr_matrix = pd.DataFrame(X_train_for_corr).corr().values
cor_features_ids = [pair for pair in np.argwhere((corr_matrix >= 0.95) | (-0.95 >= corr_matrix)) if pair[0] != pair[1]][::2]


ids_to_delete = [pair[0] for pair in cor_features_ids]
X_train = np.delete(X_train, ids_to_delete, axis=1)
X_val = np.delete(X_val, ids_to_delete, axis=1)
X_test = np.delete(X_test, ids_to_delete, axis=1)


for i in range(X_train.shape[1]):
    features = X_train[:, i]
    mean = features[features >= -998.0].mean()
    X_train[:, i] = np.where(X_train[:, i] < -990.0, mean, X_train[:, i])
    X_val[:, i] = np.where(X_val[:, i] < -990.0, mean, X_val[:, i])
    X_test[:, i] = np.where(X_test[:, i] < -990.0, mean, X_test[:, i])

for i in [
    3, 8, 9, 12, 14, 16, 17, 19, 22, 25
]:
    X_train[:, i] = np.log(X_train[:, i] -  X_train[:, i].min() + 1e-10)
    X_val[:, i] = np.log(X_val[:, i] -  X_val[:, i].min() + 1e-10)
    X_test[:, i] = np.log(X_test[:, i] -  X_test[:, i].min() + 1e-10)


mean, std = X_train.mean(axis=0), X_train.std(axis=0)
X_train = (X_train - mean) / std
X_test = (X_test - mean) / std
X_val = (X_val - mean) / std

X_train_categorial = X_train[:, 18]
X_train_new_features = np.zeros((X_train.shape[0], 4))
X_train = np.delete(X_train, 18, axis=1)

X_val_categorial = X_val[:, 18]
X_val_new_features = np.zeros((X_val.shape[0], 4))
X_val = np.delete(X_val, 18, axis=1)

X_test_categorial = X_test[:, 18]
X_test_new_features = np.zeros((X_test.shape[0], 4))
X_test = np.delete(X_test, 18, axis=1)

vals = np.unique(X_val_categorial)

for i, v in enumerate(vals):
    X_train_new_features[:, i] = np.where(X_train_categorial == v, 1, 0)
    X_val_new_features[:, i] = np.where(X_val_categorial == v, 1, 0)
    X_test_new_features[:, i] = np.where(X_test_categorial == v, 1, 0)
    
X_train = np.concatenate([X_train, X_train_new_features], axis=-1)
X_val = np.concatenate([X_val, X_val_new_features], axis=-1)
X_test = np.concatenate([X_test, X_test_new_features], axis=-1)

X_train = np.concatenate([X_train, np.ones(X_train.shape[0]).reshape(-1, 1)], axis=-1)
X_val = np.concatenate([X_val, np.ones(X_val.shape[0]).reshape(-1, 1)], axis=-1)
X_test = np.concatenate([X_test, np.ones(X_test.shape[0]).reshape(-1, 1)], axis=-1)

train_cond = 1
for i in range(24):
    l_bound, u_bound = np.quantile(X_train[:, i], 0.01), np.quantile(X_train[:, i], 0.99) 
    train_cond &= (X_train[:, i] >= l_bound) & (X_train[:, i] <= u_bound)


X_train = X_train[train_cond.astype(bool), :]
y_train = y_train[train_cond.astype(bool)]

X_train = np.concatenate([
    X_train, X_train[:, :-1] ** 2
] + [
    (X_train[:, i] * X_train[:, j]).reshape(-1, 1) for i, j in combinations(range(X_train.shape[1] - 5), r=2)
], axis=-1)

X_val = np.concatenate([
    X_val, X_val[:, :-1] ** 2
] + [
    (X_val[:, i] * X_val[:, j]).reshape(-1, 1) for i, j in combinations(range(X_val.shape[1] - 5), r=2)
], axis=-1)

X_test = np.concatenate([
    X_test, X_test[:, :-1] ** 2
] + [
    (X_test[:, i] * X_test[:, j]).reshape(-1, 1) for i, j in combinations(range(X_test.shape[1] - 5), r=2)
], axis=-1)

def cross_validation(y, x, test_ids, lambda_):
    test_index = k_indices[k]
    
    x_test = x[test_ids]
    y_test = y[test_ids]
    x_train = np.delete(x, test_ids, axis=0)
    y_train = np.delete(y, test_ids, axis=0)

    _, w = ridge_regression(y_train, x_train, lambda_)
    y_pred = sigmoid(x_test @ w)

    return roc_auc_score(y_test, y_pred)

k_fold = 5
lambdas = np.logspace(-16, 1, 100)
k_indices = build_k_indices(y_train, k_fold, seed)

roc_aucs = {}

for lambda_ in tqdm(lambdas):
    roc_auc_lambda = []
    for k in range(k_fold):
        test_ids = k_indices[k]
        roc_auc_ = cross_validation(y_train, X_train, test_ids, lambda_)
        roc_auc_lambda.append(roc_auc_)
    roc_aucs[lambda_] = np.mean(roc_auc_lambda)

best_lambda = lambdas[np.argmax([v for k, v in roc_aucs.items()])]
print(f"The best lambda is {best_lambda}")

_, w = ridge_regression(y_train, X_train, lambda_=best_lambda)
y_pred = sigmoid(X_val @ w)

tr = np.linspace(0.000001, 0.999999, 10000)
f1s = []
for t in tqdm(tr):
    f1s.append(f1(np.where(y_pred < t, 0, 1), np.where(y_val == -1, 0, 1)))
best_tr = tr[np.argmax(f1s)]
print(f"The best threshold is {best_tr}")



OUTPUT_PATH = 'results.csv'
_, w = ridge_regression(
    np.concatenate([y_train, y_val], axis=0),
    np.concatenate([X_train, X_val], axis=0), 
    lambda_=best_lambda
)
y_pred = np.where(sigmoid(X_test @ w) < best_tr, -1, 1)
create_csv_submission(ids_test, y_pred, OUTPUT_PATH)






