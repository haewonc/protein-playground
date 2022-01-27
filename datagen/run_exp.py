import re
import pandas as pd 
from sklearn.metrics import accuracy_score, mean_squared_error

DDIR = "../../pdb/exp/"
EXP_NAMES = ["{}trial{}/".format(DDIR, i) for i in range(1, 9)]

def exp(name, clf):
    result_ptn = {
        "acc": [],
        "mse": []
    }

    result_non = {
        "acc": [],
        "mse": []
    }

    names = ["acc", "mse"]
    metrics = [accuracy_score, mean_squared_error]

    col_names = []

    for exp in EXP_NAMES:
        train = pd.read_csv(exp+"train.csv", sep=',')
        train = train.drop('label', axis=1)
        cols = pd.get_dummies(train).columns
        col_names += cols.tolist()

    col_names = list(set(col_names))

    for exp in EXP_NAMES:
        train = pd.read_csv(exp+"train.csv", sep=',')
        test_non = pd.read_csv(exp+"test_non.csv", sep=',')
        test_ptn = pd.read_csv(exp+"test_ptn.csv", sep=',')
        
        y_train = train["label"]
        X_train = train.drop("label", axis=1)
        X_train = pd.get_dummies(X_train)

        y_test_non = test_non["label"]
        X_test_non = test_non.drop("label", axis=1)
        X_test_non = pd.get_dummies(X_test_non)

        y_test_ptn = test_ptn["label"]
        X_test_ptn = test_ptn.drop("label", axis=1)
        X_test_ptn = pd.get_dummies(X_test_ptn)

        X_train = X_train.reindex(columns=col_names, fill_value=0)
        X_test_non = X_test_non.reindex(columns=col_names, fill_value=0)
        X_test_ptn = X_test_ptn.reindex(columns=col_names, fill_value=0)

        clf.fit(X_train, y_train)

        pred_non = clf.predict(X_test_non)
        pred_ptn = clf.predict(X_test_ptn)

        for name, metric in zip(names, metrics):
            result_ptn[name].append(metric(y_test_ptn, pred_ptn))
            result_non[name].append(metric(y_test_non, pred_non))

    indices = ["{}".format(i) for i in range(1, 9)]
    results = [indices, result_ptn["acc"], result_ptn["mse"], result_non["acc"], result_non["mse"]]
    results = pd.DataFrame(results).T

    org_cols = results.columns.tolist()
    result_cols = ["TRIAL", "PTN ACC", "PTN MSE", "NON ACC", "NON MSE"]
    rename_dict = {}
    for org_col, result_col in zip(org_cols, result_cols):
        rename_dict[org_col] = result_col
    results = results.rename(rename_dict, axis=1)

    results.to_csv("results/{}.csv".format(name), index=False)
    print("Results saved.")