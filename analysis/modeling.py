import pandas as pd
import numpy as np
from os.path import dirname, join
from itertools import combinations
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score


CUR_DIR = dirname(__file__)
DATA_DIR = join(CUR_DIR, '../data')

data = pd.read_csv(join(DATA_DIR, 'full_data.csv'))

# Akaike Information Criterion (AIC) 
def AIC(SSE, n, k):
    '''
    SSE - sum of squared errors
    n - sample size
    k - number of parameters
    '''
    return n * np.log(SSE/n) + 2 * k

# Bayesian Information Criterion (BIC)
def BIC(SSE, n, k):
    '''
    SSE - sum of squared errors
    n - sample size
    k - number of parameters
    '''
    return n * np.log(SSE/n) + k * np.log(n)

# Mean Absolute Error (MAE)
def MAE(true, pred):
    return np.mean(np.abs(true-pred))

# Root Mean Square Error (RMSE)
def RMSE(true, pred):
    return np.sqrt(np.sum(np.power((true-pred), 2)))

# Sum of Squared Errors (SSE)
def SSE(true, pred):
    return np.sum(np.power((true-pred), 2))

# Predictive models
# Which features best explain task complexity across the whole dataset?
# Considered features:
#   The three trial color distribution features (WinDom - winnner dominance, Lead, and Imbal - imbalance),
#   RuleType - count-twice vs. count-as vs. None,
#   CurrActive - number of currently applicable rules,
#   RuleProximity - if current rule is a contraction/revision, the distance to the relevant rule,
#   Scope - specific or general rule.
def modeling():
    features = ["windom", "lead", "imbal", "rule_type", "current_active", "rule_proximity", "scope"]
    results_acc = []
    results_rt = []

    for r in range(1, len(features) + 1):
        for subset in combinations(features, r):
            feature_list = list(subset)

            X = pd.get_dummies(data[feature_list])
            if "rule_type_none" in X.columns:
                X.drop(columns=["rule_type_none"], inplace=True)
            y_acc = data["correct"]
            y_rt = data["rt"] / 1000

            lda = LinearDiscriminantAnalysis()
            lda.fit(X, y_acc)
            y_pred = lda.predict(X)
            accuracy_lda = accuracy_score(y_acc, y_pred)
            sse_acc = SSE(y_acc, y_pred)
            mae_acc = MAE(y_acc, y_pred)
            rmse_acc = RMSE(y_acc, y_pred)

            linreg = LinearRegression()
            linreg.fit(X, y_rt)
            y_pred_rt = linreg.predict(X)
            sse_rt = SSE(y_rt, y_pred_rt)
            mae_rt = MAE(y_rt, y_pred_rt)
            rmse_rt = RMSE(y_rt, y_pred_rt)

            k = len(feature_list)
            N = len(y_acc)

            aic_acc = AIC(sse_acc, N, k)
            bic_acc = BIC(sse_acc, N, k)
            aic_rt = AIC(sse_rt, N, k)
            bic_rt = BIC(sse_rt, N, k)

            results_acc.append({
                "features": feature_list, 
                "accuracy": accuracy_lda, 
                "AIC": aic_acc, 
                "BIC": bic_acc, 
                "SSE": sse_acc, 
                "MAE": mae_acc, 
                "RMSE": rmse_acc
            })

            results_rt.append({
                "features": feature_list, 
                "accuracy": accuracy_lda, 
                "AIC": aic_rt, 
                "BIC": bic_rt, 
                "SSE": sse_rt, 
                "MAE": mae_rt, 
                "RMSE": rmse_rt
            })

    results_acc_df = pd.DataFrame(results_acc)
    results_rt_df = pd.DataFrame(results_rt)

    print("Accuracy")
    print("Lowest AIC")
    print(results_acc_df.loc[results_acc_df["AIC"].idxmin(), ["AIC", "BIC", "features", "accuracy"]])
    print("\nLowest BIC")
    print(results_acc_df.loc[results_acc_df["BIC"].idxmin(), ["AIC", "BIC", "features", "accuracy"]])

    best_features_acc = results_acc_df.loc[results_acc_df["AIC"].idxmin(), "features"]
    X = pd.get_dummies(data[best_features_acc])
    if "rule_type_none" in X.columns:
        X.drop(columns=["rule_type_none"], inplace=True)
    y = data["correct"]
    lda = LinearDiscriminantAnalysis()
    lda.fit(X, y)
    y_pred = lda.predict(X)
    accuracy_lda = accuracy_score(y, y_pred)
    print(f"Accuracy: {accuracy_lda:.2f}")
    coefficients_lda = pd.DataFrame(lda.coef_, columns=X.columns)
    print(coefficients_lda)

    print("\nResponse Time")
    print("Lowest AIC")
    print(results_rt_df.loc[results_rt_df["AIC"].idxmin(), ["AIC", "BIC", "features", "MAE"]])
    print("\nLowest BIC")
    print(results_rt_df.loc[results_rt_df["BIC"].idxmin(), ["AIC", "BIC", "features", "MAE"]])

    best_features_rt = results_rt_df.loc[results_rt_df["BIC"].idxmin(), "features"]
    X = pd.get_dummies(data[best_features_rt])
    if "rule_type_none" in X.columns:
        X.drop(columns=["rule_type_none"], inplace=True)
    y = data["rt"] / 1000
    linreg = LinearRegression()
    linreg.fit(X, y)
    y_pred = linreg.predict(X)
    mae = MAE(y, y_pred)
    print(f"MAE: {mae:.2f}")
    coefficients_linreg = pd.DataFrame(linreg.coef_.reshape(1, -1), columns=X.columns)
    print(coefficients_linreg)

modeling()