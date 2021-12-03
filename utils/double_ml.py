import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

# get Q(1,x) and Q(0,x) given fitted Q
def get_outcome_pred(q, X_Q, binary=False, ATT=False):
    X_copy = X_Q.copy()
    X_copy["treatment"] = 1
    y_hat1 = q.predict_proba(X_copy) if binary else q.predict(X_copy)
    X_copy["treatment"] = 0
    y_hat0 = q.predict_proba(X_copy) if binary else q.predict(X_copy)
    if ATT:
        t = X_Q["treatment"]
        return y_hat1[t==1], y_hat0[t==1]
    return y_hat1, y_hat0

# estimate ATE/ATT given fitted Q
def outcome_est(q, X_Q, binary=False, ATT=False):
    y_hat1, y_hat0 = get_outcome_pred(q, X_Q, binary, ATT)
    if ATT:
        return np.mean((y_hat1 - y_hat0)[t==1])
    return np.mean(y_hat1 - y_hat0)
        
# estimate ATE/ATT given fitted g
def iptw_est(g, t, X, y, ATT=False):
    ps = g.predict_proba(X)[:,1]
    weight = (t/ps) - (1-t)/(1-ps) 
    if ATT:
        return np.mean((weight * y)[t==1])
    return np.mean(weight * y)

# estimate ATE/ATT w/ double ML technique given fitted Q, g
def double_ML(q, g, t, X, X_Q, y, binary=False, ATT=False):
    y_hat1, y_hat0 = get_outcome_pred(q, X_Q, binary, ATT)
    ps = g.predict_proba(X)[:,1]
    if ATT:
        return (np.mean((
            y_hat1 - y_hat0 + t/ps * (y - y_hat1) - (1-t)/(1-ps) * (y - y_hat0))[t==1]
            ))
    return (np.mean(
        y_hat1 - y_hat0 + t/ps * (y - y_hat1) - (1-t)/(1-ps) * (y - y_hat0)
        ))

# estimate influence curve
def phi(q, g, t, X, X_Q, y, tau, binary=False, ATT=False):
    y_hat1, y_hat0 = get_outcome_pred(q, X_Q, binary, ATT)
    ps = g.predict_proba(X)[:,1]
    if ATT:
        return (y_hat1 - y_hat0 + 
        t/ps * (y - y_hat1) - 
        (1-t)/(1-ps) * (y - y_hat0) -
        tau)[t==1]
    return (y_hat1 - y_hat0 + 
            t/ps * (y - y_hat1) - 
            (1-t)/(1-ps) * (y - y_hat0) -
            tau)

def cross_fit_doubleML(Q_model, g_model, t, X, y, k=10, ATT=False, Q_model_params={}, g_model_params={}):
    X_Q = X.copy()
    X_Q["treatment"] = t
    binary = ((y==0) | (y==1)).all()
    if binary:
        kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)
    else:
        kf = KFold(n_splits=k, shuffle=True, random_state=0)

    learned_Q, learned_g, estimates = [], [], []
    # cross fit to get estimate of treatment effect
    for train_index, fit_index in kf.split(X, y):
        # split data
        X_train, X_fit = X.iloc[train_index], X.iloc[fit_index]
        X_Q_train, X_Q_fit = X_Q.iloc[train_index], X_Q.iloc[fit_index]
        t_train, t_fit = t.iloc[train_index], t.iloc[fit_index]
        y_train, y_fit = y.iloc[train_index], y.iloc[fit_index]
        # get double ML estimate
        Q = Q_model(**Q_model_params).fit(X_Q_train, y_train) if Q_model_params else Q_model().fit(X_Q_train, y_train)
        g = g_model(**g_model_params).fit(X_train, t_train) if g_model_params else g_model().fit(X_train, t_train)
        tau_hat = double_ML(Q, g, t_fit, X_fit, X_Q_fit, y_fit, ATT)
        learned_Q.append(Q)
        learned_g.append(g)
        estimates.append(tau_hat)
    tau_hat = np.mean(estimates) # NOTE: fine to take mean of means if N/k sufficiently large?
    
    # get std errs (reuse learned nuisance functions from above)
    pred_vars = []
    i = 0
    for _, fit_index in kf.split(X):
        X_fit, X_Q_fit, t_fit, y_fit = X.iloc[fit_index], X_Q.iloc[fit_index], t.iloc[fit_index], y.iloc[fit_index]
        Q, g = learned_Q[i], learned_g[i]
        phi_hat = phi(Q, g, t_fit, X_fit, X_Q_fit, y_fit, tau_hat, binary, ATT)
        pred_vars.append(np.mean(phi_hat ** 2))
        i += 1
    var_hat = np.mean(pred_vars)
    stderr = (var_hat / len(X)) ** 0.5
    
    return tau_hat, stderr



