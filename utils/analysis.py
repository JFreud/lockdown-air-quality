from utils.double_ml import *
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, log_loss
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from scipy.stats import ttest_ind
from scipy.stats import t

# =================== dataframe gymnastics ===================

wf = pd.read_csv("data/wf.csv")
wf["temp2"] = wf["temp"] ** 2
wf["l_aqi"] = np.log(1 + wf["aqi"])
wf["l_pm"] = np.log(1 + wf["pm"])
wf.drop('Unnamed: 0', axis = 1)

city_yb = pd.read_csv("data/city_yb.csv")
city_yb.drop('Unnamed: 0', axis = 1)

def make_wf2020(city_var=False):
    wf2020 = wf[(wf["daynum"] >= 8401) & (wf["daynum"]<= 8461)].dropna(
        subset = ['aqi', 'pm']
    )
    # weather variable transforms
    wf2020["temp2"] = wf2020["temp"] ** 2
    wf2020["l_aqi"] = np.log(1 + wf2020["aqi"])
    wf2020["l_pm"] = np.log(1 + wf2020["pm"])

    # merge with city economic and environmental variables
    if (city_var):
        city_yb_clean = city_yb.dropna()
        wf2020 = wf2020.merge(city_yb_clean, on='city_code').dropna(
            subset = ['sec_city', 'gdp_city', 'pgdp_city', 
                    'firm_city', 'gonglu', 'emit_ww', 'emit_so1', 'emi_dust1',
                    'aqi', 'pm'])
    
    # add column for day of first treatment
    treated = wf2020[wf2020['treat'] == 1]
    treated = treated[['daynum', 'city_code']].groupby('city_code')
    first = treated.apply(lambda x: x.sort_values(by = 'daynum', ascending=True).head(1))
    first = {city:day for day, city in first.values}
    wf2020 = wf2020.assign(first = [first.get(city, 0) for city in wf2020['city_code']])

    # make dummy variables for cities, days, weeks pre and post treatment
    wf2020['cities'] = wf2020['city_code'].astype('category')
    wf2020['days'] = wf2020['daynum'].astype('category')
    wf2020["week_coef"] = np.floor((wf2020["daynum"] - wf2020["first"])/7).astype(int)
    # set -1 lead and untreated to NaN so they don't get week0 dummy TODO: remove and do this post?
    wf2020["week_coef"] = np.where((wf2020["week_coef"] == -1), np.NaN, wf2020["week_coef"])
    wf2020.loc[wf2020["first"] == 0, 'week_coef'] = np.NaN
    # wf2020["week_coef"][wf2020["first"] == 0] = np.NaN
    wf2020["week_coef"] = wf2020["week_coef"].astype('category')

    wf2020 = pd.get_dummies(wf2020, columns=['cities', 'days'], drop_first=True)
    wf2020 = pd.get_dummies(wf2020, columns=['week_coef'])
    return wf2020

# get subset of wf2020 that is either treated on given day or never treated
def get_group(wf2020, treat_day):
    return wf2020[(wf2020['first'] == treat_day) | (wf2020['first'] == 0)]
    
# get counts of number treated on each day
def get_day_count(wf2020):
    treated = wf2020[wf2020['treat'] == 1]
    treated = treated[['daynum', 'city_code']].groupby('city_code')
    first = treated.apply(lambda x: x.sort_values(by = 'daynum', ascending=True).head(1))

    day, count = np.unique(first.daynum, return_counts = True)
    num_cities = {d:c for d,c in zip(day, count)}
    return day, count, num_cities


def get_model_string(m, m_params):
    if m == LinearRegression or m == LogisticRegression:
        return "Lin./Log. Reg"
    if m == RandomForestRegressor:
        return "RF (depth " + str(m_params['max_depth']) + ")"
    if m == RandomForestClassifier:
        return "RF(depth " + str(m_params['max_depth']) + ")"
    if m == XGBRegressor:
        return "XGBoost"
    if m == XGBClassifier:
        return "XGBoost"

def get_var_string(s):
    if s == 'aqi':
        return 'AQI'
    if s == 'l_aqi':
        return 'log AQI'
    if s == 'pm':
        return 'PM'
    else:
        return 'log PM'

# =================== estimation functions ===================

# two period DiD on specified day
def single_period_estimate(wf2020, treat_day, outcome_var, confounder_list, 
                           Q_model_class, g_model_class, Q_model_params={}, g_model_params={}):

    # average into one pre and post period
    wf2020 = wf2020.copy()
    group = get_group(wf2020, treat_day)
    group.loc[:, 'pre'] = group['daynum'] < treat_day
    group = group.groupby(['city_code', 'pre']).mean().reset_index('pre')
    compact = group[~group['pre']]
    out = group[outcome_var].values
    compact.loc[:, 'Y1-Y0'] = out[~group['pre']] - out[group['pre']]

    compact = compact.reset_index()
    outcome = compact['Y1-Y0']
    treatment = compact['treat']
    confounders = compact[confounder_list]
    
    g = treatment_k_fold_fit_and_predict(make_g_model, X=confounders, A=treatment, n_splits=10,
                                        model_class=g_model_class, model_params=g_model_params)
    Q0,Q1 = outcome_k_fold_fit_and_predict(make_Q_model, X=confounders, y=outcome, 
                                          A=treatment, n_splits=10, output_type="continuous",
                                          model_class=Q_model_class, model_params=Q_model_params)
    
    data_and_nuisance_estimates = pd.DataFrame({'g': g, 'Q0': Q0, 'Q1': Q1, 'A': treatment, 'Y': outcome})
    tau_hat, std_hat = att_aiptw(**data_and_nuisance_estimates)
    
    
    Q_model = make_Q_model(Q_model_class, Q_model_params)
    g_model = make_g_model(g_model_class, g_model_params)
    
    # include the treatment as input feature
    X_w_treatment = confounders.copy()
    X_w_treatment["A"] = treatment
    
    Q_model.fit(X_w_treatment, outcome)
    g_model.fit(confounders, treatment)

    return tau_hat, std_hat, Q_model, g_model


# test model fit for two period DiD on the given model classes
def test_single_models(wf2020, treat_day, outcome_var, confounder_list, 
                       Q_model_class, g_model_class, Q_model_params={}, g_model_params={},
                       only_treated=False):
    wf2020 = wf2020.copy()
    group = get_group(wf2020, treat_day)

    group.loc[:, 'pre'] = group['daynum'] < treat_day
    group = group.groupby(['city_code', 'pre']).mean().reset_index('pre')
    compact = group[~group['pre']]
    out = group[outcome_var].values
    compact.loc[:, 'Y1-Y0'] = out[~group['pre']] - out[group['pre']]

    compact = compact.reset_index()
    outcome = compact['Y1-Y0']
    treatment = compact['treat']
    confounders = compact[confounder_list]

    X_w_treatment = confounders.copy()
    X_w_treatment["treatment"] = treatment

    Q_mses = []
    mse_baselines = []
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    for train_index, test_index in kf.split(X_w_treatment, outcome):
        X_train, X_test = X_w_treatment.loc[train_index], X_w_treatment.loc[test_index]
        y_train, y_test = outcome.loc[train_index], outcome.loc[test_index]
        Q_model = Q_model_class(**Q_model_params)
        Q_model.fit(X_train, y_train)
        if only_treated:
            treated_indices = X_test['treatment'] == 1
            X_test = X_test[treated_indices]
            y_test = y_test[treated_indices]
            y_train = y_train.loc[X_train['treatment'] == 1]
        y_pred = Q_model.predict(X_test)
        Q_mse = mean_squared_error(y_test, y_pred)
        baseline_mse = mean_squared_error(y_train.mean()*np.ones_like(y_test), y_test)
        Q_mses.append(Q_mse)
        mse_baselines.append(baseline_mse)
    
    X = confounders.copy()
    g_ces = []
    ce_baselines = []
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    for train_index, test_index in kf.split(X, treatment):
        X_train, X_test= X.loc[train_index], X.loc[test_index]
        a_train, a_test = treatment.loc[train_index], treatment.loc[test_index]
        g_model = g_model_class(**g_model_params)
        g_model.fit(X_train, a_train)
        a_pred = g_model.predict_proba(X_test)[:,1]
        g_ce = log_loss(a_test, a_pred)
        baseline_ce = log_loss(a_test, a_train.mean()*np.ones_like(a_test))
        g_ces.append(g_ce)
        ce_baselines.append(baseline_ce)

    return np.mean(Q_mses), np.mean(g_ces), np.mean(mse_baselines), np.mean(ce_baselines)

# given model for g get predictions from single DiD (for checking overlap)
def get_ps(g_model, wf2020, treat_day, outcome_var, confounder_list):
    wf2020 = wf2020.copy()
    group = get_group(wf2020, treat_day)
    group.loc[:, 'pre'] = group['daynum'] < treat_day
    group = group.groupby(['city_code', 'pre']).mean().reset_index('pre')
    compact = group[~group['pre']]
    out = group[outcome_var].values
    compact.loc[:, 'Y1-Y0'] = out[~group['pre']] - out[group['pre']]

    compact = compact.reset_index()
    confounders = compact[confounder_list]

    ps = g_model.predict_proba(confounders)[:,1]
    return ps

# aggregate ATT estimates from multiple periods into one w/ inverse variance weighting scheme
def multi_period_estimate(wf2020, outcome_var, confounder_list,
                          Q_model_class, g_model_class, Q_model_params={}, g_model_params={}):
    wf2020 = wf2020.copy()
    wf2020['A'] = (wf2020['daynum'] == wf2020['first']).astype('int64')
    wf2020['pastweek_mean'] = wf2020.groupby('daynum')[outcome_var].transform(
        lambda x: pd.Series.rolling(x, window=7).mean()
    )

    # create week rolling mean for each city/day combo
    # dummy = wf2020.groupby('city_code')[[outcome_var, 'daynum']].rolling(window=7, on='daynum').mean().reset_index()
    # dummy = dummy.rename(columns={outcome_var: outcome_var+"_avg"})
    # wf2020 = pd.merge(wf2020, dummy, on = ['city_code', 'daynum'])

    # wf2020['diff'] = wf2020.sort_values(by = 'daynum')[outcome_var+"_avg"] \
    #     - wf2020.sort_values(by = 'daynum').groupby('city_code')[outcome_var+"_avg"].shift(7)
    # wf2020 = wf2020.dropna(subset= ['diff'])


    wf2020['diff'] = wf2020.sort_values(by = 'daynum')[outcome_var] \
            - wf2020.sort_values(by = 'daynum').groupby('city_code')[outcome_var].shift(1)
    wf2020 = wf2020.dropna(subset= ['diff'])

    outcome = wf2020['diff']
    treatment = wf2020['A']
    confounders = wf2020[confounder_list]

    Q_model = make_Q_model(Q_model_class, Q_model_params)
    X_w_treatment = confounders.copy()
    X_w_treatment["treatment"] = treatment

    g_model = make_g_model(g_model_class, g_model_params)

    res = dict()
    Q_model.fit(X_w_treatment, outcome)
    g_model.fit(confounders, treatment)

    _, _, num_cities = get_day_count(wf2020)

    for day in np.unique(wf2020['daynum']):
        df = wf2020[wf2020['daynum'] == day]
        outcome_t = df['diff']
        treatment_t = df['A']
        confounders_t = df[confounder_list]
        
        if df['A'].sum() == 0 or num_cities[day] < 2:
            continue
        
        X1 = confounders_t.copy()
        X0 = confounders_t.copy()
        X1["treatment"] = 1
        X0["treatment"] = 0
        
        Q0 = Q_model.predict(X0)
        Q1 = Q_model.predict(X1)
        g = g_model.predict_proba(confounders_t)[:,1]
        
        est, sd = att_aiptw(Q0, Q1, g, treatment_t, outcome_t)
        res[day] = (est, sd)
    
    inv_var = np.array([1/v**2 for p,v in res.values()])
    point = np.array([p for p,v in res.values()])    

    tau_hat = (point * inv_var).sum()/inv_var.sum()
    std_hat = np.sqrt(1/inv_var.sum())
    
    return tau_hat, std_hat, Q_model, g_model, res

# check model fit for the staggered DiD on the given models
def test_multi_models(wf2020, outcome_var, confounder_list,
                      Q_model_class, g_model_class, Q_model_params={}, g_model_params={},
                      only_treated=False):
    wf2020 = wf2020.copy()
    wf2020['A'] = (wf2020['daynum'] == wf2020['first']).astype('int64')

    wf2020['diff'] = wf2020.sort_values(by = 'daynum')[outcome_var] \
            - wf2020.sort_values(by = 'daynum').groupby('city_code')[outcome_var].shift(1)
    wf2020 = wf2020.dropna(subset= ['diff'])

    outcome = wf2020['diff']
    treatment = wf2020['A']
    confounders = wf2020[confounder_list]

    X_w_treatment = confounders.copy()
    X_w_treatment["treatment"] = treatment

    Q_mses = []
    mse_baselines = []
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    for train_index, test_index in kf.split(X_w_treatment, outcome):
        X_train, X_test = X_w_treatment.iloc[train_index], X_w_treatment.iloc[test_index]
        y_train, y_test = outcome.iloc[train_index], outcome.iloc[test_index]
        Q_model = Q_model_class(**Q_model_params)
        Q_model.fit(X_train, y_train)
        
        # see how well we can predict outcome on treated (spoiler: we can't)
        if only_treated:
            treated_indices = X_test['treatment'] == 1
            X_test = X_test[treated_indices]
            y_test = y_test[treated_indices]
            y_train = y_train.loc[X_train['treatment'] == 1]
        y_pred = Q_model.predict(X_test)
        Q_mse = mean_squared_error(y_test, y_pred)
        baseline_mse = mean_squared_error(y_train.mean()*np.ones_like(y_test), y_test)
        Q_mses.append(Q_mse)
        mse_baselines.append(baseline_mse)
    
    X = confounders.copy()
    g_ces = []
    ce_baselines = []
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    for train_index, test_index in kf.split(X, treatment):
        X_train, X_test= X.iloc[train_index], X.iloc[test_index]
        a_train, a_test = treatment.iloc[train_index], treatment.iloc[test_index]
        g_model = g_model_class(**g_model_params)
        g_model.fit(X_train, a_train)
        a_pred = g_model.predict_proba(X_test)[:,1]
        g_ce = log_loss(a_test, a_pred)
        baseline_ce = log_loss(a_test, a_train.mean()*np.ones_like(a_test))
        g_ces.append(g_ce)
        ce_baselines.append(baseline_ce)

    return np.mean(Q_mses), np.mean(g_ces), np.mean(mse_baselines), np.mean(ce_baselines)


# given model for g get predictions from staggered DiD (for checking overlap)
def get_multi_ps(g_model, wf2020, confounder_list):
    confounders = wf2020[confounder_list]
    ps = g_model.predict_proba(confounders)[:,1]
    return ps


# =================== sensitivity/robustness functions ===================


def two_period_parallel_trends(wf2020, treat_day, confounder_list, Q_model, g_model):
    res = dict()
    for period in range(treat_day - 7*4, treat_day, 7):

        df = wf2020[(wf2020['daynum'] >= period) & (wf2020['daynum'] < period + 7)]

        
        df.loc[:, 'pre'] = df['daynum'] < treat_day
        df = df.groupby(['city_code']).mean().reset_index()
        
        treated = df[df['first'] == treat_day]
        control = df[df['first'] == 0]
        
        confounders_t = treated[confounder_list]
        confounders_c = control[confounder_list]
        
        Xt = confounders_t.copy()
        Xc = confounders_c.copy()
        
        Xt['A'] = 0
        Xc['A'] = 0
        
        
        Qt = Q_model.predict(Xt)
        Qc = Q_model.predict(Xc)
        
        res[period] = welch_ttest(Qt, Qc)
        
    return res

def welch_ttest(x1, x2):
    
    n1 = x1.size
    n2 = x2.size
    
    m1 = np.mean(x1)
    m2 = np.mean(x2)
    
    v1 = np.var(x1, ddof=1)
    v2 = np.var(x2, ddof=1)
    
    pooled_se = np.sqrt(v1 / n1 + v2 / n2)
    delta = m1-m2
    
    tstat = delta /  pooled_se
    df = (v1 / n1 + v2 / n2)**2 / (v1**2 / (n1**2 * (n1 - 1)) + v2**2 / (n2**2 * (n2 - 1)))
    
    # two side t-test
    p = 2 * t.cdf(-abs(tstat), df)
    
    # upper and lower bounds
    lb = delta - t.ppf(1-0.05/(2),df)*pooled_se 
    ub = delta + t.ppf(1-0.05/(2),df)*pooled_se
  
    return pd.DataFrame(np.array([tstat,df,p,delta,lb,ub]).reshape(1,-1),
                         columns=['T statistic','df','pvalue 2 sided','Difference in mean','lb','ub'])



# check overlap
