from utils.double_ml import *
import pandas as pd
import numpy as np

# =================== dataframe gymnastics ===================
wf = pd.read_csv("data/wf.csv")
wf["temp2"] = wf["temp"] ** 2
wf["l_aqi"] = np.log(1 + wf["aqi"])
wf["l_pm"] = np.log(1 + wf["pm"])
wf.drop('Unnamed: 0', axis = 1)

city_yb = pd.read_csv("data/city_yb.csv")
city_yb.drop('Unnamed: 0', axis = 1)

def make_wf2020():
    wf2020 = wf[(wf["daynum"] >= 8401) & (wf["daynum"]<= 8461)].dropna(
        subset = ['aqi', 'pm']
    )
    # weather variable transforms
    wf2020["temp2"] = wf2020["temp"] ** 2
    wf2020["l_aqi"] = np.log(1 + wf2020["aqi"])
    wf2020["l_pm"] = np.log(1 + wf2020["pm"])

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
    

def get_day_count(wf2020):
    treated = wf2020[wf2020['treat'] == 1]
    treated = treated[['daynum', 'city_code']].groupby('city_code')
    first = treated.apply(lambda x: x.sort_values(by = 'daynum', ascending=True).head(1))

    day, count = np.unique(first.daynum, return_counts = True)
    num_cities = {d:c for d,c in zip(day, count)}
    return day, count, num_cities


# =================== estimation functions ===================


def single_period_estimate(wf2020, treat_day, outcome_var, confounder_list, 
                           Q_model_class, g_model_class, Q_model_params={}, g_model_params={}):
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

    return tau_hat, std_hat


def multi_period_estimate(wf2020, outcome_var, confounder_list,
                          Q_model_class, g_model_class, Q_model_params={}, g_model_params={}):
    wf2020 = wf2020.copy()
    wf2020['A'] = (wf2020['daynum'] == wf2020['first']).astype('int64')
    wf2020['pastweek_mean'] = wf2020.groupby('daynum')[outcome_var].transform(
        lambda x: pd.Series.rolling(x, window=7).mean()
    )


    wf2020['diff'] = wf2020.sort_values(by = 'daynum')[outcome_var] \
            - wf2020.sort_values(by = 'daynum').groupby('city_code')[outcome_var].shift(7)
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
    
    return tau_hat, std_hat



# =================== sensitivity/robustness functions ===================

def parametric_parallel_trends():
    return


def two_period_parallel_trends(treat_day, week=False):
    return

