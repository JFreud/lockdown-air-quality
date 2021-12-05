from double_ml import *
import pandas as pd
import numpy as np

# =================== dataframe gymnastics ===================
wf = pd.read_csv("data/wf.csv")
wf["temp2"] = wf["temp"] ** 2
wf["l_aqi"] = np.log(1 + wf["aqi"])
wf["l_pm"] = np.log(1 + wf["pm"])

city_yb = pd.read_csv("data/city_yb.csv")


def make_wf2020(wf):
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
    wf2020["week_coef"][wf2020["first"] == 0] = np.NaN
    wf2020["week_coef"] = wf2020["week_coef"].astype('category')

    wf2020 = pd.get_dummies(wf2020, columns=['cities', 'days'], drop_first=True)
    wf2020 = pd.get_dummies(wf2020, columns=['week_coef'])
    return wf2020

# get subset of wf2020 that is either treated on given day or never treated
def get_group(wf2020, treat_day):
    return wf2020[(wf2020['first'] == treat_day) | (wf2020['first'] == 0)]
    


# =================== estimation functions ===================


def single_period_estimate(wf2020, treat_day, outcome_var, confounder_list, 
                           Q_model_class, g_model_class, Q_model_params={}, g_model_params={}):
    group = get_group(wf2020, treat_day)
    group['pre'] = group['daynum'] < treat_day
    group = group.groupby(['city_code', 'pre']).mean().reset_index('pre')
    compact = group[~group['pre']]
    out = group[outcome_var].values
    compact['Y1-Y0'] = out[~group['pre']] - out[group['pre']]

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


def multi_period_estimate(wf2020, outcome):
    return



# =================== sensitivity/robustness functions ===================

def parametric_parallel_trends():
    return


def two_period_parallel_trends(treat_day, week=False):
    return

