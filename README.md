# lockdown-air-quality
Estimating causal effects of COVID lockdown on air quality based on He et al. 2020


```
data/                     -- original data at https://github.com/yhyhpan/COVID19_LOCKDOWN/tree/master/data
  | city_yb.csv           -- city covariates
  | wf.csv                -- treatment and weather data
plots/
utils/
  | analysis.py           -- modeling and sensitivity analysis functions
  | double_ml.py          -- double ML functions
double_ML_results.ipynb   -- estimates on data
model_fits.ipynb          -- different model fits for both approaches
overlap.ipynb             -- checking overlap conditions
replicate.ipynb           -- replication of He at al. analyses
report.pdf                -- writeup
```
