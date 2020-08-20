import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf

from auxiliary.auxiliary import get_results_local_regression
from auxiliary.auxiliary import get_results_regression
from auxiliary.auxiliary import process_results
from auxiliary.auxiliary import simulate_data


# original data
original_data = pd.read_stata("data/Bronzini-Iachini_dataset.dta")
linear = smf.ols(
    """INVSALES ~ largem + treatsmall + treatlarge + ssmall + slarge + streatsmall
        + streatlarge""",
    original_data,
).fit(cov_type="cluster", cov_kwds={"groups": original_data["score"]})

# number of major runs
num_runs = 10
num_obs = 360
num_bootstrap_runs = 250
true_treatment_effect = 0.08

# true dgp is a polynomial of one model with different slope and intercept
# on each side of the treatment threshold
true_model = {
    "polynomials": 1,
    "coefficients": {
        "untreated": np.array([-0.05, -0.02]),
        "treated": np.array([0.08, 0.03]),
    },
}
results_linear_dgp = get_results_regression(
    num_runs, num_obs, num_bootstrap_runs, true_model
)
processed_results_linear_dgp = process_results(
    results_linear_dgp, true_treatment_effect
)

# true dgp is polynomial of four
true_model = {
    "polynomials": 4,
    "coefficients": {
        "untreated": np.array([-0.05, -0.0005, -0.000005, -5e-7, -5e-9]),
        "treated": np.array([0.08, -0.0008, -0.000008, -8e-7, -8e-9]),
    },
}
results_nonlinear_dgp = get_results_regression(
    num_runs, num_obs, num_bootstrap_runs, true_model
)
processed_results_nonlinear_dgp = process_results(
    results_nonlinear_dgp, true_treatment_effect
)

# alternative to local linear regression
num_runs = 1000
num_obs = 3600
start = 3
width = 30
true_model = {
    "polynomials": 4,
    "coefficients": {
        "untreated": np.array([-0.05, -0.0005, -0.000005, -5e-7]),
        "treated": np.array([0.08, -0.0008, -0.000008, -8e-7]),
    },
    "superscript": (0.01, 0.5),
}
subset = np.array([np.arange(2), np.arange(4), np.arange(6), np.arange(8)])
get_results_local_regression(num_runs, num_obs, true_model, start, width, subset)

polynomials = 3
coefficients = {
    "untreated": np.array([-0.05, -0.0005, -0.000005, -5e-7]),
    "treated": np.array([0.08, -0.0008, -0.000008, -8e-7]),
}
superscript = (0.000001, 2.5)
num_obs = 3600

data = simulate_data(num_obs, coefficients, polynomials, superscript)[0]

small = data.loc[data["large"] == 0, :]
large = data.loc[data["large"] == 1, :]

mean_small = small.groupby("score").mean()
mean_small.reset_index(inplace=True)
mean_large = large.groupby("score").mean()
mean_large.reset_index(inplace=True)

sns.scatterplot(mean_small["score"], mean_small["scaled_investment"])

sns.scatterplot(mean_large["score"], mean_large["scaled_investment"])
