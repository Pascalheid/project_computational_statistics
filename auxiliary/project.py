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

sns.scatterplot(original_data["s"], linear.resid)
# number of major runs
num_runs = 15
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
    num_runs, num_obs, num_bootstrap_runs, true_model, "random_cluster"
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
    num_runs, num_obs, num_bootstrap_runs, true_model, "random_cluster"
)
processed_results_nonlinear_dgp = process_results(
    results_nonlinear_dgp, true_treatment_effect
)

# alternative to local linear regression
results = {}
error_dists = ["normal", "inverse", "random_cluster", "random_homo"]
for error_dist in error_dists:
    num_runs = 10
    num_obs = 1000
    start_local = 5
    start_jma = 5
    width = 50
    true_model = {
        "polynomials": 5,
        "coefficients": {
            "untreated": np.array([-0.05, -0.0005, -0.00005, -5e-7, -5e-9, -1e-11]),
            "treated": np.array([0.08, -0.0008, -0.00008, -8e-7, -5e-9, -4e-11]),
        },
    }
    subset = np.array([np.arange(2), np.arange(4), np.arange(6), np.arange(8)])
    results[error_dist] = get_results_local_regression(
        num_runs,
        num_obs,
        true_model,
        start_local,
        start_jma,
        width,
        subset,
        error_dist=error_dist,
    )


polynomials = 4
coefficients = {
    "untreated": np.array([-0.05, -0.0005, -0.000005, -5e-7, -5e-9]),
    "treated": np.array([0.08, -0.0008, -0.000008, -8e-7, -8e-9]),
}
superscript = (0, 0)
num_obs = 1000

data, error = simulate_data(num_obs, coefficients, polynomials, superscript, "bla")

small = data.loc[data["large"] == 0, :]
large = data.loc[data["large"] == 1, :]

mean_small = small.groupby("score").mean()
mean_small.reset_index(inplace=True)
mean_large = large.groupby("score").mean()
mean_large.reset_index(inplace=True)

sns.scatterplot(mean_small["score"], mean_small["scaled_investment"])
sns.scatterplot(data["score"], error)

sns.scatterplot(mean_large["score"], mean_large["scaled_investment"])
