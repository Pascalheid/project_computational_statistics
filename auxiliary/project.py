import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from joblib import delayed
from joblib import Parallel

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

small_original = original_data.loc[original_data["largem"] == 0, :]
mean_small_or = small_original.groupby("s").mean()
mean_small_or.reset_index(inplace=True)
sns.scatterplot(mean_small_or["s"], mean_small_or["INVSALES"])


sns.scatterplot(original_data["s"], linear.resid)
# number of major runs
np.random.seed(123)
num_runs = 1000
num_bootstrap_runs = 200
true_treatment_effect = 0.08
true_model = {
    "polynomials": 1,
    "coefficients": {
        "untreated": np.array([-0.05, -0.0016]),
        "treated": np.array([0.08, 0.0003]),
    },
}
# true dgp is a polynomial of one model with different slope and intercept
# on each side of the treatment threshold
results_linear_dgp = {}
processed_results_linear_dgp = {}
for error_distr in ["random_cluster", "normal", "inverse"]:
    results_linear_dgp[error_distr] = {}
    processed_results_linear_dgp[error_distr] = {}
    for num_obs in [100, 200, 360, 600, 1000]:
        results_linear_dgp[error_distr][str(num_obs)] = get_results_regression(
            num_runs, num_obs, num_bootstrap_runs, true_model, error_distr
        )
        processed_results_linear_dgp[error_distr][str(num_obs)] = process_results(
            results_linear_dgp[error_distr][str(num_obs)], true_treatment_effect
        )

processed_results_linear_dgp = {}
for error_distr in ["homoskedastic", "random_cluster", "normal", "inverse"]:
    processed_results_linear_dgp[error_distr] = {}
    for num_obs in [100, 200, 360, 600, 1000]:
        processed_results_linear_dgp[error_distr][str(num_obs)] = process_results(
            results_linear_dgp[error_distr][str(num_obs)], true_treatment_effect
        )

# true dgp is polynomial of four
np.random.seed(123)
num_runs = 1000
num_bootstrap_runs = 200
true_treatment_effect = 0.08
true_model = {
    "polynomials": 4,
    "coefficients": {
        "untreated": np.array([-0.05, -0.00016, -0.00006, -5e-6, -5e-8]),
        "treated": np.array([0.08, 0.00002, 0.00009, -8e-6, -8e-8]),
    },
}
# true dgp is a polynomial of one model with different slope and intercept
# on each side of the treatment threshold
results_nonlinear_dgp = {}
processed_results_nonlinear_dgp = {}
for error_distr in ["homoskedastic", "random_cluster", "normal", "inverse"]:
    results_nonlinear_dgp[error_distr] = {}
    processed_results_nonlinear_dgp[error_distr] = {}
    for num_obs in [100, 200, 360, 600, 1000]:
        results_nonlinear_dgp[error_distr][str(num_obs)] = get_results_regression(
            num_runs, num_obs, num_bootstrap_runs, true_model, error_distr
        )
        processed_results_nonlinear_dgp[error_distr][str(num_obs)] = process_results(
            results_nonlinear_dgp[error_distr][str(num_obs)], true_treatment_effect
        )


def loop(error_dist):
    result = {}
    num_runs = 1000
    num_obs = 1000
    start_local = 10
    start_jma = 10
    width = 35
    true_model = {
        "polynomials": 4,
        "coefficients": {
            "untreated": np.array([-0.05, -0.00016, -0.00006, -5e-6, -5e-8]),
            "treated": np.array([0.08, 0.00002, 0.00009, -8e-6, -8e-8]),
        },
    }
    subset = np.array([np.arange(2), np.arange(4), np.arange(6), np.arange(8)])
    result[error_dist] = get_results_local_regression(
        num_runs,
        num_obs,
        true_model,
        start_local,
        start_jma,
        width,
        subset,
        error_dist=error_dist,
    )

    return result


np.random.seed(123)
results_1 = Parallel(n_jobs=4, verbose=50)(
    delayed(loop)(error_dist)
    for error_dist in ["homoskedastic", "random_cluster", "normal", "inverse"]
)

processed_results = {}
for number, error_dist in enumerate(
    ["homoskedastic", "random_cluster", "normal", "inverse"]
):
    processed_results[error_dist] = (
        results_1[number][error_dist].groupby("Model").mean()
    )

for s in [0, 0.015, 0.03]:
    polynomials = 4
    coefficients = {
        "untreated": np.array([-0.05, -0.00016, -0.00006, -5e-6, -5e-8]),
        "treated": np.array([0.08, 0.00002, 0.00009, -8e-6, -8e-8]),
    }
    superscript = (s, 0.5)
    num_obs = 100000

    data, error = simulate_data(
        num_obs, coefficients, polynomials, superscript, "random_cluster"
    )

    small = data.loc[data["large"] == 0, :]
    large = data.loc[data["large"] == 1, :]

    mean_small = small.groupby("score").mean()
    mean_small.reset_index(inplace=True)
    mean_large = large.groupby("score").mean()
    mean_large.reset_index(inplace=True)

    sns.scatterplot(mean_small["score"], mean_small["scaled_investment"])
sns.scatterplot(data["score"], error)

sns.scatterplot(mean_large["score"], mean_large["scaled_investment"])
