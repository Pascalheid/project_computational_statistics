import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from auxiliary.auxiliary import get_results_regression
from auxiliary.auxiliary import process_results

# original data
original_data = pd.read_stata("data/Bronzini-Iachini_dataset.dta")
linear = smf.ols(
    """INVSALES ~ largem + treatsmall + treatlarge + ssmall + slarge + streatsmall
        + streatlarge""",
    original_data,
).fit(cov_type="cluster", cov_kwds={"groups": original_data["score"]})

# number of major runs
num_runs = 5
num_obs = 3600
num_bootstrap_runs = 200
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

true_model = {
    "polynomials": 4,
    "coefficients": {
        "untreated": np.array([-0.05, -0.02, -0.0009, -0.000004, 0.0000008]),
        "treated": np.array([0.08, 0.03, 0.0001, 0.000005, -0.0000002]),
    },
}
results_nonlinear_dgp = get_results_regression(
    num_runs, num_obs, num_bootstrap_runs, true_model
)
processed_results_nonlinear_dgp = process_results(
    results_nonlinear_dgp, true_treatment_effect
)
