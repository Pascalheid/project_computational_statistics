import numpy as np
import pandas as pd
import quadprog
import statsmodels.formula.api as smf
from scipy.stats import skewnorm

# import seaborn as sns
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import LeaveOneOut
# from sklearn.model_selection import cross_val_score
# from functools import partial

# x = np.loadtxt("auxiliary/x.txt")
# y = np.loadtxt("auxiliary/y.txt")
# subset = np.array([[0, 1], [0, 2], [0, 1, 3]])
# a,b,c = jackknife_averaging(y, x, subset)

# num_obs = x.shape[0]

# # Running it by hand
# fitted_values = np.zeros(num_obs)
# for row in range(num_obs):
#     x_row = x[row]
#     y_row = y[row]
#     x_temp = np.delete(x, row, axis=0)
#     y_temp = np.delete(y, row, axis=0)
#     averaged_beta = jackknife_averaging(y_temp, x_temp, subset)[1]
#     fitted_values[row] = x_row @ averaged_beta
# residuals = y - fitted_values
# mse = (residuals.T @ residuals) / num_obs


def jackknife_averaging(data, subset):
    """
    calulates the averaged coefficients across several linear regression models
    according to the Jackknife Model Averaging (Hansen, Racine (2012)).

    Parameters
    ----------
    data : pd.DataFrame
        first column consists of the dependent variable and the others
        of the regressors over which the averaging is supposed to be performed.
    subset : np.array
        This array contains in each row the index of the column of the x
        matrix to indicate which regressors should be added for this model.
        Each row, hence, describes one model.

    Returns
    -------
    weights : np.array
        the optimal weights to average the coefficients.
    averaged_beta : np.array
        the averaged coefficients.
    expected_test_mse : float
        the expected test MSE when applying the averaged coefficients.

    """
    y = data.iloc[:, 0].astype(float).to_numpy()
    x = data.iloc[:, 1:].astype(float).to_numpy()
    num_obs = x.shape[0]
    num_regressors = x.shape[1]
    num_models = subset.shape[0]

    beta_all = np.zeros((num_regressors, num_models))
    transformed_residuals_all = np.zeros((num_obs, num_models))

    for model in range(num_models):
        x_model = x[:, subset[model]]
        beta_model = np.linalg.inv(x_model.T @ x_model) @ x_model.T @ y
        beta_all[subset[model], model] = beta_model

        residuals_model = y - x_model @ beta_model
        transformer = np.diag(x_model @ np.linalg.inv(x_model.T @ x_model) @ x_model.T)
        transformed_residuals_all[:, model] = residuals_model * (1 / (1 - transformer))

    weights = quadprog.solve_qp(
        transformed_residuals_all.T @ transformed_residuals_all,
        np.zeros(num_models),
        np.hstack(
            (
                np.ones((num_models, 1)),
                np.identity(num_models),
                -np.identity(num_models),
            )
        ),
        np.hstack((np.ones(1), np.zeros(num_models), -np.ones(num_models))),
        1,
    )[0]
    averaged_beta = beta_all @ weights
    expected_test_mse = (
        weights.T @ (transformed_residuals_all.T @ transformed_residuals_all) @ weights
    ) / num_obs

    # # Running it by hand
    # fitted_values = np.zeros(num_obs)
    # for row in range(num_obs):
    #     x_row = x_model[row]
    #     x_temp = np.delete(x_model, row, axis=0)
    #     y_temp = np.delete(y, row, axis=0)
    #     fitted_values[row] =  x_row @ np.linalg.inv(x_temp.T @ x_temp) @ x_temp.T @ y_temp
    # residuals = y - fitted_values

    return weights, averaged_beta, expected_test_mse


def simulate_data(num_obs, coefficients, polynomials=1, curvature=(0, 0)):
    """
    Simulate data with different polynomials for small firms
    without any treatment effect for large firms with a flat dependent variable
    around zero.

    Parameters
    ----------
    num_obs : int
        the total number of firms.
    coefficients : dict
        dictinairy with keys "untreated" and "treated" both holding a numpy array
        of length polynomials. The first float in each numpy array corresponds
        to the coeffcient for polynomial zero.
    polynomials : int, optional
        the amount of polynomials for each untreated and treated firms.
        The default is 1.
    curvature : tuple
        indicates the coefficient and superscript of a curvature regressors.

    Returns
    -------
    data : pandas DataFrame
        holds the simulated independent as well as dependent variables.

    """
    # create empty data frame for data
    data = pd.DataFrame(
        index=pd.Index(np.arange(num_obs), name="firm"),
        columns=["large", "score", "scaled_investment"],
    )
    # draw size of the firm
    data["large"] = np.random.binomial(1, 0.5, num_obs)
    data["small"] = 1 - data["large"]
    value_counts = data["large"].value_counts().to_dict()
    num_small = value_counts[0]
    num_large = value_counts[1]
    # get scores for large firms
    loc = 90
    scale = 20
    score_large = pd.DataFrame(
        skewnorm.rvs(-5, loc=loc, scale=scale, size=num_large), columns=["score"]
    )
    array = score_large.loc[(score_large["score"] <= 90) & (score_large["score"] >= 80)]

    # flatten peak for normal distribution
    score_large.loc[(score_large["score"] <= 90) & (score_large["score"] >= 80)] = (
        np.random.uniform(80, 92, len(array))
    ).reshape((len(array), 1))

    # make sure no value is below zero or above 100
    if len(score_large.loc[score_large["score"] < 0]) > 0:
        score_large.loc[score_large["score"] < 0] = np.random.choice(
            score_large.loc[(score_large["score"] >= 0) & score_large["score"] <= 100]
            .to_numpy()
            .flatten(),
            size=len(score_large.loc[score_large["score"] < 0]),
        ).reshape(len(score_large.loc[score_large["score"] < 0]), 1)
    if len(score_large.loc[score_large["score"] > 100]) > 0:
        score_large.loc[score_large["score"] > 100] = np.random.choice(
            score_large.loc[(score_large["score"] >= 0) & score_large["score"] <= 100]
            .to_numpy()
            .flatten(),
            size=len(score_large.loc[score_large["score"] > 100]),
        ).reshape(len(score_large.loc[score_large["score"] > 100]), 1)

    # round the numbers to the next integer
    score_large = score_large.round()
    data.loc[data["large"] == 1, "score"] = score_large.values

    # get scores for small firms
    loc = 87
    scale = 15
    num_normal = int(4 / 5 * num_small)
    score_small_1 = pd.DataFrame(
        skewnorm.rvs(-2, loc=loc, scale=scale, size=num_normal), columns=["score"]
    )
    # adjust for uniform like lower tail
    score_small_2 = pd.DataFrame(
        np.random.uniform(20, 55, num_small - num_normal), columns=["score"]
    )
    score_small = pd.concat([score_small_1, score_small_2])
    score_small.loc[score_small["score"] > 100] = np.random.choice(
        score_small.loc[(score_small["score"] >= 0) & score_small["score"] <= 100]
        .to_numpy()
        .flatten(),
        size=len(score_small.loc[score_small["score"] > 100]),
    ).reshape(len(score_small.loc[score_small["score"] > 100]), 1)
    score_small = score_small.round()

    data.loc[data["large"] == 0, "score"] = score_small.values

    # get treatment variable based on score
    data.loc[data["score"] >= 75, "treated"] = 1
    data.loc[data["score"] < 75, "treated"] = 0
    # normalize score
    # data = data.astype(int)
    data["score"] = data["score"] - 75

    error = (
        0.05 - 0.05 * np.abs(data["score"].astype(float).to_numpy()) / 100
    ) * np.random.normal(size=num_obs)

    # sns.scatterplot(data["score"], error)

    # simulated dependent variable
    # extract polynomials
    treated = []
    untreated = []
    for poly in np.arange(polynomials + 1):
        string_untreated = "untreated_score_" + str(poly)
        untreated.append(string_untreated)
        data[string_untreated] = data["small"] * (data["score"] ** poly)
        string_treated = "treated_score_" + str(poly)
        treated.append(string_treated)
        data[string_treated] = data["treated"] * data["small"] * (data["score"] ** poly)

    data["scaled_investment"] = (
        (coefficients["untreated"] * data[untreated].astype(float).to_numpy()).sum(
            axis=1
        )
        + (coefficients["treated"] * data[treated].astype(float).to_numpy()).sum(axis=1)
        - curvature[0] * data["small"] * (np.abs(data["score"]) ** curvature[1])
        + curvature[0]
        * data["treated"]
        * data["small"]
        * (np.abs(data["score"]) ** curvature[1])
        + error
    )
    data = data.astype(float)
    data = data[["scaled_investment", "large", "score"]]

    return data


def bandwidth_selection_jma(width, gridsize, data, subset):
    bandwidth = np.linspace(1, width, gridsize)

    mse = []
    betas = []
    for h in bandwidth:
        data_temp = data.loc[data["score"].between(-h, h)]
        results_temp = jackknife_averaging(data_temp, subset)
        betas.append(results_temp[1])
        mse.append(results_temp[2])

    min_index = np.argmin(mse)
    min_mse = mse[min_index]
    min_betas = betas[min_index]

    return min_betas, min_mse


def get_results_regression(num_runs, num_obs, num_bootstrap_runs, true_model):
    """
    obtains the results from a Monte Carlo simulation in which I get the coeffcients
    for polynomial regression models and the Jackknife model averaging for
    several data sets following the same DGP.

    Parameters
    ----------
    num_runs : int
        number of major simulation runs.
    num_obs : int
        number of firms per simulated data set.
    num_bootstrap_runs : int
        number of bootstrap runs for a single data set to get the confidence
        intervals for Jackknife Model Averaging.
    true_model : dict
        has all necessary keywords for the data simulation ``simulate_data ``
        as keys.

    Returns
    -------
    results : pd.DataFrame
        data frame containing the parameter for the treatment effect and
        an indicator of whether the confidence interval covers the true
        treatment effect parameter per model and per run.

    """
    # set seed
    np.random.seed(123)

    # create empty dataframe for results
    models = [
        "polynomial_0",
        "polynomial_1",
        "polynomial_2",
        "polynomial_3",
        "AIC",
        "JMA",
    ]
    index = pd.MultiIndex.from_product(
        [np.arange(num_runs), models], names=["Run", "Model"]
    )
    results = pd.DataFrame(columns=["Treatment Effect", "95% Coverage"], index=index)

    # true specifications for the data simulation
    polynomials = true_model["polynomials"]
    coefficients = true_model["coefficients"]
    if "superscript" in true_model:
        superscript = true_model["superscript"]
    else:
        superscript = (0, 0)
    true_treatment_effect = coefficients["treated"][0]

    for run in np.arange(num_runs):

        # plain data
        data = simulate_data(num_obs, coefficients, polynomials, superscript)

        # add covariates
        data.loc[data["score"] >= 0, "treated"] = 1
        data.loc[data["score"] < 0, "treated"] = 0

        # add interactions to account for different functional forms on both sides of cutoff
        columns = ["scaled_investment"]
        interactions = 3
        for poly in np.arange(interactions + 1):
            data["untreated_score_" + str(poly)] = (1 - data["large"]) * (
                data["score"] ** poly
            )
            data["treated_score_" + str(poly)] = (
                data["treated"] * (1 - data["large"]) * (data["score"] ** poly)
            )
            columns.append("untreated_score_" + str(poly))
            columns.append("treated_score_" + str(poly))
        data = data[columns]

        # get coverage and coefficient for single polynomial regressions
        aic = np.zeros(interactions + 1)
        for poly in np.arange(interactions + 1):
            regressors = []
            for number in np.arange(poly + 1):
                regressors.append("untreated_score_" + str(number))
                regressors.append("treated_score_" + str(number))
            fit = smf.ols(
                "scaled_investment ~ -1 +" + " + ".join(regressors[:]), data,
            ).fit(cov_type="HC0")
            results.loc[
                (run, "polynomial_" + str(poly)), "Treatment Effect"
            ] = fit.params[1]
            if (
                true_treatment_effect >= fit.conf_int(0.05)[0][1]
                and true_treatment_effect <= fit.conf_int(0.05)[1][1]
            ):
                results.loc[(run, "polynomial_" + str(poly)), "95% Coverage"] = 1
            else:
                results.loc[(run, "polynomial_" + str(poly)), "95% Coverage"] = 0
            aic[poly] = fit.aic
        min_index = np.argmin(aic)
        results.loc[(run, "AIC"), :] = results.loc[(run, slice(None)), :].iloc[
            min_index
        ]

        # get coverage and coefficients for Jackknife Model Averaging
        subset = np.array([np.arange(2), np.arange(4), np.arange(6), np.arange(8)])
        jma_results = jackknife_averaging(data, subset=subset)
        results.loc[(run, "JMA"), "Treatment Effect"] = jma_results[1][1]
        # to get the confidence intervals for coverage I bootstrap from the data
        beta = np.zeros(num_bootstrap_runs)
        for bootstrap_run in np.arange(num_bootstrap_runs):
            data_new = data.sample(n=num_obs, replace=True)
            beta[bootstrap_run] = jackknife_averaging(data_new, subset)[1][1]
        lower_ci = np.percentile(beta, 2.5)
        upper_ci = np.percentile(beta, 97.5)
        if true_treatment_effect >= lower_ci and true_treatment_effect <= upper_ci:
            results.loc[(run, "JMA"), "95% Coverage"] = 1
        else:
            results.loc[(run, "JMA"), "95% Coverage"] = 0
    results = results.astype(float)

    return results


def process_results(results, true_treatment_effect):
    """
    translates the raw results from the Monte Carlos Simulation into processed
    results showing the estimated treatment effect, its standard deviation,
    the root mean squared error and the coverage probability of the
    95% confidence interval across different models.

    Parameters
    ----------
    results : pd.DataFrame
        the results from ``get_results_regression``.
    true_treatment_effect : float
        the true treatment effect of the underlying DGP.

    Returns
    -------
    processed_results : pd.DataFrame
        the resulting processed results.

    """
    index = ["Treatment Effect", "Bias", "Standard Error", "RMSE", "95% Coverage"]
    columns = results.index.to_frame().loc[(0, slice(None)), "Model"].to_numpy()
    columns = np.sort(columns)
    processed_results = pd.DataFrame(index=index, columns=columns)
    processed_results.index.name = "Statistic"

    processed_results.loc[["Treatment Effect", "95% Coverage"]] = (
        results.groupby("Model").mean().T
    )
    processed_results.loc[["Bias"]] = (
        processed_results.loc[["Treatment Effect"]].to_numpy() - true_treatment_effect
    )
    processed_results.loc[["Standard Error"]] = (
        results.groupby("Model")["Treatment Effect"].std().to_frame().T.to_numpy()
    )
    processed_results = processed_results.astype(float)

    processed_results.loc[["RMSE"]] = (
        processed_results.loc[["Bias"]].to_numpy() ** 2
        + processed_results.loc[["Standard Error"]].to_numpy() ** 2
    ) ** 0.5

    return processed_results


# def bandwidth_selection(bound, gridsize, df):
#     bandwidth = np.linspace(1, bound, gridsize)

#     rslt_err = dict()
#     for label in ["below", "above"]:

#         rslt_err[label] = list()

#         for h in bandwidth:

#             if label == "below":
#                 df_subset = df.loc[df["difshare"].between(-h, +0.00)]
#             else:
#                 df_subset = df.loc[df["difshare"].between(+0.00, +h)]

#             y = df_subset[["outcomenext"]]
#             x = df_subset[["difshare", "treated"]]
#             df_subset["treated"] = 1
#             df_subset.loc[df_subset["difshare"] < 0, "treated"] = 0

#             results = cross_val_score_p(model, x, y)
#             err = results.mean() * (-1)
#             rslt_err[label].append(err)

#     # Type tranformations
#     for label in ["below", "above"]:
#         rslt_err[label] = np.array(rslt_err[label])
#     rslt_err["error"] = (rslt_err["above"][::-1] + rslt_err["below"]) / 2

#     pkl.dump(rslt_err, open("rslt.rdd.pkl", "wb"))


# df = pd.read_stata("data/individ_final.dta")

# num_points = 10
# bandwidth = np.linspace(0.01, 0.50, num_points)

# scoring = "neg_mean_squared_error"
# model = LinearRegression()
# cv = LeaveOneOut()

# cross_val_score_p = partial(cross_val_score, scoring=scoring, cv=cv)
# rslt_err = dict()
# for label in ["below", "above"]:

#     rslt_err[label] = list()

#     h = 0.01
#     if label == "below":
#         df_subset = df.loc[df["difshare"].between(-h, +h)]
#     else:
#         df_subset = df.loc[df["difshare"].between(+0.00, +h)]

#     y = df_subset[["outcomenext"]]
#     x = df_subset[["difshare"]]

#     results = cross_val_score_p(model, x, y)
#     err = results.mean() * (-1)
#     rslt_err[label].append(err)

# # Type tranformations
# for label in ["below", "above"]:
#     rslt_err[label] = np.array(rslt_err[label])
# rslt_err["error"] = (rslt_err["above"][::-1] + rslt_err["below"]) / 2

# # Running it by hand
# fitted_values = np.zeros(num_obs)
# for row in range(num_obs):
#     x_row = x[row]
#     x_temp = np.delete(x, row, axis=0)
#     y_temp = np.delete(y, row, axis=0)
#     fitted_values[row] =  x_row @ np.linalg.inv(x_temp.T @ x_temp) @ x_temp.T @ y_temp
# residuals = y - fitted_values
# mse = residuals.T @ residuals / num_obs

# mse_1 = np.zeros(num_obs)
# for row in range(num_obs):
#     x_temp = np.delete(x, row, axis=0)
#     y_temp = np.delete(y, row, axis=0)
#     fitted_values =  x_temp @ np.linalg.inv(x_temp.T @ x_temp) @ x_temp.T @ y_temp
#     residuals = y_temp - fitted_values
#     mse_1[row] = residuals.T @ residuals / num_obs
# mse_1 = mse_1.mean()
