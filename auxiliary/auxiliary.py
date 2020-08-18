import numpy as np
import pandas as pd
import quadprog
import seaborn as sns
from scipy.stats import skewnorm

x = np.loadtxt("auxiliary/x.txt")
y = np.loadtxt("auxiliary/y.txt")
subset = np.array([[0, 1], [0, 2], [0, 1, 3]])


def jackknife_averaging(y, x, subset):
    """
    calulates the averaged coefficients across several linear regression models
    according to the Jackknife Model Averaging (Hansen, Racine (2012)).

    Parameters
    ----------
    y : np.array
        the dependent variable of the rergression model.
    x : np.array
        the matrix containing all the regressors needed for the competing
        models across which is averaged.
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
            weights.T
            @ (transformed_residuals_all.T @ transformed_residuals_all)
            @ weights
        ) / num_obs

        return weights, averaged_beta, expected_test_mse

        # # Running it by hand
        # fitted_values = np.zeros(num_obs)
        # for row in range(num_obs):
        #     x_row = x_model[row]
        #     x_temp = np.delete(x_model, row, axis=0)
        #     y_temp = np.delete(y, row, axis=0)
        #     fitted_values[row] =  x_row @ np.linalg.inv(x_temp.T @ x_temp) @ x_temp.T @ y_temp
        # residuals = y - fitted_values


def simulate_data(num_obs, coefficients, polynomials=1):
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
    score_large.loc[score_large["score"] < 0] = np.random.choice(
        score_large.loc[(score_large["score"] >= 0) & score_large["score"] <= 100]
        .to_numpy()
        .flatten(),
        size=len(score_large.loc[score_large["score"] < 0]),
    )
    score_large.loc[score_large["score"] > 100] = np.random.choice(
        score_large.loc[(score_large["score"] >= 0) & score_large["score"] <= 100]
        .to_numpy()
        .flatten(),
        size=len(score_large.loc[score_large["score"] > 100]),
    )

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

    sns.scatterplot(data["score"], error)

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
        + error
    )
    data = data.astype(float)

    return data
