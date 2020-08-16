import numpy as np
import quadprog

x = np.loadtxt("auxiliary/x.txt")
y = np.loadtxt("auxiliary/y.txt")
subset = np.array([[0, 1], [0, 2], [0, 1, 3]])


def jackknife_averaging(y, x, subset):

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
