# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
from scipy.stats import skewnorm

df = pd.read_stata("data/Bronzini-Iachini_dataset.dta")

# a = (
#     smf.ols("INVSALES ~ treat", data)
#     .fit(cov_type="cluster", cov_kwds={"groups": data["score"]})
#     .summary()
# )

# sns.scatterplot(x=data["score"], y=data["INVSALES"])
# sns.distplot(data["score"])
# sns.jointplot(data["score"], data["INVSALES"])

# mean_small = data.groupby("score").mean()
# sns.scatterplot(x=mean_small.index, y=mean_small["INVSALES"])

b = (
    smf.ols(
        """INVSALES ~ largem + treatsmall + treatlarge + ssmall + slarge + streatsmall
        + streatlarge""",
        df,
    )
    .fit(cov_type="cluster", cov_kwds={"groups": df["score"]})
    .summary()
)
c = smf.ols(
    """INVSALES ~ largem + treatsmall + treatlarge + ssmall + slarge + streatsmall +
    streatlarge""",
    df,
).fit(cov_type="cluster", cov_kwds={"groups": df["score"]})

# large = data.loc[data["largem"] == 1.0]
# sns.distplot(small["score"], bins=100)
# sns.distplot(large["score"], bins=100)
# sns.scatterplot(small["score"])


def simulate_data(num_obs, polynomials=1):

    # create empty data frame for data
    data = pd.DataFrame(
        index=pd.Index(np.arange(num_obs), name="firm"),
        columns=["large", "score", "scaled_investment"],
    )
    # draw size of the firm
    data["large"] = np.random.binomial(1, 0.5, num_obs)
    num_large, num_small = data["large"].value_counts()
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
        1 - np.abs(data["score"].astype(float).to_numpy()) / 100
    ) * np.random.normal(size=num_obs)

    sns.scatterplot(data["score"], error)

    # simulated dependent variable
    beta_large = 0.05
    treatment_effect = 0.5
    data["scaled_investment"] = (
        beta_large * data["large"]
        + treatment_effect * (1 - data["large"]) * data["treated"]
        + -0.002 * (1 - data["large"]) * data["score"]
        + error
    )
    data = data.astype(float)
    bla = data.groupby(["score", "large"]).mean()
    bla.reset_index(inplace=True)
    sns.scatterplot(
        bla.loc[bla["large"] == 0, "score"],
        bla.loc[bla["large"] == 0, "scaled_investment"],
    )
    # sns.distplot(score_small, bins=100)
