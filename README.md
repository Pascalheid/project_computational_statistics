[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/Pascalheid/project_computational_statistics/master)
[![Build Status](https://travis-ci.com/Pascalheid/project_computational_statistics.svg?branch=master)](https://travis-ci.com/Pascalheid/project_computational_statistics)

## Stacking/ Jackknife Model Averaging and its Use for Regression Discontinuity Design

In the notebook [Project.ipynb](https://github.com/Pascalheid/project_computational_statistics/blob/master/Project.ipynb) I explore to which extent the Jackknife Model Averaging (JMA) (based on stacking) suggested by Hansen and Racine (2012) might be an improvement over the typical use of model selection via the Aikake information criterion (AIC) in Regression Discontinuity Design (RDD). In a second step I further investigate even beyond that JMA might be employed to determine the optimal bandwidth in RDD. For this I run two simulations based on the RDD setup found in Bronzini and Iachini (2014).

My main sources are:

> Bronzini, R., & Iachini, E. (2014). Are incentives for R&D effective? Evidence from a regression discontinuity approach. *American Economic Journal: Economic Policy, 6*(4), 100-134.
> Hansen, B. E., & Racine, J. S. (2012). Jackknife model averaging. *Journal of Econometrics, 167*(1), 38-46.

## Replication

I set up Travis as a continous integration of my jupyter notebook to ensure reproducibility. As my simulations take quite long, I set it up such that Travis relies on loading the previously run simulations to check whether the notebook build successfully instead of running them itself. This is represented in the code of my notebook. It gives you the option to simply load the simulation results or run the simulations again yourself. 
The best way to access my notebook is to use mybinder by clicking on the badge at the very top. It loads the whole environment needed to run the code in notebook. 
