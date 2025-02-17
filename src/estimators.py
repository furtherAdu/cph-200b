import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.neighbors import NearestNeighbors
from lifelines import KaplanMeierFitter
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor


def kaplan_meier(df, time_col='time', event_col='DEATH_EVENT', observed_col='N_observed', propensity_col=None, alpha=0.05):
    """
    Calculates the Kaplan-Meier estimate from a DataFrame.

    Args:
        df: DataFrame with time_col and event_col columns.

    Returns:
        DataFrame with columns: 'time', 'survival_prob', 'cumulative_deaths', 'N_observed', and confidence intervals
    """
    # get max time
    max_time = df[time_col].max()

    # get total number of individuals observed at time t
    n_t = df.groupby(time_col)[[time_col]].count().rename(columns={time_col: observed_col})
    n_t = n_t.sort_index(ascending=False).cumsum().sort_index()[observed_col]

    # reindex to include all times; backfill values
    n_t = n_t.reindex(range(max_time + 1)).bfill()

    # calculate number of events by time
    events_by_time = df.groupby(time_col)[[event_col]].sum()

    # reindex to include all times; fill nan with 0
    events_by_time = events_by_time.reindex(range(max_time + 1)).fillna(value=0)

    # sort by time
    events_by_time.sort_index(inplace=True)

    # calculate n individuals who experience event at time t
    d_t = events_by_time[event_col]

    # weight by propensity
    if propensity_col is not None:
        for t in n_t.index:
            n_t.loc[t] = (df.loc[df[time_col] >= t, propensity_col]).sum() # sum of propensities at risk at time t
            d_t.loc[t] = (df.loc[df[time_col] == t, event_col] * df.loc[df[time_col] == t, propensity_col]).sum()

    # Calculate cumulative survival probability
    S_hat_t = (1 - (d_t / n_t)).cumprod()

    # calculate confidence intervals
    # per https://real-statistics.com/survival-analysis/kaplan-meier-procedure/confidence-interval-for-the-survival-function/
    exp_term = (
            norm.cdf(1 - alpha/2)
            / np.log(S_hat_t)
            * np.sqrt((d_t / (n_t * (n_t - d_t))).cumsum())
    )
    # variance = np.cumsum(d_t / (n_t * (n_t - d_t)))
    # std_error = np.sqrt(variance)
    ci_lower = S_hat_t ** np.exp(-exp_term)
    ci_upper = S_hat_t ** np.exp(exp_term)

    # create output DataFrame
    km_df = pd.DataFrame({
        'time': events_by_time.index,
        'survival_prob': S_hat_t,
        'cumulative_events': events_by_time[event_col].cumsum(),
        observed_col: n_t,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    })

    return km_df


def nearest_neighbor_km(data, patient_features, time_col, event_col, n_neighbors=5):
    """
    Estimates patient-specific survival curves using a nearest-neighbor Kaplan-Meier approach.

    Args:
        data: Pandas DataFrame containing patient data.
        patient_features: List of column names representing patient features.
        time_col: Name of the column representing time-to-event.
        event_col: Name of the column representing the event indicator (1 for event, 0 for censored).
        n_neighbors: Number of nearest neighbors to consider.

    Returns:
        A dictionary where keys are patient indices and values are KaplanMeierFitter objects
        fitted to the nearest neighbors of that patient.
        Returns None if there are not enough patients.
    """

    if len(data) < n_neighbors:
        print("Not enough patients for the specified number of neighbors.")
        return None

    X = data[patient_features].values
    knn = NearestNeighbors(n_neighbors=n_neighbors + 1)  # +1 to exclude self
    knn.fit(X)

    patient_km_fits = {}
    for i in range(len(data)):
        distances, indices = knn.kneighbors(X[i].reshape(1, -1))
        neighbor_indices = indices[0][1:]  # Exclude the patient itself
        neighbor_data = data.iloc[neighbor_indices]

        kmf = KaplanMeierFitter()
        kmf.fit(neighbor_data[time_col], neighbor_data[event_col])
        patient_km_fits[i] = kmf

    return patient_km_fits

def unadjusted_DM_estimator(data, treatment_var, outcome_var, **kwargs):
    EY1 = data.loc[data[treatment_var] == 1, outcome_var].mean()
    EY0 = data.loc[data[treatment_var] == 0, outcome_var].mean()
    tau = EY1 - EY0
    return tau

def ipw_estimator(data, treatment_var, outcome_var, covariates):
    X = data[covariates]
    T = data[treatment_var]
    Y = data[outcome_var]
    
    propensity_model = GradientBoostingClassifier(random_state=40)
    pi = propensity_model.fit(X, T).predict_proba(X)[:, 1]

    tau = ((T * Y/ pi) - ((1-T) * Y / (1 - pi))).mean()

    return tau

def t_learner(data, treatment_var, outcome_var, covariates):
    # https://statisticaloddsandends.wordpress.com/2022/05/20/t-learners-s-learners-and-x-learners/
    X = data[covariates]
    T = data[treatment_var]
    Y = data[outcome_var]

    models = {
        0: GradientBoostingClassifier(random_state=40),
        1: GradientBoostingClassifier(random_state=40)
    }

    models[0].fit(X[T == 0], Y[T == 0])
    models[1].fit(X[T == 1], Y[T == 1])

    mu0 = models[0].predict(X).mean()
    mu1 = models[1].predict(X).mean()

    tau = mu1 - mu0

    return tau

def s_learner(data, treatment_var, outcome_var, covariates):
    # https://statisticaloddsandends.wordpress.com/2022/05/20/t-learners-s-learners-and-x-learners/
    XT = data[[*covariates, treatment_var]]
    Y = data[outcome_var]

    model = GradientBoostingClassifier(random_state=40)
    model.fit(XT, Y)

    X_treated = XT.copy()
    X_treated[treatment_var] = 1
    X_control = XT.copy()
    X_control[treatment_var] = 0
    
    mu1 = model.predict(X_treated).mean()
    mu0 = model.predict(X_control).mean()

    tau = mu1 - mu0

    return tau

def x_learner(data, treatment_var, outcome_var, covariates):
    # https://statisticaloddsandends.wordpress.com/2022/05/20/t-learners-s-learners-and-x-learners/
    X = data[covariates]
    T = data[treatment_var]
    Y = data[outcome_var]

    # t-learner
    models = {
        0: GradientBoostingClassifier(random_state=40),
        1: GradientBoostingClassifier(random_state=40)
    }

    models[0].fit(X[T == 0], Y[T == 0])
    models[1].fit(X[T == 1], Y[T == 1])

    # build ITE estimator
    tau_models = {
      0: GradientBoostingRegressor(random_state=40),
      1: GradientBoostingRegressor(random_state=40)
    }
    
    tau_models[0].fit(X[T==0], models[1].predict(X[T==0]) - Y[T==0])
    tau_models[1].fit(X[T==1], Y[T==1] - models[0].predict(X[T==1]))

    # get propensity score
    propensity_model = GradientBoostingClassifier(random_state=40)
    pi = propensity_model.fit(X, T).predict_proba(X)[:, 1]

    tau = pi * tau_models[0].predict(X) + (1-pi) * tau_models[1].predict(X)

    return tau.mean()

def aipw_estimator(data, treatment_var, outcome_var, covariates):
    X = data[covariates]
    XT = data[[*covariates, treatment_var]]
    T = data[treatment_var]
    Y = data[outcome_var]

    # propensity model
    propensity_model = GradientBoostingClassifier(random_state=40)
    pi = propensity_model.fit(X, T).predict_proba(X)[:, 1]

    # s-learner outcome model
    outcome_model = GradientBoostingClassifier(random_state=40)
    outcome_model.fit(XT, Y)

    X_treated = XT.copy()
    X_treated[treatment_var] = 1
    X_control = XT.copy()
    X_control[treatment_var] = 0

    mu1 = outcome_model.predict(X_treated)
    mu0 = outcome_model.predict(X_control)
    
    # AIPW (doubly robust)
    ipw_term = T / pi * (Y - mu1) - (1 - T) / (1 - pi) * (Y - mu0)
    s_learner_term = mu1 - mu0

    tau = (s_learner_term + ipw_term).mean()

    return tau

