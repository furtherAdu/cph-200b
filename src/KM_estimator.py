import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.neighbors import NearestNeighbors
from lifelines.utils import concordance_index
from lifelines import KaplanMeierFitter


def kaplan_meier(df, time_col='time', event_col='DEATH_EVENT', observed_col='N_observed', alpha=0.05):
    """
    Calculates the Kaplan-Meier estimate from a DataFrame.

    Args:
        df: DataFrame with time_col and event_col columns.

    Returns:
        DataFrame with columns: 'time', 'survival_prob', 'cumulative_deaths', 'N_observed', and confidence intervals
    """
    # get max time
    max_time = df['time'].max()

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


def evaluate_c_index(data, patient_km_fits, time_col, event_col):
    """
    Evaluates the performance of the nearest-neighbor KM estimator using the C-index.

    Args:
        data: Pandas DataFrame containing patient data.
        patient_km_fits: Dictionary of KaplanMeierFitter objects returned by nearest_neighbor_km.
        time_col: Name of the column representing time-to-event.
        event_col: Name of the column representing the event indicator.

    Returns:
        The C-index.
    """

    true_times = data[time_col].values
    true_events = data[event_col].values
    predicted_median_survival_times = []

    for i in range(len(data)):
        kmf = patient_km_fits[i]
        predicted_median_survival_times.append(kmf.median_survival_time_)

    return concordance_index(true_times, predicted_median_survival_times, true_events)
