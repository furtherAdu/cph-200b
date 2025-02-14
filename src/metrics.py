import torch
import pandas as pd
from scipy.stats import chi2, norm
from lifelines.utils import concordance_index


def cox_partial_likelihood(risk_scores, c, t, weights=None):
    event_idxs = (c.squeeze() == 1).nonzero()

    if weights is None:
        weights = torch.ones_like(risk_scores)

    # Compute the log-likelihood
    log_likelihood = torch.tensor([0.0], requires_grad=True)
    for i in event_idxs:
        at_risk = (t >= t[i]).squeeze()  # Breslow's method for ties

        log_likelihood = (
            log_likelihood 
            + torch.log(weights[i])  # following log product rule
            + risk_scores[i] 
            - torch.logsumexp(risk_scores[at_risk] + torch.log(weights[at_risk]), dim=0)
            )

    return -log_likelihood / c.sum()  # Return negative log-likelihood for minimization


def log_rank_test(df, time_col, event_col, group_col, propensity_col=None):
    df = df.sort_values(time_col)
    events = sorted(df[time_col].unique())

    results = pd.DataFrame(columns=['o', 'e', 'v'],
                            index=events)
    results.index.name = event_col

    for j in events:
        risk = df[df[time_col] >= j]
        cases = df[(df[time_col] == j) & (df[event_col] == 1)]
        
        # define aggregating function
        if propensity_col is None:
            agg = lambda x: x.shape[0]
        else:
            agg = lambda x: x[propensity_col].sum() 

        n_1j = agg(risk[risk[group_col] == 1])
        n_0j = agg(risk[risk[group_col] == 0])
        o_1j = agg(cases[cases[group_col] == 1])
        n_j = agg(risk)
        o_j = agg(cases)
        
        if n_j == 1: # handle division by zero error
            continue

        e_j = o_j * (n_1j / n_j) 
        v_j = n_0j * n_1j * o_j * (n_j - o_j) / (n_j**2 * (n_j - 1))

        results.loc[j, 'e'] = e_j
        results.loc[j, 'v'] = v_j
        results.loc[j, 'o'] = o_1j

    z = (results['o'] - results['e']).sum() / (results['v'].sum()**.5) # Mantel-Haenszel
    p = 2 * (1 - chi2.cdf(abs(z), 1)) # two-sided
    # p = 2 * (1 - norm.cdf(abs(z), 1)) # two-sided


    return z, p

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
