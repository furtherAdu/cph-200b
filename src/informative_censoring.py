import pandas as pd
import numpy as np
from scipy.stats import spearmanr, percentileofscore
from copy import deepcopy

# method to check that the censoring event C = 1 is random and does not depend on the survival time or patient features
def check_informative_censoring(df, time_col='duration', event_col='event', alpha=0.05):
    """
    Checks if censoring is independent of survival time and patient features.

    Args:
        df (pd.DataFrame): DataFrame with survival data.
        event_col (str): Name of the event indicator column (1 for event, 0 for censored).
        alpha (float, optional): Significance level for tests. Defaults to 0.05.

    Returns:
        dict: A dictionary containing test results
    """

    results = {}
    df_name = df.name
    feature_cols = list(set(df.columns) - set([event_col]))
    test = spearmanr

    for feature in feature_cols:
        # get spearman correlations
        test_results = test(df[event_col], df[feature])

        # update results
        results[feature] = {
            "statistic": test_results.statistic,
            "p_value": test_results.pvalue,
            "informative_censoring": test_results.pvalue < alpha,
        }

    # to dataframe
    results = pd.DataFrame.from_dict(results)
    results.name = df_name

    # rename index
    results = results.reset_index()
    index_cols = ['feature', f'{test.__name__}_results']
    results.columns.values[[0, 1]] = index_cols
    results = results.set_index(index_cols)

    return results

def generate_semi_synthetic_dataset(df, time_col='duration', event_col='time', alpha=0.05, max_loops=3):
    """ Procedure to generate semi-synthetic versions of datasets that introduces a controllable level of bias in the
     censoring events via time-dependent censoring.

     Larger values for p_late assume late censoring bias, larger values for p_early assume early censoring bias.

    threshold_quantile (float): threshold defining early or late events, based on time from censored data
    p_early (float): probability of censoring early events
    p_late (float): probability of censoring late events
    alpha (float): significance threshold for spearman r correlation between time_col and event_col

    """
    df_name = df.name
    df_synthetic = deepcopy(df)

    # # shuffle dataset
    # df_synthetic = df_synthetic.sample(frac=1)

    # init censoring probability utils
    original_censored_times = df.query(f'{event_col} == 0')[time_col]
    rng = np.random.default_rng(seed=40)

    # init metrics
    pvalue = spearmanr(df_synthetic[time_col], df_synthetic[event_col]).pvalue
    informative_censoring = pvalue < alpha
    any_uncensored_rows = df_synthetic[event_col].sum() > 0
    
    # init loop count
    loop = 0

    print(f'Creating semi-synthetic dataset for {df_name}')
    while (informative_censoring) and (any_uncensored_rows) and (loop < max_loops):

        # loop through dataset
        for index, row in df_synthetic.iterrows():

            if not informative_censoring:
                break

            uncensored_event = row[event_col] == 1

            if uncensored_event:
                
                # censor uncensored event with probability p_censoring
                p_censoring = abs(percentileofscore(original_censored_times, row[time_col])/100 - .5) * 2
                if rng.random() < p_censoring:
                    df_synthetic.loc[index, event_col] = 0  # censor row
                
                # check dependence
                pvalue = spearmanr(df_synthetic[time_col], df_synthetic[event_col]).pvalue
                informative_censoring = pvalue < alpha

                # check if there are rows left to censor
                any_uncensored_rows = df_synthetic[event_col].sum() > 0
        
        loop += 1
    
    if not any_uncensored_rows:
        print('No more uncensored samples; failed to remove time-dependent bias.')

    else:
        synthetic_censoring_fraction = 1 - (df_synthetic[event_col].sum() / df[event_col].sum())
        print(f'Fraction of original uncensored events censored: {synthetic_censoring_fraction:.03f}')
        print(f'Spearman r correlation p-value between time_col and event_col: {pvalue:.03f}\n')

    return df_synthetic