from src.data_dict import feature_config, clinical_feature_type
from src.directory import csv_paths
import pandas as pd
import numpy as np
from pycox.datasets import flchain, gbsg, metabric, nwtco
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr


def get_descriptive_stats(dataset_name=None, df=None):
    if df is None:
        df = pd.read_csv(csv_paths[dataset_name])
    clinical_features = [f for f in feature_config[dataset_name] if clinical_feature_type[dataset_name][f] == 'numerical']
    descriptive_stats = df[clinical_features].describe().loc[['mean', 'std']]
    return descriptive_stats.to_dict()


descriptive_stats = dict(
    unos=get_descriptive_stats('unos'),
    faisalabad=get_descriptive_stats('faisalabad'),
    flchain=get_descriptive_stats('flchain', df=flchain.read_df()),
    gbsg=get_descriptive_stats('gbsg', df=gbsg.read_df()),
    metabric=get_descriptive_stats('metabric', df=metabric.read_df()),
    nwtco=get_descriptive_stats('nwtco', df=nwtco.read_df())
)

def regress_out_confounds(data, confounds):
    """Takes xarray data array and confounds, returns deconfounded data array

    data: 2D array (samples x features)
    C: 2D array (samples x confounds), containing same number of samples as data

    return:
        X_dec: the deconfounded dataset

    Calculations based off equations (2) - (4) here:
    https://www.sciencedirect.com/science/article/pii/S1053811918319463?via%3Dihub#sec2
    """
    
    C_pi = np.linalg.pinv(confounds.T @ confounds)  # moore-penrose pseudoinverse of confounds
    b_hatX = C_pi @ confounds.T @ data  # confound parameter estimates
    X_dec = data - confounds @ b_hatX     # deconfounded X

    return X_dec

def detect_conditional_confounders(df, outcome_col, conditional_col, alpha=.05):
    X_dec = regress_out_confounds(data=df[[outcome_col]].to_numpy(),
                                  confounds=df[[conditional_col]].to_numpy())
    

    potential_confounders = list(set(df.columns) - set([outcome_col, conditional_col]))
    confounders = pd.DataFrame(index=potential_confounders, columns=['Confounder'])

    for col_name, col in confounders.iterrows():
        pvalue = spearmanr(X_dec, col).pvalue
        confounders.loc[col_name] = pvalue < alpha

    return confounders


def test_unconfoundedness_by_feature(df, outcome_col, conditional_col, feature_cols, alpha=0.05):
    confounders = pd.DataFrame(index=feature_cols, 
                               columns=['corr',
                                        'p_value'])
    confounders.index.name = 'feature'

    Y = df[outcome_col].to_numpy()
    T = df[conditional_col].to_numpy()
    
    for feature in feature_cols:
        # split data
        X = df[[feature]].to_numpy()
        X_train, X_test, T_train, T_test, Y_train, Y_test = train_test_split(X, T, Y, test_size=0.2, random_state=40)

        # standardize X
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # fit model 
        model = LogisticRegression()
        model.fit(X_train, T_train) # fit model
        
        # get propensity scores
        propensity_scores = model.predict_proba(X_test)[:, 1]

        # calculate correlation between outcome and propensity
        corr = spearmanr(Y_test, propensity_scores)

        # # get propensity scores
        # propensity_scores = get_propensity_scores(df, T_col=conditional_col, X_cols=[feature])

        # corr = spearmanr(Y, propensity_scores)

        confounders.loc[feature] = corr.statistic, corr.pvalue
    
    # determine confound by significance testing
    confounders.loc[:, 'confounder'] = confounders.loc[:, 'p_value'] < alpha

    return confounders


def get_propensity_scores(df, T_col, X_cols):
    T = df[T_col].to_numpy()
    
    # split data
    X = df[X_cols].to_numpy()

    # standardize X
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # fit model 
    model = LogisticRegression()
    model.fit(X, T) # fit model
    
    # get propensity score
    propensity_score = model.predict_proba(X)[:, 1]

    return propensity_score