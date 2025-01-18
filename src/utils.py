from src.data_dict import feature_config, clinical_feature_type
from src.directory import csv_paths
import pandas as pd


def get_descriptive_stats(dataset_name):
    df = pd.read_csv(csv_paths[dataset_name])
    clinical_features = [f for f in feature_config[dataset_name] if clinical_feature_type[dataset_name][f] == 'numerical']
    descriptive_stats = df[clinical_features].describe().loc[['mean', 'std']]
    return descriptive_stats.to_dict()


descriptive_stats = dict(
    unos=get_descriptive_stats('unos'),
    faisalabad=get_descriptive_stats('faisalabad')
)
