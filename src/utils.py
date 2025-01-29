from src.data_dict import feature_config, clinical_feature_type
from src.directory import csv_paths
import pandas as pd
from pycox.datasets import flchain, gbsg, metabric, nwtco


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