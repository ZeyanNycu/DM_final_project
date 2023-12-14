import fnmatch
import os
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import numpy as np
from sklearn.feature_selection import mutual_info_classif
import numpy as np
from sklearn.preprocessing import StandardScaler
from gensim.models import Word2Vec
temporal_embedding = [ [f'{y}/{m}'] for y in range(1980,2024) for m in range(1,13)]

def construct_dataset(path):
    temporal_model = Word2Vec(sentences=temporal_embedding, vector_size=17, window=3, min_count=1, workers=4)
    temporal_model.save("temporal_embedding_model")
    files = glob(os.path.join(path,'spotify_api_*.csv'))
    df = pd.concat((pd.read_csv(file) for file in files), ignore_index=True)
    df.drop(['track_name','track_id','artist(s)_name'], axis=1,inplace=True)
    df.fillna(df.mean(),inplace=True)
    z_score_scaler = StandardScaler()
    df[df.columns] = z_score_scaler.fit_transform(df[df.columns])
    return df

def construct_dataset_temporal(path):
    temporal_model = Word2Vec(sentences=temporal_embedding, vector_size=17, window=3, min_count=1, workers=4)
    files = glob(os.path.join(path,'spotify_api_*.csv'))
    df = pd.concat((pd.read_csv(file) for file in files), ignore_index=True)
    df.drop(['track_name','track_id','artist(s)_name'], axis=1,inplace=True)
    df.fillna(df.mean(),inplace=True)
    temporal_vector = []
    for _,row in df.iterrows():
        temporal_vector.append(temporal_model.wv[f"{int(row['released_year'])}/{int(row['released_month'])}"])
    z_score_scaler = StandardScaler()
    df[df.columns] = z_score_scaler.fit_transform(df[df.columns])
    temporal_vector = np.array(temporal_vector)
    return df,temporal_vector