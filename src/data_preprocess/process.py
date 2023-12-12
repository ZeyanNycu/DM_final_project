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

def construct_dataset(path):
    files = glob(os.path.join(path,'spotify_api_*.csv'))
    df = pd.concat((pd.read_csv(file) for file in files), ignore_index=True)
    df.drop(['track_name','track_id','artist(s)_name'], axis=1,inplace=True)
    df.fillna(df.mean(),inplace=True)
    z_score_scaler = StandardScaler()
    df[df.columns] = z_score_scaler.fit_transform(df[df.columns])
    return df