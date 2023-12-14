import os
import pandas as pd
from sklearn.cluster import KMeans,DBSCAN, SpectralClustering
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import silhouette_score
from sklearn_extra.cluster import KMedoids
import numpy as np

def create_model(df,method,num_clusters,random_state=42):
    if(method == "kmeans"):
        model = KMeans(n_clusters=num_clusters, random_state=42)
        model.fit(df)
        df['Cluster'] = model.labels_
    elif(method == "dbscan"):
        dbscan = DBSCAN(eps=4, min_samples=2)
        df['Cluster'] = dbscan.fit_predict(df)
    elif(method == "kmedoids"):
        kmedoids = KMedoids(n_clusters=num_clusters, random_state=42)
        kmedoids.fit(df)
        df['Cluster'] = kmedoids.labels_
    elif(method == "spectralclustering"):
        spectral_clustering = SpectralClustering(n_clusters=num_clusters, random_state=42)
        spectral_clustering.fit(df)
        df['Cluster'] = spectral_clustering.labels_
    else:
        rank = []
        a = pd.cut(df[method], bins=5, labels=False)
        df['Cluster'] = a
    return df
def do_cluster(output_dir,df,num_clusters,method,action):
    print('The following are the setting of doing clustering')
    print(f"Output directory:{output_dir}")
    print(f"The number of cluster we want to cluster:{num_clusters}")
    # Initialize the KMeans model
    df = create_model(df,method,num_clusters, random_state=42)
    silhouette_avg = silhouette_score(df.drop('Cluster', axis=1), df['Cluster'])
    print(f"Silhouette Coefficient: {silhouette_avg}")
    if("visualize(3D)" in action):
        # Apply t-SNE for dimensionality reduction
        tsne = TSNE(n_components=3, random_state=42)
        X_tsne = tsne.fit_transform(df.drop('Cluster', axis=1))

        # Create a new DataFrame with the reduced dimensions
        df_tsne = pd.DataFrame(X_tsne, columns=['Dimension 1', 'Dimension 2', 'Dimension 3'])
        df_tsne['Cluster'] = df['Cluster']

        # Visualize in 3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot for each cluster
        for cluster_label in range(num_clusters):
            cluster_data = df_tsne[df_tsne['Cluster'] == cluster_label]
            ax.scatter(cluster_data['Dimension 1'], cluster_data['Dimension 2'], cluster_data['Dimension 3'], label=f'Cluster {cluster_label}')

        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_zlabel('Dimension 3')
        ax.set_title('Clustering with 3D t-SNE Visualization')
        ax.legend()
        plt.show()
    if("visualize(2D)" in action):
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(df.drop('Cluster', axis=1))

        # Create a new DataFrame with the reduced dimensions
        df_tsne = pd.DataFrame(X_tsne, columns=['Dimension 1', 'Dimension 2'])
        df_tsne['Cluster'] = df['Cluster']
        plt.scatter(df_tsne['Dimension 1'],df_tsne['Dimension 2'], c=df_tsne['Cluster'], cmap='viridis')
        plt.title('DBSCAN 聚类结果')
        plt.xlabel('Feature1')
        plt.ylabel('Feature2')
        plt.show()
    
