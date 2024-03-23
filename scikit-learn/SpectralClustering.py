import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score

data = pd.read_csv('Country-data.csv')

income = data[['income']]

silhouette = []

for i in range(2, 11):
    sc = SpectralClustering(n_clusters = i, affinity = 'nearest_neighbors', random_state = 42)
    silhouette.append(silhouette_score(income, sc.fit_predict(income)))

sc = SpectralClustering(n_clusters = 2, affinity = 'nearest_neighbors', random_state = 42)
sc_ = sc.fit_predict(income)
data['cluster'] = sc_

plt.figure(figsize = (20, 7))
plt.title('Spectral Clustering', size = 18, fontweight = 'bold')
plt.scatter(data['country'], income, c = data['cluster'], cmap = 'plasma')
plt.xticks(rotation = 90)
plt.xlabel('Country', fontweight = 'bold')
plt.ylabel('Income', fontweight = 'bold')
plt.tight_layout()
plt.show()

plt.title('Silhouette', size = 18, fontweight = 'bold')
plt.scatter(range(2, 11), silhouette)
plt.plot(range(2, 11), silhouette)
plt.xlabel('k', fontweight = 'bold')
plt.ylabel('Quality of clusters', fontweight = 'bold')
plt.show()