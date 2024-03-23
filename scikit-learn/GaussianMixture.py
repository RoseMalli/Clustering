import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

data = pd.read_csv('Country-data.csv')

income = data[['income']]

silhouette = []

for i in range(2, 11):
    gm = GaussianMixture(n_components = i, covariance_type = 'full', random_state = 42)
    silhouette.append(silhouette_score(income, gm.fit_predict(income)))

gm = GaussianMixture(n_components = 2, covariance_type = 'full', random_state = 42)
gm = gm.fit_predict(income )
data['cluster'] = gm

plt.figure(figsize = (20, 7))
plt.title('Gaussian Clustering', size = 18, fontweight = 'bold')
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