import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.cluster import KMeans

#create 20 random x y coords between (0,0) and (100,100)
#separate so easier to add to dataframe
xcoords = []
ycoords = []

for x in range(20):
    xcoords.append((random.randint(0,99)))
    ycoords.append((random.randint(0,99)))


#convert to a pandasa dataframe
df = pd.DataFrame({'x': xcoords, 'y': ycoords})

#initialise a kmeans with 2 centroids
kmeans = KMeans(n_clusters = 3)
#fit the kmeans to the data
kmeans.fit(df)

y_predict = kmeans.predict(df)
centres = kmeans.cluster_centers_

figure = plt.figure()

plt.scatter(df['x'], df['y'], c = y_predict, s=50)

plt.scatter(centres[:,0], centres[:,1], c='black', s=50, alpha = 0.5)
