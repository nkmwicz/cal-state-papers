import numpy as np
from scripts.cluster import create_cluster

array = np.load("embeds.npy", allow_pickle=True)
clusters = create_cluster(array, num_clusters=3)
print(clusters)
