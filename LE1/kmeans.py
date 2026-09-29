import numpy as np

def assign_clusters(data,centroids):
    clusters = []
    for point in data:
        distances=[]
        for c in centroids:
            dist = minkowski_dist(point,c,2)
            distances.append(dist)
        nc = distances.index(min(distances))
        clusters.append(nc)
    return np.array(clusters)

def calc_cen(data,clusters,k):
    centroids=[]
    for cluster_number in range(k):
        cluster_points = data[clusters==cluster_number]
        if len(cluster_points)>0:
            centroid = np.mean(cluster_points,axis=0)
        else:
            centroid = data[0]
        centroids.append(centroid)
    return np.array(centroids)



def kmeans(data,k,max_iter=100):
    rand_idx = np.random.choice(len(data),k,replace=False)
    centroids = data[rand_idx].copy()

    for _ in range(max_iter):
        c = assign_clusters(data,centroids)
        new_cen = calc_cen(data,c,k)
        if np.array_equal(new_cen,c):
            break
        centroids=new_cen
    return centroids,c
