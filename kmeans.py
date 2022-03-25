"""
K-means clustering from scratch.
"""

import numpy as np

class K_Means:
    def __init__(self, c=3, n_iter=100, threshold=0.01, random_state=127):
        self.c = c
        self.n_iter = n_iter
        self.threshold = threshold
        # ensure reproducible results
        np.random.seed(random_state)
        
    def fit_predict(self, X: np.array):
        self.X = X
        self.m = X.shape[0]
  
        # randomly initialize examples from the input data as centroids
        rand_sample_indices = np.random.choice(a=self.m, size=self.c, replace=False)
        self.centroids = [self.X[i] for i in rand_sample_indices]

        # optimization phase
        for _ in range(self.n_iter):
            # update clusters
            self.clusters = self.assign_clusters(self.centroids)
            # keep track of previous centroids and then update centroids
            self.previous_centroids = self.centroids
            self.centroids = self.update_centroids(self.clusters)
            # check convergence
            if self.converged(self.previous_centroids, self.centroids, self.threshold):
                print("Early stopping because threshold was reached.")
                break
        # return the cluster labels either after we converge or reach the max interations
        return self.cluster_labels(self.clusters)

    def cluster_labels(self, clusters):
        """
        This method takes a list of lists with example indices per cluster as input and assigns 
        the corresponding cluster ID to each instance.
        """
        labels = self.m * [0]
        for cluster_id,cluster in enumerate(clusters):
            for example_i in cluster:
                labels[example_i] = cluster_id
        return labels

    def assign_clusters(self, centroids):
        """
        Returns a list of lists with cluster memberships based on closest distance. 
        """
        clusters = [[] for _ in range(self.c)]
        for example_idx,row in enumerate(self.X):
            cluster_id = self.closest_centroid(row, centroids)
            clusters[cluster_id] += [example_idx]
        return clusters

    def closest_centroid(self, row, centroids):
        """
        Return the cluster ID (index) of the closest centroid based on manhattan distance.
        """
        return np.argmin([self.manhattan_distance(row, centroid) for centroid in centroids])
        
    def manhattan_distance(self, x, y):
        """
        Manhattan distance for two vectors. Manhattan is preferred over Euclidean distance when
        dealing with large dimensional data (e.g. text).
        """
        return sum(abs(x - y))

    def update_centroids(self, clusters):
        """
        This method computes the mean point of all examples in each
        cluster. The resulting mean will be an n-dimensional point and form 
        the updated cluster centroid.

        clusters: A list of lists of cluster indices 
        """
        centroids = []
        for example_indices in clusters:
            # cluster mean is n dimensional instance
            centroids += [np.mean(self.X[example_indices], axis=0)] 
        return centroids

    def converged(self, previous_centroids, centroids, threshold):
        """
        This method computes the distances between the previous and new centroids and checks
        whether the distance is 0 or below some predefined threshold.
        
        returns: bool
        """
        distances = [self.manhattan_distance(previous_centroids[i], centroids[i]) for i in range(self.c)]
        return sum(distances) <= threshold
