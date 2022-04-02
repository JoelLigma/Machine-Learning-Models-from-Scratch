"""
Fuzzy-C-Means clustering from scratch.

Based on the paper:
'FCM: THE FUZZY c-MEANS CLUSTERING ALGORITHM' by Bezdeck, Ehrlich & Full (1983)

Please note:
-----------
- In the case that the ground truth is known. cluster IDs may vary due to random 
  initialization of the partition matrix. 
  However, this does not affect the actual cluster groupings. The instances will 
  still be in the same cluster, just under a different ID. Simply match the cluster IDs
  in the correct order by e.g. using pd.Series.map({0:2, 1:1, 2:0}) to match the cluster 
  labels to the ground truth before checking the accuracy score. 
"""

import numpy as np

class Fuzzy_C_Means:
    def __init__(self, c=3, n_iter=100, threshold=0.001, m=2, random_state=127, dist_metric="Euclidean", mem_matrix=None):
        """
        Constraints
        ----------
        c: 2 <= c <= no. of rows
        m: fuzzy parameter >= 1
        """
        self.c = c
        self.n_iter = n_iter
        self.threshold = threshold
        self.dist_metric = dist_metric
        self.m = m 
        self.mem_matrix = mem_matrix
        # ensure reproducible results
        np.random.seed(random_state)
        
    def fit_predict(self, X: np.array):
        self.X = X
        self.n_rows = X.shape[0]
        # check user inputs
        self._check_inputs(self.m, self.c, self.n_rows)

        if isinstance(self.mem_matrix,(np.ndarray, np.generic)):
            print("Partition matrix provided.")
            self.partition_matrix = self.mem_matrix
        else:
            self.partition_matrix = np.random.dirichlet(alpha=np.ones(self.c),size=self.n_rows).T
        
        # initialization of distance_sum variable
        distance_sum = 0
        
        # optimization phase
        for i in range(self.n_iter):
            # Step 1: (re)compute the centroids
            self.centroids = self._compute_cluster_centers(self.X, self.partition_matrix, self.c, self.m)
            # Step 2: calculate the distances between all instances and the centroids
            self.distances = self._compute_similarity_measures(self.X, self.centroids, self.dist_metric)
            # Step 3: update the partition matrix based on the distances
            old_partition_matrix = self.partition_matrix
            self.partition_matrix = self._update_partition_matrix(self.distances, self.m)
            # Step 4: check convergence through objective function
            is_convereged, distance_sum = self._converged(self.X, self.centroids, old_partition_matrix, distance_sum, self.m, self.threshold)
            print("Distance sum:", distance_sum)
            if is_convereged:
                print(f"Early stopping after {i+1} iterations because threshold was reached.")
                break

        # return the cluster labels either after we converge or reach the max interations
        self.labels = self._cluster_labels(self.partition_matrix)
        return self.partition_matrix.T

    def _compute_similarity_measures(self, X, centroids, measure) -> np.array:
        """
        Supports Euclidean and Manhattan distance. Distances will be calculated between 
        each instance and centroid.
        """
        
        row_wise_distances = np.empty((self.c, X.shape[0]))
        if measure == "Euclidean":
            for i,centroid in enumerate(centroids):
                 row_wise_distances[i] = np.sqrt(np.sum(np.power(np.subtract(X, centroid), 2), axis=1))
        elif measure == "Manhattan":
             for i,centroid in enumerate(centroids):
                 row_wise_distances[i] = np.sum(np.abs(np.subtract(X, centroid)))
        else:
            raise ValueError(f"The following similarity measure is not supported: {measure}")
        return row_wise_distances

    def _update_partition_matrix(self, dists, m):
        """
        This method updates the partition matrix using the result of the distances computed in
        the _compute_similarity_measures() method.

        The weightings are computed as follows w_ij = 1 / (dii/dii)**p + c_sigma_{i=1}(dii/dic)**p
        """
        p = (2/(m-1))
        partition_matrix = []
        
        for row in dists.T:
            row_result = []
            for i in row:
                # compute membership weightings
                row_result += [1/sum([(i/j)**p for j in row])]
            partition_matrix += [row_result]

        # handle ZeroDivisionError by converting nan to 1 (if applicable)
        partition_matrix = np.nan_to_num(partition_matrix, nan=1)
        return np.array(partition_matrix).T

    def _converged(self, X, centroids, old_partition_matrix, distance_sum, m, threshold) -> bool:   
        """
        Objective function.
        """
        raised_partition_matrix = np.power(old_partition_matrix, m)
        distances = []
        for i, centroid in enumerate(centroids): 
            distances += [np.multiply(np.sum(np.power(np.subtract(X, centroid), 2), axis=1), raised_partition_matrix[i])]
        # aggregate the result to compute overall error to check convergence threshold
        new_distance_sum = sum(distances).sum()
        # return whether or not the stopping critera has been reached
        return abs(new_distance_sum - distance_sum) <= threshold, new_distance_sum

    def _cluster_labels(self, partition_matrix):
        return np.argmax(partition_matrix.T, axis=1)

    def _compute_cluster_centers(self, X, partition_matrix, c, m):
        """
        Calculate cluster center vectors V_ij.

        V_ij = sigma_{from k=1 to n} (partition_matrix{ik})**m * X_{kj} / 
               sigma_{from k=1 to n} (partition_matrix{ij})**m  
        """
        # raise each element in the weight matrix to the power of m
        raised_partition_matrix = np.power(partition_matrix, m)
        # compute the sum of each raised weight matrix column to recieve the denominators
        # the denominator will be of length == no. of clusters
        denominator = np.sum(raised_partition_matrix, axis=1)
        
        numerator = []
        for i in range(c):
            temp = []
            for j in range(X.shape[1]):
                temp += [np.multiply(X.T[j], raised_partition_matrix[i]).sum()]
            numerator += [temp]

        #print("Numerator:\n", numerator)
        #print("Denominator:\n", denominator)
        # finally, compute the centroids: divide numerator arrays by denominator element
        centroids = np.array([np.divide(numerator[i], denominator[i]) for i in range(c)])
        return centroids

    def _check_inputs(self, m, c, n_rows):
        if m < 1:
            raise ValueError(f"The following value for m is not accepted: {m}. Please select a value >= 1.")
        elif c < 2:
            raise ValueError(f"The following value for c is not accepted: {c}. Please select a value >= 2 and >= n_rows.")
        elif n_rows <= c:
            raise ValueError(f"The following value for c is not accepted: {c}. Please select a value >= 2 and >= n_rows.")
