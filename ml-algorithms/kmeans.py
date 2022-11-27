
import numpy as np

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum(x1-x2)**2)

class KMeans(object):
    def __init__(self, K=5, max_iters: int = 100, plot_steps: bool = False):
        self.K: int = K
        self.max_iters: int = max_iters
        self.plot_steps: bool = plot_steps
        
        # list of sample indices for each cluster
        self.clusters = [[] for _ in range(self.K)]
        # the centers of (mean vector) for eacxh cluster
        self.centroids = []
        
    def fit(self, X: np.ndarray):
        self.X = X
        self.n_samples, self.n_features = X.shape
        
        # initialize centroids
        random_samples_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_samples_idxs]
        
        # optimize clusters
        for _ in range(self.max_iters):
            # assign samples to closest centroids (create clusters)
            self.clusters = self._create_clusters(self.centroids)
            
            # calculate new centroids from the clusters
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)
            
            if self._is_converged(centroids_old, self.centroids):
                break
            
        # classify samples as the index of their clusters
        return self._get_cluster_labels(self.clusters)
            
    def _create_clusters(self, centroids):
       # assign the smaples to the closest centroids
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
           centroid_idx = self._closest_centroid(sample, centroids)
           clusters[centroid_idx].append(idx)
           
        return clusters
    
    def _closest_centroid(self, sample, centroids):
        # distance of the current smaple to each centroid
        distances = [euclidean_distance(sample, point) for point in centroids]
        closest_idx = np.argmin(distances)
        return closest_idx
    
    def _get_centroids(self, clusters):
        # assign mean value of clusters to centroids
        centroids = np.zeros(self.K, self.n_features)
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster_idx], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids
    
    def _is_converged(self, centroid_old, centroids):
        # distances between old and new centroids, for all centroids
        distacnes = [euclidean_distance(centroid_old[i], centroids[i]) for i in range(self.K)]
        return sum(distacnes) == 0
    
    def _get_cluster_labels(self, clusters):
        # each sample will get the label of the cluster it was assigned to
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
                
        return labels