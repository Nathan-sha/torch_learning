import numpy as np

class Kmeans:
    def __init__(self, n_clusters, max_iters=300, tol=1e-4, seed=42):
        self.n_clusters = n_clusters
        self.max_iters, self.tol, self.seed = max_iters, tol, seed
        self.center = None
    
    def fit(self, x):
        N, dim = x.shape

        if N < self.n_clusters:
            return "error N < n_clusters"
        
        np.random.seed(self.seed)
        idx = np.random.randint(0, N, self.n_clusters)
        self.center = x[idx].copy()

        iter_num = 0
        error_sum = 0
        
        
        iter_num += 1
        if error_sum <= self.tol or iter_num >= self.max_iters:
            return self.center
        else:

    # 广播写法
    def L1_error(self, centers, x):  
        n_clusters = centers.shape[0]
        dim = x.shape[1]
        error_sum = 0
        for n in range(n_clusters):
            for num in x:
                L1_error = num - centers[n]
                abs_error = 0
                for i in range(dim):
                    abs_error += np.abs(L1_error[i])

                error_sum += abs_error
        return error_sum
    




            