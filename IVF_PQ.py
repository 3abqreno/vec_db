import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans
import pickle
import logging
from sklearn.metrics.pairwise import euclidean_distances
logging.basicConfig(level=logging.INFO)
class IVF_PQ_Index:
    def __init__(self, n_subvectors,n_bits, n_clusters,random_state=42,dimension=70):
        #! number of bits per subvector codeword
        self.n_bits = n_bits
        #! number of subvectors 
        self.n_subvectors=n_subvectors
        #! number of clusters per subvector
        self.n_clusters_per_subvector = 2**n_bits
        #! dimension of the vectors
        self.dimension = dimension
        #! number of IVF clusters
        self.n_clusters=n_clusters
        #! IVF cluster centres
        self.cluster_centers = None
        self.random_state=42
        self.inverted_index = {}

        if dimension % n_subvectors != 0:
            raise ValueError("dimension needs to be a multiple of n_subvectors")

        self.subvector_estimators = [
            [
                MiniBatchKMeans(
                    n_clusters=self.n_clusters_per_subvector,
                    init='random',
                    max_iter=50,
                    random_state=42,
                    batch_size=1000
                ) 
                for _ in range(self.n_subvectors)
            ]
            for _ in range(self.n_clusters)
        ]
        logging.info(f"Created subvector estimators {self.subvector_estimators[0]!r}")

        self.is_trained = False

    def fit(self, vectors):
        if self.is_trained:
            raise ValueError("IVF is already trained.")
        #! fit the data to the kmeans
        kmeans=MiniBatchKMeans(n_clusters=self.n_clusters,random_state=self.random_state,verbose=1,batch_size=10000)
        labels=kmeans.fit_predict(vectors)
        self.cluster_centers=kmeans.cluster_centers_
        #! separate labels for subclusters training
        label_indices = {label: np.where(labels == label)[0] for label in np.unique(labels)}
        #! train the subclusters and assign the codewords and create the inverted index
        for i in range(self.n_clusters):
            indices = label_indices[i]
            self._train_subclusters(vectors[indices],self.subvector_estimators[i])
            codewords = self._add(vectors[indices],self.subvector_estimators[i])
            self.inverted_index[i] = (codewords, indices)
        self.is_trained = True
        

    def _train_subclusters(self, vectors,estimators):
        for i in range(self.n_subvectors):
            #! to slice arrays
            data_slicer=self.dimension//self.n_subvectors 
            estimator = estimators[i]
            subvectors = vectors[:, i * data_slicer : (i + 1) * data_slicer]
            logging.info(f"Fitting KMeans for the {i+1}-th subvectors")
            estimator.fit(subvectors)

    def _encode(self, vectors,estimators):
        #! result array to store the codewords
        result = np.empty((vectors.shape[0], self.n_subvectors), dtype=np.uint32)
        for i in range(self.n_subvectors):
            #! predict the assigned cluster for each group of subvectors
            estimator =estimators[i]
            #! to slice arrays
            data_slicer=self.dimension//self.n_subvectors
            query = vectors[:, i * data_slicer : (i + 1) * data_slicer]
            result[:, i] = estimator.predict(query)

        return result
    def _add(self, vectors,estimators):
        codewords = self._encode(vectors,estimators)
        codewords = codewords.astype(np.uint16)
        return codewords

    def _subvector_distance(self, queries,codewords,cluster_index):
        if not self.is_trained:
            raise ValueError("Index is not trained yet")
        #! table to store the distances between the query subvectors and the codewords subvector
        distances_table = np.zeros(( queries.shape[0],self.n_subvectors,self.n_clusters_per_subvector), dtype=np.float32)
        #! calculate the distance between the queries sub vectors and the clusters centers for each subvector
        for i in range(self.n_subvectors):
            #! to slice arrays
            data_slicer=self.dimension//self.n_subvectors
            query = queries[:, i * data_slicer : (i + 1) * data_slicer]
            centers = self.subvector_estimators[cluster_index][i].cluster_centers_  
            distances_table[:, i, :] = euclidean_distances(query, centers, squared=True)
        #! calculate the distance between the query vectors and the codewords
        distances = np.zeros((queries.shape[0], len(codewords)), dtype=np.float32)
        for i in range(self.n_subvectors):
            distances += distances_table[:, i, codewords[:, i]]
        return distances

    def _searchPQ(self, query_vectors, codewords,cluster_index,n_neighbors=1):
        if not self.is_trained:
            raise ValueError("Index is not trained yet")
        #! calculate the distance between the query vector and the codewords
        distances = self._subvector_distance(query_vectors, codewords,cluster_index)
        #! get the nearest vectors indices
        nearest_vector_indices = np.argsort(distances, axis=1)[:, :n_neighbors]
        return nearest_vector_indices
    def retreive(self,query_vector,cosine_similarity,get_row,n_clusters=3,n_neighbors=10):
        #! calculate the similarities between the query vector and the cluster centers
        similarities = np.array([cosine_similarity(query_vector, center) for center in self.cluster_centers]).squeeze()
        #! get the n nearest clusters
        nearest_clusters = np.argpartition(similarities, -n_clusters)[-n_clusters:]
        #! get nearest n vectors
        similarities = np.empty((0,))
        vectors = np.empty((0, ))
        for cluster in nearest_clusters:
            #! get the codewords and indices of the vectors in the cluster
            codewords, indices = self.inverted_index[cluster]
            nearest_vector_indices=self._searchPQ(query_vectors=query_vector.reshape(1,-1),codewords=codewords,n_neighbors=n_neighbors*2,cluster_index=cluster).flatten()
            new_similarities = np.array([cosine_similarity(query_vector, get_row(i)) for i in indices[nearest_vector_indices]]).squeeze()
            vectors = np.append(vectors,indices[nearest_vector_indices])
            similarities = np.append(similarities, new_similarities) 
        nearest_arrays = np.argpartition(similarities, -n_neighbors)[-n_neighbors:]
        return vectors[nearest_arrays]
    def save_index(self,file_name):
        with open(file_name, 'wb') as f:
            np.save(f, self.cluster_centers)
            pickle.dump(self.inverted_index, f)
            subvector_centers = [
            [estimator.cluster_centers_ for estimator in estimators]
            for estimators in self.subvector_estimators
            ]
            pickle.dump(subvector_centers, f)    
    def load_index(self, file_name):
        with open(file_name, 'rb') as f:
            self.cluster_centers = np.load(f)
            self.inverted_index = pickle.load(f)
            self.subvector_estimators = pickle.load(f)
        self.n_clusters=self.cluster_centers.shape[0]
        self.n_subvectors=len(self.subvector_estimators[0])
        self.n_clusters_per_subvector=len(self.subvector_estimators[0][0].cluster_centers_)
        self.dimension=self.cluster_centers.shape[1]
        self.n_bits=int(np.log2(self.n_clusters_per_subvector))
        self.is_trained = True
