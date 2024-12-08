import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans
import pickle
class IVF_Flat_Index:
    def __init__(self, n_clusters, file_name,random_state=4):
        self.n_clusters = n_clusters
        self.cluster_centers = None
        self.inverted_index = {}
        self.random_state=random_state
        self.file_name=file_name
    #! get the index clusters
    def fit(self, data):
        #! fit the data to the kmeans
        kmeans=MiniBatchKMeans(n_clusters=self.n_clusters, random_state=self.random_state, max_iter=10000, tol=1e-6, verbose=1, batch_size=10000)
        labels=kmeans.fit_predict(data)
        self.cluster_centers=kmeans.cluster_centers_
        #! create the inverted index
        for i,label in enumerate(labels):
            if label not in self.inverted_index:
                self.inverted_index[label]=[]
            self.inverted_index[label].append(i)
        #! convert the lists to numpy arrays
        for key in self.inverted_index:
            self.inverted_index[key]=np.array(self.inverted_index[key])
        
    #! save index to a file
    def save(self):
        with open(self.file_name, 'wb') as f:
            np.save(f, self.cluster_centers)
            pickle.dump(self.inverted_index, f)
    def load(self):
        with open(self.file_name, 'rb') as f:
            self.cluster_centers = np.load(f)
            self.inverted_index = pickle.load(f)
            self.n_clusters=self.cluster_centers.shape[0]
    #? To be edited instead of flat search
    def retrieve(self,query_vector,n_clusters,n_arrays,cosine_similarity,get_row):
        #! calculate the similarities between the query vector and the cluster centers
        similarities = np.array([cosine_similarity(query_vector, center) for center in self.cluster_centers]).squeeze()
        #! get the n nearest clusters
        nearest_clusters = np.argpartition(similarities, -n_clusters)[-n_clusters:]
        #! get nearest n arrays within nearest k clusters 
        vectors_indices = [self.inverted_index[cluster] for cluster in nearest_clusters]
        all_vectors_indices = np.concatenate(vectors_indices)
        similarities = np.array([cosine_similarity(query_vector, get_row(i)) for i in all_vectors_indices]).squeeze()
        # print(similarities.shape)
        #! get nearest n arrays overall
        nearest_arrays = np.argpartition(similarities, -n_arrays)[-n_arrays:]
        return all_vectors_indices[nearest_arrays]
