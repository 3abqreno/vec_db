# Vector Database Indexing Project
# Table of Contents

- [Overview](#overview)
- [Project Structure](#structure)
- [System Constraints](#constraints)
- [Usage](#usage)
- [Additional Information](#info)
- [Final Results](#results)
- [Contributors](#contributors)

---
## 🌐 Overview <a name="overview"></a>
This project focuses on building a semantic search engine using vectorized databases. The goal is to create efficient indexing mechanisms for large-scale vector databases, enabling fast and accurate retrieval of similar vectors. The project involves implementing and evaluating different indexing strategies, such as IVF (Inverted File Index) and PQ (Product Quantization), to handle databases of varying sizes (1M, 10M, 15M, and 20M rows).

## 🏗️ Project Structure <a name="structure"></a>
The project is organized into several key files and components:

### 1. **`vec_db.py`**
   - **Purpose**: This file contains the core implementation of the vector database and indexing logic.
   - **Key Functions**:
     - **`generate_database(size: int)`**: Generates a random database of a given size.
     - **`_build_index()`**: Builds the index on the generated data.
     - **`_write_vectors_to_file(vectors: np.ndarray)`**: Writes the randomly generated data to disk using memory-mapped files.
     - **`_get_num_records()`**: Calculates the number of records in the database.
     - **`insert_records(rows: np.ndarray)`**: Inserts new records into the database and rebuilds the index.
     - **`get_one_row(row_num: int)`**: Retrieves a single row from the database without loading the entire file into memory.
     - **`get_all_rows()`**: Retrieves all rows from the database (loads the entire file into memory).
     - **`retrieve(query: np.ndarray, top_k: int)`**: Retrieves the top K nearest vectors given a query vector.
     - **`_cal_score(vec1, vec2)`**: Calculates the cosine similarity between two vectors.

### 2. **`IVF_Flat.py`**
   - **Purpose**: Implements the IVF (Inverted File Index) with flat indexing.
   - **Key Features**:
     - **`IVF_Flat_Index`**: A class that handles the creation and management of the IVF index.
     - **`fit(data)`**: Fits the index to the provided data using KMeans clustering.
     - **`retrieve(query_vector, n_clusters, n_arrays, cosine_similarity, get_row)`**: Retrieves the nearest vectors using the IVF index.

### 3. **`IVF_PQ.py`**
   - **Purpose**: Implements the IVF (Inverted File Index) with Product Quantization (PQ) for more efficient indexing.
   - **Key Features**:
     - **`IVF_PQ_Index`**: A class that handles the creation and management of the IVF-PQ index.
     - **`fit(vectors)`**: Fits the index to the provided data using KMeans clustering and subvector quantization.
     - **`retreive(query_vector, cosine_similarity, get_row, n_clusters, n_neighbors)`**: Retrieves the nearest vectors using the IVF-PQ index.

### 4. **`evaluation.py`**
   - **Purpose**: Provides functions to evaluate the performance of the indexing mechanisms.
   - **Key Functions**:
     - **`run_queries(db, np_rows, top_k, num_runs)`**: Runs a given number of queries against the index and returns the indices of the nearest neighbors.
     - **`eval(results: List[Result])`**: Evaluates the results against the ground-truth and calculates the score.

### 5. **`requirements.txt`**
   - **Purpose**: Lists the required Python packages for the project.
   - **Dependencies**:
     - `numpy`
     - `scikit-learn`

## 🔒 System Constraints <a name="constraints"></a>
The project has specific constraints for different database sizes:

| DB Size | Peak RAM Usage (Retrieval) | Time Limit (Retrieval) | Min Accepted Score | Max Index Size |
|---------|----------------------------|------------------------|--------------------|----------------|
| 1M      | 20 MB                      | 3 sec                  | -5000              | 50 MB          |
| 10M     | 50 MB                      | 6 sec                  | -5000              | 100 MB         |
| 15M     | 50 MB                      | 8 sec                  | -5000              | 150 MB         |
| 20M     | 50 MB                      | 10 sec                 | -5000              | 200 MB         |

## 🚀 Usage <a name="usage"></a>
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/3abqreno/vec_db.git
   ```
2. **Install Required Dependencies**:
    ```bash 
    pip install -r requirements.txt
    ```
3.  **Choose desired database size in** [evaluation.py](evaluation.py):
    ```python 
    db = VecDB(db_size=1000000)  # For 1M rows
    ```
4. **Build a new database or use an existing one from the new_db flag in** [vec_db.py](vec_db.py):
    ```python
    def __init__(self, database_file_path = "saved_db.dat", index_file_path = "index", new_db = True, db_size = None) -> None:
    ```
## ℹ️ Additional Information <a name="info"></a>
1. The generated database is saved as a file named ```save_db.dat```
2. The built index is composed of multiple files saved within the folder ```index```
3. For implementation details you can check [ADB_Project__Final_Submission_Document.pdf](ADB_Project__Final_Submission_Document):

## 📊 Final Results <a name="results"></a>
| DB Size | Accuracy | Time  |  Ram |
|---------|----------|-------|------|
| 1M      |  0.0    | 0.20   | 0.0  |
| 10M     |  0.0    | 2.11   | 0.25 |
| 15M     |  0.0    | 2.64   | 0.23 |
| 20M     |  0.0B   | 6.21   | 0.25 |

## ✍️ Contributors <a name = "contributors"></a>

<table>
  <tr>
   <td align="center">
    <a href="https://github.com/Omar-Said-4" target="_black">
    <img src="https://avatars.githubusercontent.com/u/87082462?v=4"  alt="Omar Said"/>
    <br />
    <sub><b>Omar Said</b></sub></a>
    </td>
   <td align="center">
    <a href="https://github.com/MostafaMagdyy" target="_black">
    <img src="https://avatars.githubusercontent.com/u/97239596?v=4" alt="Mostafa Magdy"/>
    <br />
    <sub><b>Mostafa Magdy</b></sub></a>
    </td>
<td align="center">
    <a href="https://github.com/nouraymanh" target="_black">
    <img src="https://avatars.githubusercontent.com/u/102790603?v=4" alt="Nour Ayman"/>
    <br />
    <sub><b>Nour Ayman</b></sub></a>
    </td>
<td align="center">
    <a href="https://github.com/3abqreno" target="_black">
    <img src="https://avatars.githubusercontent.com/u/102177769?v=4" alt="Abdelrahman Mohamed"/>
    <br />
    <sub><b>Abdelrahman Mohamed</b></sub></a>
    </td>
  </tr>
</table>
