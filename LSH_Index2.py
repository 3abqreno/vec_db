import numpy as np
from binary_file import BinaryFile 
class LSHForest:
    def __init__(self):
        self.num_tables = 8
        self.dim = 70
        self.num_hashes = 15
        self.random_vectors = [np.random.randn(self.dim, self.num_hashes) for _ in range(self.num_tables)]

    def lsh_training(self, rows):
        hash_tables = []
        for i in range(self.num_tables):
            hash_table = {}
            for row in rows:
                hash_value = self.hash(row[1:], i)
                if hash_value in hash_table:
                    hash_table[hash_value].append(row)
                else:
                    hash_table[hash_value] = [row]
            hash_tables.append(hash_table)
        return hash_tables

    def hash(self, input_vector, table_index):
        norm_input_vector = input_vector / np.linalg.norm(input_vector)
        norm_random_vectors = self.random_vectors[table_index] / np.linalg.norm(self.random_vectors[table_index], axis=0)
        cos_sim = np.dot(norm_input_vector, norm_random_vectors)
        bools = (cos_sim > 0).astype('int')
        return int(''.join(bools.astype('str')), 2)

    def write_lsh_hash_tables(self, hash_tables, file_name, position_file_name):
        for i, hash_table in enumerate(hash_tables):
            open(f"{file_name}_{i}.bin", 'w').close()
            open(f"{position_file_name}_{i}.bin", 'w').close()

            bfh = BinaryFile(f"{file_name}_{i}.bin")
            bfh_pos = BinaryFile(f"{position_file_name}_{i}.bin")

            for hash_value, rows in hash_table.items():
                hash_dict = [{"id": int(row[0]), "embed": row[1:]} for row in rows]
                first_position, last_position = bfh.insert_records(hash_dict)
                bfh_pos.insert_position(hash_value, [first_position, last_position])