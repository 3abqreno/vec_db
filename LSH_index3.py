import numpy as np

class RandomHash:
    def __init__(self, dim, num_hashes):
        self.hash_functions = [np.random.randn(dim) for _ in range(num_hashes)]
    
    def hash(self, vector, length):
        """
        Generate a hash for a vector.
        Args:
            vector (array): Input vector.
            length (int): Number of hash functions to use.
        Returns:
            tuple: Hash value (variable-length).
        """
        return tuple((np.dot(f, vector) > 0).astype(int) for f in self.hash_functions[:length])

def synchascend(x, s, lsh_trees, c, m):
    """
    Bottom-up traversal in LSH Forest.
    Args:
        x (list): Depths of leaf nodes for each tree.
        s (list): Leaf nodes for each tree.
        lsh_trees (list): List of LSH trees.
        c (float): Constant multiplier for total candidates.
        m (int): Minimum number of distinct points.
    Returns:
        set: Candidate points.
    """
    max_x = max(x)
    P = set()  # Initialize the set of candidate points

    while max_x > 0 and (len(P) < c * len(lsh_trees) or len(set(P)) < m):
        for i in range(len(lsh_trees)):
            if x[i] == max_x:
                # Add descendants of the current node to P
                P.update(lsh_trees[i].get_descendants(s[i]))
                # Move to the parent node
                s[i] = lsh_trees[i].get_parent(s[i], x[i])
                # Update depth
                x[i] -= 1
        
        # Decrease the global depth
        max_x -= 1
    
    return P


class LSHNode:
    def __init__(self, parent=None):
        self.children = {}  # Hash bucket to child node mapping
        self.points = set()  # Points stored in this node (leaf)
        self.parent = parent  # Pointer to the parent node


class LSHTree:
    def __init__(self, max_depth):
        self.root = LSHNode()
        self.max_depth = max_depth

    def insert(self, vector, point_id, hasher):
        """
        Inserts a point into the tree.
        Args:
            vector (array): The point's vector.
            point_id (int): Unique identifier for the point.
            hasher (RandomHash): Hashing class to compute g(p, x).
        """
        node = self.root
        for depth in range(1, self.max_depth + 1):
            label = hasher.hash(vector, depth)
            if label not in node.children:
                # Create a new child node and set its parent
                node.children[label] = LSHNode(parent=node)
            node = node.children[label]
        node.points.add(point_id)


    def descend(self, vector, hasher):
        """
        Top-down traversal to find the deepest matching leaf node.
        Args:
            vector (array): Query vector.
            hasher (RandomHash): Hashing class to compute g(p, x).
        Returns:
            (int, LSHNode): Depth reached and the corresponding leaf node.
        """
        node = self.root
        depth = 0
        for depth in range(1, self.max_depth + 1):
            label = hasher.hash(vector, depth)
            if label in node.children:
                node = node.children[label]
            else:
                break
        return depth - 1, node

    def get_descendants(self, node):
        """
        Collect all points in the subtree rooted at a given node.
        Args:
            node (LSHNode): Node from which to collect points.
        Returns:
            set: Set of points in the subtree.
        """
        if not node.children:
            return node.points
        points = set(node.points)
        for child in node.children.values():
            points.update(self.get_descendants(child))
        return points

    def get_parent(self, node, depth):
        """
        Get the parent of a given node by maintaining a traversal path.
        Args:
            node (LSHNode): Node to find the parent for.
            depth (int): Current depth of the node.
        Returns:
            LSHNode: Parent node.
        """
        return node.parent


class LSHForest:
    def __init__(self, num_trees, max_depth, dim):
        """
        Initialize the LSH Forest.
        Args:
            num_trees (int): Number of trees in the forest.
            max_depth (int): Maximum depth for each tree.
            dim (int): Dimensionality of the data.
        """
        self.trees = [LSHTree(max_depth) for _ in range(num_trees)]
        self.hashers = [RandomHash(dim, max_depth) for _ in range(num_trees)]
        self.max_depth = max_depth

    def insert(self, vector, point_id):
        """
        Insert a point into all trees.
        Args:
            vector (array): The point's vector.
            point_id (int): Unique identifier for the point.
        """
        for tree, hasher in zip(self.trees, self.hashers):
            tree.insert(vector, point_id, hasher)

    def query(self, vector, m, c):
        """
        Query the forest for nearest neighbors.
        Args:
            vector (array): Query vector.
            m (int): Number of distinct neighbors to find.
            c (float): Multiplier for total candidates.
        Returns:
            set: Candidate points.
        """
        # Top-down phase
        x = []
        s = []
        for tree, hasher in zip(self.trees, self.hashers):
            depth, leaf = tree.descend(vector, hasher)
            x.append(depth)
            s.append(leaf)

        # Bottom-up phase using SYNCHASCEND
        return synchascend(x, s, self.trees, c, m)



