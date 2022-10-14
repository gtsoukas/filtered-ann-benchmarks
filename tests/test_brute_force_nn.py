"""tests for brute force top-k nearest neighbor search"""

import unittest

import numpy as np

from filtered_ann_benchmarks.brute_force_nn import knn_query


class BruteForceTesting(unittest.TestCase):
    def test_knn_query(self):
        """A simple test for the knn_query function"""
        query_embedding = np.array([1, 2, 3])
        search_space = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
        ids = np.array(["a", "b", "c", "d"])
        k = 2
        result = knn_query(
            query_embedding,
            search_space,
            ids,
            k,
        )
        # expecting largest inner product at first index
        expected = ["d", "c"]

        self.assertEqual(result, expected)


if __name__ == "__main__":
    unittest.main()
