"""Brute force top-k nearest neighbor search"""

from typing import List

import numpy as np


def knn_query(
    query_embedding: np.array,
    search_space: np.array,
    search_space_ids: np.array,
    k: int = 100,
) -> List:
    """Returns top k nearest items from search_space by inner product"""
    all_scores = query_embedding.dot(search_space.T)
    top_k_idx = np.argsort(all_scores)[::-1][:k]
    nearest_object_ids = search_space_ids[top_k_idx].tolist()
    return nearest_object_ids


# Typically, application of this function will requre precomputed lists of
# search_space_indices_filtered and search_space_ids_filtered in order to
# achieve low latency.
def filtered_knn_query(
    query_embedding: np.array,
    search_space: np.array,
    search_space_indices_filtered: np.array,
    search_space_ids_filtered: np.array,
    k: int = 100,
) -> List:
    """Like knn_query but with restriction of search space"""
    filtered_search_space = search_space.take(search_space_indices_filtered, axis=0)
    return knn_query(
        query_embedding, filtered_search_space, search_space_ids_filtered, k=k
    )
