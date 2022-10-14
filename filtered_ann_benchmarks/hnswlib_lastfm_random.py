"""Module for benchmarking filtered ANN with hnswlib"""

from typing import Callable, List

import hnswlib
import numpy as np

from filtered_ann_benchmarks.benchmark import Benchmark
from filtered_ann_benchmarks.datasets import LastfmRandom


class HnswlibLastfmRandom(Benchmark):
    """Class for benchmarking filterd version of hnswlib on last.fm dataset
    with random subspaces
    """

    def __init__(self):
        self.hnsw_ef_construction = 128
        self.hnsw_m = 64
        self.hnsw_ef = 10
        self.hnswlib_threads = 4

        self._data = LastfmRandom()

        print("Indexing...")
        self.ann = hnswlib.Index(space="ip", dim=self._data.item_embeddings.shape[1])
        self.ann.init_index(
            len(self._data.item_ids),
            ef_construction=self.hnsw_ef_construction,
            M=self.hnsw_m,
        )
        # Controlling the recall by setting ef:
        # higher ef leads to better accuracy, but slower search
        self.ann.set_ef(self.hnsw_ef)
        # Set number of threads used during batch search/construction
        # By default using all available cores
        self.ann.set_num_threads(self.hnswlib_threads)
        # Can't add items ids since only integer type ids are accepted
        self.ann.add_items(self._data.item_embeddings)

    @property
    def data(self):
        return self._data

    def get_algorithm_summary(self) -> str:
        """Returns a short description of the search algorithm and its parameters"""
        return (
            f"hnswlib({self.hnsw_ef_construction}, {self.hnsw_m},"
            f"{self.hnsw_ef}, {self.hnswlib_threads})"
        )

    def knn_query(
        self,
        query_embedding: np.array,
        k: int,
        filter_func: Callable = None,
        category_id: str = None,
    ) -> List[str]:
        """Returns ids of top k nearest items
        where the search space is restricted by a filter callable.
        Only inner product is currently supported"""
        # we have to map the id to an index first, since hnswlib only accepts intege labels
        def hnswlib_filter_func(item_idx):
            return filter_func(self._data.item_ids[item_idx])

        labels, _ = self.ann.knn_query(query_embedding, k, filter=hnswlib_filter_func)
        # labels, _ = self.ann.knn_query(query_embedding, k, filter=lambda x: True)
        # also we have to map indices back to ids
        hnswlib_item_ids = self._data.item_ids[labels[0]].tolist()
        return hnswlib_item_ids


if __name__ == "__main__":
    benchmark = HnswlibLastfmRandom()
    benchmark.run()
