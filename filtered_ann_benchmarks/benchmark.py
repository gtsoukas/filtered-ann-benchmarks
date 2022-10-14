"""Modul for abstract filtered ann benchmarks"""

from abc import ABC, abstractmethod
import os.path
import statistics
import time
from typing import Callable, List

import matplotlib.pyplot as plt
import numpy as np

from filtered_ann_benchmarks.brute_force_nn import filtered_knn_query

RESULTS_DIR = "results"


class Benchmark(ABC):
    """Abstract base class for defining filtered vector search benchmarks

    Subclasses are responsible for datasets, subspaces and vector search
    implementation
    """

    @property
    @abstractmethod
    def data(self):
        """Returns the dataset used for benchmarking"""

    @abstractmethod
    def get_algorithm_summary(self) -> str:
        """Returns a short description of the search algorithm and its parameters"""

    @abstractmethod
    def knn_query(
        self,
        query_embedding: np.array,
        k: int,
        filter_func: Callable = None,
        category_id: str = None,
    ) -> List[str]:
        """Returns ids of top k nearest items from search_space by inner product

        filter_function: item_id -> true if item is allowed to be returned by
        the query

        Implementations should either use filter_func or category_id but not
        both.
        """

    def run(self):
        """Executes the benchmark"""
        k = 10
        user_embeddings, item_embeddings, _, item_ids = self.data.get_data()
        subspace_sizes = []
        median_latencies = []
        median_recalls = []
        min_recalls = []
        bf_median_latencies = []
        for filtered_item_ids, category_id in self.data.subspaces():
            subspace_size = len(filtered_item_ids)
            subspace_sizes.append(subspace_size)
            print(
                f"Querying {subspace_size} from {item_embeddings.shape[0]} elements..."
            )
            filtered_item_ids_set = set(filtered_item_ids)

            def filter_func(idx):
                return idx in filtered_item_ids_set

            latencies = []
            bf_latencies = []
            recalls = []
            filtered_idxs = np.array(
                [idx for idx, id in enumerate(item_ids) if filter_func(id)]
            )
            filtered_ids = item_ids.take(filtered_idxs)
            for user_embedding in user_embeddings:
                start = time.time()
                # It might seem a little unfair to count the time for the filter
                # callback functions into the latency...
                ann_item_ids = self.knn_query(
                    user_embedding, k, filter_func, category_id=category_id
                )
                latency = 1000 * (time.time() - start)
                latencies.append(latency)

                # brute force nn for recall computation
                start = time.time()
                bf_item_ids = filtered_knn_query(
                    user_embedding, item_embeddings, filtered_idxs, filtered_ids, k=k
                )
                bf_latency = 1000 * (time.time() - start)
                bf_latencies.append(bf_latency)
                ann_recall = float(len(set(bf_item_ids) & set(ann_item_ids))) / float(k)
                recalls.append(ann_recall)

            median_latency = statistics.median(latencies)
            bf_median_latency = statistics.median(bf_latencies)
            median_recall = statistics.median(recalls)
            median_latencies.append(median_latency)
            bf_median_latencies.append(bf_median_latency)
            median_recalls.append(median_recall)
            min_recalls.append(min(recalls))
            print(f"... took {median_latency:.02f} ms per query (median)")
            print(f"Median ANN recall: {median_recall}")

        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)

        alpha = 0.7

        plt.scatter(subspace_sizes, median_latencies, label="ANN", alpha=alpha)
        plt.scatter(
            subspace_sizes, bf_median_latencies, label="brute force NN", alpha=alpha
        )
        plt.xlabel("search space size (number of vectors)")
        plt.ylabel("median milliseconds per query")
        plt.title(f"{self.data.summary()} {self.get_algorithm_summary()} latency")
        plt.legend()
        plt.grid()
        plt.savefig(f"results/{self.__class__.__name__}_latency.svg")
        plt.close()

        plt.scatter(subspace_sizes, median_recalls, label="median", alpha=alpha)
        plt.scatter(subspace_sizes, min_recalls, label="min", alpha=alpha)
        plt.xlabel("search space size (number of vectors)")
        plt.ylabel("recall")
        plt.title(f"{self.data.summary()} {self.get_algorithm_summary()} recall")
        plt.legend()
        plt.grid()
        plt.savefig(f"results/{self.__class__.__name__}_recall.svg")
        plt.close()
