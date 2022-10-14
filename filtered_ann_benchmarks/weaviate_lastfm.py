"""Module for benchmarking filtered ANN with hnswlib"""

from typing import Callable, List
import uuid

import numpy as np
import weaviate

from filtered_ann_benchmarks.benchmark import Benchmark
from filtered_ann_benchmarks.datasets import LastfmGenres


class WeaviateLastfm(Benchmark):
    """Class for benchmarking weaviate"""

    def __init__(self):

        self._data = LastfmGenres()

        print("Configuring weaviate...")
        self.client = weaviate.Client("http://localhost:8080")

        # delete all classes
        self.client.schema.delete_all()

        schema = {
            "classes": [
                {
                    "class": "Item",
                    # explicitly tell Weaviate not to vectorize anything, we are
                    # providing the vectors ourselves
                    "vectorizer": "none",
                    "description": "An item",
                    "properties": [
                        {
                            "name": "itemId",
                            "dataType": ["string"],
                        },
                        {
                            "name": "categoryId",
                            "dataType": ["string"],
                        },
                    ],
                    "vectorIndexConfig": {"distance": "dot"},
                },
            ]
        }

        self.client.schema.create(schema)
        print("Loading into weaviate...")
        # Python client specific configurations can be set with `client.batch.configure`
        # the settings can be applied to both `objects` AND `references`.
        # You have to only set them once.
        self.client.batch.configure(
            # `batch_size` takes an `int` value to enable auto-batching
            # (`None` is used for manual batching)
            batch_size=100,
            # dynamically update the `batch_size` based on import speed
            dynamic=False,
            # `timeout_retries` takes an `int` value to retry on time outs
            timeout_retries=3,
            # checks for batch-item creation errors
            # this is the default in weaviate-client >= 3.6.0
            callback=weaviate.util.check_batch_result,
        )

        # It is unclear to me, how we can ensure that weaviate filters to exacly
        # all matching categories. As a workaround we assemble an artificial
        # category id
        with self.client.batch as batch:
            for idx, item_embedding in enumerate(self._data.item_embeddings):
                genre_id = self._data.artists_genres[idx]
                genre = (
                    "unknown" if genre_id == -1 else self._data.genres[int(genre_id)]
                )
                object_props = {
                    "itemId": str(self._data.item_ids[idx]),
                    "categoryId": genre,
                }
                batch.add_data_object(
                    object_props, "Item", uuid.uuid1(), vector=item_embedding.tolist()
                )

    @property
    def data(self):
        return self._data

    def get_algorithm_summary(self) -> str:
        """Returns a short description of the search algorithm and its parameters"""
        return "weaviate w/ hnsw defaults"

    def knn_query(
        self,
        query_embedding: np.array,
        k: int,
        filter_func: Callable = None,
        category_id: str = None,
    ) -> List[str]:
        near_vector = {"vector": query_embedding.tolist()}
        query = (
            self.client.query.get(
                "Item", ["itemId", "categoryId", "_additional {distance}"]
            )
            .with_near_vector(near_vector)
            .with_limit(k)
        )
        if category_id == LastfmGenres.FULL_SPACE_LABEL:
            query_result = query.do()
        elif category_id.startswith(LastfmGenres.LARGE_SUBSPACE_LABEL_PREFIX):
            genre = category_id[len(LastfmGenres.LARGE_SUBSPACE_LABEL_PREFIX) :]
            where_filter = {
                "path": ["categoryId"],
                "operator": "NotEqual",
                "valueString": genre,
            }
            query_result = query.with_where(where_filter).do()
        else:
            where_filter = {
                "path": ["categoryId"],
                "operator": "Equal",
                "valueString": category_id,
            }
            query_result = query.with_where(where_filter).do()
        weaviate_item_ids = [x["itemId"] for x in query_result["data"]["Get"]["Item"]]
        return weaviate_item_ids


if __name__ == "__main__":
    benchmark = WeaviateLastfm()
    benchmark.run()
