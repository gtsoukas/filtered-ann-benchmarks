"""Datasets for filtered ANN benchmarks"""

from abc import ABC, abstractmethod
from collections import Counter
import os.path
from typing import List, Tuple
import urllib.request
import zipfile

import h5py
from implicit.datasets.lastfm import get_lastfm
from implicit.utils import augment_inner_product_matrix
import implicit
import numpy as np

DATA_DIR = "data"


# Adapted version from ann-benchmarks.com
def lastfm(
    n_dimensions=64, test_size=100
) -> Tuple[np.array, np.array, np.array, np.array,]:
    """Returns last.fm embeddings

    The first cal computes embeddings and caches them. Subsequent calls return
    cached data
    """
    cache_file_path = f"{DATA_DIR}/lastfm_{n_dimensions}_{test_size}.h5"

    if os.path.isfile(cache_file_path):
        with h5py.File(cache_file_path, "r") as h5f:
            user_embeddings = h5f["user_embeddings"][:]
            item_embeddings = h5f["item_embeddings"][:]
            user_ids = h5f["user_ids"].asstr()[:]
            item_ids = h5f["item_ids"].asstr()[:]
            return user_embeddings, item_embeddings, user_ids, item_ids

    elif not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    # This tests out ANN methods for retrieval on simple matrix factorization
    # based recommendation algorithms. The idea being that the query/test
    # vectors are user factors and the train set are item factors from
    # the matrix factorization model.

    # Since the predictor is a dot product, we transform the factors first
    # as described in this
    # paper: https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/XboxInnerProduct.pdf  # noqa
    # This hopefully replicates the experiments done in this post:
    # http://www.benfrederickson.com/approximate-nearest-neighbours-for-recommender-systems/  # noqa

    # The dataset is from "Last.fm Dataset - 360K users":
    # http://www.dtic.upf.edu/~ocelma/MusicRecommendationDataset/lastfm-360K.html  # noqa

    # This requires the implicit package to generate the factors
    # (on my desktop/gpu this only takes 4-5 seconds to train - but
    # could take 1-2 minutes on a laptop)

    # train an als model on the lastfm data
    artist_ids, user_ids, play_counts = get_lastfm()
    model = implicit.als.AlternatingLeastSquares(factors=n_dimensions - 1)
    model.fit(implicit.nearest_neighbours.bm25_weight(play_counts, K1=100, B=0.8))

    # transform item factors so that each one has the same norm,
    # and transform the user factors such by appending a 0 column
    _, item_factors = augment_inner_product_matrix(model.item_factors)
    user_factors = np.append(
        model.user_factors, np.zeros((model.user_factors.shape[0], 1)), axis=1
    )

    # only query the first test_size users (speeds things up signficantly
    # without changing results)
    # user_factors = user_factors#[:test_size]

    # after that transformation a cosine lookup will return the same results
    # as the inner product on the untransformed data

    with h5py.File(cache_file_path, "w") as h5f:
        h5f.create_dataset("user_embeddings", data=item_factors[:test_size])
        h5f.create_dataset("item_embeddings", data=user_factors)
        h5f.create_dataset("user_ids", data=user_ids[:test_size])
        h5f.create_dataset("item_ids", data=artist_ids)

    # Assuming item and user embeddings are accidentally swapped.
    # It also would be more plausible if there are more users than items
    # use: user_embeddings, item_embeddings, user_ids, item_ids
    return item_factors[:test_size], user_factors, user_ids[:test_size], artist_ids


def lastfm_genres(artists: np.array) -> Tuple[np.array, np.array]:
    """Returns a list af genres so that each artist is mapped to a genre

    Approx. 60% of artists are mapped to the unknown genre -1

    Genre data is taken from http://www.cp.jku.at/datasets/LFM-1b/

    Citation:
    Large-scale Analysis of Group-specific Music Genre Taste From Collaborative
    Tags
    Schedl, M. and Ferwerda, B.
    Proceedings of the 19th IEEE International Symposium on Multimedia (ISM
    2017), Taichung, Taiwan, December 2017.
    """

    cache_file_path = f"{DATA_DIR}/LFM-1b_UGP_genres_allmusic.h5"

    missing_genre_id = -1

    if os.path.isfile(cache_file_path):
        with h5py.File(cache_file_path, "r") as h5f:
            artist_genre = h5f["artist_genre"][:]
            genres = h5f["genres"].asstr()[:]
            return artist_genre, genres

    elif not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    url = "http://www.cp.jku.at/datasets/LFM-1b/LFM-1b_UGP.zip"
    filehandle, _ = urllib.request.urlretrieve(url)
    zip_file_object = zipfile.ZipFile(filehandle, "r")
    with zip_file_object.open("LFM-1b_UGP/genres_allmusic.txt") as genres_file:
        genres = str(genres_file.read(), "utf-8").split("\n")

    def extract_tuple(line):
        cols = line.split("\t")
        artist = cols[0].lower()
        genre_id = (
            int(cols[1].strip())
            if len(cols) > 1 and len(cols[1].strip()) > 0
            else missing_genre_id
        )
        return (artist, genre_id)

    with zip_file_object.open(
        "LFM-1b_UGP/LFM-1b_artist_genres_allmusic.txt"
    ) as artist_genre_file:
        lines = str(artist_genre_file.read(), "utf-8").split("\n")
        artists_genre_map = dict(map(extract_tuple, lines))
        artist_genre = np.array(
            list(
                map(
                    lambda a: artists_genre_map[a.strip()]
                    if a.strip() in artists_genre_map
                    else missing_genre_id,
                    artists,
                )
            )
        )

    with h5py.File(cache_file_path, "w") as h5f:
        h5f.create_dataset("artist_genre", data=artist_genre)
        h5f.create_dataset("genres", data=genres)

    return artist_genre, genres


class Dataset(ABC):
    """Abstract base class for defining a dataset for the filtered vector
    search benchmarks
    """

    @abstractmethod
    def summary(self) -> str:
        """Returns a short description of the dataset and its subspaces"""

    @abstractmethod
    def get_data(self) -> Tuple[np.array, np.array, np.array, np.array]:
        """Returns user_embeddings, item_embeddings, user_ids, item_ids"""

    @abstractmethod
    def subspaces(self) -> Tuple[List, str]:
        """Yields item ids of subspaces of ascending size and a
        string identifier (category) per subspace.

        The last subspace returned should be the full search space.
        """


class LastfmRandom(Dataset):
    """Last.fm dataset with random subspaces"""

    def __init__(self):
        print("Preparing/loading lastfm data...")
        (
            self.user_embeddings,
            self.item_embeddings,
            self.user_ids,
            self.item_ids,
        ) = lastfm()

        self.random_indices = np.arange(len(self.item_ids))
        np.random.shuffle(self.random_indices)
        self.fractions_of_space = [
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            0.95,
            0.98,
            1.0,
        ]

    def get_data(self):
        """Returns user_embeddings, item_embeddings, user_ids, item_ids"""
        return self.user_embeddings, self.item_embeddings, self.user_ids, self.item_ids

    def summary(self) -> str:
        """Returns a short description of the dataset and its subspaces"""
        return "lastfm random subspaces"

    def subspaces(self):
        """Yields random subspaces of ascending size"""
        for frac in self.fractions_of_space:
            filtered_idxs = self.random_indices[: int(len(self.item_ids) * frac)]
            filtered_ids = self.item_ids.take(filtered_idxs)
            yield filtered_ids, str(frac)


class LastfmGenres(Dataset):
    """Last.fm dataset with genre subspaces"""

    UNKNOWN_GENRE_ID = -1
    UNKNOWN_GENRE_LABEL = "unknown"
    LARGE_SUBSPACE_LABEL_PREFIX = "All - "
    FULL_SPACE_LABEL = "All"

    def __init__(self):
        print("Preparing/loading lastfm data...")
        (
            self.user_embeddings,
            self.item_embeddings,
            self.user_ids,
            self.item_ids,
        ) = lastfm()
        print("Preparing/loading genre data...")
        self.artists_genres, self.genres = lastfm_genres(self.item_ids)

    def get_data(self):
        """Returns user_embeddings, item_embeddings, user_ids, item_ids"""
        return self.user_embeddings, self.item_embeddings, self.user_ids, self.item_ids

    def summary(self) -> str:
        """Returns a short description of the dataset and its subspaces"""
        return "lastfm by genre"

    def subspaces(self):
        """Yields genre subspaces of ascending size"""

        genre_freq = list(Counter(self.artists_genres).items())
        genre_freq.sort(key=lambda x: x[1])

        # small subspaces
        for genre_id, _ in genre_freq:
            genre = "unknown" if genre_id == -1 else self.genres[genre_id]
            print(f"Genre: {genre}")
            filtered_idxs = np.where(self.artists_genres == genre_id)[0]
            filtered_ids = self.item_ids.take(filtered_idxs).reshape(-1).tolist()
            yield filtered_ids, genre

        # large subspaces
        # large subspaces are constructed by taking the full space and
        # subtracting a single genre
        for genre_id, _ in genre_freq[::-1]:
            genre = self.LARGE_SUBSPACE_LABEL_PREFIX + (
                "unknown" if genre_id == -1 else self.genres[genre_id]
            )
            print(genre)
            filtered_idxs = np.where(self.artists_genres == genre_id)[0]
            filtered_ids = self.item_ids.take(filtered_idxs).reshape(-1).tolist()
            subspace_ids = list(set(self.item_ids.tolist()) - set(filtered_ids))
            yield subspace_ids, genre

        # full space
        print(self.FULL_SPACE_LABEL)
        yield self.item_ids.tolist(), self.FULL_SPACE_LABEL
