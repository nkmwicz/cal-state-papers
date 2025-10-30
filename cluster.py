import random
import numpy as np
from openai import OpenAI
import os
import dotenv
import umap
from sklearn.cluster import SpectralClustering

dotenv.load_dotenv()


def embed_items(texts):
    """
    Embed text into vectors.

    Parameters:
        texts (list): The collection of texts to be embedded.
    Returns:
        embeddings (list): A list of embedded vectors.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    texts = [text.replace("\n", " ") for text in texts]
    if isinstance(texts, str):
        texts = [texts]
    texts = [t.replace("\n", " ") for t in texts]
    response = client.embeddings.create(input=texts, model="text-embedding-3-small")

    return [data.embedding for data in response.data]


def reduce_dimensions(embeddings, n_components=50, random_state=42):
    """
    Reduce the dimensionality of embeddings using UMAP
    Parameters:
        embeddings (list): The list of embedded vectors.
        n_components (int): The number of dimensions to reduce to.

    Returns:
        reduced_embeddings (np.ndarray): The reduced dimensionality embeddings.
    """
    reducer = umap.UMAP(
        n_components=n_components,
        random_state=random_state,
        metric="cosine",
        min_dist=0.1,
        n_neighbors=15,
    )
    X_reducer = reducer.fit_transform(embeddings)
    return X_reducer


def normalize_rows(mat, eps=1e-10):
    """
    Normalize the rows of a matrix.

    Parameters:
        mat (np.ndarray): The input matrix.
        eps (float): A small value to avoid division by zero.

    Returns:
        np.ndarray: The row-normalized matrix.
    """
    import numpy as np

    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / (norms + eps)


def kmeans_plus_plus_init(
    X: np.ndarray, k: int, random_state: int | None = None, return_indices: bool = False
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Construct initial cluster centroids using k-means++ algorithm for cosine similarity clustering with embedded vectors.

    Parameters:
        X (np.ndarray): The input data points (embedded vectors).
        k (int): The number of clusters.
        random_state (int): Seed for random number generator.
    Returns:
        np.ndarray: The initial cluster centroids.
    """
    n_samples, n_features = X.shape

    rng = np.random.default_rng(random_state)

    Xw = X.astype(np.float64, copy=True)
    norms = np.linalg.norm(Xw, axis=1, keepdims=True)
    Xw = Xw / (norms + 1e-10)

    first = int(rng.integers(0, n_samples))
    centroids_idx = [first]
    C = Xw[[first], :]  # (1, d)

    for _ in range(1, k):
        sims = Xw @ C.T  # (n, m)
        max_sim = np.max(sims, axis=1)  # (n,)
        d = 1.0 - max_sim  # cosine "distance" on unit sphere
        d2 = d * d
        total = d2.sum()
        if not np.isfinite(total) or total <= 1e-18:
            # All points are identical (or numerical collapse): pick random unseen
            remaining = np.setdiff1d(
                np.arange(n_samples), np.array(centroids_idx), assume_unique=False
            )
            next_idx = int(rng.choice(remaining))
        else:
            probs = d2 / total
            next_idx = int(rng.choice(n_samples, p=probs))

        centroids_idx.append(next_idx)
        C = Xw[centroids_idx, :]  # update centers matrix

    centroids_idx = np.array(centroids_idx, dtype=int)
    centroids = Xw[centroids_idx, :]
    return (centroids_idx, centroids) if return_indices else centroids


# def assign_similarities()
def setup_clusters(centroids, embeds, clusters):
    """
    Sets embeds into clusters based on cosine similarity to centroids.

    Parameters:
        centroids (np.ndarray): The cluster centroids.
        embeds (np.ndarray): The embedded vectors.
        clusters (list[list]): The list of clusters to populate. A list for each cluster.
    Returns:

    """
    arr_norm = normalize_rows(embeds)
    sims = arr_norm @ centroids.T  # cosine similarity
    # max_sim_indices = np.argmax(sims, axis=1)
    max_sim = np.max(sims, axis=1).reshape(-1, 1)
    margin = 0.10
    mask = sims >= (max_sim - margin)
    item, group = np.where(mask)
    for idx, num in enumerate(group):
        clusters[num].append(int(item[idx]))
    return sims, clusters


def create_cluster(embed_list, num_clusters=10):
    """
    Create clusters from embedded

    Parameters:

    Returns:
        list: A list of clusters.
    """
    # Normalize the embedded vectors
    arr = np.asarray(embed_list, dtype=object)
    if arr.dtype == object or arr.ndim == 1:
        arr = np.array(list(arr), dtype=np.float64)
    else:
        arr = arr.astype(np.float64, copy=False)

    # Reduce dimensions for clustering
    # arr = reduce_dimensions(arr, n_components=50)

    # create random cluster centroid for first iteration
    centroids_idx, centroids = kmeans_plus_plus_init(
        arr, num_clusters, random_state=42, return_indices=True
    )
    average_cent_dist_change = np.inf
    groups = [[] for _ in range(num_clusters)]
    min_dist = 1e-6

    clusters = setup_clusters(centroids, arr, groups)
    # while average_cent_dist_change > min_dist:
    # assign points to nearest centroid
    # clusters = [
    #     np.mean(arr_norm[grp], axis=0) for grp in groups
    # ]  # these clusters aren't normalized?

    return clusters


def spectral_clusters(embeddings, n_clusters=10, random_state=42):
    """
    Create clusters using spectral clustering.

    Parameters:
        embeddings (list): The list of embedded vectors.
        n_clusters (int): The number of clusters.

    Returns:
        labels (np.ndarray): The cluster labels for each embedding.
    """

    embeddings = reduce_dimensions(embeddings, n_components=50)

    X = np.asarray(embeddings, dtype=np.float64)
    spectral = SpectralClustering(
        n_clusters=n_clusters,
        affinity="nearest_neighbors",
        assign_labels="kmeans",
        random_state=random_state,
    )
    labels = spectral.fit_predict(X)
    return labels
