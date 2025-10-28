from pydoc import text
import numpy as np
from openai import OpenAI


def embed_items(texts):
    """
    Embed text into vectors.

    Parameters:
        texts (list): The collection of texts to be embedded.
    Returns:
        embeddings (list): A list of embedded vectors.
    """
    client = OpenAI()
    texts = [text.replace("\n", " ") for text in texts]
    if isinstance(texts, str):
        texts = [texts]
    response = client.embeddings.create(input=texts, model="text-embedding-3-small")

    return [data.embedding for data in response.data]


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


def kmeans_plus_plus_init(X: np.ndarray, k: int) -> np.ndarray:
    """
    Construct initial cluster centroids using k-means++ algorithm for cosine similarity clustering with embedded vectors.

    Parameters:
        X (np.ndarray): The input data points (embedded vectors).
        k (int): The number of clusters.
    Returns:
        np.ndarray: The initial cluster centroids.
    """
    n_samples, n_features = X.shape

    Xw = X.astype(np.float64, copy=True)
    norms = np.linalg.norm(Xw, axis=1, keepdims=True)
    Xw = Xw / (norms + 1e-10)

    first = np.random.randint(0, n_samples)
    centroids = [Xw[first]]

    for _ in range(1, k):
        dists = np.array([min(np.dot(x, c) for c in centroids) for x in Xw])
        probs = dists / dists.sum()
        next_centroid = np.random.choice(n_samples, p=probs)
        centroids.append(Xw[next_centroid])

    return np.array(centroids)


def create_cluster(embed_list, num_clusters=10):
    """
    Create clusters from embedded

    Parameters:

    Returns:
        list: A list of clusters.
    """
    # Normalize the embedded vectors
    embed_list = normalize_rows(np.array(embed_list))

    # create random cluster centroid for first iteration
    random_indices = np.random.choice(len(embed_list), num_clusters, replace=False)
    centroids = embed_list[random_indices]
