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
