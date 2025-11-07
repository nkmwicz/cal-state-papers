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
def setup_clusters(
    centroids: np.ndarray,
    embeds: np.ndarray,
    clusters: list[list[int]],
    soft_margin: float = 0.1,
    soft_assign: bool = True,
    iter: int = 0,
) -> tuple[list[int], np.ndarray, int]:
    """
    Sets embeds into clusters based on cosine similarity to centroids.

    Parameters:
        centroids (np.ndarray): The cluster centroids.
        embeds (np.ndarray): The embedded vectors.
        clusters (list[list]): The list of clusters to populate. A list for each cluster.
        soft_margin (float): The margin for soft assignment.
        soft_assign (bool): Whether to use soft assignment or hard assignment.
    Returns:
        embed_clust (list[int]): The list of clusters (0-k clusters) for each embed. Align to embeds.
        centroids (np.ndarray): The final centroids after clustering.
        iter (int): The number of iterations taken to converge.
    """
    centroids_list = centroids
    centroids = centroids_list[-1]
    X = embeds.astype(np.float64, copy=False)
    Xn = normalize_rows(X)
    C = centroids.astype(np.float64, copy=False)
    sims = Xn @ C.T  # cosine similarity
    # max_sim_indices = np.argmax(sims, axis=1)

    if soft_assign:
        best_sim = np.max(sims, axis=1).reshape(-1, 1)  # gets values
        # soft assignment within margin
        sim_diffs = np.diff(np.sort(sims, axis=1), axis=1)
        marg_num = (1 - soft_margin) * 100
        print(marg_num)
        margin = np.percentile(sim_diffs, marg_num)
        soft_mask = sims >= (best_sim - margin)
        item, group = np.where(soft_mask)
        for idx, num in enumerate(group):
            clusters[num].append(int(item[idx]))

        exclusive_clusters: list[list[int]] = []
        for idx, cluster in enumerate(clusters):
            clust_set = set(cluster)
            other_clust_sets = [
                set(clusters[i]) for i in range(len(clusters)) if i != idx
            ]
            exclusive_clust = clust_set - set().union(*other_clust_sets)
            exclusive_clusters.append(list(exclusive_clust))

        new_centers: list[np.ndarray] = []
        # new_centers = np.array([np.mean(Xn[grp], axis=0) for grp in exclusive_clusters])
        new_centers: list[np.ndarray] = []
        for idx, (exclusive_grp, all_grp) in enumerate(
            zip(exclusive_clusters, clusters)
        ):
            if len(exclusive_grp) > 0:
                new_centers.append(np.mean(Xn[exclusive_grp], axis=0))
            else:
                # Find the point in this cluster that is least similar to other clusters
                other_cluster_points = []
                for jdx, other_grp in enumerate(clusters):
                    if jdx != idx:
                        other_cluster_points.extend(other_grp)

                if len(other_cluster_points) > 0 and len(all_grp) > 0:
                    # Compute similarity of each point in this cluster to other clusters
                    cluster_points = Xn[all_grp]
                    other_points = Xn[other_cluster_points]

                    # Average similarity to other cluster points
                    cross_sims = cluster_points @ other_points.T
                    avg_cross_sim = np.mean(cross_sims, axis=1)

                    # Choose point with lowest avg similarity to other clusters
                    most_distinctive_local_idx = np.argmin(avg_cross_sim)
                    most_distinctive_global_idx = all_grp[most_distinctive_local_idx]
                    new_centers.append(Xn[most_distinctive_global_idx])
                else:
                    # Last resort: keep current centroid
                    new_centers.append(C[idx])
    else:
        best_sim = np.argmax(sims, axis=1)  # gets indices
        for idx, num in enumerate(best_sim):
            clusters[num].append(int(idx))
        new_centers: list[np.ndarray] = [np.mean(Xn[grp], axis=0) for grp in clusters]

    centroids_list = list(centroids_list)
    centroids_list.append(new_centers)

    last_two = centroids_list[-2:]

    threshold = 1e-12
    last_centroid = last_two[0]
    this_centroid = last_two[1]

    change_norm = np.linalg.norm(last_centroid - this_centroid)

    if iter == 100 or change_norm < threshold:
        embed_clust = [0 for n in range(embeds.shape[0])]
        for idx, cluster in enumerate(clusters):
            for clus in cluster:
                embed_clust[clus] = idx
        return embed_clust, this_centroid, iter
    else:
        reset_clusters = [[] for _ in range(len(centroids))]
        centroids_list = np.array(centroids_list)
        iter += 1
        return setup_clusters(
            centroids_list,
            Xn,
            reset_clusters,
            soft_assign=soft_assign,
            soft_margin=soft_margin,
            iter=iter,
        )


def return_representative_samples(embeds, clusters, centroids):
    """
    Returns the 10 most representative samples from each cluster based on cosine similarity to the cluster centroid.

    Parameters:
        embeds (np.ndarray): The embedded vectors.
        clusters (list[int]): The list of clusters (0-k clusters) for each embed. Align to embeds.
        centroids (list[np.ndarray]): The centroid of each cluster.

    Returns:
        Representative samples (list[list[int]]): A list of lists containing the indices of the 10 most representative samples for each cluster.
    """
    representat_samples = []
    X = embeds.astype(np.float64, copy=False)
    Xn = normalize_rows(X)
    C = np.array(centroids).astype(np.float64, copy=False)
    sims = Xn @ C.T  # cosine similarity
    for cluster_idx in range(len(centroids)):
        cluster_indices = [i for i, c in enumerate(clusters) if c == cluster_idx]
        cluster_sims = sims[cluster_indices, cluster_idx]
        top_indices = np.argsort(cluster_sims)[-10:][::-1]  # Top 10 indices
        representat_samples.append([cluster_indices[i] for i in top_indices])
    return representat_samples


def create_cluster(embed_list, num_clusters=10, soft_assign=True, soft_margin=0.1):
    """
    Create clusters from embedded

    Parameters:
        embed_list (list): The list of embedded vectors.
        num_clusters (int): The number of clusters.
        soft_assign (bool): Whether to use soft assignment or hard assignment.
        soft_margin (float): The margin for soft assignment.

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
    centroids = [centroids]
    average_cent_dist_change = np.inf
    groups = [[] for _ in range(num_clusters)]
    min_dist = 1e-6

    clusters, new_centroids, iters = setup_clusters(
        centroids, arr, groups, soft_assign=soft_assign, soft_margin=soft_margin, iter=0
    )
    representative_samples = return_representative_samples(arr, clusters, new_centroids)
    # while average_cent_dist_change > min_dist:
    # assign points to nearest centroid
    # clusters = [
    #     np.mean(arr_norm[grp], axis=0) for grp in groups
    # ]  # these clusters aren't normalized?

    return clusters, representative_samples, iters


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
