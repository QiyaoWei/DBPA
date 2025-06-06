import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jensenshannon, cdist

# In the experiments, energy dist is used to reproduce the AISTATS results
def compute_energy_distance(X, Y, distance = 'cosine'):
    n = len(X)
    m = len(Y)
    # Compute pairwise distances
    if distance == 'cosine':
        dists_XY = cdist(X, Y, distance)
        dists_XX = cdist(X, X, distance)
        dists_YY = cdist(Y, Y, distance)
    elif distance == 'l1':
        dists_XY = cdist(X, Y, 'minkowski', p=1)
        dists_XX = cdist(X, X, 'minkowski', p=1)
        dists_YY = cdist(Y, Y, 'minkowski', p=1)
    elif distance == 'l2':
        dists_XY = cdist(X, Y, 'minkowski', p=2)
        dists_XX = cdist(X, X, 'minkowski', p=2)
        dists_YY = cdist(Y, Y, 'minkowski', p=2)
    else:
        raise ValueError(f"Invalid distance metric: {distance}")

    # Compute the terms
    term1 = (2.0 / (n * m)) * np.sum(dists_XY)
    term2 = (1.0 / n**2) * np.sum(dists_XX)
    term3 = (1.0 / m**2) * np.sum(dists_YY)

    energy_distance = term1 - term2 - term3
    return energy_distance, dists_XY, dists_XX, dists_YY

def permutation_test_energy(X, Y, num_permutations=1000, distance='cosine'):
    combined = np.vstack((X, Y))
    n = len(X)
    E_values = []
    for _ in range(num_permutations):
        np.random.shuffle(combined)
        perm_X = combined[:n]
        perm_Y = combined[n:]
        E_perm, dists_XY, dists_XX, dists_YY = compute_energy_distance(perm_X, perm_Y, distance=distance)
        E_values.append(E_perm)
    return np.array(E_values)

def compute_energy_distance_fn(baseline_embeddings1, baseline_embeddings2, distance='cosine'):
    E_n, dists_XY, dists_XX, dists_YY = compute_energy_distance(baseline_embeddings1, baseline_embeddings2, distance=distance)
    E_values = permutation_test_energy(baseline_embeddings1, baseline_embeddings2, num_permutations=500, distance=distance)
    p_value = np.mean(E_values >= E_n)
    return E_n, p_value

# In the experiments, JSD is used to reproduce the SFLLM results
def calculate_cosine_similarities(embeddings1, embeddings2=None):
    if embeddings2 is None:
        similarities = cosine_similarity(embeddings1)
        return similarities[np.triu_indices_from(similarities, k=1)]
    else:
        return cosine_similarity(embeddings1, embeddings2).flatten()

def jensen_shannon_divergence_and_pvalue(arr1, arr2, num_bootstraps=1000, bins=30):
    def calculate_jsd(x, y):
        hist1, _ = np.histogram(x, bins=bins, density=True)
        hist2, _ = np.histogram(y, bins=bins, density=True)
        return jensenshannon(hist1, hist2)
    
    observed_jsd = calculate_jsd(arr1, arr2)
    combined = np.concatenate([arr1, arr2])
    n1, n2 = len(arr1), len(arr2)
    
    bootstrap_jsds = []
    for _ in range(num_bootstraps):
        resampled = np.random.choice(combined, size=n1+n2, replace=True)
        bootstrap_arr1, bootstrap_arr2 = resampled[:n1], resampled[n1:]
        bootstrap_jsd = calculate_jsd(bootstrap_arr1, bootstrap_arr2)
        bootstrap_jsds.append(bootstrap_jsd)
    
    p_value = np.mean(np.array(bootstrap_jsds) >= observed_jsd)
    jsd_std = np.std(bootstrap_jsds)
    
    return observed_jsd, p_value, jsd_std