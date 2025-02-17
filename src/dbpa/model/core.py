import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jensenshannon, cdist

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