"""Scientific metrics implementation extracted without changing calculation order."""

import numpy as np


def precision_at_k(recommended, relevant, k):
    recommended_k = recommended[:k]
    hits = len(set(recommended_k) & relevant)
    return hits / k


def recall_at_k(recommended, relevant, k):
    recommended_k = recommended[:k]
    hits = len(set(recommended_k) & relevant)
    return hits / len(relevant)


def dcg_at_k(recommended, relevant, k):
    dcg = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            dcg += 1 / np.log2(i + 2)
    return dcg


def ndcg_at_k(recommended, relevant, k):
    dcg = dcg_at_k(recommended, relevant, k)
    ideal_dcg = sum(
        1 / np.log2(i + 2)
        for i in range(min(len(relevant), k))
    )
    if ideal_dcg == 0:
        return 0.0
    return dcg / ideal_dcg


def coverage_at_k(all_recommendations, all_items, k):
    recommended_items = set()
    for recs in all_recommendations.values():
        recommended_items.update(recs[:k])
    return len(recommended_items) / len(all_items)


def gap(recommendations, item_interaction_count, k):
    """
    Group Average Popularity (Abdollahpouri et al. 2019)
    """
    user_avgs = []
    for recs in recommendations.values():
        recs_k = recs[:k]
        if len(recs_k) == 0:
            continue
        avg = np.mean([item_interaction_count.get(item, 0) for item in recs_k])
        user_avgs.append(avg)
    return np.mean(user_avgs)


def delta_gap(recommendations, user_profiles, item_interaction_count, k):
    """
    Delta GAP (Abdollahpouri et al. 2019)
    Względna różnica między popularnością rekomendacji a profilem użytkownika.
    recommendations: dict {user_id: [item_id, ...]}
    user_profiles: dict {user_id: [item_id, ...]}
    item_interaction_count: dict {item_id: liczba interakcji w train}
    """
    gap_recs = gap(recommendations, item_interaction_count, k)

    profile_avgs = []
    for user_id, items in user_profiles.items():
        if len(items) == 0:
            continue
        avg = np.mean([item_interaction_count.get(item, 0) for item in items])
        profile_avgs.append(avg)

    if len(profile_avgs) == 0:
        return np.nan

    gap_profiles = np.mean(profile_avgs)

    if gap_profiles == 0 or np.isnan(gap_profiles):
        return np.nan

    return (gap_recs - gap_profiles) / gap_profiles


def diversity_at_k(all_recommendations, item_factors, k):
    """
    ALS-only. Mean pairwise cosine distance within each top-K list,
    averaged across users. Higher = more diverse.
    item_factors: np.ndarray of shape (n_items, n_factors)
    all_recommendations: dict {user_id: [item_idx, ...]}
    """
    from numpy.linalg import norm

    user_diversities = []

    for recs in all_recommendations.values():
        recs_k = recs[:k]
        if len(recs_k) < 2:
            continue

        vecs = item_factors[recs_k]
        norms = norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        vecs_normalized = vecs / norms
        sim_matrix = vecs_normalized @ vecs_normalized.T

        n = len(recs_k)
        upper = [
            1 - sim_matrix[i, j]
            for i in range(n)
            for j in range(i + 1, n)
        ]

        user_diversities.append(np.mean(upper))

    if len(user_diversities) == 0:
        return np.nan

    return np.mean(user_diversities)
