"""Scientific baselines implementation extracted without changing calculation order."""

import numpy as np
from .metrics import precision_at_k, recall_at_k, dcg_at_k, ndcg_at_k, coverage_at_k, gap, delta_gap, diversity_at_k


def evaluate_popular_test(train_df, test_df, K=10, return_per_user=True):
    """
    Popular baseline on final test set.

    Recommendations:
    - globally most popular items from train_df,
    - excluding items already seen by the user in train_df.

    To match ALS evaluation, test interactions with users/items unseen
    in train_df are dropped.
    """

    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df["user_id"] = train_df["user_id"].astype(str)
    train_df["item_id"] = train_df["item_id"].astype(str)

    test_df["user_id"] = test_df["user_id"].astype(str)
    test_df["item_id"] = test_df["item_id"].astype(str)

    train_users = set(train_df["user_id"].unique())
    train_items = set(train_df["item_id"].unique())

    test_df = test_df[
        test_df["user_id"].isin(train_users)
        & test_df["item_id"].isin(train_items)
    ].reset_index(drop=True)

    true_items = (
        test_df
        .groupby("user_id")["item_id"]
        .apply(set)
        .to_dict()
    )

    valid_users = [
        user_id for user_id, items in true_items.items()
        if len(items) > 0
    ]

    if len(valid_users) == 0:
        empty_results = {
            f"precision@{K}": np.nan,
            f"recall@{K}": np.nan,
            f"ndcg@{K}": np.nan,
            f"coverage@{K}": np.nan,
            f"delta_gap@{K}": np.nan,
            f"diversity@{K}": np.nan,
            "users_evaluated": 0,
        }

        if return_per_user:
            return empty_results, {}

        return empty_results

    item_interaction_count = (
        train_df
        .groupby("item_id")
        .size()
        .to_dict()
    )

    popular_items = (
        train_df["item_id"]
        .value_counts()
        .index
        .tolist()
    )

    user_profiles = (
        train_df
        .groupby("user_id")["item_id"]
        .apply(list)
        .to_dict()
    )

    all_recommendations = {}

    precisions = []
    recalls = []
    ndcgs = []
    user_ndcg = {}

    for user_id in valid_users:
        seen_items = set(user_profiles.get(user_id, []))

        recommended_items = [
            item for item in popular_items
            if item not in seen_items
        ][:K]

        relevant_items = true_items[user_id]

        all_recommendations[user_id] = recommended_items

        p = precision_at_k(recommended_items, relevant_items, K)
        r = recall_at_k(recommended_items, relevant_items, K)
        n = ndcg_at_k(recommended_items, relevant_items, K)

        precisions.append(p)
        recalls.append(r)
        ndcgs.append(n)

        if return_per_user:
            user_ndcg[user_id] = n

    results = {
        f"precision@{K}": float(np.mean(precisions)),
        f"recall@{K}": float(np.mean(recalls)),
        f"ndcg@{K}": float(np.mean(ndcgs)),
        f"coverage@{K}": coverage_at_k(
            all_recommendations,
            train_items,
            K
        ),
        f"delta_gap@{K}": delta_gap(
            all_recommendations,
            user_profiles,
            item_interaction_count,
            K
        ),
        # diversity liczymy na faktorach itemów ALS, więc dla Popular zostawiamy NA
        f"diversity@{K}": np.nan,
        "users_evaluated": len(valid_users),
    }

    if return_per_user:
        return results, user_ndcg

    return results


def evaluate_popular_validation(train_df, val_df, K=10):
    """
    Popular baseline evaluated on a given train/validation split.

    Recommendations are the globally most popular items from train_df,
    excluding items already seen by the user in train_df.

    To match ALS evaluation, validation interactions with users/items
    unseen in train_df are dropped.
    """

    train_df = train_df.copy()
    val_df = val_df.copy()

    train_df["user_id"] = train_df["user_id"].astype(str)
    train_df["item_id"] = train_df["item_id"].astype(str)

    val_df["user_id"] = val_df["user_id"].astype(str)
    val_df["item_id"] = val_df["item_id"].astype(str)

    train_users = set(train_df["user_id"].unique())
    train_items = set(train_df["item_id"].unique())

    val_df = val_df[
        val_df["user_id"].isin(train_users)
        & val_df["item_id"].isin(train_items)
    ].reset_index(drop=True)

    true_items = (
        val_df
        .groupby("user_id")["item_id"]
        .apply(set)
        .to_dict()
    )

    valid_users = [
        user_id for user_id, items in true_items.items()
        if len(items) > 0
    ]

    if len(valid_users) == 0:
        return {
            f"precision@{K}": np.nan,
            f"recall@{K}": np.nan,
            f"ndcg@{K}": np.nan,
            f"coverage@{K}": np.nan,
            f"delta_gap@{K}": np.nan,
            f"diversity@{K}": np.nan,
            "users_evaluated": 0,
        }

    item_interaction_count = (
        train_df
        .groupby("item_id")
        .size()
        .to_dict()
    )

    popular_items = (
        train_df["item_id"]
        .value_counts()
        .index
        .tolist()
    )

    user_profiles = (
        train_df
        .groupby("user_id")["item_id"]
        .apply(list)
        .to_dict()
    )

    all_recommendations = {}

    precisions = []
    recalls = []
    ndcgs = []

    for user_id in valid_users:
        seen_items = set(user_profiles.get(user_id, []))

        recommended_items = [
            item for item in popular_items
            if item not in seen_items
        ][:K]

        relevant_items = true_items[user_id]

        all_recommendations[user_id] = recommended_items

        precisions.append(
            precision_at_k(recommended_items, relevant_items, K)
        )
        recalls.append(
            recall_at_k(recommended_items, relevant_items, K)
        )
        ndcgs.append(
            ndcg_at_k(recommended_items, relevant_items, K)
        )

    results = {
        f"precision@{K}": float(np.mean(precisions)),
        f"recall@{K}": float(np.mean(recalls)),
        f"ndcg@{K}": float(np.mean(ndcgs)),
        f"coverage@{K}": coverage_at_k(
            all_recommendations,
            train_items,
            K
        ),
        f"delta_gap@{K}": delta_gap(
            all_recommendations,
            user_profiles,
            item_interaction_count,
            K
        ),
        f"diversity@{K}": np.nan,
        "users_evaluated": len(valid_users),
    }

    return results
