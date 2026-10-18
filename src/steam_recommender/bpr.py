"""Scientific bpr implementation extracted without changing calculation order."""

import numpy as np
from scipy.sparse import csr_matrix
from implicit.bpr import BayesianPersonalizedRanking
from .metrics import precision_at_k, recall_at_k, dcg_at_k, ndcg_at_k, coverage_at_k, gap, delta_gap, diversity_at_k


class BPRPipeline:
    def __init__(
        self,
        factors=64,
        iterations=100,
        learning_rate=0.01,
        regularization=0.01,
        random_state=42,
        num_threads=8,
        verify_negative_samples=True,
    ):
        self.factors = factors
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.random_state = random_state
        self.num_threads = num_threads
        self.verify_negative_samples = verify_negative_samples

        self.model = None

        self.user2idx = None
        self.item2idx = None
        self.idx2user = None
        self.idx2item = None

        self.train_df = None
        self.val_df = None
        self.user_item_train = None

    # =========================
    # MAPPING
    # =========================

    def fit_mapping(self, train_df):
        self.user2idx = {u: i for i, u in enumerate(train_df["user_id"].unique())}
        self.item2idx = {i: j for j, i in enumerate(train_df["item_id"].unique())}

        self.idx2user = {i: u for u, i in self.user2idx.items()}
        self.idx2item = {i: g for g, i in self.item2idx.items()}

    def apply_mapping(self, df):
        df = df.copy()
        df["user_id"] = df["user_id"].map(self.user2idx)
        df["item_id"] = df["item_id"].map(self.item2idx)
        return df.dropna()

    def transform_test(self, test_df):
        test_df = test_df.copy()
        test_df["user_id"] = test_df["user_id"].map(self.user2idx)
        test_df["item_id"] = test_df["item_id"].map(self.item2idx)
        return test_df.dropna()

    # =========================
    # FIT
    # =========================

    def fit(self, train_df, val_df=None):
        self.fit_mapping(train_df)

        train_df = self.apply_mapping(train_df)

        if val_df is not None:
            val_df = self.apply_mapping(val_df)
            self.val_df = val_df

        self.train_df = train_df

        # BPR traktuje każdy niezerowy wpis jako pozytywną interakcję.
        # Wartość wpisu ustawiamy na 1, czyli sygnał binarny.
        self.user_item_train = csr_matrix(
            (
                np.ones(len(train_df), dtype=np.float32),
                (
                    train_df["user_id"].astype(int),
                    train_df["item_id"].astype(int),
                ),
            ),
            shape=(len(self.user2idx), len(self.item2idx)),
        )

        self.model = BayesianPersonalizedRanking(
            factors=self.factors,
            learning_rate=self.learning_rate,
            regularization=self.regularization,
            iterations=self.iterations,
            random_state=self.random_state,
            num_threads=self.num_threads,
            verify_negative_samples=self.verify_negative_samples,
        )

        self.model.fit(self.user_item_train)

    # =========================
    # FAST EVALUATE — NDCG ONLY
    # =========================

    def evaluate_fast(self, K=10):
        val_true_items = (
            self.val_df
            .groupby("user_id")["item_id"]
            .apply(set)
            .to_dict()
        )

        valid_users = [
            u for u, items in val_true_items.items()
            if len(items) > 0
        ]

        if len(valid_users) == 0:
            return np.nan

        all_ids, _ = self.model.recommend(
            userid=valid_users,
            user_items=self.user_item_train[valid_users],
            N=K,
            filter_already_liked_items=True,
        )

        ndcgs = []

        for i, user_id in enumerate(valid_users):
            relevant_items = val_true_items[user_id]
            recommended_items = list(all_ids[i])

            ndcgs.append(
                ndcg_at_k(recommended_items, relevant_items, K)
            )

        return float(np.mean(ndcgs))

    # =========================
    # FULL EVALUATE
    # =========================

    def evaluate_on(
        self,
        eval_df,
        all_items,
        item_interaction_count,
        K=10,
        return_per_user=False,
        verbose=True,
    ):
        true_items = (
            eval_df
            .groupby("user_id")["item_id"]
            .apply(set)
            .to_dict()
        )

        valid_users = [
            u for u, items in true_items.items()
            if len(items) > 0
        ]

        if len(valid_users) == 0:
            results = {
                f"precision@{K}": np.nan,
                f"recall@{K}": np.nan,
                f"ndcg@{K}": np.nan,
                f"coverage@{K}": np.nan,
                f"delta_gap@{K}": np.nan,
                f"diversity@{K}": np.nan,
                "users_evaluated": 0,
            }

            if return_per_user:
                return results, {}, {}

            return results, {}

        all_ids, _ = self.model.recommend(
            userid=valid_users,
            user_items=self.user_item_train[valid_users],
            N=K,
            filter_already_liked_items=True,
        )

        precisions = []
        recalls = []
        ndcgs = []

        all_recommendations = {}
        per_user_ndcg = {}

        for i, user_id in enumerate(valid_users):
            relevant_items = true_items[user_id]
            recommended_items = list(all_ids[i])

            all_recommendations[user_id] = recommended_items

            p = precision_at_k(recommended_items, relevant_items, K)
            r = recall_at_k(recommended_items, relevant_items, K)
            n = ndcg_at_k(recommended_items, relevant_items, K)

            precisions.append(p)
            recalls.append(r)
            ndcgs.append(n)

            if return_per_user:
                original_user_id = self.idx2user[user_id]
                per_user_ndcg[original_user_id] = n

        user_profiles = (
            self.train_df
            .groupby("user_id")["item_id"]
            .apply(list)
            .to_dict()
        )

        diversity = diversity_at_k(
            all_recommendations,
            self.model.item_factors,
            K,
        )

        results = {
            f"precision@{K}": float(np.mean(precisions)),
            f"recall@{K}": float(np.mean(recalls)),
            f"ndcg@{K}": float(np.mean(ndcgs)),
            f"coverage@{K}": coverage_at_k(
                all_recommendations,
                all_items,
                K,
            ),
            f"delta_gap@{K}": delta_gap(
                all_recommendations,
                user_profiles,
                item_interaction_count,
                K,
            ),
            f"diversity@{K}": diversity,
            "users_evaluated": len(valid_users),
        }

        if verbose:
            print(f"Users evaluated: {len(valid_users)}")
            print(f"Precision@{K}: {results[f'precision@{K}']:.4f}")
            print(f"Recall@{K}:    {results[f'recall@{K}']:.4f}")
            print(f"NDCG@{K}:      {results[f'ndcg@{K}']:.4f}")
            print(f"Coverage@{K}:  {results[f'coverage@{K}']:.4f}")
            print(f"Delta GAP@{K}: {results[f'delta_gap@{K}']:.4f}")
            print(f"Diversity@{K}: {results[f'diversity@{K}']:.4f}")

        if return_per_user:
            return results, all_recommendations, per_user_ndcg

        return results, all_recommendations

    # =========================
    # REFIT FULL + TEST
    # =========================

    def refit_full_and_test(
        self,
        train_df,
        test_df,
        K=10,
        return_per_user=False,
        verbose=True,
    ):
        self.fit(train_df, val_df=None)

        all_items = set(self.item2idx.values())

        item_interaction_count = (
            self.train_df
            .groupby("item_id")
            .size()
            .to_dict()
        )

        test_mapped = self.transform_test(test_df)

        if verbose:
            print("\n--- TEST SET ---")

        return self.evaluate_on(
            test_mapped,
            all_items=all_items,
            item_interaction_count=item_interaction_count,
            K=K,
            return_per_user=return_per_user,
            verbose=verbose,
        )
