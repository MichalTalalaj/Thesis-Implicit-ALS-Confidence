"""Scientific als implementation extracted without changing calculation order."""

import numpy as np
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares
from .metrics import precision_at_k, recall_at_k, dcg_at_k, ndcg_at_k, coverage_at_k, gap, delta_gap, diversity_at_k


class ALSPipeline:
    def __init__(
        self,
        factors=50,
        iterations=20,
        alpha=40,
        regularization=0.01,
        random_state=42,
        signal_strength_fn=None,
        num_threads=2,
    ):
        self.factors = factors
        self.iterations = iterations
        self.alpha = alpha
        self.regularization = regularization
        self.random_state = random_state
        self.signal_strength_fn = signal_strength_fn
        self.num_threads = num_threads

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
        self.user2idx = {
            u: i for i, u in enumerate(train_df["user_id"].unique())
        }
        self.item2idx = {
            item: j for j, item in enumerate(train_df["item_id"].unique())
        }

        self.idx2user = {
            idx: user for user, idx in self.user2idx.items()
        }
        self.idx2item = {
            idx: item for item, idx in self.item2idx.items()
        }

    def apply_mapping(self, df):
        df = df.copy()
        df["user_id"] = df["user_id"].map(self.user2idx)
        df["item_id"] = df["item_id"].map(self.item2idx)
        return df.dropna().reset_index(drop=True)

    # =========================
    # SIGNAL STRENGTH
    # =========================
    def add_signal_strength(self, df):
        df = df.copy()

        if self.signal_strength_fn is not None:
            df["signal_strength"] = self.signal_strength_fn(df, self.alpha)
        else:
            df["signal_strength"] = 1 + self.alpha * np.log1p(
                df["playtime_forever"]
            )

        return df

    # =========================
    # BUILD MATRIX
    # =========================
    def build_user_item_matrix(self, df):
        return csr_matrix(
            (
                df["signal_strength"].astype(np.float32),
                (
                    df["user_id"].astype(int),
                    df["item_id"].astype(int)
                )
            ),
            shape=(len(self.user2idx), len(self.item2idx))
        )

    # =========================
    # TRAIN ON TRAIN/VAL
    # =========================
    def fit(self, train_df, val_df):
        self.fit_mapping(train_df)

        train_df = self.apply_mapping(train_df)
        val_df = self.apply_mapping(val_df)

        train_df = self.add_signal_strength(train_df)
        val_df = self.add_signal_strength(val_df)

        self.train_df = train_df
        self.val_df = val_df

        self.user_item_train = self.build_user_item_matrix(train_df)

        self.model = AlternatingLeastSquares(
            factors=self.factors,
            iterations=self.iterations,
            regularization=self.regularization,
            random_state=self.random_state,
            num_threads=self.num_threads,
        )

        self.model.fit(self.user_item_train)

    # =========================
    # RECOMMEND SINGLE USER
    # =========================
    def recommend(self, user_id, N=10):
        item_ids, scores = self.model.recommend(
            userid=user_id,
            user_items=self.user_item_train[user_id],
            N=N,
            filter_already_liked_items=True
        )

        return [
            (self.idx2item[item_idx], float(score))
            for item_idx, score in zip(item_ids, scores)
        ]

    # =========================
    # SAFE TEST / EVAL MAPPING
    # =========================
    def transform_eval(self, eval_df):
        eval_df = eval_df.copy()
        eval_df["user_id"] = eval_df["user_id"].map(self.user2idx)
        eval_df["item_id"] = eval_df["item_id"].map(self.item2idx)
        return eval_df.dropna().reset_index(drop=True)

    # =========================
    # FAST EVALUATE — ONLY MEAN NDCG
    # Used by Optuna
    # =========================
    def evaluate_fast(self, K=10):
        """
        Fast validation evaluation for Optuna.

        Returns only mean NDCG@K.
        No per-user dicts.
        No coverage.
        No delta_gap.
        No diversity.

        This keeps Optuna trials as cheap as possible.
        """

        val_true_items = (
            self.val_df.groupby("user_id")["item_id"]
            .apply(set)
            .to_dict()
        )

        valid_users = [
            user_id for user_id, items in val_true_items.items()
            if len(items) > 0
        ]

        if len(valid_users) == 0:
            return np.nan

        all_ids, _ = self.model.recommend(
            userid=valid_users,
            user_items=self.user_item_train[valid_users],
            N=K,
            filter_already_liked_items=True
        )

        ndcgs = []

        for row_idx, user_id in enumerate(valid_users):
            relevant_items = val_true_items[user_id]
            recommended_items = list(all_ids[row_idx])

            ndcgs.append(
                ndcg_at_k(recommended_items, relevant_items, K)
            )

        return float(np.mean(ndcgs))

    # =========================
    # FULL EVALUATE — FINAL MODELS
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
        """
        Full evaluation for final validation/test reporting.

        If return_per_user=True:
            returns per-user NDCG mapped back to original user_id.

        This function is NOT used inside Optuna objective.
        """

        true_items = (
            eval_df.groupby("user_id")["item_id"]
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
                return empty_results, {}, {}

            return empty_results, {}

        all_ids, _ = self.model.recommend(
            userid=valid_users,
            user_items=self.user_item_train[valid_users],
            N=K,
            filter_already_liked_items=True
        )

        precisions = []
        recalls = []
        ndcgs = []

        all_recommendations = {}
        user_ndcg = {} if return_per_user else None

        for row_idx, mapped_user_id in enumerate(valid_users):
            relevant_items = true_items[mapped_user_id]
            recommended_items = list(all_ids[row_idx])

            all_recommendations[mapped_user_id] = recommended_items

            p = precision_at_k(recommended_items, relevant_items, K)
            r = recall_at_k(recommended_items, relevant_items, K)
            n = ndcg_at_k(recommended_items, relevant_items, K)

            precisions.append(p)
            recalls.append(r)
            ndcgs.append(n)

            if return_per_user:
                original_user_id = self.idx2user[mapped_user_id]
                user_ndcg[original_user_id] = n

        user_profiles = (
            self.train_df.groupby("user_id")["item_id"]
            .apply(list)
            .to_dict()
        )

        diversity = diversity_at_k(
            all_recommendations,
            self.model.item_factors,
            K
        )

        results = {
            f"precision@{K}": float(np.mean(precisions)),
            f"recall@{K}": float(np.mean(recalls)),
            f"ndcg@{K}": float(np.mean(ndcgs)),
            f"coverage@{K}": coverage_at_k(
                all_recommendations,
                all_items,
                K
            ),
            f"delta_gap@{K}": delta_gap(
                all_recommendations,
                user_profiles,
                item_interaction_count,
                K
            ),
            f"diversity@{K}": diversity,
            "users_evaluated": len(valid_users),
        }

        if verbose:
            print(f"Users evaluated: {len(valid_users)}")
            print(f"Precision@{K}:   {results[f'precision@{K}']:.4f}")
            print(f"Recall@{K}:      {results[f'recall@{K}']:.4f}")
            print(f"NDCG@{K}:        {results[f'ndcg@{K}']:.4f}")
            print(f"Coverage@{K}:    {results[f'coverage@{K}']:.4f}")
            print(f"Delta GAP@{K}:   {results[f'delta_gap@{K}']:.4f}")
            print(f"Diversity@{K}:   {results[f'diversity@{K}']:.4f}")

        if return_per_user:
            return results, all_recommendations, user_ndcg

        return results, all_recommendations

    # =========================
    # REFIT FULL TRAIN + TEST
    # =========================
    def refit_full_and_test(
        self,
        train_df,
        test_df,
        K=10,
        return_per_user=False,
        verbose=True,
    ):
        """
        Retrain model on full train data and evaluate on test set.

        Use return_per_user=True for final statistical tests.
        Use return_per_user=False for ordinary reporting.
        """

        df = train_df.copy()

        self.fit_mapping(df)

        df = self.apply_mapping(df)
        df = self.add_signal_strength(df)

        self.train_df = df
        self.val_df = None

        self.user_item_train = self.build_user_item_matrix(df)

        self.model = AlternatingLeastSquares(
            factors=self.factors,
            iterations=self.iterations,
            regularization=self.regularization,
            random_state=self.random_state,
            num_threads=self.num_threads,
        )

        self.model.fit(self.user_item_train)

        if verbose:
            print("Model retrained on full dataset")

        all_items = set(self.item2idx.values())

        item_interaction_count = (
            self.train_df.groupby("item_id")
            .size()
            .to_dict()
        )

        test_mapped = self.transform_eval(test_df)

        if verbose:
            print("\n--- TEST SET ---")

        return self.evaluate_on(
            test_mapped,
            all_items,
            item_interaction_count,
            K=K,
            return_per_user=return_per_user,
            verbose=verbose,
        )
