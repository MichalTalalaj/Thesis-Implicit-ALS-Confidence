"""Scientific splitting implementation extracted without changing calculation order."""

import numpy as np


def random_user_split_train_val(
    users_items_filtered,
    val_frac=0.1,
    seed=None,
):
    """
    Random per-user train/val split.

    Parameters
    ----------
    users_items_filtered : pd.DataFrame
        Must contain a 'user_id' column.

    val_frac : float
        Fraction of interactions per user for validation.

    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    train_df, val_df : pd.DataFrame
    """

    df = users_items_filtered.copy()

    # shuffle interactions within each user
    df = (
        df.groupby("user_id", group_keys=False)
          .sample(frac=1, random_state=seed)
          .reset_index(drop=True)
    )

    # number of interactions per user
    group_sizes = df.groupby("user_id")["user_id"].transform("size")

    # position within shuffled user history
    pos = df.groupby("user_id").cumcount()

    # val size
    n_val = np.ceil(val_frac * group_sizes).astype(int)

    # masks
    val_mask = pos < n_val
    train_mask = ~val_mask

    # splits
    train_df = df[train_mask].reset_index(drop=True)
    val_df = df[val_mask].reset_index(drop=True)

    return train_df, val_df


def train_test_split_first(df, test_frac=0.1, seed=42):

    df = (
        df.groupby("user_id", group_keys=False)
          .sample(frac=1, random_state=seed)
          .reset_index(drop=True)
    )

    sizes = df.groupby("user_id")["user_id"].transform("size")
    pos = df.groupby("user_id").cumcount()

    n_test = np.ceil(test_frac * sizes).astype(int)

    test_mask = pos < n_test
    trainval_mask = ~test_mask

    return df[trainval_mask].copy(), df[test_mask].copy()
