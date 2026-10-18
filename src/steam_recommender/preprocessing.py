"""The thesis's single-pass filtering, preserving row order and inferred dtypes."""


def prepare_interactions(users_items):
    """Cap at 120000 minutes, filter items at five rows, then users at five rows.

    This is deliberately not iterative and does not deduplicate observations.
    Zero-playtime ownership records remain. Return (filtered, cold_start), where
    cold_start contains users below five after the item filter, not all removed
    rows. Input indices and order are retained; the caller's frame is not mutated.
    """
    # cap playtime (2000h)
    users_items = users_items.copy()
    users_items['playtime_forever'] = users_items['playtime_forever'].clip(upper=120000)

    # 1. usuń gry (itemy) występujące mniej niż 5 razy
    item_counts = users_items['item_id'].value_counts()
    valid_items = item_counts[item_counts >= 5].index

    df_step1 = users_items[users_items['item_id'].isin(valid_items)].copy()

    # 2. policz użytkowników po filtrze itemów
    user_counts = df_step1['user_id'].value_counts()

    # 3. podział na valid i cold start
    valid_users = user_counts[user_counts >= 5].index
    cold_start_users = user_counts[user_counts < 5].index

    # 4. finalne zbiory
    users_items_filtered = df_step1[df_step1['user_id'].isin(valid_users)].copy()
    cold_start_users = df_step1[df_step1['user_id'].isin(cold_start_users)].copy()

    return users_items_filtered, cold_start_users
