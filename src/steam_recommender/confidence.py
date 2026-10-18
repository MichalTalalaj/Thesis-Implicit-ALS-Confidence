"""Scientific confidence implementation extracted without changing calculation order."""

import numpy as np
import pandas as pd


def linear_signal(df, alpha=1.0):
    return 1 + alpha * df["playtime_forever"]


def log_signal(df, alpha=1.0):
    return 1 + alpha * np.log1p(df["playtime_forever"])


def log_saturated_signal(df, alpha=1.0):
    x = np.log1p(df["playtime_forever"])
    return 1 + alpha * (x / (1 + x))


def make_log_inside_beta_signal(beta):
    def log_inside_beta_signal(df, alpha=1.0):
        return 1 + alpha * np.log1p(df["playtime_forever"] + beta)

    return log_inside_beta_signal


def make_constant_signal(gamma):
    """
    Stała ufność, niezależna od danych: c_{u,i} = 1 + gamma.

    Odpowiednik czystego binarnego ALS z pojedynczą globalną wagą.
    Parametr alpha jest ignorowany (zachowany dla zgodności sygnatury).
    """
    def constant_signal(df, alpha=1.0):
        return pd.Series(1.0 + gamma, index=df.index, dtype=float)

    return constant_signal


def get_four_variant_signal(function_name, beta=None):
    if function_name == "linear":
        return linear_signal

    if function_name == "log":
        return log_signal

    if function_name == "log_saturation":
        return log_saturated_signal

    if function_name == "log_inside_beta":
        if beta is None or pd.isna(beta):
            raise ValueError("log_inside_beta requires beta.")
        return make_log_inside_beta_signal(float(beta))

    raise ValueError(f"Unknown function: {function_name}")


def get_sensitivity_signal(function_name, beta=None):
    if function_name == "linear":
        return linear_signal

    if function_name == "log":
        return log_signal

    if function_name == "log_saturation":
        return log_saturated_signal

    if function_name == "log_inside_beta":
        if pd.isna(beta):
            raise ValueError("log_inside_beta requires beta.")
        return make_log_inside_beta_signal(beta)

    raise ValueError(f"Unknown function: {function_name}")


def get_signal_fn(function_name, beta=None, gamma=None):
    if function_name == "linear":
        return linear_signal

    if function_name == "log":
        return log_signal

    if function_name == "log_saturation":
        return log_saturated_signal

    if function_name == "log_inside_beta":
        if beta is None or pd.isna(beta):
            raise ValueError("log_inside_beta requires beta.")
        return make_log_inside_beta_signal(float(beta))

    if function_name == "constant":
        if gamma is None or pd.isna(gamma):
            raise ValueError("constant requires gamma.")
        return make_constant_signal(float(gamma))

    raise ValueError(f"Unknown function: {function_name}")


def mean_signal_strength(train_df, signal_fn, alpha):
    signal = signal_fn(train_df, alpha)
    return float(np.mean(signal))
