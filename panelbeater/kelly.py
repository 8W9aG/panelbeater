"""Kelly functions."""

# pylint: disable=too-many-locals,too-many-branches,line-too-long
import numpy as np


def calculate_full_kelly_path_aware(row, sim_df):
    """
    Comprehensive Kelly sizing for options:
    Path-aware, variance-penalized.

    Returns:
        f_star: The Kelly fraction (0.0 to 1.0+)
        mean_r: The Expected Return Percentage (e.g., 0.15 for 15%)
    """
    entry_price = row["ask"]
    if entry_price <= 0:
        return 0.0, 0.0

    # Get current spot from the start of the simulation paths
    current_spot = sim_df.iloc[0, 0] if not sim_df.empty else 0

    intrinsic_value = 0.0
    if row["type"] == "call":
        intrinsic_value = max(0, current_spot - row["strike"])
    elif row["type"] == "put":
        intrinsic_value = max(0, row["strike"] - current_spot)

    # If the option costs less than the cash you can get for it right now,
    # it is a data error. This creates "risk-free" variance which breaks Kelly.
    if entry_price < intrinsic_value:
        # Log a warning if you have a logger, otherwise just return 0 leverage
        # e.g., print(f"Arb detected on {row['symbol']}: Cost {entry_price} < Intrinsic {intrinsic_value}")
        return 0.0, 0.0

    # Do not recalculate using calculate_distribution_exits(row, sim_df)
    tp_target = row["tp_target"]
    sl_target = row["sl_target"]

    # If the target is below entry, we know mathematically the Max Return is negative.
    if tp_target <= entry_price:
        return 0.0, 0.0  # Guaranteed loss on 'winning' paths -> Negative EV.

    # 2. Force path_matrix to be 2D (Rows: Time, Cols: Paths)
    path_matrix = np.atleast_2d(sim_df.values)

    # Ensure rows are time and columns are paths
    if path_matrix.shape[0] == 1 and len(sim_df.index) > 1:
        path_matrix = path_matrix.T

    path_outcomes = []
    num_paths = path_matrix.shape[1]

    # 3. Path-Aware Logic: Check for TP/SL hits before expiry
    for col in range(num_paths):
        single_path = path_matrix[:, col]

        # Calculate max/min payoffs reached during the life of the path
        if row["type"] == "call":
            best_payoff = np.max(single_path) - row["strike"]
            worst_payoff = np.min(single_path) - row["strike"]
        else:
            best_payoff = row["strike"] - np.min(single_path)
            worst_payoff = row["strike"] - np.max(single_path)

        # Outcome mapping based on dynamic targets
        if best_payoff >= tp_target:
            path_outcomes.append((tp_target - entry_price) / entry_price)
        elif worst_payoff <= sl_target:
            path_outcomes.append((sl_target - entry_price) / entry_price)
        else:
            # No hit: use the terminal payoff at expiration
            terminal_price = single_path[-1]
            if row["type"] == "call":
                terminal_payoff = max(0, terminal_price - row["strike"])
            else:
                terminal_payoff = max(0, row["strike"] - terminal_price)

            path_outcomes.append((terminal_payoff - entry_price) / entry_price)

    # 4. Variance-Aware Kelly Calculation
    path_returns = np.array(path_outcomes)
    mean_r = np.mean(path_returns)
    var_r = np.var(path_returns)

    # Pure Kelly: f* = E[r] / Var(r)
    if var_r > 0 and mean_r > 0:
        f_star = mean_r / var_r
    else:
        f_star = 0.0

    # SAFETY CAP: Never leverage more than X (e.g., 5.0) on a single option
    # regardless of how good the math looks.
    f_star = min(f_star, 5.0)

    # CHANGED: Return mean_r directly (e.g., 0.36 for 36%)
    # instead of mean_r * entry_price
    return f_star, mean_r
