from __future__ import annotations


def compute_r_multiple(side: str, entry_price: float, exit_price: float, stop_price: float) -> float:
    """Compute normalized R-multiple for a finished trade.

    LONG:  R = (exit - entry) / (entry - stop)
    SHORT: R = (entry - exit) / (stop - entry)
    """
    normalized_side = side.upper()

    if normalized_side == "LONG":
        denominator = entry_price - stop_price
        if denominator <= 0:
            raise ValueError("Invalid LONG risk definition: entry_price must be > stop_price")
        return (exit_price - entry_price) / denominator

    if normalized_side == "SHORT":
        denominator = stop_price - entry_price
        if denominator <= 0:
            raise ValueError("Invalid SHORT risk definition: stop_price must be > entry_price")
        return (entry_price - exit_price) / denominator

    raise ValueError(f"Unsupported side: {side}")
