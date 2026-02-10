MODE_500_USD = {
    "initial_balance": 250.0,
    "trade_size_usdt": 50.0,
    "max_trades_per_day": 10,
    "target_daily_profit_usdt": 2.0,
    "max_daily_loss_usdt": 1.0,
    "buy_threshold_pct": -0.3,   # buy after -0.3% move
    "sell_threshold_pct": 0.3,   # sell after +0.3% move
    "stop_loss_pct": 0.5,        # stop loss at -0.5% from entry
    "take_profit_pct": 1.0,      # take profit at +1.0% from entry
    "fee_rate": 0.001
}
