from app.trading.paper_wallet import PaperWallet

def run():
    wallet = PaperWallet()

    print("Initial balance:", wallet.balances)

    # BUY 0.001 BTC @ 50,000 USDT
    wallet.buy("BTC/USDT", price=50000, quantity=0.001)
    print("After BUY:", wallet.balances)

    # SELL 0.001 BTC @ 51,000 USDT
    wallet.sell("BTC/USDT", price=51000, quantity=0.001)
    print("After SELL:", wallet.balances)

    pnl = wallet.total_pnl({"BTC": 51000})
    print("Total PnL:", pnl)

    print("Trades:")
    for trade in wallet.trades:
        print(vars(trade))

if __name__ == "__main__":
    run()
