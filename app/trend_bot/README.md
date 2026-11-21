# Trend bot

An automated execution of the daily BTC trend rule from `HYP-DAILY-TREND-BTC-001` (`research/hypotheses/daily_trend_btc_v1.md`). Every day after the 00:00 UTC close, the bot holds 0, 1/3, 2/3 or all of its capital in BTC, according to how many of the 50/100/200-day moving averages the close is above.

**What to expect:** the rule trades a few times per month. Its documented benefit is avoiding large drawdowns, not fast profits. In-sample (2018–2026) it beat buy-and-hold on risk-adjusted terms, but that history is public knowledge; the clean forward test runs until 2028-11-21. Nothing here guarantees profit.

## Safety design

- **Spot only:** long or flat. No leverage, no futures, no shorting.
- **Capped sleeve:** the bot tracks its own USDT/BTC sleeve, starting from `TREND_BOT_CAPITAL_USDT`, and never trades more than that sleeve. Other funds in the account are not touched.
- **Key checks (live):** the bot refuses keys that can withdraw, and refuses to run if the account does not hold the sleeve.
- **One run per daily candle:** a rerun on the same day does nothing. Orders carry deterministic client IDs.
- **Kill switch:** create the file `STOP` in the state directory and the bot stops trading.
- **Records:** `state.json` plus an append-only `ledger.jsonl`. Optional Telegram messages after every run and on errors.

## Stage 1: paper (running now)

GitHub Actions runs `.github/workflows/trend-bot-paper.yml` daily at 00:10 UTC, with a 1,000 USDT simulated sleeve. It uses public prices only and needs no keys. The results are committed to `research/paper/trend_bot_v1/`.

Let it run 2–4 weeks and check that its targets and trades match the rule.

## Stage 2: server (outside the US)

Binance blocks trading from US IPs, so live mode needs a server elsewhere, for example a small Ubuntu VPS in Europe (about 5 USD per month).

1. Create an Ubuntu 24.04 server and log in as root.
2. Install:
   ```bash
   curl -fsSL https://raw.githubusercontent.com/joelramone/crypto-scalping-core/main/deploy/trend_bot_vps_setup.sh | bash
   ```
   This installs the bot in **paper** mode with a daily systemd timer and prints the server's public IP.
3. Check a run: `systemctl start trend-bot.service && journalctl -u trend-bot.service -n 30`.

## Stage 3: live, small capital

1. In Binance, create a **new** API key:
   - enable **Spot & Margin Trading** only;
   - leave **withdrawals disabled**;
   - restrict access to the server IP printed by the installer.
2. Move exactly the capital you want the bot to manage into Spot as USDT. Start small, with an amount you can afford to lose entirely.
3. Edit `/etc/trend-bot.env`:
   ```
   TREND_BOT_MODE=live
   TREND_BOT_CAPITAL_USDT=<your cap, e.g. 100>
   TREND_BOT_STATE_DIR=/home/trendbot/.trend_bot_live
   BINANCE_API_KEY=...
   BINANCE_API_SECRET=...
   ```
4. Run once by hand and read the output: `systemctl start trend-bot.service && journalctl -u trend-bot.service -n 30`.

To stop it: `touch /home/trendbot/.trend_bot_live/STOP`, or `systemctl disable --now trend-bot.timer`.

## Notes

- **Fees:** if BNB pays your fees, keep some BNB in the account. The sleeve only tracks USDT and BTC.
- **Missed days:** if the server is down at 00:05 UTC, `Persistent=true` runs the bot when it comes back, using the latest closed candle.
- **Rule changes:** never change the rule on the live bot. Test changes as new hypotheses first.
