#!/usr/bin/env bash
# Install the trend bot on a fresh Ubuntu 22.04/24.04 server (outside the US).
# Run as root:  curl -fsSL https://raw.githubusercontent.com/joelramone/crypto-scalping-core/main/deploy/trend_bot_vps_setup.sh | bash
#
# It installs the code and a systemd timer (00:05 UTC daily) in PAPER mode.
# Going live is a separate, explicit step: see app/trend_bot/README.md.
set -euo pipefail

REPO=https://github.com/joelramone/crypto-scalping-core.git
APP_USER=trendbot
APP_DIR=/opt/trend-bot
ENV_FILE=/etc/trend-bot.env

apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq git python3 python3-venv >/dev/null
id -u "$APP_USER" >/dev/null 2>&1 || useradd --system --create-home --shell /usr/sbin/nologin "$APP_USER"

if [ -d "$APP_DIR/.git" ]; then
  git -C "$APP_DIR" pull -q --ff-only
else
  git clone -q "$REPO" "$APP_DIR"
fi
python3 -m venv "$APP_DIR/.venv"
"$APP_DIR/.venv/bin/pip" install -q pandas==2.2.3 numpy requests pydantic pytest
chown -R "$APP_USER":"$APP_USER" "$APP_DIR"
sudo -u "$APP_USER" bash -c "cd $APP_DIR && .venv/bin/python -m pytest -q -p no:cacheprovider app/trend_bot/test_bot.py"

if [ ! -f "$ENV_FILE" ]; then
  cat > "$ENV_FILE" <<'EOF'
# Trend bot configuration. Keep this file private (chmod 600).
TREND_BOT_MODE=paper
TREND_BOT_CAPITAL_USDT=1000
TREND_BOT_STATE_DIR=/home/trendbot/.trend_bot_paper
# For live mode (see README): a NEW key WITHOUT withdrawals, restricted to this server's IP.
BINANCE_API_KEY=
BINANCE_API_SECRET=
# Optional Telegram notifications
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
EOF
fi
chown root:"$APP_USER" "$ENV_FILE"
chmod 640 "$ENV_FILE"

cat > /etc/systemd/system/trend-bot.service <<EOF
[Unit]
Description=Daily BTC trend bot
Wants=network-online.target
After=network-online.target

[Service]
Type=oneshot
User=$APP_USER
WorkingDirectory=$APP_DIR
EnvironmentFile=$ENV_FILE
ExecStart=$APP_DIR/.venv/bin/python -m app.trend_bot.bot
EOF

cat > /etc/systemd/system/trend-bot.timer <<'EOF'
[Unit]
Description=Run the BTC trend bot daily after the 00:00 UTC close

[Timer]
OnCalendar=*-*-* 00:05:00 UTC
Persistent=true
RandomizedDelaySec=60

[Install]
WantedBy=timers.target
EOF

systemctl daemon-reload
systemctl enable --now trend-bot.timer
echo "Installed. Public IP for the Binance key restriction: $(curl -fsS https://api.ipify.org || echo unknown)"
systemctl list-timers trend-bot.timer --no-pager
