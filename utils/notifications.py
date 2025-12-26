"""
Notifications - Alertes Trading
Discord Webhooks + Email (optionnel)
"""
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass
import urllib.request
import urllib.error

from config.settings import DISCORD_WEBHOOK_URL

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TradeAlert:
    """Alerte de trade"""
    symbol: str
    action: str  # BUY, SELL, CLOSE
    price: float
    quantity: int
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    pnl: Optional[float] = None
    reason: Optional[str] = None


@dataclass
class SignalAlert:
    """Alerte de signal"""
    symbol: str
    direction: str  # buy, sell
    strength: float
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward: float
    reasons: List[str] = None


@dataclass
class DailySummary:
    """Resume quotidien"""
    date: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_pnl: float
    portfolio_value: float
    open_positions: int


# =============================================================================
# DISCORD NOTIFIER
# =============================================================================

class DiscordNotifier:
    """
    Envoi de notifications via Discord Webhook

    Gratuit et simple - pas besoin de bot
    """

    def __init__(self, webhook_url: str = None):
        self.webhook_url = webhook_url or DISCORD_WEBHOOK_URL
        self.enabled = bool(self.webhook_url)

        if not self.enabled:
            logger.warning("Discord webhook not configured. Notifications disabled.")

    def _send_webhook(self, payload: Dict) -> bool:
        """
        Envoie un message via webhook Discord
        """
        if not self.enabled:
            return False

        try:
            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(
                self.webhook_url,
                data=data,
                headers={'Content-Type': 'application/json'}
            )

            with urllib.request.urlopen(req, timeout=10) as response:
                return response.status == 204

        except urllib.error.HTTPError as e:
            logger.error(f"Discord webhook HTTP error: {e.code}")
            return False
        except urllib.error.URLError as e:
            logger.error(f"Discord webhook URL error: {e.reason}")
            return False
        except Exception as e:
            logger.error(f"Discord webhook error: {e}")
            return False

    def send_trade_alert(self, alert: TradeAlert) -> bool:
        """
        Envoie une alerte de trade
        """
        # Emoji selon action
        emoji_map = {
            'BUY': ':green_circle:',
            'SELL': ':red_circle:',
            'CLOSE': ':white_circle:',
            'TP1': ':moneybag:',
            'STOP_LOSS': ':octagonal_sign:'
        }
        emoji = emoji_map.get(alert.action, ':black_circle:')

        # Couleur selon action
        color_map = {
            'BUY': 0x00FF00,    # Vert
            'SELL': 0xFF0000,   # Rouge
            'CLOSE': 0x808080,  # Gris
            'TP1': 0xFFD700,    # Or
            'STOP_LOSS': 0xFF4500  # Rouge orange
        }
        color = color_map.get(alert.action, 0x000000)

        # Construire embed
        fields = [
            {"name": "Symbol", "value": alert.symbol, "inline": True},
            {"name": "Prix", "value": f"${alert.price:.2f}", "inline": True},
            {"name": "Quantite", "value": str(alert.quantity), "inline": True}
        ]

        if alert.stop_loss:
            fields.append({"name": "Stop Loss", "value": f"${alert.stop_loss:.2f}", "inline": True})

        if alert.take_profit:
            fields.append({"name": "Take Profit", "value": f"${alert.take_profit:.2f}", "inline": True})

        if alert.pnl is not None:
            pnl_str = f"+${alert.pnl:.2f}" if alert.pnl >= 0 else f"-${abs(alert.pnl):.2f}"
            pnl_emoji = ":chart_with_upwards_trend:" if alert.pnl >= 0 else ":chart_with_downwards_trend:"
            fields.append({"name": f"P&L {pnl_emoji}", "value": pnl_str, "inline": True})

        if alert.reason:
            fields.append({"name": "Raison", "value": alert.reason, "inline": False})

        embed = {
            "title": f"{emoji} {alert.action} - {alert.symbol}",
            "color": color,
            "fields": fields,
            "footer": {"text": "Trading Bot"},
            "timestamp": datetime.utcnow().isoformat()
        }

        payload = {"embeds": [embed]}
        return self._send_webhook(payload)

    def send_signal_alert(self, alert: SignalAlert) -> bool:
        """
        Envoie une alerte de signal detecte
        """
        emoji = ":arrow_up:" if alert.direction == 'buy' else ":arrow_down:"
        color = 0x00FF00 if alert.direction == 'buy' else 0xFF0000

        # Strength indicator
        strength_bar = "█" * int(alert.strength * 10) + "░" * (10 - int(alert.strength * 10))

        fields = [
            {"name": "Direction", "value": alert.direction.upper(), "inline": True},
            {"name": "Force", "value": f"`{strength_bar}` ({alert.strength:.0%})", "inline": True},
            {"name": "R:R", "value": f"1:{alert.risk_reward:.1f}", "inline": True},
            {"name": "Entry", "value": f"${alert.entry_price:.2f}", "inline": True},
            {"name": "Stop Loss", "value": f"${alert.stop_loss:.2f}", "inline": True},
            {"name": "Take Profit", "value": f"${alert.take_profit:.2f}", "inline": True}
        ]

        if alert.reasons:
            reasons_str = "\n".join([f"• {r}" for r in alert.reasons])
            fields.append({"name": "Raisons", "value": reasons_str, "inline": False})

        embed = {
            "title": f"{emoji} SIGNAL DETECTE - {alert.symbol}",
            "color": color,
            "fields": fields,
            "footer": {"text": "Trading Bot - Signal Alert"},
            "timestamp": datetime.utcnow().isoformat()
        }

        payload = {"embeds": [embed]}
        return self._send_webhook(payload)

    def send_daily_summary(self, summary: DailySummary) -> bool:
        """
        Envoie le resume quotidien
        """
        # Calculer win rate
        win_rate = (summary.winning_trades / summary.total_trades * 100) if summary.total_trades > 0 else 0

        # Emoji selon P&L
        pnl_emoji = ":trophy:" if summary.total_pnl >= 0 else ":disappointed:"

        # Couleur selon P&L
        color = 0x00FF00 if summary.total_pnl >= 0 else 0xFF0000

        pnl_str = f"+${summary.total_pnl:.2f}" if summary.total_pnl >= 0 else f"-${abs(summary.total_pnl):.2f}"

        fields = [
            {"name": ":chart_with_upwards_trend: Trades", "value": str(summary.total_trades), "inline": True},
            {"name": ":white_check_mark: Gagnants", "value": str(summary.winning_trades), "inline": True},
            {"name": ":x: Perdants", "value": str(summary.losing_trades), "inline": True},
            {"name": ":dart: Win Rate", "value": f"{win_rate:.1f}%", "inline": True},
            {"name": f"{pnl_emoji} P&L Jour", "value": pnl_str, "inline": True},
            {"name": ":bank: Portfolio", "value": f"${summary.portfolio_value:,.2f}", "inline": True},
            {"name": ":briefcase: Positions", "value": str(summary.open_positions), "inline": True}
        ]

        embed = {
            "title": f":calendar: Resume du {summary.date}",
            "color": color,
            "fields": fields,
            "footer": {"text": "Trading Bot - Daily Summary"},
            "timestamp": datetime.utcnow().isoformat()
        }

        payload = {"embeds": [embed]}
        return self._send_webhook(payload)

    def send_error_alert(self, error: str, context: str = "") -> bool:
        """
        Envoie une alerte d'erreur
        """
        embed = {
            "title": ":warning: ERREUR TRADING BOT",
            "color": 0xFF0000,
            "fields": [
                {"name": "Contexte", "value": context or "N/A", "inline": False},
                {"name": "Erreur", "value": f"```{error[:1000]}```", "inline": False}
            ],
            "footer": {"text": "Trading Bot - Error Alert"},
            "timestamp": datetime.utcnow().isoformat()
        }

        payload = {"embeds": [embed]}
        return self._send_webhook(payload)

    def send_startup_message(self) -> bool:
        """
        Envoie un message au demarrage
        """
        embed = {
            "title": ":rocket: Trading Bot Demarre",
            "color": 0x00BFFF,
            "description": "Le bot de trading a demarre avec succes.",
            "fields": [
                {"name": "Mode", "value": "Paper Trading / Live Trading", "inline": True},
                {"name": "Strategie", "value": "Swing Trading Hybride", "inline": True}
            ],
            "footer": {"text": "Trading Bot"},
            "timestamp": datetime.utcnow().isoformat()
        }

        payload = {"embeds": [embed]}
        return self._send_webhook(payload)

    def send_simple_message(self, message: str, title: str = "Trading Bot") -> bool:
        """
        Envoie un message simple
        """
        embed = {
            "title": title,
            "description": message,
            "color": 0x808080,
            "timestamp": datetime.utcnow().isoformat()
        }

        payload = {"embeds": [embed]}
        return self._send_webhook(payload)


# =============================================================================
# NOTIFICATION MANAGER
# =============================================================================

class NotificationManager:
    """
    Gestionnaire centralise des notifications

    Gere plusieurs canaux: Discord, Email (futur), etc.
    """

    def __init__(self):
        self.discord = DiscordNotifier()
        self.enabled = True

    def toggle(self, enabled: bool):
        """Active/desactive les notifications"""
        self.enabled = enabled

    def notify_trade(
        self,
        symbol: str,
        action: str,
        price: float,
        quantity: int,
        **kwargs
    ):
        """Notifie un trade"""
        if not self.enabled:
            return

        alert = TradeAlert(
            symbol=symbol,
            action=action,
            price=price,
            quantity=quantity,
            stop_loss=kwargs.get('stop_loss'),
            take_profit=kwargs.get('take_profit'),
            pnl=kwargs.get('pnl'),
            reason=kwargs.get('reason')
        )

        self.discord.send_trade_alert(alert)

    def notify_signal(
        self,
        symbol: str,
        direction: str,
        strength: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        risk_reward: float,
        reasons: List[str] = None
    ):
        """Notifie un signal"""
        if not self.enabled:
            return

        alert = SignalAlert(
            symbol=symbol,
            direction=direction,
            strength=strength,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward=risk_reward,
            reasons=reasons
        )

        self.discord.send_signal_alert(alert)

    def notify_daily_summary(
        self,
        date: str,
        total_trades: int,
        winning_trades: int,
        losing_trades: int,
        total_pnl: float,
        portfolio_value: float,
        open_positions: int
    ):
        """Notifie le resume quotidien"""
        if not self.enabled:
            return

        summary = DailySummary(
            date=date,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            total_pnl=total_pnl,
            portfolio_value=portfolio_value,
            open_positions=open_positions
        )

        self.discord.send_daily_summary(summary)

    def notify_error(self, error: str, context: str = ""):
        """Notifie une erreur"""
        if not self.enabled:
            return

        self.discord.send_error_alert(error, context)

    def notify_startup(self):
        """Notifie le demarrage"""
        if not self.enabled:
            return

        self.discord.send_startup_message()


# =============================================================================
# SINGLETON
# =============================================================================

_notification_manager = None


def get_notification_manager() -> NotificationManager:
    """Retourne l'instance singleton"""
    global _notification_manager
    if _notification_manager is None:
        _notification_manager = NotificationManager()
    return _notification_manager
