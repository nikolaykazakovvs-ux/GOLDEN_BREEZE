"""
Prop Guardian - Risk Management Core
Защита от нарушения правил проп-компаний

Автоматически:
- Определяет тир по балансу
- Отслеживает дневной убыток
- Блокирует торговлю при достижении лимитов
- Рассчитывает безопасный размер позиции
"""

import logging
from datetime import datetime, timezone, date
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

from aimodule.config_prop_rules import (
    TierRules,
    AccountType,
    detect_tier,
    get_tier_rules,
    PROP_FIRMS
)

logger = logging.getLogger(__name__)


class RiskStatus(Enum):
    """Статус риск-проверки"""
    OK = "ok"                           # Можно торговать
    WARNING = "warning"                 # Близко к лимиту
    DAILY_LIMIT_HIT = "daily_limit"     # Дневной лимит достигнут
    DRAWDOWN_LIMIT_HIT = "drawdown"     # Общая просадка достигнута
    MAX_POSITIONS = "max_positions"     # Максимум позиций
    WEEKEND_BLOCKED = "weekend"         # Блокировка на выходных


@dataclass
class RiskCheckResult:
    """Результат проверки риска"""
    allowed: bool
    status: RiskStatus
    message: str
    
    # Детали
    daily_pnl: float = 0.0
    daily_limit: float = 0.0
    daily_remaining: float = 0.0
    
    equity: float = 0.0
    drawdown_limit: float = 0.0
    drawdown_remaining: float = 0.0
    
    current_positions: int = 0
    max_positions: int = 0


@dataclass
class TradingSession:
    """Информация о торговой сессии"""
    date: date
    starting_equity: float
    current_equity: float
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    trades_count: int = 0
    
    @property
    def total_pnl(self) -> float:
        return self.realized_pnl + self.unrealized_pnl


class PropGuardian:
    """
    🛡️ Prop Guardian - Защита аккаунта проп-компании
    
    Основные функции:
    1. Автоматическое определение тира по балансу
    2. Расчёт лимитов (дневной убыток, общая просадка)
    3. Проверка разрешения на торговлю
    4. Расчёт безопасного размера позиции
    5. Логирование всех действий
    """
    
    def __init__(
        self,
        initial_balance: float,
        firm: str = "traders_mastery",
        account_type: Optional[AccountType] = None,
        high_water_mark: Optional[float] = None
    ):
        """
        Args:
            initial_balance: Начальный баланс аккаунта
            firm: Название проп-компании
            account_type: Тип аккаунта (если известен)
            high_water_mark: Максимальный баланс (для trailing drawdown)
        """
        self.firm = firm
        self.initial_balance = initial_balance
        self.high_water_mark = high_water_mark or initial_balance
        
        # Определяем тир и правила
        self.tier_amount, self.rules = detect_tier(initial_balance, firm)
        
        # Если тип аккаунта указан явно - получаем правила для него
        if account_type:
            specific_rules = get_tier_rules(firm, account_type, initial_balance)
            if specific_rules:
                self.rules = specific_rules
        
        # Рассчитываем абсолютные лимиты
        self._calculate_limits()
        
        # Текущая сессия
        self.session: Optional[TradingSession] = None
        
        # Статистика
        self.blocked_trades: int = 0
        self.warnings_issued: int = 0
        
        logger.info(f"🛡️ PropGuardian initialized:")
        logger.info(f"   Firm: {firm}")
        logger.info(f"   Tier: {self.rules.tier_name}")
        logger.info(f"   Balance: ${initial_balance:,.2f}")
        logger.info(f"   Daily Loss Limit: ${self.daily_loss_limit:,.2f} ({self.rules.max_daily_loss_percent}%)")
        logger.info(f"   Max Drawdown: ${self.total_drawdown_limit:,.2f} ({self.rules.max_total_drawdown_percent}%)")
    
    def _calculate_limits(self):
        """Расчёт абсолютных лимитов"""
        # Дневной лимит убытка (от начального баланса)
        self.daily_loss_limit = self.initial_balance * (self.rules.max_daily_loss_percent / 100)
        
        # Общий лимит просадки (от high water mark)
        self.total_drawdown_limit = self.high_water_mark * (self.rules.max_total_drawdown_percent / 100)
        
        # Минимальный допустимый эквити
        self.min_equity = self.high_water_mark - self.total_drawdown_limit
    
    def start_session(self, current_equity: float):
        """
        Начало новой торговой сессии (дня)
        
        Args:
            current_equity: Текущий эквити
        """
        today = date.today()
        
        # Если сессия уже существует для сегодня - не пересоздаём
        if self.session and self.session.date == today:
            return
        
        self.session = TradingSession(
            date=today,
            starting_equity=current_equity,
            current_equity=current_equity
        )
        
        # Обновляем high water mark если эквити вырос
        if current_equity > self.high_water_mark:
            self.high_water_mark = current_equity
            self._calculate_limits()
            logger.info(f"📈 New High Water Mark: ${current_equity:,.2f}")
        
        logger.info(f"📅 New trading session started: {today}")
        logger.info(f"   Starting equity: ${current_equity:,.2f}")
        logger.info(f"   Daily loss limit: ${self.daily_loss_limit:,.2f}")
    
    def update_equity(self, current_equity: float, realized_pnl: float = 0.0):
        """
        Обновление текущего эквити
        
        Args:
            current_equity: Текущий эквити
            realized_pnl: Реализованный P&L за сегодня
        """
        if not self.session:
            self.start_session(current_equity)
        
        self.session.current_equity = current_equity
        self.session.realized_pnl = realized_pnl
        self.session.unrealized_pnl = current_equity - self.session.starting_equity - realized_pnl
    
    def check_trade_allowance(
        self,
        current_equity: float,
        current_daily_pnl: float,
        current_positions: int = 0
    ) -> RiskCheckResult:
        """
        🔍 Проверка разрешения на торговлю
        
        Args:
            current_equity: Текущий эквити
            current_daily_pnl: Текущий дневной P&L (может быть отрицательным)
            current_positions: Количество открытых позиций
            
        Returns:
            RiskCheckResult с результатом проверки
        """
        # Обновляем сессию
        if not self.session:
            self.start_session(current_equity)
        
        # Базовый результат
        result = RiskCheckResult(
            allowed=True,
            status=RiskStatus.OK,
            message="Trading allowed",
            daily_pnl=current_daily_pnl,
            daily_limit=self.daily_loss_limit,
            daily_remaining=self.daily_loss_limit + current_daily_pnl,
            equity=current_equity,
            drawdown_limit=self.total_drawdown_limit,
            drawdown_remaining=current_equity - self.min_equity,
            current_positions=current_positions,
            max_positions=self.rules.max_positions
        )
        
        # ❌ Проверка 1: Дневной лимит убытка
        if current_daily_pnl <= -self.daily_loss_limit:
            result.allowed = False
            result.status = RiskStatus.DAILY_LIMIT_HIT
            result.message = f"🚫 DAILY LOSS LIMIT HIT: ${current_daily_pnl:,.2f} <= -${self.daily_loss_limit:,.2f}"
            self.blocked_trades += 1
            logger.error(result.message)
            return result
        
        # ❌ Проверка 2: Общая просадка
        if current_equity <= self.min_equity:
            result.allowed = False
            result.status = RiskStatus.DRAWDOWN_LIMIT_HIT
            result.message = f"🚫 DRAWDOWN LIMIT HIT: Equity ${current_equity:,.2f} <= Min ${self.min_equity:,.2f}"
            self.blocked_trades += 1
            logger.error(result.message)
            return result
        
        # ❌ Проверка 3: Максимум позиций
        if current_positions >= self.rules.max_positions:
            result.allowed = False
            result.status = RiskStatus.MAX_POSITIONS
            result.message = f"🚫 MAX POSITIONS: {current_positions} >= {self.rules.max_positions}"
            self.blocked_trades += 1
            logger.warning(result.message)
            return result
        
        # ❌ Проверка 4: Выходные (если запрещено)
        if not self.rules.weekend_holding:
            now = datetime.now(timezone.utc)
            # Пятница после 21:00 UTC или суббота/воскресенье
            if now.weekday() == 4 and now.hour >= 21:
                result.allowed = False
                result.status = RiskStatus.WEEKEND_BLOCKED
                result.message = "🚫 WEEKEND HOLDING NOT ALLOWED - Close positions before weekend"
                logger.warning(result.message)
                return result
            if now.weekday() in [5, 6]:
                result.allowed = False
                result.status = RiskStatus.WEEKEND_BLOCKED
                result.message = "🚫 WEEKEND TRADING NOT ALLOWED"
                return result
        
        # ⚠️ Предупреждение: близко к дневному лимиту (80%+)
        daily_usage = abs(current_daily_pnl) / self.daily_loss_limit if self.daily_loss_limit > 0 else 0
        if daily_usage >= 0.8 and current_daily_pnl < 0:
            result.status = RiskStatus.WARNING
            result.message = f"⚠️ WARNING: {daily_usage*100:.1f}% of daily loss limit used"
            self.warnings_issued += 1
            logger.warning(result.message)
        
        # ⚠️ Предупреждение: близко к общей просадке (80%+)
        drawdown_usage = (self.high_water_mark - current_equity) / self.total_drawdown_limit if self.total_drawdown_limit > 0 else 0
        if drawdown_usage >= 0.8:
            result.status = RiskStatus.WARNING
            result.message = f"⚠️ WARNING: {drawdown_usage*100:.1f}% of max drawdown used"
            self.warnings_issued += 1
            logger.warning(result.message)
        
        return result
    
    def get_safe_lot_size(
        self,
        risk_amount: float,
        stop_loss_pips: float,
        pip_value: float = 10.0,  # $10 per pip per lot for XAUUSD
        symbol: str = "XAUUSD"
    ) -> float:
        """
        💰 Расчёт безопасного размера лота
        
        Args:
            risk_amount: Сумма риска в долларах
            stop_loss_pips: Размер стоп-лосса в пипсах
            pip_value: Стоимость пипса на 1 лот
            symbol: Торговый символ
            
        Returns:
            Размер лота (округлённый до 0.01)
        """
        if stop_loss_pips <= 0:
            logger.error("Stop loss pips must be positive")
            return 0.0
        
        # Базовый расчёт лота
        lot_size = risk_amount / (stop_loss_pips * pip_value)
        
        # Ограничения по правилам
        lot_size = min(lot_size, self.rules.max_lot_size)
        
        # Ограничение по плечу
        # Примерный расчёт: 1 лот XAUUSD ≈ $100,000 контракт
        contract_value = 100_000  # Для золота
        max_lot_by_leverage = (self.initial_balance * self.rules.max_leverage) / contract_value
        lot_size = min(lot_size, max_lot_by_leverage)
        
        # Округляем до 0.01
        lot_size = round(lot_size, 2)
        
        # Минимум 0.01
        lot_size = max(lot_size, 0.01)
        
        logger.info(f"📊 Safe lot size: {lot_size} (risk=${risk_amount}, SL={stop_loss_pips} pips)")
        
        return lot_size
    
    def get_risk_amount(
        self,
        risk_percent: float = 1.0,
        use_remaining_daily: bool = True
    ) -> float:
        """
        Расчёт суммы риска на сделку
        
        Args:
            risk_percent: Процент риска от баланса
            use_remaining_daily: Учитывать оставшийся дневной лимит
            
        Returns:
            Сумма риска в долларах
        """
        # Базовый риск от баланса
        base_risk = self.initial_balance * (risk_percent / 100)
        
        if use_remaining_daily and self.session:
            # Не рисковать больше оставшегося дневного лимита
            daily_pnl = self.session.total_pnl
            remaining = self.daily_loss_limit + daily_pnl  # daily_pnl может быть отрицательным
            
            if remaining < base_risk:
                logger.warning(f"⚠️ Reducing risk to remaining daily: ${remaining:.2f}")
                return max(remaining, 0)
        
        return base_risk
    
    def get_status_report(self) -> dict:
        """Получение отчёта о текущем статусе"""
        session_info = {}
        if self.session:
            session_info = {
                "date": str(self.session.date),
                "starting_equity": self.session.starting_equity,
                "current_equity": self.session.current_equity,
                "realized_pnl": self.session.realized_pnl,
                "unrealized_pnl": self.session.unrealized_pnl,
                "total_pnl": self.session.total_pnl,
                "trades_count": self.session.trades_count
            }
        
        return {
            "firm": self.firm,
            "tier": self.rules.tier_name,
            "initial_balance": self.initial_balance,
            "high_water_mark": self.high_water_mark,
            "daily_loss_limit": self.daily_loss_limit,
            "total_drawdown_limit": self.total_drawdown_limit,
            "min_equity": self.min_equity,
            "max_positions": self.rules.max_positions,
            "max_lot_size": self.rules.max_lot_size,
            "max_leverage": self.rules.max_leverage,
            "blocked_trades": self.blocked_trades,
            "warnings_issued": self.warnings_issued,
            "session": session_info
        }
    
    def __repr__(self) -> str:
        return (
            f"PropGuardian("
            f"tier={self.rules.tier_name}, "
            f"balance=${self.initial_balance:,.2f}, "
            f"daily_limit=${self.daily_loss_limit:,.2f})"
        )


class RiskError(Exception):
    """Исключение при нарушении риск-лимитов"""
    
    def __init__(self, message: str, result: RiskCheckResult):
        super().__init__(message)
        self.result = result
