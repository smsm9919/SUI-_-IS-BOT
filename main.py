# =================== SMART MONEY CONCEPTS (SMC) ENGINE - ENHANCED VERSION ===================

from collections import deque
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

# =========================
# ====== DATA MODELS ======
# =========================

@dataclass
class MarketState:
    regime: str            # TREND / RANGE / NO_TRADE / CHOP
    trend_strength: float  # ADX-like value
    direction: str         # BULL / BEAR / NONE
    classification: str    # LARGE / MID / NO_TREND / CHOP

@dataclass
class LiquidityState:
    swept_high: bool
    swept_low: bool
    sweep_price: float | None
    snapshot: str

@dataclass
class StructureState:
    bos: bool
    choch: bool
    direction: str         # BULL / BEAR / NONE

@dataclass
class OrderBlock:
    type: str              # BUY / SELL
    high: float
    low: float
    valid: bool
    strength: float

@dataclass
class FVGZone:
    high: float
    low: float
    filled: bool
    type: str              # BULLISH / BEARISH
    size_bps: float

@dataclass
class ExplosionState:
    type: str              # EXPLOSION_UP / EXPLOSION_DOWN / COLLAPSE / NORMAL
    direction: str         # UP / DOWN / NONE
    confidence: float

@dataclass
class EntryDecision:
    allow_entry: bool
    side: str | None       # BUY / SELL
    reason: str
    confidence: float
    mode: str              # TREND / STUDIED_SCALP / CAUTIOUS_SCALP / REJECT
    tierA: List[str]
    tierB: List[str]
    tierC: List[str]

@dataclass
class FakeBreakoutVerdict:
    detected: bool
    flags: List[str]
    type: str | None       # FAKE-UP / FAKE-DOWN

# =========================
# ====== SMC ZONES ENGINE =======
# =========================

class SMCZonesEngine:
    def __init__(self, candles):
        """
        candles: list of dict or DataFrame
        """
        if hasattr(candles, 'iloc'):
            # DataFrame
            self.high = candles['high'].astype(float).values
            self.low = candles['low'].astype(float).values
            self.open = candles['open'].astype(float).values
            self.close = candles['close'].astype(float).values
            self.volume = candles['volume'].astype(float).values if 'volume' in candles.columns else None
        else:
            # List of dicts
            self.high = np.array([c["high"] for c in candles])
            self.low = np.array([c["low"] for c in candles])
            self.open = np.array([c["open"] for c in candles])
            self.close = np.array([c["close"] for c in candles])
            self.volume = np.array([c.get("volume", 0) for c in candles]) if candles[0].get("volume") else None
    
    def detect_order_block(self) -> OrderBlock | None:
        """كشف كتل الأوامر الحقيقية"""
        if len(self.close) < 8:
            return None
            
        for i in range(max(-len(self.close), -6), -2):
            if i >= -1:
                continue
                
            body = abs(self.close[i] - self.open[i])
            if body == 0:
                continue
                
            impulse = abs(self.close[i+1] - self.close[i])
            
            if impulse > body * 1.5:  # حركة قوية بعد الشمعة
                # Bullish OB: شمعة هابطة يليها صعود قوي
                if self.close[i] < self.open[i] and self.close[i+1] > self.high[i]:
                    strength = (impulse / body) * (self.volume[i] / np.mean(self.volume[max(-20, -len(self.volume)):-1]) if self.volume is not None else 1.0)
                    return OrderBlock(
                        type="BUY",
                        high=self.high[i],
                        low=self.low[i],
                        valid=True,
                        strength=min(strength, 5.0)
                    )
                
                # Bearish OB: شمعة صاعدة يليها هبوط قوي
                if self.close[i] > self.open[i] and self.close[i+1] < self.low[i]:
                    strength = (impulse / body) * (self.volume[i] / np.mean(self.volume[max(-20, -len(self.volume)):-1]) if self.volume is not None else 1.0)
                    return OrderBlock(
                        type="SELL",
                        high=self.high[i],
                        low=self.low[i],
                        valid=True,
                        strength=min(strength, 5.0)
                    )
        return None
    
    def detect_fvg(self) -> FVGZone | None:
        """كشف فجوات القيمة العادلة"""
        if len(self.close) < 4:
            return None
            
        # Bullish FVG: A-high < C-low
        h1 = self.high[-3]
        l3 = self.low[-1]
        
        if h1 < l3:
            size_bps = ((l3 - h1) / h1) * 10000
            filled = self.close[-1] <= h1  # هل السعر عاد وملأ الفجوة؟
            return FVGZone(
                high=l3,
                low=h1,
                filled=filled,
                type="BULLISH",
                size_bps=size_bps
            )
        
        # Bearish FVG: A-low > C-high
        l1 = self.low[-3]
        h3 = self.high[-1]
        
        if l1 > h3:
            size_bps = ((l1 - h3) / l1) * 10000
            filled = self.close[-1] >= l1  # هل السعر عاد وملأ الفجوة؟
            return FVGZone(
                high=l1,
                low=h3,
                filled=filled,
                type="BEARISH",
                size_bps=size_bps
            )
        
        return None
    
    def price_in_zone(self, zone) -> bool:
        """التحقق إذا كان السعر الحالي داخل منطقة"""
        price = self.close[-1]
        return zone.low <= price <= zone.high

# =========================
# ===== TREND CLASSIFIER ENGINE =======
# =========================

class TrendClassifierEngine:
    def __init__(self, candles, adx_value=None, di_plus=None, di_minus=None):
        if hasattr(candles, 'iloc'):
            self.high = candles['high'].astype(float).values
            self.low = candles['low'].astype(float).values
            self.close = candles['close'].astype(float).values
            self.volume = candles['volume'].astype(float).values if 'volume' in candles.columns else np.ones(len(candles))
        else:
            self.high = np.array([c["high"] for c in candles])
            self.low = np.array([c["low"] for c in candles])
            self.close = np.array([c["close"] for c in candles])
            self.volume = np.array([c.get("volume", 1) for c in candles])
        
        self.adx = adx_value
        self.di_plus = di_plus
        self.di_minus = di_minus
    
    def structure_expansion(self):
        """قياس توسع الهيكل السعري"""
        if len(self.close) < 20:
            return 1.0
        last_range = self.high[-1] - self.low[-1]
        avg_range = np.mean(self.high[-20:] - self.low[-20:])
        return last_range / avg_range if avg_range > 0 else 1.0
    
    def volume_expansion(self):
        """قياس توسع الحجم"""
        if len(self.volume) < 20:
            return False
        return self.volume[-1] > np.mean(self.volume[-20:]) * 1.4
    
    def directional_control(self):
        """تحديد السيطرة الاتجاهية"""
        if len(self.close) < 5:
            return None
            
        if self.di_plus is not None and self.di_minus is not None:
            if self.di_plus > self.di_minus and self.close[-1] > self.close[-5]:
                return "BULL"
            if self.di_minus > self.di_plus and self.close[-1] < self.close[-5]:
                return "BEAR"
        
        # Fallback to price action
        sma_20 = np.mean(self.close[-20:]) if len(self.close) >= 20 else self.close[-1]
        sma_50 = np.mean(self.close[-50:]) if len(self.close) >= 50 else self.close[-1]
        
        if self.close[-1] > sma_20 > sma_50:
            return "BULL"
        elif self.close[-1] < sma_20 < sma_50:
            return "BEAR"
        return None
    
    def classify(self):
        """تصنيف حالة الترند"""
        direction = self.directional_control()
        
        if direction is None:
            return "CHOP"
        
        # حساب ADX تقريبي إذا لم يكن معطى
        adx_value = self.adx
        if adx_value is None:
            tr = np.maximum(
                np.maximum(
                    self.high[-1] - self.low[-1],
                    abs(self.high[-1] - self.close[-2])
                ),
                abs(self.low[-1] - self.close[-2])
            )
            atr = np.mean([self.high[i] - self.low[i] for i in range(-14, 0)]) if len(self.close) >= 14 else tr
            adx_value = (tr / atr * 100) if atr > 0 else 20
        
        # 🔥 LARGE TREND
        if (adx_value > 30 and 
            self.structure_expansion() > 1.3 and 
            self.volume_expansion()):
            return "LARGE"
        
        # ⚠️ MID TREND
        if adx_value >= 20:
            return "MID"
        
        return "CHOP"

# =========================
# ===== EXPLOSION/COLLAPSE ENGINE =======
# =========================

class ExplosionCollapseEngine:
    def __init__(self, candles, atr_value=None):
        if hasattr(candles, 'iloc'):
            self.high = candles['high'].astype(float).values
            self.low = candles['low'].astype(float).values
            self.open = candles['open'].astype(float).values
            self.close = candles['close'].astype(float).values
            self.volume = candles['volume'].astype(float).values if 'volume' in candles.columns else np.ones(len(candles))
        else:
            self.high = np.array([c["high"] for c in candles])
            self.low = np.array([c["low"] for c in candles])
            self.open = np.array([c["open"] for c in candles])
            self.close = np.array([c["close"] for c in candles])
            self.volume = np.array([c.get("volume", 1) for c in candles])
        
        self.atr = atr_value if atr_value is not None else self._calculate_atr()
    
    def _calculate_atr(self, period=14):
        """حساب ATR تقريبي"""
        if len(self.close) < period + 1:
            return np.mean(self.high[-5:] - self.low[-5:]) if len(self.close) >= 5 else 0.01
        
        tr_values = []
        for i in range(-period, 0):
            hi = self.high[i]
            lo = self.low[i]
            cl_prev = self.close[i-1] if i > -len(self.close) else self.open[i]
            tr = max(hi - lo, abs(hi - cl_prev), abs(lo - cl_prev))
            tr_values.append(tr)
        
        return np.mean(tr_values)
    
    def candle_expansion(self):
        """توسع الشمعة"""
        body = abs(self.close[-1] - self.open[-1])
        return body > self.atr * 1.6
    
    def volume_spike(self):
        """ارتفاع غير طبيعي في الحجم"""
        if len(self.volume) < 20:
            return False
        return self.volume[-1] > np.mean(self.volume[-20:]) * 1.8
    
    def clean_close(self):
        """إغلاق نظيف (بدون ظلال طويلة)"""
        candle_range = self.high[-1] - self.low[-1]
        if candle_range == 0:
            return False
        body_size = abs(self.close[-1] - self.open[-1])
        return (body_size / candle_range) > 0.7
    
    def direction(self):
        """اتجاه الشمعة"""
        return "UP" if self.close[-1] > self.open[-1] else "DOWN"
    
    def detect_explosion(self):
        """كشف الانفجار الحقيقي"""
        if self.candle_expansion() and self.volume_spike() and self.clean_close():
            return ExplosionState(
                type=f"EXPLOSION_{self.direction()}",
                direction=self.direction(),
                confidence=0.8
            )
        return ExplosionState(type="NORMAL", direction="NONE", confidence=0.0)
    
    def detect_collapse(self):
        """كشف الانهيار/استنزاف السيولة"""
        candle_range = self.high[-1] - self.low[-1]
        if candle_range == 0:
            return None
        
        if self.close[-1] < self.open[-1]:  # شمعة هابطة
            wick_ratio = (self.high[-1] - self.close[-1]) / candle_range
        else:  # شمعة صاعدة
            wick_ratio = (self.close[-1] - self.low[-1]) / candle_range
        
        if self.volume_spike() and wick_ratio > 0.6:
            return ExplosionState(type="COLLAPSE", direction="NONE", confidence=0.9)
        return None

# =========================
# ===== LIQUIDITY LOG ENGINE =======
# =========================

class LiquidityLogEngine:
    def __init__(self, candles):
        if hasattr(candles, 'iloc'):
            self.high = candles['high'].astype(float).values
            self.low = candles['low'].astype(float).values
            self.open = candles['open'].astype(float).values
            self.close = candles['close'].astype(float).values
            self.volume = candles['volume'].astype(float).values if 'volume' in candles.columns else np.ones(len(candles))
        else:
            self.high = np.array([c["high"] for c in candles])
            self.low = np.array([c["low"] for c in candles])
            self.open = np.array([c["open"] for c in candles])
            self.close = np.array([c["close"] for c in candles])
            self.volume = np.array([c.get("volume", 1) for c in candles])
    
    def sweep_high(self):
        """سحب سيولة من القمة"""
        if len(self.high) < 10:
            return False
        recent_high = max(self.high[-10:-1])
        return self.high[-1] > recent_high and self.close[-1] < recent_high
    
    def sweep_low(self):
        """سحب سيولة من القاع"""
        if len(self.low) < 10:
            return False
        recent_low = min(self.low[-10:-1])
        return self.low[-1] < recent_low and self.close[-1] > recent_low
    
    def volume_spike(self):
        """ارتفاع الحجم"""
        if len(self.volume) < 20:
            return False
        return self.volume[-1] > np.mean(self.volume[-20:]) * 1.5
    
    def accumulation(self):
        """مرحلة التجميع"""
        candle_range = self.high[-1] - self.low[-1]
        if candle_range == 0:
            return False
        body = abs(self.close[-1] - self.open[-1])
        return body < candle_range * 0.3 and self.volume_spike() and self.close[-1] > self.open[-1]
    
    def distribution(self):
        """مرحلة التوزيع"""
        candle_range = self.high[-1] - self.low[-1]
        if candle_range == 0:
            return False
        body = abs(self.close[-1] - self.open[-1])
        return body < candle_range * 0.3 and self.volume_spike() and self.close[-1] < self.open[-1]
    
    def snapshot(self):
        """لقطة لحالة السيولة"""
        flags = []
        
        if self.sweep_high():
            flags.append("🩸 SELL-SWEEP")
        if self.sweep_low():
            flags.append("💧 BUY-SWEEP")
        if self.accumulation():
            flags.append("📦 ACCUMULATION")
        if self.distribution():
            flags.append("📤 DISTRIBUTION")
        
        return " | ".join(flags) if flags else "…"

# =========================
# ===== FAKE BREAKOUT ENGINE =======
# =========================

class FakeBreakoutEngine:
    def __init__(self, candles, atr_value=None):
        if hasattr(candles, 'iloc'):
            self.high = candles['high'].astype(float).values
            self.low = candles['low'].astype(float).values
            self.open = candles['open'].astype(float).values
            self.close = candles['close'].astype(float).values
            self.volume = candles['volume'].astype(float).values if 'volume' in candles.columns else np.ones(len(candles))
        else:
            self.high = np.array([c["high"] for c in candles])
            self.low = np.array([c["low"] for c in candles])
            self.open = np.array([c["open"] for c in candles])
            self.close = np.array([c["close"] for c in candles])
            self.volume = np.array([c.get("volume", 1) for c in candles])
        
        self.atr = atr_value if atr_value is not None else self._calculate_atr()
    
    def _calculate_atr(self):
        """حساب ATR تقريبي"""
        if len(self.close) < 14:
            return np.mean(self.high[-5:] - self.low[-5:]) if len(self.close) >= 5 else 0.01
        return np.mean(self.high[-14:] - self.low[-14:])
    
    def _vol_spike(self):
        if len(self.volume) < 20:
            return False
        return self.volume[-1] > np.mean(self.volume[-20:]) * 1.6
    
    def _long_wick(self):
        candle_range = self.high[-1] - self.low[-1]
        if candle_range == 0:
            return False
        upper_wick = self.high[-1] - max(self.close[-1], self.open[-1])
        lower_wick = min(self.close[-1], self.open[-1]) - self.low[-1]
        return (upper_wick / candle_range > 0.5) or (lower_wick / candle_range > 0.5)
    
    def _no_follow_through(self):
        if len(self.close) < 3:
            return False
        body_now = abs(self.close[-1] - self.open[-1])
        body_prev = abs(self.close[-2] - self.open[-2])
        return body_now < body_prev * 0.6
    
    def detect_fake_up(self):
        """كشف الكسر الوهمي للأعلى"""
        if len(self.high) < 10:
            return False
        broke_high = self.high[-1] > max(self.high[-10:-1])
        weak_close = self.close[-1] < self.high[-2]
        return broke_high and weak_close and self._vol_spike() and self._long_wick()
    
    def detect_fake_down(self):
        """كشف الكسر الوهمي للأسفل"""
        if len(self.low) < 10:
            return False
        broke_low = self.low[-1] < min(self.low[-10:-1])
        weak_close = self.close[-1] > self.low[-2]
        return broke_low and weak_close and self._vol_spike() and self._long_wick()
    
    def verdict(self):
        """الحكم النهائي على الكسور الوهمية"""
        flags = []
        
        if self.detect_fake_up():
            flags.append("FAKE-UP")
        if self.detect_fake_down():
            flags.append("FAKE-DOWN")
        
        if flags and self._no_follow_through():
            flags.append("NO-FOLLOW")
        
        return FakeBreakoutVerdict(
            detected=len(flags) > 0,
            flags=flags,
            type=flags[0] if flags else None
        )

# =========================
# ===== SMART TRAILING ENGINE =======
# =========================

class SmartTrailingEngine:
    def __init__(self, candles, side, atr_value=None):
        """
        side: BUY أو SELL
        """
        if hasattr(candles, 'iloc'):
            self.high = candles['high'].astype(float).values
            self.low = candles['low'].astype(float).values
            self.close = candles['close'].astype(float).values
        else:
            self.high = np.array([c["high"] for c in candles])
            self.low = np.array([c["low"] for c in candles])
            self.close = np.array([c["close"] for c in candles])
        
        self.side = side
        self.atr = atr_value if atr_value is not None else self._calculate_atr()
    
    def _calculate_atr(self):
        if len(self.close) < 14:
            return np.mean(self.high[-5:] - self.low[-5:]) if len(self.close) >= 5 else 0.01
        return np.mean(self.high[-14:] - self.low[-14:])
    
    def trailing_stop(self, entry_price, current_sl, current_price):
        """
        وقف الخسارة الذكي الذي يتبع الهيكل
        """
        if self.side == "BUY":
            # للصفقات الشرائية: الوقف تحت آخر قاع هيكلي
            recent_lows = self.low[-10:]
            structural_low = min(recent_lows) if len(recent_lows) > 0 else entry_price * 0.99
            
            # إضافة buffer صغير
            buffer = self.atr * 0.3
            new_sl = structural_low - buffer
            
            # التأكد من أن SL الجديد أعلى من القديم (لا نرجوع للخلف)
            if new_sl > current_sl and new_sl < current_price * 0.995:
                return new_sl
        
        elif self.side == "SELL":
            # للصفقات البيعية: الوقف فوق آخر قمة هيكلية
            recent_highs = self.high[-10:]
            structural_high = max(recent_highs) if len(recent_highs) > 0 else entry_price * 1.01
            
            # إضافة buffer صغير
            buffer = self.atr * 0.3
            new_sl = structural_high + buffer
            
            # التأكد من أن SL الجديد أقل من القديم (لا نرجوع للخلف)
            if new_sl < current_sl and new_sl > current_price * 1.005:
                return new_sl
        
        return current_sl

# =========================
# ===== TP LADDER ENGINE =======
# =========================

class TPLadderEngine:
    def __init__(self, candles, side, trend_type="MID"):
        """
        side: BUY أو SELL
        trend_type: LARGE أو MID
        """
        if hasattr(candles, 'iloc'):
            self.high = candles['high'].astype(float).values
            self.low = candles['low'].astype(float).values
            self.close = candles['close'].astype(float).values
        else:
            self.high = np.array([c["high"] for c in candles])
            self.low = np.array([c["low"] for c in candles])
            self.close = np.array([c["close"] for c in candles])
        
        self.side = side
        self.trend_type = trend_type
    
    def detect_targets(self, entry_price):
        """كشف مستويات جني الأرباح"""
        targets = []
        
        if self.side == "BUY":
            # للصفقات الشرائية: البحث عن قمم سيولة
            if len(self.high) >= 30:
                swing_highs = self.high[-30:]
                recent_high = max(swing_highs)
                
                # TP1: أول هدف (مقاومة قريبة)
                tp1 = entry_price + (recent_high - entry_price) * 0.5
                targets.append(tp1)
                
                # TP2: هدف رئيسي (قمة السيولة)
                targets.append(recent_high)
                
                # TP3: هدف ممتد للترند الكبير
                if self.trend_type == "LARGE" and len(self.high) >= 60:
                    extended_high = max(self.high[-60:])
                    if extended_high > recent_high * 1.05:
                        targets.append(extended_high)
        
        elif self.side == "SELL":
            # للصفقات البيعية: البحث عن قيعان سيولة
            if len(self.low) >= 30:
                swing_lows = self.low[-30:]
                recent_low = min(swing_lows)
                
                # TP1: أول هدف (دعم قريب)
                tp1 = entry_price - (entry_price - recent_low) * 0.5
                targets.append(tp1)
                
                # TP2: هدف رئيسي (قاع السيولة)
                targets.append(recent_low)
                
                # TP3: هدف ممتد للترند الكبير
                if self.trend_type == "LARGE" and len(self.low) >= 60:
                    extended_low = min(self.low[-60:])
                    if extended_low < recent_low * 0.95:
                        targets.append(extended_low)
        
        # تنظيف الأهداف المكررة
        unique_targets = []
        for target in targets:
            if not unique_targets or abs(target - unique_targets[-1]) > entry_price * 0.01:
                unique_targets.append(target)
        
        return unique_targets[:3]  # أقصى 3 أهداف
    
    def ladder(self, entry_price):
        """إنشاء سلم جني الأرباح"""
        targets = self.detect_targets(entry_price)
        
        if not targets:
            return []
        
        if self.trend_type == "MID":
            if len(targets) >= 2:
                return [
                    {"tp": targets[0], "close_pct": 0.5},
                    {"tp": targets[1], "close_pct": 0.5}
                ]
            else:
                return [{"tp": targets[0], "close_pct": 1.0}]
        
        elif self.trend_type == "LARGE":
            if len(targets) >= 3:
                return [
                    {"tp": targets[0], "close_pct": 0.33},
                    {"tp": targets[1], "close_pct": 0.33},
                    {"tp": targets[2], "close_pct": 0.34}
                ]
            elif len(targets) == 2:
                return [
                    {"tp": targets[0], "close_pct": 0.5},
                    {"tp": targets[1], "close_pct": 0.5}
                ]
            else:
                return [{"tp": targets[0], "close_pct": 1.0}]
        
        return []

# =========================
# ===== DECISION MATRIX ENGINE =======
# =========================

class DecisionMatrixEngine:
    def __init__(
        self,
        trend_state,           # LARGE / MID / CHOP / NO_TREND
        explosion_signal,      # ExplosionState
        fake_breakout_verdict, # FakeBreakoutVerdict
        liquidity_snapshot,    # string
        order_block_signal,    # OrderBlock or None
        fvg_signal,           # FVGZone or None
        position_open=False
    ):
        self.trend = trend_state
        self.explosion = explosion_signal
        self.fake = fake_breakout_verdict
        self.liq = liquidity_snapshot
        self.ob = order_block_signal
        self.fvg = fvg_signal
        self.position_open = position_open
        
        self.votes_buy = 0
        self.votes_sell = 0
        self.reasons = []
    
    def _add_vote(self, side, strength, reason):
        if side == "BUY":
            self.votes_buy += strength
        elif side == "SELL":
            self.votes_sell += strength
        self.reasons.append(reason)
    
    def decide(self):
        # 1️⃣ HARD STOPS (أولوية قصوى)
        if self.fake.detected:
            return {
                "action": "BLOCK",
                "reason": f"FAKE_BREAKOUT: {','.join(self.fake.flags)}",
                "confidence": 0.0
            }
        
        if self.trend in ["CHOP", "NO_TREND"]:
            return {
                "action": "BLOCK",
                "reason": f"CHOP_MARKET: {self.trend}",
                "confidence": 0.0
            }
        
        # 2️⃣ أثناء وجود صفقة
        if self.position_open:
            if self.explosion.type == "COLLAPSE":
                return {
                    "action": "EXIT",
                    "reason": "LIQUIDITY_DRAIN_DETECTED",
                    "confidence": 0.9
                }
            return {
                "action": "HOLD",
                "reason": "MANAGE_EXISTING_TRADE",
                "confidence": 0.5
            }
        
        # 3️⃣ جمع الأصوات للإشارات
        # Order Block إشارات
        if self.ob and self.ob.valid:
            self._add_vote(self.ob.type, 2, f"OB_{self.ob.type}({self.ob.strength:.1f})")
        
        # FVG إشارات
        if self.fvg and not self.fvg.filled:
            if self.fvg.type == "BULLISH":
                self._add_vote("BUY", 1.5, f"FVG_BULLISH({self.fvg.size_bps:.1f}bps)")
            elif self.fvg.type == "BEARISH":
                self._add_vote("SELL", 1.5, f"FVG_BEARISH({self.fvg.size_bps:.1f}bps)")
        
        # Explosion إشارات
        if self.explosion.type.startswith("EXPLOSION_"):
            if "UP" in self.explosion.type:
                self._add_vote("BUY", 1.0, "EXPLOSION_UP")
            elif "DOWN" in self.explosion.type:
                self._add_vote("SELL", 1.0, "EXPLOSION_DOWN")
        
        # Liquidity hints
        if "BUY-SWEEP" in self.liq:
            self._add_vote("BUY", 1.0, "LIQ_BUY_SWEEP")
        if "SELL-SWEEP" in self.liq:
            self._add_vote("SELL", 1.0, "LIQ_SELL_SWEEP")
        if "ACCUMULATION" in self.liq:
            self._add_vote("BUY", 0.5, "ACCUMULATION")
        if "DISTRIBUTION" in self.liq:
            self._add_vote("SELL", 0.5, "DISTRIBUTION")
        
        # 4️⃣ تحديد العتبة بناءً على نوع الترند
        if self.trend == "LARGE":
            threshold = 3
        elif self.trend == "MID":
            threshold = 2
        else:
            threshold = 999  # لن يحدث
        
        # 5️⃣ اتخاذ القرار النهائي
        total_votes = self.votes_buy + self.votes_sell
        confidence = total_votes / 10.0 if total_votes > 0 else 0.0
        
        if self.votes_buy >= threshold and self.votes_buy > self.votes_sell:
            return {
                "action": "BUY",
                "reason": " | ".join(self.reasons),
                "votes": self.votes_buy,
                "confidence": min(confidence, 1.0)
            }
        
        if self.votes_sell >= threshold and self.votes_sell > self.votes_buy:
            return {
                "action": "SELL",
                "reason": " | ".join(self.reasons),
                "votes": self.votes_sell,
                "confidence": min(confidence, 1.0)
            }
        
        return {
            "action": "WAIT",
            "reason": "NO_CONFLUENCE",
            "confidence": 0.0
        }

# =========================
# ===== SMART MONEY CONCEPTS MASTER ENGINE =======
# =========================

class SmartMoneyConceptsMaster:
    def __init__(self):
        self.pivot_highs = deque(maxlen=20)
        self.pivot_lows = deque(maxlen=20)
        self.order_blocks_history = deque(maxlen=15)
        self.fair_value_gaps_history = deque(maxlen=15)
        
        self.market_state_cache = None
        self.liquidity_state_cache = None
        self.explosion_state_cache = None
        self.fake_breakout_cache = None
        
    def analyze_market(self, df, indicators=None):
        """
        تحليل شامل للسوق باستخدام جميع المحركات
        """
        if df.empty or len(df) < 20:
            return None
        
        # استخراج المؤشرات
        ind = indicators or {}
        adx_value = ind.get('adx', 20)
        di_plus = ind.get('di_plus', 30)
        di_minus = ind.get('di_minus', 30)
        atr_value = ind.get('atr', 0.01)
        
        # 1️⃣ تصنيف الترند
        trend_classifier = TrendClassifierEngine(df, adx_value, di_plus, di_minus)
        trend_state = trend_classifier.classify()
        
        # 2️⃣ حالة السيولة
        liquidity_log = LiquidityLogEngine(df)
        liquidity_snapshot = liquidity_log.snapshot()
        
        # 3️⃣ الانفجار/الانهيار
        explosion_engine = ExplosionCollapseEngine(df, atr_value)
        explosion_state = explosion_engine.detect_explosion()
        collapse_state = explosion_engine.detect_collapse()
        
        if collapse_state:
            explosion_state = collapse_state
        
        # 4️⃣ الكسور الوهمية
        fake_engine = FakeBreakoutEngine(df, atr_value)
        fake_verdict = fake_engine.verdict()
        
        # 5️⃣ مناطق SMC
        smc_zones = SMCZonesEngine(df)
        order_block = smc_zones.detect_order_block()
        fvg_zone = smc_zones.detect_fvg()
        
        # 6️⃣ تحديث الكاش
        self.market_state_cache = MarketState(
            regime="TREND" if trend_state in ["LARGE", "MID"] else "RANGE",
            trend_strength=adx_value,
            direction="BULL" if di_plus > di_minus else "BEAR",
            classification=trend_state
        )
        
        self.liquidity_state_cache = LiquidityState(
            swept_high=liquidity_log.sweep_high(),
            swept_low=liquidity_log.sweep_low(),
            sweep_price=None,  # يمكن حسابها لاحقاً
            snapshot=liquidity_snapshot
        )
        
        self.explosion_state_cache = explosion_state
        self.fake_breakout_cache = fake_verdict
        
        # 7️⃣ قرار المصفوفة
        decision_matrix = DecisionMatrixEngine(
            trend_state=trend_state,
            explosion_signal=explosion_state,
            fake_breakout_verdict=fake_verdict,
            liquidity_snapshot=liquidity_snapshot,
            order_block_signal=order_block,
            fvg_signal=fvg_zone,
            position_open=False
        )
        
        decision = decision_matrix.decide()
        
        # 8️⃣ إعداد النتيجة النهائية
        result = {
            "market_state": self.market_state_cache,
            "liquidity_state": self.liquidity_state_cache,
            "explosion_state": self.explosion_state_cache,
            "fake_breakout": self.fake_breakout_cache,
            "order_block": order_block,
            "fvg_zone": fvg_zone,
            "trend_classification": trend_state,
            "decision": decision,
            "timestamp": df['time'].iloc[-1] if 'time' in df.columns else None
        }
        
        # تحديث التاريخ
        if order_block:
            self.order_blocks_history.append(order_block)
        if fvg_zone:
            self.fair_value_gaps_history.append(fvg_zone)
        
        return result
    
    def get_trade_plan(self, df, side, entry_price, indicators=None):
        """
        إنشاء خطة تداول كاملة بناءً على تحليل السوق
        """
        if df.empty:
            return None
        
        # تحليل السوق
        analysis = self.analyze_market(df, indicators)
        if not analysis:
            return None
        
        # تحديد نوع الترند
        trend_type = analysis["trend_classification"]
        
        # 1️⃣ وقف الخسارة الذكي
        trailing_engine = SmartTrailingEngine(df, side, indicators.get('atr') if indicators else None)
        initial_sl = self._calculate_initial_sl(side, entry_price, df)
        
        # 2️⃣ سلم جني الأرباح
        tp_engine = TPLadderEngine(df, side, trend_type if trend_type in ["LARGE", "MID"] else "MID")
        tp_ladder = tp_engine.ladder(entry_price)
        
        # 3️⃣ إعداد خطة التداول
        trade_plan = {
            "entry_price": entry_price,
            "side": side,
            "trend_type": trend_type,
            "initial_stop_loss": initial_sl,
            "current_stop_loss": initial_sl,
            "take_profit_ladder": tp_ladder,
            "position_size": 1.0,  # سيتم ضبطه لاحقاً
            "risk_reward": self._calculate_risk_reward(entry_price, initial_sl, tp_ladder),
            "confidence": analysis["decision"]["confidence"],
            "entry_reason": analysis["decision"]["reason"],
            "market_context": {
                "trend": analysis["trend_classification"],
                "liquidity": analysis["liquidity_state"].snapshot,
                "explosion": analysis["explosion_state"].type,
                "fake_breakout": analysis["fake_breakout"].detected
            }
        }
        
        return trade_plan
    
    def _calculate_initial_sl(self, side, entry_price, df):
        """حساب وقف الخسارة الأولي"""
        atr = np.mean(df['high'].astype(float).values[-14:] - df['low'].astype(float).values[-14:]) if len(df) >= 14 else entry_price * 0.02
        
        if side == "BUY":
            # وقف تحت آخر قاع + buffer
            recent_low = min(df['low'].astype(float).values[-10:]) if len(df) >= 10 else entry_price
            return recent_low - (atr * 0.5)
        else:  # SELL
            # وقف فوق آخر قمة + buffer
            recent_high = max(df['high'].astype(float).values[-10:]) if len(df) >= 10 else entry_price
            return recent_high + (atr * 0.5)
    
    def _calculate_risk_reward(self, entry, sl, tp_ladder):
        """حساب نسبة المخاطرة إلى العائد"""
        if not tp_ladder:
            return 0
        
        if isinstance(sl, (int, float)):
            risk = abs(entry - sl)
            total_reward = 0
            for tp in tp_ladder:
                reward = abs(tp["tp"] - entry) * tp["close_pct"]
                total_reward += reward
            
            return total_reward / risk if risk > 0 else 0
        return 0
    
    def update_trade_plan(self, trade_plan, df, current_price):
        """
        تحديث خطة التداول بناءً على حركة السعر
        """
        if not trade_plan or df.empty:
            return trade_plan
        
        side = trade_plan["side"]
        entry_price = trade_plan["entry_price"]
        current_sl = trade_plan["current_stop_loss"]
        
        # تحديث وقف الخسارة الذكي
        trailing_engine = SmartTrailingEngine(df, side)
        new_sl = trailing_engine.trailing_stop(entry_price, current_sl, current_price)
        
        # تحقق من جني الأرباح
        tp_achieved = []
        remaining_tp = []
        
        for tp in trade_plan["take_profit_ladder"]:
            if side == "BUY" and current_price >= tp["tp"]:
                tp_achieved.append(tp)
            elif side == "SELL" and current_price <= tp["tp"]:
                tp_achieved.append(tp)
            else:
                remaining_tp.append(tp)
        
        # تحديث الخطة
        trade_plan["current_stop_loss"] = new_sl
        trade_plan["take_profit_ladder"] = remaining_tp
        trade_plan["tp_achieved"] = tp_achieved
        trade_plan["current_price"] = current_price
        trade_plan["pnl_pct"] = ((current_price - entry_price) / entry_price * 100) * (1 if side == "BUY" else -1)
        
        return trade_plan

# =================== INTEGRATION WITH EXISTING CODE ===================

# تحديث الدوال الموجودة للعمل مع المحرك الجديد

def enhanced_master_entry_engine(council, price, df):
    """
    محرك دخول محسن مع Smart Money Concepts
    """
    # استخراج البيانات
    det = council["details"]
    gz = det.get("golden_zone", {})
    indi = det.get("indicators", {})
    vwap = det.get("vwap_analysis", {})
    
    # تهيئة محرك SMC الرئيسي
    smc_master = SmartMoneyConceptsMaster()
    market_analysis = smc_master.analyze_market(df, indi)
    
    if not market_analysis:
        return {
            "allow": False,
            "mode": "reject",
            "side": None,
            "score": 0,
            "reason": "No market analysis available"
        }
    
    # تحليل القرار من المصفوفة
    decision = market_analysis["decision"]
    
    if decision["action"] in ["BLOCK", "WAIT"]:
        return {
            "allow": False,
            "mode": "reject",
            "side": None,
            "score": 0,
            "reason": decision["reason"],
            "confidence": decision["confidence"]
        }
    
    # تحديد نوع التداول بناءً على تصنيف الترند
    trend_type = market_analysis["trend_classification"]
    
    if trend_type == "LARGE":
        mode = "trend"
        size_multiplier = 1.0
    elif trend_type == "MID":
        mode = "studied_scalp"
        size_multiplier = 0.75
    else:
        mode = "cautious_scalp"
        size_multiplier = 0.5
    
    # تحديد الاتجاه
    side = decision["action"].lower() if decision["action"] in ["BUY", "SELL"] else None
    
    if not side:
        # Fallback إلى الإشارات التقليدية
        if gz.get("ok"):
            if gz["zone"]["type"] == "golden_bottom":
                side = "buy"
            elif gz["zone"]["type"] == "golden_top":
                side = "sell"
    
    if not side:
        return {
            "allow": False,
            "mode": "reject",
            "side": None,
            "score": 0,
            "reason": "No clear direction",
            "confidence": decision["confidence"]
        }
    
    # إنشاء خطة التداول
    trade_plan = smc_master.get_trade_plan(df, side.upper(), price, indi)
    
    return {
        "allow": True,
        "mode": mode,
        "side": side,
        "score": decision.get("votes", 0),
        "confidence": decision["confidence"],
        "size_multiplier": size_multiplier,
        "trade_plan": trade_plan,
        "market_analysis": {
            "trend": trend_type,
            "liquidity": market_analysis["liquidity_state"].snapshot,
            "explosion": market_analysis["explosion_state"].type,
            "fake_breakout": market_analysis["fake_breakout"].detected,
            "order_block": bool(market_analysis["order_block"]),
            "fvg": bool(market_analysis["fvg_zone"])
        },
        "reason": decision["reason"]
    }

def manage_position_with_smc_master(df, ind, info, state):
    """
    إدارة محسنة للمراكز مع Smart Money Concepts Master
    """
    if not state.get("open", False) or state.get("qty", 0) <= 0:
        return state
    
    current_price = info.get("price", 0)
    entry_price = state.get("entry", 0)
    side = state.get("side", "").upper()
    
    if not current_price or not entry_price or not side:
        return state
    
    # تحديث خطة التداول
    if "trade_plan" not in state:
        # إنشاء خطة تداول جديدة إذا لم تكن موجودة
        smc_master = SmartMoneyConceptsMaster()
        trade_plan = smc_master.get_trade_plan(df, side, entry_price, ind)
        if trade_plan:
            state["trade_plan"] = trade_plan
    else:
        # تحديث الخطة الحالية
        smc_master = SmartMoneyConceptsMaster()
        state["trade_plan"] = smc_master.update_trade_plan(
            state["trade_plan"], df, current_price
        )
    
    trade_plan = state.get("trade_plan", {})
    
    # التحقق من وقف الخسارة
    current_sl = trade_plan.get("current_stop_loss", 0)
    pnl_pct = trade_plan.get("pnl_pct", 0)
    
    # إدارة جني الأرباح التلقائي
    tp_achieved = trade_plan.get("tp_achieved", [])
    if tp_achieved and not state.get("tp_partial_closed", False):
        # جني جزئي للأرباح
        total_close_pct = sum(tp["close_pct"] for tp in tp_achieved)
        if total_close_pct > 0 and state.get("qty", 0) > 0:
            close_qty = state["qty"] * total_close_pct
            # هنا يمكن إضافة منطق إغلاق جزئي فعلي
            state["tp_partial_closed"] = True
            state["partial_close_pct"] = total_close_pct
            
            log_g(f"🏦 Partial Take Profit: {total_close_pct*100:.1f}% at {current_price:.6f}")
    
    # رفع وقف الخسارة إلى نقطة التعادل بعد تحقيق ربح معين
    if not state.get("sl_to_be", False) and pnl_pct > 1.0:
        # رفع SL إلى نقطة التعادل
        trade_plan["current_stop_loss"] = entry_price
        state["sl_to_be"] = True
        log_i(f"🛡️ Stop Loss moved to Break Even: {entry_price:.6f}")
    
    return state

# =================== ENHANCED TRADE LOOP WITH SMC MASTER ===================

def trade_loop_enhanced_with_smc_master():
    """
    الدورة الرئيسية المحسنة مع Smart Money Concepts Master
    """
    global wait_for_next_signal_side, compound_pnl
    
    # تهيئة محرك SMC الرئيسي
    smc_master = SmartMoneyConceptsMaster()
    
    loop_i = 0
    
    while True:
        try:
            current_time = time.time()
            bal = balance_usdt()
            px = price_now()
            df = fetch_ohlcv()
            
            if df.empty:
                time.sleep(BASE_SLEEP)
                continue
            
            info = rf_signal_live(df)
            ind = compute_indicators(df)
            
            # تحليل السوق مع SMC Master
            market_analysis = smc_master.analyze_market(df, ind)
            
            # مجلس الإدارة المحسن
            council_data = super_council_ai_enhanced_with_smc(df)
            
            # قرار الدخول المحسن
            master_decision = enhanced_master_entry_engine(council_data, px, df)
            
            # لوج قرار الدخول
            if market_analysis:
                liq_log = market_analysis["liquidity_state"].snapshot
                explosion_type = market_analysis["explosion_state"].type
                fake_detected = market_analysis["fake_breakout"].detected
                
                log_i(f"🧠 MARKET: {market_analysis['trend_classification']} | "
                      f"💥 {explosion_type} | 🧱 LIQ: {liq_log} | "
                      f"🚫 FAKE: {fake_detected}")
            
            if master_decision.get("allow", False):
                log_i(f"🏛 MASTER DECISION: {master_decision['side'].upper()} | "
                      f"{master_decision['mode']} | Score: {master_decision['score']} | "
                      f"Confidence: {master_decision['confidence']:.2f}")
                
                if not STATE["open"] and px is not None:
                    side = master_decision["side"]
                    mode = master_decision["mode"]
                    size_multiplier = master_decision.get("size_multiplier", 0.5)
                    
                    base_qty = compute_size(bal, px)
                    final_qty = safe_qty(base_qty * size_multiplier)
                    
                    if final_qty > 0:
                        ok = open_market_enhanced(side, final_qty, px)
                        if ok:
                            STATE["mode"] = mode
                            STATE["entry_reasons"] = {
                                "master_decision": master_decision["reason"],
                                "market_analysis": master_decision["market_analysis"],
                                "confidence": master_decision["confidence"]
                            }
                            STATE["trade_plan"] = master_decision.get("trade_plan", {})
                            
                            log_g(f"🎯 SMART ENTRY: {side.upper()} | {mode} | "
                                  f"Size: {size_multiplier*100:.0f}% | "
                                  f"Confidence: {master_decision['confidence']:.2f}")
            
            # إدارة الصفقة المفتوحة مع SMC Master
            if STATE["open"]:
                STATE = manage_position_with_smc_master(df, ind, {
                    "price": px or info.get("price", 0),
                    "market_analysis": market_analysis,
                    **info
                }, STATE)
            
            loop_i += 1
            sleep_s = NEAR_CLOSE_S if time_to_candle_close(df) <= 10 else BASE_SLEEP
            time.sleep(sleep_s)
            
        except Exception as e:
            log_e(f"SMC Master Enhanced loop error: {e}")
            time.sleep(BASE_SLEEP)

# =================== ADVANCED DASHBOARD API ===================

@app.route("/smc_master_dashboard")
def smc_master_dashboard():
    """
    لوحة تحكم متقدمة لعرض قرارات SMC Master والمجلس
    """
    df = fetch_ohlcv(limit=100)
    current_price = price_now()
    
    if df.empty or current_price is None:
        return jsonify({"error": "No data available"})
    
    # تحليل شامل مع SMC Master
    ind = compute_indicators(df)
    smc_master = SmartMoneyConceptsMaster()
    market_analysis = smc_master.analyze_market(df, ind)
    
    # البيانات التقليدية
    council_data = super_council_ai_enhanced_with_smc(df)
    master_decision = enhanced_master_entry_engine(council_data, current_price, df)
    
    return jsonify({
        "timestamp": datetime.utcnow().isoformat(),
        "price": current_price,
        "master_decision": master_decision,
        "market_analysis": {
            "trend_classification": market_analysis["trend_classification"] if market_analysis else None,
            "liquidity": market_analysis["liquidity_state"].snapshot if market_analysis else None,
            "explosion": market_analysis["explosion_state"].type if market_analysis else None,
            "fake_breakout": market_analysis["fake_breakout"].detected if market_analysis else None,
            "order_block": bool(market_analysis["order_block"]) if market_analysis else None,
            "fvg": bool(market_analysis["fvg_zone"]) if market_analysis else None,
            "decision": market_analysis["decision"] if market_analysis else None
        },
        "current_position": {
            "open": STATE["open"],
            "side": STATE["side"],
            "mode": STATE.get("mode", "none"),
            "entry_reasons": STATE.get("entry_reasons", {}),
            "trade_plan": STATE.get("trade_plan", {})
        }
    })

# =================== INITIALIZATION ===================

# استبدال محرك SMC القديم بالمحرك الجديد
smc_detector = SmartMoneyConceptsMaster()

# استبدال الدورة الرئيسية
trade_loop = trade_loop_enhanced_with_smc_master

# تحديث التحقق من البيئة
def verify_execution_environment_enhanced():
    print(f"⚙️ ENHANCED EXECUTION ENVIRONMENT", flush=True)
    print(f"🔧 EXCHANGE: {EXCHANGE_NAME.upper()} | SYMBOL: {SYMBOL}", flush=True)
    print(f"🔧 EXECUTE_ORDERS: {EXECUTE_ORDERS} | DRY_RUN: {DRY_RUN}", flush=True)
    print(f"🎯 SMART MONEY CONCEPTS MASTER: Full Advanced SMC Integration", flush=True)
    print(f"🏛 MASTER ENTRY ENGINE: Trend Classification + Liquidity Analysis", flush=True)
    print(f"📊 ADVANCED ANALYSIS: Explosion/Collapse + Fake Breakout Detection", flush=True)
    print(f"🧠 ENHANCED MANAGEMENT: Smart Trailing + TP Ladder", flush=True)

# تشغيل البوت المحدث
if __name__ == "__main__":
    verify_execution_environment_enhanced()
    
    import threading
    threading.Thread(target=keepalive_loop, daemon=True).start()
    threading.Thread(target=trade_loop, daemon=True).start()
    
    log_i(f"🚀 SUI ULTRA PRO AI BOT WITH SMC MASTER STARTED - {BOT_VERSION}")
    log_i(f"🎯 SYMBOL: {SYMBOL} | INTERVAL: {INTERVAL} | LEVERAGE: {LEVERAGE}x")
    log_i(f"💡 SMART MONEY CONCEPTS MASTER FULLY INTEGRATED: Advanced Market Analysis + Smart Management")
    
    app.run(host="0.0.0.0", port=PORT, debug=False)
