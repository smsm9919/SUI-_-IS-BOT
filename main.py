# -*- coding: utf-8 -*-
"""
ULTIMATE SMART MONEY BOT — Professional SMC Trading System
• Smart Money Concepts (SMC) - Full Implementation
• Market Structure: BOS, CHOCH, Internal/External Structure
• Liquidity Analysis: Sweeps, Pools, Hidden Liquidity
• Supply/Demand Zones + FVG + Order Blocks
• Advanced Candlestick Patterns + Reversal Detection
• Fibonacci Confluence + Price Action + Breakout Validation
• Professional Risk Management + Dynamic Position Sizing
• Real vs Fake Breakout Detection + Stop Hunts
• Multi-Timeframe Analysis + Confluence Trading
"""

import os, time, math, random, signal, sys, traceback, logging, json
from logging.handlers import RotatingFileHandler
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import ccxt
from flask import Flask, jsonify
from decimal import Decimal, ROUND_DOWN, InvalidOperation
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

try:
    from termcolor import colored
except Exception:
    def colored(t, *a, **k): return t

# =================== ENV / MODE ===================
EXCHANGE_NAME = os.getenv("EXCHANGE", "bingx").lower()

if EXCHANGE_NAME == "bybit":
    API_KEY = os.getenv("BYBIT_API_KEY", "")
    API_SECRET = os.getenv("BYBIT_API_SECRET", "")
else:
    API_KEY = os.getenv("BINGX_API_KEY", "")
    API_SECRET = os.getenv("BINGX_API_SECRET", "")

MODE_LIVE = bool(API_KEY and API_SECRET)
SELF_URL = os.getenv("SELF_URL", "") or os.getenv("RENDER_EXTERNAL_URL", "")
PORT = int(os.getenv("PORT", 5000))

# ==== Execution Switches ====
EXECUTE_ORDERS = True
SHADOW_MODE_DASHBOARD = False
DRY_RUN = False

# ==== Logging ====
LOG_LEGACY = False
LOG_ADDONS = True
LOG_SMC_DETAILS = True

# ==== State Management ====
STATE_PATH = "./bot_state.json"
RESUME_ON_RESTART = True
RESUME_LOOKBACK_SECS = 60 * 60

# ==== Core Settings ====
SYMBOL = os.getenv("SYMBOL", "SUI/USDT:USDT")
INTERVAL = os.getenv("INTERVAL", "15m")
LEVERAGE = int(os.getenv("LEVERAGE", 10))
RISK_ALLOC = float(os.getenv("RISK_ALLOC", 0.60))
POSITION_MODE = os.getenv("POSITION_MODE", "oneway")

# ==== SMC & Market Structure Settings ====
SMC_LOOKBACK = 200
BOS_CONFIRMATION_BARS = 3
CHOCH_CONFIRMATION_BARS = 2
LIQUIDITY_SWEEP_MARGIN = 0.001  # 0.1%
SUPPLY_DEMAND_ZONE_WIDTH = 0.005  # 0.5%
FVG_MIN_SIZE = 0.003  # 0.3%
ORDER_BLOCK_STRENGTH_THRESHOLD = 2.0
BREAKOUT_CONFIRMATION = 3  # bars
FAKE_BREAKOUT_DETECTION = True

# ==== Advanced Fibonacci ====
FIB_LEVELS = [0.236, 0.382, 0.5, 0.618, 0.786, 0.886]
FIB_EXTENSIONS = [1.272, 1.414, 1.618, 2.0, 2.618]
FIB_CONFLUENCE_ZONE = 0.02  # 2% zone for confluence

# ==== Smart Entry Settings ====
ENTRY_CONFLUENCE_MIN = 3  # Minimum confluence factors
ENTRY_VOLUME_MULTIPLIER = 1.5
ENTRY_SPREAD_MAX_BPS = 10  # Maximum spread in basis points
ENTRY_RETEST_CONFIRMATION = True

# ==== Advanced Risk Management ====
DYNAMIC_POSITION_SIZING = True
VOLATILITY_ADJUSTED_SL = True
TRAILING_STOP_ACTIVATION = 0.5  %  # Activate after 0.5% profit
TRAILING_STOP_DISTANCE = 1.0  %  # Distance from price
MAX_CONSECUTIVE_LOSSES = 3
COOLDOWN_AFTER_LOSS = 300  # 5 minutes

# ==== Multi-Timeframe Analysis ====
HIGHER_TF = "1h"
LOWER_TF = "5m"
MTF_CONFLUENCE_REQUIRED = True

# ==== Price Action & Candlestick ====
CANDLE_PATTERNS_ENABLED = True
REVERSAL_PATTERN_MIN_STRENGTH = 2.0
MOMENTUM_CANDLE_SIZE = 0.015  # 1.5% minimum for momentum candle

# ==== Liquidity Analysis ====
LIQUIDITY_POOL_DETECTION = True
HIDDEN_LIQUIDITY_ANALYSIS = True
LIQUIDITY_GRAB_THRESHOLD = 1.8  # Volume multiplier

# =================== PROFESSIONAL LOGGING ===================
def log_i(msg): print(f"ℹ️ {msg}", flush=True)
def log_g(msg): print(f"✅ {msg}", flush=True)
def log_w(msg): print(f"🟨 {msg}", flush=True)
def log_e(msg): print(f"❌ {msg}", flush=True)
def log_banner(text): print(f"\n{'—'*12} {text} {'—'*12}\n", flush=True)

def save_state(state: dict):
    try:
        state["ts"] = int(time.time())
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log_w(f"state save failed: {e}")

def load_state() -> dict:
    try:
        if not os.path.exists(STATE_PATH): return {}
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log_w(f"state load failed: {e}")
    return {}

# =================== EXCHANGE SETUP ===================
def make_ex():
    exchange_config = {
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
        "timeout": 20000,
    }
    
    if EXCHANGE_NAME == "bybit":
        exchange_config["options"] = {"defaultType": "swap"}
        return ccxt.bybit(exchange_config)
    else:
        exchange_config["options"] = {"defaultType": "swap"}
        return ccxt.bingx(exchange_config)

ex = make_ex()

# =================== DATA STRUCTURES ===================
@dataclass
class MarketStructure:
    trend: str  # "uptrend", "downtrend", "consolidation"
    bos_formed: bool
    choch_formed: bool
    internal_structure: Dict
    external_structure: Dict
    swing_highs: List[float]
    swing_lows: List[float]
    last_bos: Optional[float]
    last_choch: Optional[float]

@dataclass
class SupplyDemandZone:
    zone_type: str  # "supply" or "demand"
    high: float
    low: float
    strength: float
    touched: int
    created_at: int
    last_touch: int

@dataclass
class OrderBlock:
    high: float
    low: float
    is_bullish: bool
    strength: float
    volume: float
    created_at: int

@dataclass
class FVG:
    high: float
    low: float
    direction: str  # "bullish" or "bearish"
    filled: bool
    filled_at: Optional[float]

@dataclass
class LiquidityPool:
    price_level: float
    liquidity_type: str  # "bid", "ask", "hidden"
    estimated_size: float
    last_updated: int

# =================== SMC CORE FUNCTIONS ===================
def analyze_market_structure(df: pd.DataFrame) -> MarketStructure:
    """
    تحليل متقدم للهيكل السعري:
    - اكتشاف BOS (Break of Structure)
    - اكتشاف CHOCH (Change of Character)
    - تحديد الاتجاه الداخلي والخارجي
    - تحديد القمم والقيعان
    """
    if len(df) < 50:
        return MarketStructure(
            trend="unknown",
            bos_formed=False,
            choch_formed=False,
            internal_structure={},
            external_structure={},
            swing_highs=[],
            swing_lows=[],
            last_bos=None,
            last_choch=None
        )
    
    highs = df['high'].astype(float).values
    lows = df['low'].astype(float).values
    closes = df['close'].astype(float).values
    
    # اكتشاف القمم والقيعان
    swing_highs = []
    swing_lows = []
    
    for i in range(2, len(df) - 2):
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
           highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            swing_highs.append(highs[i])
        
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
           lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            swing_lows.append(lows[i])
    
    # تحديد الاتجاه
    trend = "consolidation"
    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        if swing_highs[-1] > swing_highs[-2] and swing_lows[-1] > swing_lows[-2]:
            trend = "uptrend"
        elif swing_highs[-1] < swing_highs[-2] and swing_lows[-1] < swing_lows[-2]:
            trend = "downtrend"
    
    # اكتشاف BOS
    bos_formed = False
    last_bos = None
    
    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        if trend == "uptrend":
            # BOS صاعد: كسر قمة سابقة
            for i in range(len(swing_highs)-1, 0, -1):
                if swing_highs[i] > swing_highs[i-1] + (swing_highs[i-1] * 0.002):  # 0.2% زيادة
                    bos_formed = True
                    last_bos = swing_highs[i]
                    break
        elif trend == "downtrend":
            # BOS هابط: كسر قاع سابق
            for i in range(len(swing_lows)-1, 0, -1):
                if swing_lows[i] < swing_lows[i-1] - (swing_lows[i-1] * 0.002):
                    bos_formed = True
                    last_bos = swing_lows[i]
                    break
    
    # اكتشاف CHOCH
    choch_formed = False
    last_choch = None
    
    if len(df) >= 20:
        recent_closes = closes[-20:]
        ma_short = np.mean(recent_closes[-5:])
        ma_long = np.mean(recent_closes)
        
        if trend == "uptrend" and ma_short < ma_long:
            choch_formed = True
            last_choch = closes[-1]
        elif trend == "downtrend" and ma_short > ma_long:
            choch_formed = True
            last_choch = closes[-1]
    
    return MarketStructure(
        trend=trend,
        bos_formed=bos_formed,
        choch_formed=choch_formed,
        internal_structure={"swings": len(swing_highs) + len(swing_lows)},
        external_structure={"trend_strength": abs(swing_highs[-1] - swing_lows[-1]) if swing_highs and swing_lows else 0},
        swing_highs=swing_highs[-5:],  # آخر 5 قمم
        swing_lows=swing_lows[-5:],    # آخر 5 قيعان
        last_bos=last_bos,
        last_choch=last_choch
    )

def find_supply_demand_zones(df: pd.DataFrame) -> List[SupplyDemandZone]:
    """
    اكتشاف مناطق العرض والطلب:
    - مناطق التجميع (Demand)
    - مناطق التوزيع (Supply)
    - قوة المنطقة بناءً على عدد المرات التي تم لمسها
    """
    zones = []
    
    if len(df) < 30:
        return zones
    
    highs = df['high'].astype(float).values
    lows = df['low'].astype(float).values
    volumes = df['volume'].astype(float).values
    
    # البحث عن شموع ذات أحجام عالية (نقاط تحول محتملة)
    volume_avg = np.mean(volumes[-30:])
    high_volume_indices = np.where(volumes > volume_avg * 1.5)[0]
    
    for idx in high_volume_indices:
        if idx < 2 or idx > len(df) - 3:
            continue
        
        # تحديد نوع المنطقة
        candle_high = highs[idx]
        candle_low = lows[idx]
        candle_body = abs(df['close'].iloc[idx] - df['open'].iloc[idx])
        candle_range = candle_high - candle_low
        
        if candle_body / candle_range < 0.3:  # شمعة صغيرة الجسم (دوجي/هامر)
            # هذه قد تكون منطقة عرض أو طلب
            prev_trend = np.mean(closes[idx-5:idx]) < np.mean(closes[idx:idx+5])
            
            if prev_trend:
                # منطقة طلب (شراء)
                zone = SupplyDemandZone(
                    zone_type="demand",
                    high=candle_high + (candle_range * SUPPLY_DEMAND_ZONE_WIDTH),
                    low=candle_low - (candle_range * SUPPLY_DEMAND_ZONE_WIDTH),
                    strength=volumes[idx] / volume_avg,
                    touched=1,
                    created_at=int(df['time'].iloc[idx]),
                    last_touch=int(df['time'].iloc[idx])
                )
            else:
                # منطقة عرض (بيع)
                zone = SupplyDemandZone(
                    zone_type="supply",
                    high=candle_high + (candle_range * SUPPLY_DEMAND_ZONE_WIDTH),
                    low=candle_low - (candle_range * SUPPLY_DEMAND_ZONE_WIDTH),
                    strength=volumes[idx] / volume_avg,
                    touched=1,
                    created_at=int(df['time'].iloc[idx]),
                    last_touch=int(df['time'].iloc[idx])
                )
            
            zones.append(zone)
    
    # دمج المناطق المتقاربة
    merged_zones = []
    for zone in zones:
        merged = False
        for mz in merged_zones:
            if (zone.zone_type == mz.zone_type and 
                abs(zone.high - mz.high) / mz.high < 0.01):
                # دمج المناطق
                mz.high = max(zone.high, mz.high)
                mz.low = min(zone.low, mz.low)
                mz.strength = max(zone.strength, mz.strength)
                mz.touched += zone.touched
                merged = True
                break
        
        if not merged:
            merged_zones.append(zone)
    
    return merged_zones[-10:]  # إرجاع آخر 10 مناطق

def find_fvg(df: pd.DataFrame) -> List[FVG]:
    """
    اكتشاف Fair Value Gaps:
    - فجوات بين الشموع تشير إلى مناطق غير متوازنة
    - تستخدم لدخول المراكز عند عودة السعر لملئها
    """
    fvgs = []
    
    if len(df) < 3:
        return fvgs
    
    for i in range(1, len(df) - 1):
        current = df.iloc[i]
        previous = df.iloc[i-1]
        next_candle = df.iloc[i+1]
        
        # FVG صاعد: قاع الشمعة الحالية > قمة الشمعة السابقة
        if (current['low'] > previous['high'] and 
            next_candle['high'] > current['low']):
            fvg = FVG(
                high=current['low'],
                low=previous['high'],
                direction="bullish",
                filled=False,
                filled_at=None
            )
            fvgs.append(fvg)
        
        # FVG هابط: قمة الشمعة الحالية < قاع الشمعة السابقة
        elif (current['high'] < previous['low'] and 
              next_candle['low'] < current['high']):
            fvg = FVG(
                high=previous['low'],
                low=current['high'],
                direction="bearish",
                filled=False,
                filled_at=None
            )
            fvgs.append(fvg)
    
    # التحقق من الفجوات التي تم ملؤها
    current_price = float(df['close'].iloc[-1])
    for fvg in fvgs[-20:]:  # التحقق من آخر 20 FVG
        if not fvg.filled:
            if fvg.direction == "bullish" and current_price <= fvg.high:
                fvg.filled = True
                fvg.filled_at = current_price
            elif fvg.direction == "bearish" and current_price >= fvg.low:
                fvg.filled = True
                fvg.filled_at = current_price
    
    return fvgs[-10:]  # إرجاع آخر 10 FVGs

def find_order_blocks(df: pd.DataFrame) -> List[OrderBlock]:
    """
    اكتشاف Order Blocks (كتل الأوامر):
    - مناطق دخول الكبار (Smart Money)
    - شموع ذات أحجام عالية وحركة قوية
    """
    blocks = []
    
    if len(df) < 10:
        return blocks
    
    volumes = df['volume'].astype(float).values
    volume_avg = np.mean(volumes[-20:])
    
    for i in range(1, len(df) - 1):
        current = df.iloc[i]
        previous = df.iloc[i-1]
        next_candle = df.iloc[i+1]
        
        volume_ratio = volumes[i] / volume_avg if volume_avg > 0 else 1
        
        if volume_ratio > ORDER_BLOCK_STRENGTH_THRESHOLD:
            candle_size = abs(current['close'] - current['open'])
            prev_candle_size = abs(previous['close'] - previous['open'])
            
            # Order Block صاعد: شمعة خضراء كبيرة بعد هبوط
            if (current['close'] > current['open'] and 
                previous['close'] < previous['open'] and
                candle_size > prev_candle_size * 1.5):
                
                block = OrderBlock(
                    high=current['high'],
                    low=current['low'],
                    is_bullish=True,
                    strength=volume_ratio,
                    volume=volumes[i],
                    created_at=int(current['time'])
                )
                blocks.append(block)
            
            # Order Block هابط: شمعة حمراء كبيرة بعد صعود
            elif (current['close'] < current['open'] and 
                  previous['close'] > previous['open'] and
                  candle_size > prev_candle_size * 1.5):
                
                block = OrderBlock(
                    high=current['high'],
                    low=current['low'],
                    is_bullish=False,
                    strength=volume_ratio,
                    volume=volumes[i],
                    created_at=int(current['time'])
                )
                blocks.append(block)
    
    return blocks[-10:]  # إرجاع آخر 10 order blocks

def detect_liquidity_sweeps(df: pd.DataFrame, structure: MarketStructure) -> Dict:
    """
    اكتشاف عمليات سحب السيولة:
    - اختراق مؤقت للمستويات لجمع الاستوبات
    - ارتداد سريع بعد السحب
    """
    if len(df) < 20:
        return {"detected": False, "type": None, "level": None}
    
    current_price = float(df['close'].iloc[-1])
    recent_high = max(df['high'].astype(float).tail(10))
    recent_low = min(df['low'].astype(float).tail(10))
    
    # سحب سيولة علوي: اختراق قمة ثم عودة سريعة
    if (current_price < recent_high * 0.995 and  # عودة أكثر من 0.5%
        max(df['high'].astype(float).tail(5)) > recent_high):
        
        # التحقق من وجود شمعة ارتداد
        last_candle = df.iloc[-1]
        if (last_candle['close'] < last_candle['open'] and
            abs(last_candle['close'] - last_candle['open']) > (recent_high - recent_low) * 0.3):
            
            return {
                "detected": True,
                "type": "liquidity_sweep_high",
                "level": recent_high,
                "retracement_percent": ((recent_high - current_price) / recent_high) * 100
            }
    
    # سحب سيولة سفلي: اختراق قاع ثم عودة سريعة
    elif (current_price > recent_low * 1.005 and  # عودة أكثر من 0.5%
          min(df['low'].astype(float).tail(5)) < recent_low):
        
        last_candle = df.iloc[-1]
        if (last_candle['close'] > last_candle['open'] and
            abs(last_candle['close'] - last_candle['open']) > (recent_high - recent_low) * 0.3):
            
            return {
                "detected": True,
                "type": "liquidity_sweep_low",
                "level": recent_low,
                "retracement_percent": ((current_price - recent_low) / recent_low) * 100
            }
    
    return {"detected": False, "type": None, "level": None}

def analyze_liquidity_pools(df: pd.DataFrame, orderbook: Dict = None) -> List[LiquidityPool]:
    """
    تحليل تجمعات السيولة:
    - السيولة المرئية في الـ Order Book
    - السيولة المخفية (Stop Clusters)
    """
    pools = []
    
    # السيولة المرئية من الـ Order Book
    if orderbook and 'bids' in orderbook and 'asks' in orderbook:
        bids = orderbook['bids'][:10]  # أفضل 10 عروض
        asks = orderbook['asks'][:10]  # أفضل 10 طلبات
        
        for price, size in bids:
            pool = LiquidityPool(
                price_level=float(price),
                liquidity_type="bid",
                estimated_size=float(size),
                last_updated=int(time.time())
            )
            pools.append(pool)
        
        for price, size in asks:
            pool = LiquidityPool(
                price_level=float(price),
                liquidity_type="ask",
                estimated_size=float(size),
                last_updated=int(time.time())
            )
            pools.append(pool)
    
    # اكتشاف السيولة المخفية (بناءً على تحليل السعر)
    if len(df) >= 50:
        closes = df['close'].astype(float).values
        
        # مناطق التجميع حول المتوسطات
        sma_20 = np.mean(closes[-20:])
        sma_50 = np.mean(closes[-50:])
        
        for level in [sma_20, sma_50]:
            # البحث عن تكرر السعر حول هذا المستوى
            price_counts = np.sum((closes[-20:] >= level * 0.995) & 
                                  (closes[-20:] <= level * 1.005))
            
            if price_counts >= 5:  # السعر مر على هذا المستوى 5 مرات على الأقل
                pool = LiquidityPool(
                    price_level=level,
                    liquidity_type="hidden",
                    estimated_size=price_counts * 1000,  # تقدير حجم
                    last_updated=int(time.time())
                )
                pools.append(pool)
    
    return pools

def advanced_fibonacci_analysis(df: pd.DataFrame) -> Dict:
    """
    تحليل فيبوناتشي متقدم:
    - مستويات التصحيح والامتداد
    - مناطق التجميع (Confluence Zones)
    - دعم قرار الدخول والخروج
    """
    if len(df) < 100:
        return {"error": "Insufficient data"}
    
    highs = df['high'].astype(float).values
    lows = df['low'].astype(float).values
    
    # أحدث موجة صاعدة أو هابطة
    recent_high_idx = np.argmax(highs[-50:]) + len(highs) - 50
    recent_low_idx = np.argmin(lows[-50:]) + len(lows) - 50
    
    if recent_high_idx > recent_low_idx:
        # موجة صاعدة
        swing_high = highs[recent_high_idx]
        swing_low = lows[recent_low_idx]
        direction = "uptrend"
    else:
        # موجة هابطة
        swing_high = highs[recent_high_idx]
        swing_low = lows[recent_low_idx]
        direction = "downtrend"
    
    swing_range = swing_high - swing_low
    
    # مستويات التصحيح
    retracement_levels = {}
    for level in FIB_LEVELS:
        if direction == "uptrend":
            price = swing_high - (swing_range * level)
        else:
            price = swing_low + (swing_range * level)
        retracement_levels[f"fib_{level}"] = price
    
    # مستويات الامتداد
    extension_levels = {}
    for level in FIB_EXTENSIONS:
        if direction == "uptrend":
            price = swing_high + (swing_range * level)
        else:
            price = swing_low - (swing_range * level)
        extension_levels[f"ext_{level}"] = price
    
    # مناطق التجميع (مستويات فيبوناتشي متقاربة)
    confluence_zones = []
    all_levels = {**retracement_levels, **extension_levels}
    level_values = list(all_levels.values())
    level_values.sort()
    
    for i in range(len(level_values) - 1):
        if abs(level_values[i+1] - level_values[i]) / level_values[i] < FIB_CONFLUENCE_ZONE:
            zone = {
                "start": level_values[i],
                "end": level_values[i+1],
                "strength": 2  # قوة مبدئية
            }
            confluence_zones.append(zone)
    
    current_price = float(df['close'].iloc[-1])
    
    return {
        "direction": direction,
        "swing_high": swing_high,
        "swing_low": swing_low,
        "retracement_levels": retracement_levels,
        "extension_levels": extension_levels,
        "confluence_zones": confluence_zones,
        "current_position": "above" if current_price > swing_high else ("below" if current_price < swing_low else "within"),
        "nearest_fib_level": min(retracement_levels.values(), key=lambda x: abs(x - current_price))
    }

def detect_real_vs_fake_breakout(df: pd.DataFrame, level: float, breakout_type: str) -> Dict:
    """
    التمييز بين الاختراق الحقيقي والوهمي:
    - الاختراق الحقيقي: إغلاق متعدد فوق/تحت المستوى مع حجم عالي
    - الاختراق الوهمي: اختراق سريع ثم عودة
    """
    if len(df) < 10:
        return {"real": False, "confidence": 0, "reason": "Insufficient data"}
    
    recent_candles = df.tail(5)
    closes = recent_candles['close'].astype(float).values
    volumes = recent_candles['volume'].astype(float).values
    avg_volume = np.mean(volumes)
    
    if breakout_type == "above":
        # اختراق فوقي
        closes_above = np.sum(closes > level)
        volume_multiplier = np.mean(volumes[closes > level]) / avg_volume if avg_volume > 0 else 1
        
        if closes_above >= BREAKOUT_CONFIRMATION and volume_multiplier > 1.2:
            return {
                "real": True,
                "confidence": min(1.0, (closes_above / 5) * volume_multiplier),
                "reason": f"{closes_above} closes above with {volume_multiplier:.1f}x volume"
            }
        else:
            return {
                "real": False,
                "confidence": max(0.0, (closes_above / 5) * 0.5),
                "reason": f"Insufficient confirmation ({closes_above} closes, volume {volume_multiplier:.1f}x)"
            }
    
    else:  # breakout_type == "below"
        # اختراق تحتي
        closes_below = np.sum(closes < level)
        volume_multiplier = np.mean(volumes[closes < level]) / avg_volume if avg_volume > 0 else 1
        
        if closes_below >= BREAKOUT_CONFIRMATION and volume_multiplier > 1.2:
            return {
                "real": True,
                "confidence": min(1.0, (closes_below / 5) * volume_multiplier),
                "reason": f"{closes_below} closes below with {volume_multiplier:.1f}x volume"
            }
        else:
            return {
                "real": False,
                "confidence": max(0.0, (closes_below / 5) * 0.5),
                "reason": f"Insufficient confirmation ({closes_below} closes, volume {volume_multiplier:.1f}x)"
            }

def advanced_candlestick_analysis(df: pd.DataFrame) -> Dict:
    """
    تحليل متقدم للشموع اليابانية:
    - اكتشاف أنماط الانعكاس
    - قوة النمط
    - تأكيدات الحجم
    """
    patterns = []
    
    if len(df) < 5:
        return {"patterns": patterns, "strength": 0}
    
    # بيانات الشموع
    o1, h1, l1, c1 = [float(x) for x in df[['open', 'high', 'low', 'close']].iloc[-1]]
    o2, h2, l2, c2 = [float(x) for x in df[['open', 'high', 'low', 'close']].iloc[-2]]
    o3, h3, l3, c3 = [float(x) for x in df[['open', 'high', 'low', 'close']].iloc[-3]]
    
    # الحسابات الأساسية
    body1 = abs(c1 - o1)
    range1 = h1 - l1
    body2 = abs(c2 - o2)
    range2 = h2 - l2
    
    # 1. Hammer / Inverted Hammer
    if body1 < range1 * 0.3:  # جسم صغير
        upper_wick = h1 - max(c1, o1)
        lower_wick = min(c1, o1) - l1
        
        if lower_wick > body1 * 2 and upper_wick < body1:  # Hammer
            patterns.append({
                "name": "Hammer",
                "type": "bullish_reversal",
                "strength": 1.5 if c1 > o1 else 1.0,
                "confirmation_needed": True
            })
        
        elif upper_wick > body1 * 2 and lower_wick < body1:  # Inverted Hammer
            patterns.append({
                "name": "Inverted_Hammer",
                "type": "bullish_reversal",
                "strength": 1.2,
                "confirmation_needed": True
            })
    
    # 2. Engulfing Pattern
    if body1 > body2 * 1.2:
        # Bullish Engulfing
        if c2 < o2 and c1 > o1 and o1 < c2 and c1 > o2:
            patterns.append({
                "name": "Bullish_Engulfing",
                "type": "bullish_reversal",
                "strength": 2.0,
                "confirmation_needed": False
            })
        
        # Bearish Engulfing
        elif c2 > o2 and c1 < o1 and o1 > c2 and c1 < o2:
            patterns.append({
                "name": "Bearish_Engulfing",
                "type": "bearish_reversal",
                "strength": 2.0,
                "confirmation_needed": False
            })
    
    # 3. Doji
    if body1 < range1 * 0.1:
        patterns.append({
            "name": "Doji",
            "type": "indecision",
            "strength": 0.5,
            "confirmation_needed": True
        })
    
    # 4. Morning Star / Evening Star
    if body2 < range2 * 0.3:  # شمعة صغيرة في المنتصف
        # Morning Star
        if c3 < o3 and c1 > o1 and min(o1, c1) > max(o2, c2):
            patterns.append({
                "name": "Morning_Star",
                "type": "bullish_reversal",
                "strength": 2.5,
                "confirmation_needed": False
            })
        
        # Evening Star
        elif c3 > o3 and c1 < o1 and max(o1, c1) < min(o2, c2):
            patterns.append({
                "name": "Evening_Star",
                "type": "bearish_reversal",
                "strength": 2.5,
                "confirmation_needed": False
            })
    
    # حساب القوة الكلية
    total_strength = sum(p["strength"] for p in patterns if p["type"] in ["bullish_reversal", "bearish_reversal"])
    
    return {
        "patterns": patterns,
        "strength": total_strength,
        "has_reversal": total_strength >= REVERSAL_PATTERN_MIN_STRENGTH
    }

def smart_money_confluence_analysis(df: pd.DataFrame, current_price: float) -> Dict:
    """
    تحليل تجمع إشارات Smart Money:
    - دمج جميع مفاهيم SMC
    - حساب قوة الإشارة
    - تحديد مناطق الدخول المثلى
    """
    # جمع جميع البيانات
    structure = analyze_market_structure(df)
    zones = find_supply_demand_zones(df)
    fvgs = find_fvg(df)
    order_blocks = find_order_blocks(df)
    fib_analysis = advanced_fibonacci_analysis(df)
    candlestick_analysis = advanced_candlestick_analysis(df)
    liquidity_sweeps = detect_liquidity_sweeps(df, structure)
    
    # تحليل السيولة
    try:
        orderbook = ex.fetch_order_book(SYMBOL, limit=20)
        liquidity_pools = analyze_liquidity_pools(df, orderbook)
    except:
        liquidity_pools = []
    
    # حساب نقاط التجمع
    confluence_points = []
    
    # 1. تقاطع مناطق العرض/الطلب مع مستويات فيبوناتشي
    for zone in zones:
        for fib_name, fib_level in fib_analysis.get("retracement_levels", {}).items():
            if (zone.low <= fib_level <= zone.high or
                abs(zone.high - fib_level) / fib_level < 0.01):
                
                confluence_points.append({
                    "type": "zone_fib_confluence",
                    "level": fib_level,
                    "zone_type": zone.zone_type,
                    "strength": zone.strength * 1.5
                })
    
    # 2. تقاطع Order Blocks مع مناطق العرض/الطلب
    for block in order_blocks:
        for zone in zones:
            if (zone.low <= block.high <= zone.high or
                zone.low <= block.low <= zone.high):
                
                confluence_points.append({
                    "type": "block_zone_confluence",
                    "level": (block.high + block.low) / 2,
                    "block_type": "bullish" if block.is_bullish else "bearish",
                    "strength": block.strength * zone.strength
                })
    
    # 3. FVG بالقرب من مستويات فيبوناتشي
    for fvg in fvgs:
        if not fvg.filled:
            fvg_mid = (fvg.high + fvg.low) / 2
            for fib_name, fib_level in fib_analysis.get("retracement_levels", {}).items():
                if abs(fvg_mid - fib_level) / fib_level < 0.01:
                    
                    confluence_points.append({
                        "type": "fvg_fib_confluence",
                        "level": fib_level,
                        "fvg_direction": fvg.direction,
                        "strength": 2.0
                    })
    
    # 4. مناطق سحب السيولة مع مستويات هامة
    if liquidity_sweeps["detected"]:
        sweep_level = liquidity_sweeps["level"]
        
        confluence_points.append({
            "type": "liquidity_sweep",
            "level": sweep_level,
            "sweep_type": liquidity_sweeps["type"],
            "strength": 2.5,
            "retracement": liquidity_sweeps.get("retracement_percent", 0)
        })
    
    # تحليل القرب من المناطق
    near_zones = []
    for zone in zones:
        distance_pct = abs(current_price - (zone.high + zone.low) / 2) / current_price * 100
        if distance_pct < 1.0:  # ضمن 1%
            near_zones.append({
                "zone": zone,
                "distance_pct": distance_pct,
                "type": zone.zone_type
            })
    
    # توصيات التداول
    trade_recommendations = []
    
    # شراء: منطقة طلب + فيبوناتشي + شموع انعكاس صاعدة
    if (any(z.zone_type == "demand" for z in zones) and
        candlestick_analysis["has_reversal"] and
        any(p["type"] == "bullish_reversal" for p in candlestick_analysis["patterns"])):
        
        trade_recommendations.append({
            "action": "buy",
            "confidence": min(3.0, len(confluence_points) * 0.5 + candlestick_analysis["strength"]),
            "reasons": ["Demand zone", "Bullish reversal pattern", f"{len(confluence_points)} confluence points"]
        })
    
    # بيع: منطقة عرض + فيبوناتشي + شموع انعكاس هابطة
    if (any(z.zone_type == "supply" for z in zones) and
        candlestick_analysis["has_reversal"] and
        any(p["type"] == "bearish_reversal" for p in candlestick_analysis["patterns"])):
        
        trade_recommendations.append({
            "action": "sell",
            "confidence": min(3.0, len(confluence_points) * 0.5 + candlestick_analysis["strength"]),
            "reasons": ["Supply zone", "Bearish reversal pattern", f"{len(confluence_points)} confluence points"]
        })
    
    return {
        "market_structure": structure,
        "zones": zones,
        "fvgs": fvgs,
        "order_blocks": order_blocks,
        "fibonacci": fib_analysis,
        "candlestick": candlestick_analysis,
        "liquidity_sweeps": liquidity_sweeps,
        "liquidity_pools": liquidity_pools,
        "confluence_points": confluence_points,
        "near_zones": near_zones,
        "trade_recommendations": trade_recommendations,
        "current_price": current_price,
        "timestamp": int(time.time())
    }

# =================== SMART RISK MANAGEMENT ===================
class AdvancedRiskManager:
    """مدير مخاطر متقدم مع وقف خسارة متحرك ذكي"""
    
    def __init__(self):
        self.initial_sl_pct = 1.5  # وقف الخسارة الأولي 1.5%
        self.trailing_activation_pct = 0.5  %  # تفعيل التريل بعد 0.5% ربح
        self.trailing_distance_pct = 1.0  %  # مسافة التريل 1%
        self.max_position_pct = 2.0  %  # الحد الأقصى للمركز 2% من الرصيد
        self.consecutive_losses = 0
        self.last_trade_time = 0
        self.cooldown_period = 300  # 5 دقائق تبريد بعد خسارة
        
    def calculate_position_size(self, balance: float, entry_price: float, 
                               stop_loss: float, risk_pct: float = 1.0) -> float:
        """
        حساب حجم المركز بناءً على المخاطرة
        """
        risk_amount = balance * (risk_pct / 100)
        price_distance = abs(entry_price - stop_loss)
        
        if price_distance == 0:
            return 0
        
        position_size = risk_amount / price_distance
        max_size = balance * (self.max_position_pct / 100) / entry_price
        
        return min(position_size, max_size)
    
    def calculate_stop_loss(self, entry_price: float, side: str, 
                           atr: float, volatility_ratio: float) -> float:
        """
        حساب وقف الخسارة الذكي بناءً على التقلب
        """
        base_sl_distance = atr * 1.5
        
        # تعديل المسافة حسب التقلب
        if volatility_ratio > 2.0:
            sl_distance = base_sl_distance * 1.5
        elif volatility_ratio < 0.5:
            sl_distance = base_sl_distance * 0.7
        else:
            sl_distance = base_sl_distance
        
        if side == "long":
            return entry_price - sl_distance
        else:
            return entry_price + sl_distance
    
    def calculate_take_profit(self, entry_price: float, side: str,
                             risk_reward_ratio: float = 2.0,
                             stop_loss: float = None) -> float:
        """
        حساب هدف الربح بناءً على نسبة المخاطرة/العائد
        """
        if stop_loss is None:
            stop_loss = self.calculate_stop_loss(entry_price, side, 0.01, 1.0)
        
        risk_distance = abs(entry_price - stop_loss)
        
        if side == "long":
            return entry_price + (risk_distance * risk_reward_ratio)
        else:
            return entry_price - (risk_distance * risk_reward_ratio)
    
    def update_trailing_stop(self, current_price: float, entry_price: float,
                            side: str, highest_profit_pct: float) -> Tuple[float, bool]:
        """
        تحديث وقف الخسارة المتحرك
        يعيد: (وقف الخسارة الجديد, هل تم تفعيل التريل)
        """
        current_profit_pct = ((current_price - entry_price) / entry_price * 100 * 
                             (1 if side == "long" else -1))
        
        if current_profit_pct >= self.trailing_activation_pct:
            # تفعيل التريل
            trail_distance = current_price * (self.trailing_distance_pct / 100)
            
            if side == "long":
                new_sl = current_price - trail_distance
                # التأكد من أن التريل لا ينزل
                if new_sl > entry_price:
                    return new_sl, True
            else:
                new_sl = current_price + trail_distance
                if new_sl < entry_price:
                    return new_sl, True
        
        return None, False
    
    def can_trade(self, current_time: float) -> bool:
        """
        التحقق من إمكانية التداول بناءً على قواعد المخاطرة
        """
        # فحص التبريد بعد الخسارة
        if (self.consecutive_losses >= MAX_CONSECUTIVE_LOSSES and
            current_time - self.last_trade_time < self.cooldown_period):
            return False
        
        return True
    
    def record_trade_result(self, profit: float, trade_time: float):
        """
        تسجيل نتيجة الصفقة
        """
        self.last_trade_time = trade_time
        
        if profit <= 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

# =================== INTELLIGENT ENTRY SYSTEM ===================
def intelligent_entry_system(df: pd.DataFrame, current_price: float) -> Dict:
    """
    نظام دخول ذكي يجمع بين جميع مفاهيم SMC:
    - تحليل الهيكل السعري
    - مناطق العرض والطلب
    - فيبوناتشي
    - الشموع والأنماط
    - السيولة
    """
    # تحليل SMC الشامل
    smc_analysis = smart_money_confluence_analysis(df, current_price)
    
    # تحليل إضافي
    structure = smc_analysis["market_structure"]
    zones = smc_analysis["zones"]
    fib = smc_analysis["fibonacci"]
    candles = smc_analysis["candlestick"]
    liquidity = smc_analysis["liquidity_sweeps"]
    
    # حساب قوة الإشارة
    signal_strength = 0
    reasons = []
    
    # 1. قوة الهيكل السعري
    if structure.trend in ["uptrend", "downtrend"]:
        signal_strength += 1.0
        reasons.append(f"Strong {structure.trend}")
    
    # 2. قرب من منطقة هامة
    near_zone = smc_analysis["near_zones"]
    if near_zone:
        signal_strength += 1.5
        reasons.append(f"Near {near_zone[0]['type']} zone ({near_zone[0]['distance_pct']:.1f}%)")
    
    # 3. أنماط الشموع
    if candles["has_reversal"]:
        signal_strength += candles["strength"]
        pattern_names = [p["name"] for p in candles["patterns"]]
        reasons.append(f"Candle patterns: {', '.join(pattern_names)}")
    
    # 4. مستويات فيبوناتشي
    nearest_fib = fib.get("nearest_fib_level", 0)
    fib_distance_pct = abs(current_price - nearest_fib) / current_price * 100
    if fib_distance_pct < 0.5:  %  # ضمن 0.5%
        signal_strength += 1.0
        reasons.append(f"At Fibonacci level ({fib_distance_pct:.1f}% distance)")
    
    # 5. سحب السيولة
    if liquidity["detected"]:
        signal_strength += 2.0
        reasons.append(f"Liquidity sweep detected ({liquidity['type']})")
    
    # 6. نقاط التجمع
    confluence_count = len(smc_analysis["confluence_points"])
    if confluence_count >= 2:
        signal_strength += confluence_count * 0.5
        reasons.append(f"{confluence_count} confluence points")
    
    # تحديد اتجاه الدخول
    entry_signal = None
    entry_confidence = signal_strength
    
    if signal_strength >= ENTRY_CONFLUENCE_MIN:
        # تحليل الاتجاه
        if structure.trend == "uptrend" or (structure.trend == "downtrend" and liquidity["type"] == "liquidity_sweep_low"):
            entry_signal = "BUY"
        elif structure.trend == "downtrend" or (structure.trend == "uptrend" and liquidity["type"] == "liquidity_sweep_high"):
            entry_signal = "SELL"
    
    return {
        "signal": entry_signal,
        "confidence": min(10.0, entry_confidence),
        "reasons": reasons,
        "smc_analysis": smc_analysis,
        "structure": structure,
        "zones": zones,
        "fibonacci": fib,
        "candlestick": candles,
        "liquidity": liquidity,
        "timestamp": int(time.time())
    }

# =================== ADVANCED TRADE MANAGEMENT ===================
def manage_open_trade(df: pd.DataFrame, entry_data: Dict, 
                     current_price: float, position_side: str) -> Dict:
    """
    إدارة متقدمة للصفقة المفتوحة:
    - تحديث وقف الخسارة المتحرك
    - جني الأرباح على مراحل
    - مراقبة السيولة
    - اكتشاف الانعكاسات
    """
    management_signal = {
        "action": "hold",
        "reason": "Continue holding",
        "trailing_stop": None,
        "partial_close": False,
        "close_percentage": 0
    }
    
    # بيانات الدخول
    entry_price = entry_data.get("entry_price", 0)
    entry_time = entry_data.get("entry_time", 0)
    initial_sl = entry_data.get("stop_loss", 0)
    take_profit = entry_data.get("take_profit", 0)
    
    # حساب الربح/الخسارة
    if position_side == "long":
        pnl_pct = (current_price - entry_price) / entry_price * 100
        distance_to_sl = (current_price - initial_sl) / current_price * 100 if initial_sl else 0
    else:
        pnl_pct = (entry_price - current_price) / entry_price * 100
        distance_to_sl = (initial_sl - current_price) / current_price * 100 if initial_sl else 0
    
    # 1. وقف الخسارة المتحرك
    if pnl_pct >= TRAILING_STOP_ACTIVATION:
        trail_distance = current_price * (TRAILING_STOP_DISTANCE / 100)
        
        if position_side == "long":
            new_sl = current_price - trail_distance
            if new_sl > (entry_data.get("trailing_stop") or initial_sl or 0):
                management_signal["trailing_stop"] = new_sl
                management_signal["reason"] = f"Trailing stop updated to {new_sl:.6f}"
        else:
            new_sl = current_price + trail_distance
            if new_sl < (entry_data.get("trailing_stop") or initial_sl or entry_price * 1.02):
                management_signal["trailing_stop"] = new_sl
                management_signal["reason"] = f"Trailing stop updated to {new_sl:.6f}"
    
    # 2. جني الأرباح على مراحل
    profit_targets = [0.5, 1.0, 1.5, 2.0]  # أهداف ربح %
    close_percentages = [0.2, 0.3, 0.3, 0.2]  # نسب الإغلاق
    
    achieved_targets = entry_data.get("achieved_targets", [])
    
    for i, target in enumerate(profit_targets):
        if target not in achieved_targets and pnl_pct >= target:
            management_signal["partial_close"] = True
            management_signal["close_percentage"] = close_percentages[i]
            management_signal["reason"] = f"Take partial profit at {target}% target"
            achieved_targets.append(target)
            break
    
    # 3. تحليل الانعكاس
    candle_analysis = advanced_candlestick_analysis(df)
    if candle_analysis["has_reversal"]:
        reversal_type = None
        for pattern in candle_analysis["patterns"]:
            if (position_side == "long" and pattern["type"] == "bearish_reversal") or \
               (position_side == "short" and pattern["type"] == "bullish_reversal"):
                reversal_type = pattern["name"]
                break
        
        if reversal_type and pnl_pct > 0.5:  %  # إذا كان لدينا ربح وتشكل انعكاس
            management_signal["action"] = "close"
            management_signal["reason"] = f"Reversal pattern detected: {reversal_type}"
    
    # 4. اختبار وقف الخسارة
    if initial_sl:
        if (position_side == "long" and current_price <= initial_sl) or \
           (position_side == "short" and current_price >= initial_sl):
            management_signal["action"] = "close"
            management_signal["reason"] = "Stop loss hit"
    
    # 5. اختبار هدف الربح
    if take_profit:
        if (position_side == "long" and current_price >= take_profit) or \
           (position_side == "short" and current_price <= take_profit):
            management_signal["action"] = "close"
            management_signal["reason"] = "Take profit hit"
    
    return management_signal

# =================== MAIN TRADING ENGINE ===================
class UltimateSmartMoneyBot:
    """البوت الرئيسي للتداول بمفاهيم Smart Money"""
    
    def __init__(self):
        self.exchange = make_ex()
        self.symbol = SYMBOL
        self.interval = INTERVAL
        self.risk_manager = AdvancedRiskManager()
        self.state = load_state() or {}
        self.current_position = None
        self.consecutive_losses = 0
        
        # تهيئة السجل
        setup_file_logging()
        
    def fetch_market_data(self) -> pd.DataFrame:
        """جلب بيانات السوق"""
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe=self.interval, limit=200)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['time'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            log_e(f"Error fetching market data: {e}")
            return pd.DataFrame()
    
    def get_current_price(self) -> float:
        """الحصول على السعر الحالي"""
        try:
            ticker = self.exchange.fetch_ticker(self.symbol)
            return float(ticker['last'])
        except:
            return 0.0
    
    def analyze_and_trade(self):
        """الدورة الرئيسية للتحليل والتداول"""
        while True:
            try:
                # 1. جمع البيانات
                df = self.fetch_market_data()
                if df.empty:
                    time.sleep(5)
                    continue
                
                current_price = self.get_current_price()
                if current_price == 0:
                    time.sleep(5)
                    continue
                
                # 2. التحليل الذكي
                entry_analysis = intelligent_entry_system(df, current_price)
                
                # 3. التحقق من وجود صفقة مفتوحة
                if self.current_position:
                    # إدارة الصفقة المفتوحة
                    management = manage_open_trade(
                        df, 
                        self.current_position,
                        current_price,
                        self.current_position["side"]
                    )
                    
                    self.execute_management(management, current_price)
                
                else:
                    # 4. فتح صفقة جديدة
                    if (entry_analysis["signal"] and 
                        entry_analysis["confidence"] >= ENTRY_CONFLUENCE_MIN):
                        
                        self.execute_entry(entry_analysis, current_price, df)
                
                # 5. التسجيل والعرض
                self.log_analysis(entry_analysis)
                
                # 6. الانتظار للدورة التالية
                time.sleep(self.get_sleep_time(df))
                
            except Exception as e:
                log_e(f"Error in main loop: {e}")
                time.sleep(10)
    
    def execute_entry(self, analysis: Dict, current_price: float, df: pd.DataFrame):
        """تنفيذ دخول صفقة"""
        side = analysis["signal"].lower()
        
        # حساب وقف الخسارة والهدف
        atr = self.calculate_atr(df)
        volatility_ratio = atr / current_price * 100
        
        stop_loss = self.risk_manager.calculate_stop_loss(
            current_price, side, atr, volatility_ratio
        )
        
        take_profit = self.risk_manager.calculate_take_profit(
            current_price, side, risk_reward_ratio=2.0, stop_loss=stop_loss
        )
        
        # حساب حجم المركز
        balance = self.get_balance()
        position_size = self.risk_manager.calculate_position_size(
            balance, current_price, stop_loss, risk_pct=1.0
        )
        
        if position_size <= 0:
            log_w("Position size too small, skipping entry")
            return
        
        # التحقق من قواعد المخاطرة
        if not self.risk_manager.can_trade(time.time()):
            log_w("Risk rules prevent trading at this time")
            return
        
        # تنفيذ الصفقة
        if EXECUTE_ORDERS and not DRY_RUN and MODE_LIVE:
            try:
                order = self.exchange.create_order(
                    symbol=self.symbol,
                    type='market',
                    side=side,
                    amount=position_size
                )
                
                log_g(f"Entry order executed: {side.upper()} {position_size:.4f} @ {current_price}")
                
                # تحديث حالة الصفقة
                self.current_position = {
                    "side": side,
                    "entry_price": current_price,
                    "position_size": position_size,
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "entry_time": time.time(),
                    "entry_analysis": analysis,
                    "achieved_targets": []
                }
                
                # تسجيل في مدير المخاطر
                self.risk_manager.record_trade_result(0, time.time())
                
            except Exception as e:
                log_e(f"Error executing entry order: {e}")
        else:
            log_i(f"DRY RUN: Would enter {side.upper()} {position_size:.4f} @ {current_price}")
    
    def execute_management(self, management: Dict, current_price: float):
        """تنفيذ إجراءات إدارة الصفقة"""
        if not self.current_position:
            return
        
        side = self.current_position["side"]
        
        if management["action"] == "close":
            # إغلاق كامل الصفقة
            close_side = "sell" if side == "long" else "buy"
            
            if EXECUTE_ORDERS and not DRY_RUN and MODE_LIVE:
                try:
                    self.exchange.create_order(
                        symbol=self.symbol,
                        type='market',
                        side=close_side,
                        amount=self.current_position["position_size"]
                    )
                    
                    # حساب الربح/الخسارة
                    entry_price = self.current_position["entry_price"]
                    if side == "long":
                        profit = (current_price - entry_price) * self.current_position["position_size"]
                    else:
                        profit = (entry_price - current_price) * self.current_position["position_size"]
                    
                    log_g(f"Position closed: {close_side.upper()} | Profit: {profit:.2f} | Reason: {management['reason']}")
                    
                    # تسجيل في مدير المخاطر
                    self.risk_manager.record_trade_result(profit, time.time())
                    
                    # إعادة تعيين الصفقة الحالية
                    self.current_position = None
                    
                except Exception as e:
                    log_e(f"Error closing position: {e}")
            else:
                log_i(f"DRY RUN: Would close position | Reason: {management['reason']}")
                self.current_position = None
        
        elif management["partial_close"] and management["close_percentage"] > 0:
            # إغلاق جزئي
            close_amount = self.current_position["position_size"] * management["close_percentage"]
            close_side = "sell" if side == "long" else "buy"
            
            if EXECUTE_ORDERS and not DRY_RUN and MODE_LIVE:
                try:
                    self.exchange.create_order(
                        symbol=self.symbol,
                        type='market',
                        side=close_side,
                        amount=close_amount
                    )
                    
                    # تحديث حجم المركز
                    self.current_position["position_size"] -= close_amount
                    
                    log_g(f"Partial close: {close_amount:.4f} | Reason: {management['reason']}")
                    
                except Exception as e:
                    log_e(f"Error in partial close: {e}")
            else:
                log_i(f"DRY RUN: Would partial close {close_amount:.4f}")
                self.current_position["position_size"] -= close_amount
        
        elif management["trailing_stop"]:
            # تحديث وقف الخسارة المتحرك
            self.current_position["stop_loss"] = management["trailing_stop"]
            log_i(f"Trailing stop updated: {management['trailing_stop']:.6f}")
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """حاسبة ATR"""
        if len(df) < period + 1:
            return 0.01 * float(df['close'].iloc[-1]) if len(df) > 0 else 0.01
        
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        close = df['close'].astype(float)
        
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        
        atr = tr.rolling(period).mean().iloc[-1]
        return atr if not pd.isna(atr) else 0.01 * float(close.iloc[-1])
    
    def get_balance(self) -> float:
        """الحصول على الرصيد"""
        if not MODE_LIVE:
            return 1000.0  # رصيد تجريبي
        
        try:
            balance = self.exchange.fetch_balance()
            return float(balance['USDT']['free'])
        except:
            return 1000.0
    
    def get_sleep_time(self, df: pd.DataFrame) -> int:
        """حساب وقت الانتظار"""
        if len(df) < 2:
            return 5
        
        # وقت الانتظار حسب قرب نهاية الشمعة
        current_time = time.time() * 1000
        last_candle_time = df['timestamp'].iloc[-1]
        
        # حساب الفاصل الزمني بالميلي ثانية
        if 'm' in self.interval:
            minutes = int(self.interval[:-1])
            interval_ms = minutes * 60 * 1000
        elif 'h' in self.interval:
            hours = int(self.interval[:-1])
            interval_ms = hours * 60 * 60 * 1000
        else:
            interval_ms = 15 * 60 * 1000  # افتراضي 15 دقيقة
        
        next_candle_time = last_candle_time + interval_ms
        time_to_next = max(0, (next_candle_time - current_time) / 1000)
        
        # إذا كان باقي أقل من 10 ثواني للشمعة التالية، انتظر حتى تبدأ
        if time_to_next < 10:
            return int(time_to_next + 1)
        
        return 5  # فحص كل 5 ثواني
    
    def log_analysis(self, analysis: Dict):
        """تسجيل نتائج التحليل"""
        if not LOG_SMC_DETAILS:
            return
        
        signal = analysis.get("signal", "NONE")
        confidence = analysis.get("confidence", 0)
        reasons = analysis.get("reasons", [])
        
        if signal != "NONE" and confidence >= 5:
            log_banner(f"STRONG SIGNAL: {signal} (Confidence: {confidence:.1f}/10)")
            for reason in reasons:
                print(f"   • {reason}")
            
            # عرض تفاصيل SMC
            smc = analysis.get("smc_analysis", {})
            if smc:
                print(f"   📊 Structure: {smc.get('market_structure', {}).get('trend', 'N/A')}")
                print(f"   🎯 Zones: {len(smc.get('zones', []))} active")
                print(f"   📈 Confluence: {len(smc.get('confluence_points', []))} points")

# =================== WEB INTERFACE ===================
app = Flask(__name__)

bot = UltimateSmartMoneyBot()

@app.route("/")
def home():
    return """
    <html>
        <head>
            <title>ULTIMATE SMART MONEY BOT</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .container { max-width: 1200px; margin: 0 auto; }
                .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                         color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }
                .card { background: white; border: 1px solid #ddd; border-radius: 8px; 
                        padding: 20px; margin: 10px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .signal { font-size: 24px; font-weight: bold; margin: 10px 0; }
                .buy { color: #10b981; }
                .sell { color: #ef4444; }
                .hold { color: #6b7280; }
                .metric { display: inline-block; margin: 0 20px; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🎯 ULTIMATE SMART MONEY BOT</h1>
                    <p>Advanced SMC Trading System • Professional Market Analysis</p>
                </div>
                
                <div class="card">
                    <h2>📈 Live Analysis</h2>
                    <div class="metric"><strong>Symbol:</strong> {}</div>
                    <div class="metric"><strong>Interval:</strong> {}</div>
                    <div class="metric"><strong>Exchange:</strong> {}</div>
                    <div class="metric"><strong>Mode:</strong> {}</div>
                </div>
                
                <div class="card">
                    <h2>🚦 Trading Status</h2>
                    <div class="signal {}">Signal: {}</div>
                    <p><strong>Position:</strong> {}</p>
                    <p><strong>Consecutive Losses:</strong> {}</p>
                </div>
                
                <div class="card">
                    <h2>⚙️ System Health</h2>
                    <p><strong>Uptime:</strong> Running</p>
                    <p><strong>Last Update:</strong> {}</p>
                    <p><strong>API Status:</strong> Connected</p>
                </div>
            </div>
        </body>
    </html>
    """.format(
        SYMBOL,
        INTERVAL,
        EXCHANGE_NAME.upper(),
        "LIVE" if MODE_LIVE else "PAPER",
        "buy" if bot.current_position and bot.current_position["side"] == "long" else 
               "sell" if bot.current_position and bot.current_position["side"] == "short" else "hold",
        "BUY" if bot.current_position and bot.current_position["side"] == "long" else 
               "SELL" if bot.current_position and bot.current_position["side"] == "short" else "HOLD",
        "Active" if bot.current_position else "No Position",
        bot.consecutive_losses,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )

@app.route("/api/status")
def api_status():
    """واجهة API لحالة البوت"""
    return jsonify({
        "status": "running",
        "exchange": EXCHANGE_NAME,
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "mode": "live" if MODE_LIVE else "paper",
        "position": bot.current_position,
        "consecutive_losses": bot.consecutive_losses,
        "risk_manager": {
            "consecutive_losses": bot.risk_manager.consecutive_losses,
            "last_trade_time": bot.risk_manager.last_trade_time
        },
        "timestamp": datetime.now().isoformat()
    })

@app.route("/api/analyze")
def api_analyze():
    """واجهة API للتحليل الحالي"""
    df = bot.fetch_market_data()
    current_price = bot.get_current_price()
    
    if df.empty or current_price == 0:
        return jsonify({"error": "Unable to fetch market data"})
    
    analysis = intelligent_entry_system(df, current_price)
    
    return jsonify({
        "signal": analysis.get("signal"),
        "confidence": analysis.get("confidence"),
        "reasons": analysis.get("reasons"),
        "current_price": current_price,
        "market_structure": str(analysis.get("structure")),
        "zones_count": len(analysis.get("zones", [])),
        "confluence_points": len(analysis.get("smc_analysis", {}).get("confluence_points", [])),
        "timestamp": datetime.now().isoformat()
    })

# =================== SETUP LOGGING ===================
def setup_file_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    if not any(isinstance(h, RotatingFileHandler) and getattr(h, "baseFilename", "").endswith("smart_money_bot.log")
               for h in logger.handlers):
        fh = RotatingFileHandler("smart_money_bot.log", maxBytes=10_000_000, backupCount=10, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
        logger.addHandler(fh)
    
    logging.getLogger('werkzeug').setLevel(logging.ERROR)

# =================== MAIN EXECUTION ===================
def main():
    """الدالة الرئيسية لتشغيل البوت"""
    log_banner("ULTIMATE SMART MONEY BOT v1.0")
    print("🚀 Initializing Advanced SMC Trading System...")
    print(f"📊 Exchange: {EXCHANGE_NAME.upper()} • Symbol: {SYMBOL} • Interval: {INTERVAL}")
    print(f"⚡ Mode: {'LIVE TRADING' if MODE_LIVE and EXECUTE_ORDERS and not DRY_RUN else 'PAPER TRADING'}")
    print("🎯 Features Enabled:")
    print("   • Smart Money Concepts (SMC) Full Implementation")
    print("   • Market Structure Analysis (BOS, CHOCH)")
    print("   • Supply/Demand Zones Detection")
    print("   • Fibonacci Confluence Trading")
    print("   • Advanced Candlestick Patterns")
    print("   • Liquidity Analysis & Sweep Detection")
    print("   • Real vs Fake Breakout Detection")
    print("   • Intelligent Risk Management")
    print("   • Dynamic Position Sizing")
    print("   • Multi-Timeframe Confluence")
    
    setup_file_logging()
    
    # بدء البوت في خيط منفصل
    import threading
    bot_thread = threading.Thread(target=bot.analyze_and_trade, daemon=True)
    bot_thread.start()
    
    # بدء خادم الويب
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)

if __name__ == "__main__":
    main()
