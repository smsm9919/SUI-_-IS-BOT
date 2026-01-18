# -*- coding: utf-8 -*-
"""
SUI ULTRA PRO AI BOT - الإصدار الذكي المتقدم المتكامل المتطور
• مجلس الإدارة الفائق الذكي مع 25 استراتيجية متقدمة
• نظام ركوب الترند الذكي المحترف لتحقيق أقصى ربح متتالي
• السكالب الفائق الذكي بأهداف متعددة محسوبة
• إدارة صفقات ذكية متكيفة مع قوة الترند
• نظام Footprint + Diagonal Order-Flow المتقدم
• Multi-Exchange Support: BingX & Bybit
• نظام مراكبة الأرباح الذكي
• نظام TradePlan الذكي - كل صفقة لها خطة مسبقة
"""

import os, time, math, random, signal, sys, traceback, logging, json
from logging.handlers import RotatingFileHandler
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import ccxt
from flask import Flask, jsonify
from decimal import Decimal, ROUND_DOWN, InvalidOperation
import talib
from scipy import stats

try:
    from termcolor import colored
except Exception:
    def colored(t,*a,**k): return t

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

# ==== Run mode / Logging toggles ====
LOG_LEGACY = False
LOG_ADDONS = True

# ==== Execution Switches ====
EXECUTE_ORDERS = True
SHADOW_MODE_DASHBOARD = False
DRY_RUN = False

# ==== Addon: Logging + Recovery Settings ====
BOT_VERSION = f"SUI ULTRA PRO AI v8.0 — {EXCHANGE_NAME.upper()} - SMART TRADEPLAN EDITION"
print("🚀 Booting:", BOT_VERSION, flush=True)

STATE_PATH = "./bot_state.json"
RESUME_ON_RESTART = True
RESUME_LOOKBACK_SECS = 60 * 60

# === Addons config ===
BOOKMAP_DEPTH = 50
BOOKMAP_TOPWALLS = 3
IMBALANCE_ALERT = 1.30

FLOW_WINDOW = 20
FLOW_SPIKE_Z = 1.60
CVD_SMOOTH = 8

# =================== SETTINGS ===================
SYMBOL     = os.getenv("SYMBOL", "SUI/USDT:USDT")
INTERVAL   = os.getenv("INTERVAL", "15m")
LEVERAGE   = int(os.getenv("LEVERAGE", 10))
RISK_ALLOC = float(os.getenv("RISK_ALLOC", 0.60))
POSITION_MODE = os.getenv("POSITION_MODE", "oneway")

# RF Settings - Optimized for SUI
RF_SOURCE = "close"
RF_PERIOD = int(os.getenv("RF_PERIOD", 18))
RF_MULT   = float(os.getenv("RF_MULT", 3.0))
RF_LIVE_ONLY = True
RF_HYST_BPS  = 6.0

# Indicators
RSI_LEN = 14
ADX_LEN = 14
ATR_LEN = 14

ENTRY_RF_ONLY = False
MAX_SPREAD_BPS = float(os.getenv("MAX_SPREAD_BPS", 6.0))

# Dynamic TP / trail - Optimized for SUI
TP1_PCT_BASE       = 0.45
TP1_CLOSE_FRAC     = 0.50
BREAKEVEN_AFTER    = 0.30
TRAIL_ACTIVATE_PCT = 1.20
ATR_TRAIL_MULT     = 1.8

TREND_TPS       = [0.50, 1.00, 1.80, 2.50, 3.50, 5.00, 7.00]
TREND_TP_FRACS  = [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.10]

# Dust guard
FINAL_CHUNK_QTY = float(os.getenv("FINAL_CHUNK_QTY", 50.0))
RESIDUAL_MIN_QTY = float(os.getenv("RESIDUAL_MIN_QTY", 10.0))

# Strict close
CLOSE_RETRY_ATTEMPTS = 6
CLOSE_VERIFY_WAIT_S  = 2.0

# Pacing
BASE_SLEEP   = 5
NEAR_CLOSE_S = 1

# ==== Smart Exit Tuning ===
TP1_SCALP_PCT      = 0.35/100
TP1_TREND_PCT      = 0.60/100
HARD_CLOSE_PNL_PCT = 1.10/100
WICK_ATR_MULT      = 1.5
EVX_SPIKE          = 1.8
BM_WALL_PROX_BPS   = 5
TIME_IN_TRADE_MIN  = 8
TRAIL_TIGHT_MULT   = 1.20

# ==== Golden Entry Settings ====
GOLDEN_ENTRY_SCORE = 6.0
GOLDEN_ENTRY_ADX   = 20.0
GOLDEN_REVERSAL_SCORE = 6.5

# ==== Golden Zone Constants ====
FIB_LOW, FIB_HIGH = 0.618, 0.786
MIN_WICK_PCT = 0.35
VOL_MA_LEN = 20
RSI_LEN_GZ, RSI_MA_LEN_GZ = 14, 9
MIN_DISP = 0.8

# ==== Execution & Strategy Thresholds ====
ADX_TREND_MIN = 20
DI_SPREAD_TREND = 6
RSI_MA_LEN = 9
RSI_NEUTRAL_BAND = (45, 55)
RSI_TREND_PERSIST = 3

GZ_MIN_SCORE = 6.0
GZ_REQ_ADX = 20
GZ_REQ_VOL_MA = 20
ALLOW_GZ_ENTRY = True

SCALP_TP1 = 0.40
SCALP_BE_AFTER = 0.30
SCALP_ATR_MULT = 1.6
TREND_TP1 = 1.20
TREND_BE_AFTER = 0.80
TREND_ATR_MULT = 1.8

MAX_TRADES_PER_HOUR = 8
COOLDOWN_SECS_AFTER_CLOSE = 45
ADX_GATE = 17

# ===== SUPER SCALP ENGINE =====
SCALP_MODE            = True
SCALP_EXECUTE         = True
SCALP_SIZE_FACTOR     = 0.35
SCALP_ADX_GATE        = 12.0
SCALP_MIN_SCORE       = 3.5
SCALP_IMB_THRESHOLD   = 1.00
SCALP_VOL_MA_FACTOR   = 1.20
SCALP_COOLDOWN_SEC    = 8
SCALP_RESPECT_WAIT    = False
SCALP_TP_SINGLE_PCT   = 0.35
SCALP_BE_AFTER_PCT    = 0.15
SCALP_ATR_TRAIL_MULT  = 1.0

# ===== SUPER COUNCIL ENHANCEMENTS =====
COUNCIL_AI_MODE = True
TREND_EARLY_DETECTION = True
MOMENTUM_ACCELERATION = True
VOLUME_CONFIRMATION = True
PRICE_ACTION_INTELLIGENCE = True

# أوزان التصويت الذكية المحسنة
WEIGHT_ADX = 1.8
WEIGHT_RSI = 1.4
WEIGHT_MACD = 1.6
WEIGHT_VOLUME = 1.3
WEIGHT_FLOW = 1.7
WEIGHT_GOLDEN = 2.0
WEIGHT_CANDLES = 1.4
WEIGHT_MOMENTUM = 1.6
WEIGHT_FOOTPRINT = 1.8
WEIGHT_DIAGONAL = 1.7
WEIGHT_EARLY_TREND = 2.0
WEIGHT_BREAKOUT = 2.2
WEIGHT_MARKET_STRUCTURE = 1.9
WEIGHT_VOLATILITY = 1.2
WEIGHT_SENTIMENT = 1.5

# ===== INTELLIGENT TREND MANAGEMENT =====
TREND_RIDING_AI = True
DYNAMIC_TP_ADJUSTMENT = True
ADAPTIVE_TRAILING = True
TREND_STRENGTH_ANALYSIS = True

# إعدادات ركوب الترند الذكية
TREND_FOLLOW_MULTIPLIER = 1.5
WEAK_TREND_EARLY_EXIT = True
STRONG_TREND_HOLD = True
TREND_REENTRY_STRATEGY = True

# ===== FLOW/FOOTPRINT Council Boost =====
FLOW_IMB_RATIO          = 1.6
FLOW_STACK_DEPTH        = 4
FLOW_ABSORB_PCTL        = 0.95
FLOW_ABSORB_MAX_TICKS   = 2
FP_WINDOW               = 3
FP_SCORE_BUY            = (2, 1.0)
FP_SCORE_SELL           = (2, 1.0)
FP_SCORE_ABSORB_PENALTY = (-1, -0.5)
DIAG_SCORE_BUY          = (2, 1.0)
DIAG_SCORE_SELL         = (2, 1.0)

# =================== PROFIT ACCUMULATION SYSTEM ===================
COMPOUND_PROFIT_REINVEST = True
PROFIT_REINVEST_RATIO = 0.4  # 40% من الأرباح يعاد استثمارها
MIN_COMPOUND_BALANCE = 50.0
PROFIT_TARGET_DAILY = 5.0  # هدف ربح يومي 5%

# =================== SMART TRADEPLAN SYSTEM ===================
class TradePlan:
    """خطة تداول ذكية لكل صفقة - قلب البوت الجديد"""
    def __init__(self, side, trend_class):
        self.side = side                # 'buy' أو 'sell'
        self.trend_class = trend_class  # 'mid' أو 'large'
        
        self.entry_reason = {
            "liquidity": None,     # 'sweep_high' / 'sweep_low' / 'liquidity_grab'
            "structure": None,     # 'BOS' / 'CHoCH' / 'breakout'
            "zone": None,          # 'OB' / 'FVG' / 'range_extreme'
            "confirmation": None   # 'rejection' / 'engulfing' / 'displacement'
        }
        
        self.invalidation = None      # مستوى الإبطال
        self.sl = None                # وقف الخسارة
        self.tp_targets = []          # أهداف الربح [tp1, tp2, tp3]
        self.tp_fractions = []        # نسب الإغلاق لكل هدف
        
        self.trailing_mode = None     # 'structure' / 'hybrid' / 'atr'
        self.breakeven_rule = None    # 'after_tp1' / 'after_structure'
        
        self.created_at = time.time()
        self.valid = False
        self.rr_expected = 0.0
        
        # حالة التنفيذ
        self.tp_hits = [False, False, False]
        self.sl_moved_to_be = False
        
    def is_valid(self):
        """التحقق من صحة الخطة"""
        return all([
            self.sl is not None,
            len(self.tp_targets) > 0,
            self.invalidation is not None,
            self.valid,
            self.rr_expected >= 1.5
        ])
    
    def summary(self):
        """ملخص الخطة"""
        return {
            "side": self.side,
            "trend_class": self.trend_class,
            "entry_reason": self.entry_reason,
            "sl": self.sl,
            "tp_targets": self.tp_targets,
            "invalidation": self.invalidation,
            "rr_expected": self.rr_expected,
            "valid": self.valid
        }

class MarketStructureAnalyzer:
    """محلل هيكل السوق المتقدم"""
    def __init__(self):
        self.swing_highs = []
        self.swing_lows = []
        self.structure_levels = []
        self.liquidity_zones = []
        
    def analyze_structure(self, df):
        """تحليل هيكل السوق"""
        try:
            if len(df) < 20:
                return False
                
            high = df['high'].astype(float)
            low = df['low'].astype(float)
            
            # تحديد القمم والقيعان
            swing_window = 5
            highs = high.rolling(swing_window, center=True).max()
            lows = low.rolling(swing_window, center=True).min()
            
            swing_highs = []
            swing_lows = []
            
            for i in range(swing_window, len(df)-swing_window):
                if high.iloc[i] == highs.iloc[i]:
                    swing_highs.append({
                        'price': high.iloc[i],
                        'index': i,
                        'time': df['time'].iloc[i]
                    })
                
                if low.iloc[i] == lows.iloc[i]:
                    swing_lows.append({
                        'price': low.iloc[i],
                        'index': i,
                        'time': df['time'].iloc[i]
                    })
            
            # تحديث البيانات
            self.swing_highs = swing_highs[-10:] if swing_highs else []
            self.swing_lows = swing_lows[-10:] if swing_lows else []
            
            # تحديد مستويات الهيكل
            self._calculate_structure_levels()
            
            return True
            
        except Exception as e:
            return False
    
    def _calculate_structure_levels(self):
        """حساب مستويات الهيكل الرئيسية"""
        try:
            levels = []
            
            # استخدام القمم والقيعان كمستويات هيكلية
            for swing in self.swing_highs:
                levels.append({
                    'price': swing['price'],
                    'type': 'resistance',
                    'strength': 1
                })
            
            for swing in self.swing_lows:
                levels.append({
                    'price': swing['price'],
                    'type': 'support',
                    'strength': 1
                })
            
            # تحديد أقرب مستويات
            self.structure_levels = sorted(levels, key=lambda x: x['price'])
            
        except Exception as e:
            self.structure_levels = []

# =================== ADVANCED MARKET ANALYSIS ===================
class AdvancedMarketAnalyzer:
    def __init__(self):
        self.market_phases = []
        self.volatility_regime = "normal"
        self.trend_strength = 0.0
        self.support_resistance = []
        
    def detect_market_phase(self, df):
        """اكتشاف مرحلة السوق الحالية"""
        try:
            close = df['close'].astype(float)
            high = df['high'].astype(float)
            low = df['low'].astype(float)
            
            # حساب المتوسطات المتحركة
            sma_20 = close.rolling(20).mean()
            sma_50 = close.rolling(50).mean()
            sma_200 = close.rolling(200).mean()
            
            # تحديد الترند
            price_vs_20 = close.iloc[-1] > sma_20.iloc[-1]
            price_vs_50 = close.iloc[-1] > sma_50.iloc[-1]
            price_vs_200 = close.iloc[-1] > sma_200.iloc[-1]
            
            ma_alignment = (sma_20.iloc[-1] > sma_50.iloc[-1] > sma_200.iloc[-1])
            
            if price_vs_200 and ma_alignment:
                return "strong_bull"
            elif price_vs_200 and not ma_alignment:
                return "bull"
            elif not price_vs_200 and ma_alignment:
                return "weak_bull"
            elif not price_vs_200 and not ma_alignment:
                return "bear"
            else:
                return "neutral"
                
        except Exception as e:
            return "neutral"
    
    def calculate_support_resistance(self, df, window=20):
        """حساب مستويات الدعم والمقاومة"""
        try:
            high = df['high'].astype(float)
            low = df['low'].astype(float)
            
            resistance = high.rolling(window).max()
            support = low.rolling(window).min()
            
            current_price = df['close'].iloc[-1]
            
            # إيجاد أقرب مستويات الدعم والمقاومة
            above_support = support[support < current_price].tail(3)
            below_resistance = resistance[resistance > current_price].head(3)
            
            return {
                'support_levels': above_support.tolist(),
                'resistance_levels': below_resistance.tolist(),
                'current_position': (current_price - above_support.iloc[-1] if len(above_support) > 0 else 0) / 
                                  (below_resistance.iloc[0] - above_support.iloc[-1] if len(above_support) > 0 and len(below_resistance) > 0 else 1)
            }
        except Exception as e:
            return {'support_levels': [], 'resistance_levels': [], 'current_position': 0.5}
    
    def analyze_volatility_regime(self, df):
        """تحليل نظام التقلب الحالي"""
        try:
            high = df['high'].astype(float)
            low = df['low'].astype(float)
            close = df['close'].astype(float)
            
            true_range = np.maximum(high - low, 
                                  np.maximum(abs(high - close.shift(1)), 
                                           abs(low - close.shift(1))))
            atr = true_range.rolling(14).mean()
            current_atr = atr.iloc[-1]
            avg_atr = atr.mean()
            
            volatility_ratio = current_atr / avg_atr if avg_atr > 0 else 1.0
            
            if volatility_ratio > 1.5:
                return "high", volatility_ratio
            elif volatility_ratio < 0.7:
                return "low", volatility_ratio
            else:
                return "normal", volatility_ratio
                
        except Exception as e:
            return "normal", 1.0

# =================== ENHANCED TRADE MANAGER ===================
class SmartTradeManager:
    def __init__(self):
        self.trade_history = []
        self.daily_profit = 0.0
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.win_rate = 0.0
        self.avg_win = 0.0
        self.avg_loss = 0.0
        
    def record_trade(self, side, entry, exit_price, quantity, profit, duration):
        """تسجيل الصفقة في السجل"""
        trade = {
            'timestamp': datetime.now(),
            'side': side,
            'entry': entry,
            'exit': exit_price,
            'quantity': quantity,
            'profit': profit,
            'duration': duration,
            'profit_pct': (profit / (entry * quantity)) * 100 if entry * quantity > 0 else 0
        }
        
        self.trade_history.append(trade)
        self.daily_profit += profit
        
        if profit > 0:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            
        # تحديث إحصائيات الأداء
        self.calculate_performance_metrics()
        
    def calculate_performance_metrics(self):
        """حساب مقاييس الأداء"""
        if not self.trade_history:
            return
            
        wins = [t for t in self.trade_history if t['profit'] > 0]
        losses = [t for t in self.trade_history if t['profit'] <= 0]
        
        self.win_rate = len(wins) / len(self.trade_history) * 100
        
        if wins:
            self.avg_win = sum(t['profit'] for t in wins) / len(wins)
        if losses:
            self.avg_loss = abs(sum(t['profit'] for t in losses) / len(losses))
            
    def get_trade_suggestions(self):
        """الحصول على اقتراحات تداول ذكية بناءً على الأداء"""
        suggestions = []
        
        if self.consecutive_losses >= 3:
            suggestions.append("REDUCE_SIZE: خسائر متتالية - تقليل حجم الصفقة")
            
        if self.win_rate < 40:
            suggestions.append("REVIEW_STRATEGY: نسبة نجاح منخفضة - مراجعة الاستراتيجية")
            
        if self.avg_loss > self.avg_win * 1.5:
            suggestions.append("ADJUST_STOP_LOSS: متوسط الخسارة أكبر من متوسط الربح - تعديل وقف الخسارة")
            
        return suggestions
    
    def get_optimal_position_size(self, balance, risk_per_trade=0.02):
        """حساب حجم الصفقة الأمثل بناءً على الأداء"""
        base_size = balance * risk_per_trade
        
        # تعديل الحجم بناءً على الأداء
        if self.consecutive_wins >= 3:
            size_multiplier = min(2.0, 1.0 + (self.consecutive_wins * 0.1))
        elif self.consecutive_losses >= 2:
            size_multiplier = max(0.5, 1.0 - (self.consecutive_losses * 0.2))
        else:
            size_multiplier = 1.0
            
        return base_size * size_multiplier

# إنشاء المحللين والمديرين
market_analyzer = AdvancedMarketAnalyzer()
trade_manager = SmartTradeManager()
market_structure = MarketStructureAnalyzer()

# =================== PROFESSIONAL LOGGING ===================
def log_i(msg): 
    print(f"ℹ️ {datetime.now().strftime('%H:%M:%S')} {msg}", flush=True)

def log_g(msg): 
    print(f"✅ {datetime.now().strftime('%H:%M:%S')} {msg}", flush=True)

def log_w(msg): 
    print(f"🟨 {datetime.now().strftime('%H:%M:%S')} {msg}", flush=True)

def log_e(msg): 
    print(f"❌ {datetime.now().strftime('%H:%M:%S')} {msg}", flush=True)

def log_banner(text): 
    print(f"\n{'—'*12} {text} {'—'*12}\n", flush=True)

def save_state(state: dict):
    try:
        state["ts"] = int(time.time())
        state["trade_stats"] = {
            "daily_profit": trade_manager.daily_profit,
            "consecutive_wins": trade_manager.consecutive_wins,
            "consecutive_losses": trade_manager.consecutive_losses,
            "win_rate": trade_manager.win_rate
        }
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        log_i(f"state saved → {STATE_PATH}")
    except Exception as e:
        log_w(f"state save failed: {e}")

def load_state() -> dict:
    try:
        if not os.path.exists(STATE_PATH): return {}
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            state = json.load(f)
            
        # استعادة إحصائيات التداول
        if "trade_stats" in state:
            trade_manager.daily_profit = state["trade_stats"].get("daily_profit", 0.0)
            trade_manager.consecutive_wins = state["trade_stats"].get("consecutive_wins", 0)
            trade_manager.consecutive_losses = state["trade_stats"].get("consecutive_losses", 0)
            trade_manager.win_rate = state["trade_stats"].get("win_rate", 0.0)
            
        return state
    except Exception as e:
        log_w(f"state load failed: {e}")
    return {}

# =================== EXCHANGE FACTORY ===================
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

# =================== EXCHANGE-SPECIFIC ADAPTERS ===================
def exchange_specific_params(side, is_close=False):
    if EXCHANGE_NAME == "bybit":
        if POSITION_MODE == "hedge":
            return {"positionSide": "Long" if side == "buy" else "Short", "reduceOnly": is_close}
        return {"positionSide": "Both", "reduceOnly": is_close}
    else:
        if POSITION_MODE == "hedge":
            return {"positionSide": "LONG" if side == "buy" else "SHORT", "reduceOnly": is_close}
        return {"positionSide": "BOTH", "reduceOnly": is_close}

def exchange_set_leverage(exchange, leverage, symbol):
    try:
        if EXCHANGE_NAME == "bybit":
            exchange.set_leverage(leverage, symbol)
        else:
            exchange.set_leverage(leverage, symbol, params={"side": "BOTH"})
        log_g(f"✅ {EXCHANGE_NAME.upper()} leverage set: {leverage}x")
    except Exception as e:
        log_w(f"⚠️ set_leverage warning: {e}")

# =================== MARKET SPECS ===================
MARKET = {}
AMT_PREC = 0
LOT_STEP = None
LOT_MIN  = None

def load_market_specs():
    global MARKET, AMT_PREC, LOT_STEP, LOT_MIN
    try:
        ex.load_markets()
        MARKET = ex.markets.get(SYMBOL, {})
        AMT_PREC = int((MARKET.get("precision", {}) or {}).get("amount", 0) or 0)
        LOT_STEP = (MARKET.get("limits", {}) or {}).get("amount", {}).get("step", None)
        LOT_MIN  = (MARKET.get("limits", {}) or {}).get("amount", {}).get("min",  None)
        log_i(f"🎯 {SYMBOL} specs → precision={AMT_PREC}, step={LOT_STEP}, min={LOT_MIN}")
    except Exception as e:
        log_w(f"load_market_specs: {e}")

def ensure_leverage_mode():
    try:
        exchange_set_leverage(ex, LEVERAGE, SYMBOL)
        log_i(f"📊 {EXCHANGE_NAME.upper()} position mode: {POSITION_MODE}")
    except Exception as e:
        log_w(f"ensure_leverage_mode: {e}")

try:
    load_market_specs()
    ensure_leverage_mode()
except Exception as e:
    log_w(f"exchange init: {e}")

# =================== LOGGING SETUP ===================
def setup_file_logging():
    """إعداد التسجيل المهني مع قمع رسائل Werkzeug"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    if not any(isinstance(h, RotatingFileHandler) and getattr(h, "baseFilename", "").endswith("bot.log")
               for h in logger.handlers):
        fh = RotatingFileHandler("bot.log", maxBytes=5_000_000, backupCount=7, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s [%(filename)s:%(lineno)d]"))
        logger.addHandler(fh)
    
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logger.addHandler(ch)
    
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    logging.getLogger('ccxt.base.exchange').setLevel(logging.INFO)
    
    log_i("🔄 Professional logging ready - File rotation + Werkzeug suppression")

setup_file_logging()

# =================== HELPERS ===================
_consec_err = 0
last_loop_ts = time.time()

def _fmt(x,n=6):
    try: return f"{float(x):.{n}f}"
    except: return str(x)

def _pct(x):
    try: return f"{float(x):.2f}%"
    except: return str(x)

def last_scalar(x, default=0.0):
    """يرجع float من آخر عنصر; يقبل Series/np.ndarray/list/float."""
    try:
        if isinstance(x, pd.Series): return float(x.iloc[-1])
        if isinstance(x, (list, tuple, np.ndarray)): return float(x[-1])
        if x is None: return float(default)
        return float(x)
    except Exception:
        return float(default)

def safe_get(ind: dict, key: str, default=0.0):
    """يقرأ مؤشر من dict ويحوّله scalar أخير."""
    if ind is None: 
        return float(default)
    val = ind.get(key, default)
    return last_scalar(val, default=default)

def _ind_brief(ind):
    if not ind: return "n/a"
    
    # استخراج قيم scalar بأمان
    adx = safe_get(ind, 'adx', 0)
    di_spread = safe_get(ind, 'di_spread', 0)
    rsi = safe_get(ind, 'rsi', 0)
    rsi_ma = safe_get(ind, 'rsi_ma', 0)
    atr = safe_get(ind, 'atr', 0)
    
    return (f"ADX={adx:.1f} DI={di_spread:.1f} | "
            f"RSI={rsi:.1f}/{rsi_ma:.1f} | "
            f"ATR={atr:.4f}")

def _council_brief(c):
    if not c: return "n/a"
    return f"B:{c.get('b',0)}/{_fmt(c.get('score_b',0),1)} | S:{c.get('s',0)}/{_fmt(c.get('score_s',0),1)}"

def _flow_brief(f):
    if not f: return "n/a"
    parts=[f"Δz={_fmt(f.get('delta_z','n/a'),2)}", f"CVD={_fmt(f.get('cvd_last','n/a'),0)}", f"trend={f.get('cvd_trend','?')}"]
    if f.get("spike"): parts.append("SPIKE")
    return " ".join(parts)

def print_position_snapshot(reason="OPEN", color=None):
    try:
        side   = STATE.get("side")
        open_f = STATE.get("open",False)
        qty    = STATE.get("qty"); px = STATE.get("entry")
        mode   = STATE.get("mode","trend")
        lev    = globals().get("LEVERAGE",0)
        tp1    = globals().get("TP1_PCT_BASE",0)
        be_a   = globals().get("BREAKEVEN_AFTER",0)
        trailA = globals().get("TRAIL_ACTIVATE_PCT",0)
        atrM   = globals().get("ATR_TRAIL_MULT",0)
        bal    = balance_usdt()
        spread = STATE.get("last_spread_bps")
        council= STATE.get("last_council")
        ind    = STATE.get("last_ind")
        flow   = STATE.get("last_flow")

        if color is None:
            icon = "🟢" if side=="buy" else "🔴"
        else:
            icon = "🟢" if str(color).lower()=="green" else "🔴"

        log_i(f"{icon} {reason} — POSITION SNAPSHOT")
        log_i(f"SIDE: {side} | QTY: {_fmt(qty)} | ENTRY: {_fmt(px)} | LEV: {lev}× | MODE: {mode} | OPEN: {open_f}")
        log_i(f"TP1: {_pct(tp1)} | BE@: {_pct(be_a)} | TRAIL: act≥{_pct(trailA)}, ATR×{atrM} | SPREAD: {_fmt(spread,2)} bps")
        log_i(f"IND: {_ind_brief(ind)}")
        log_i(f"COUNCIL: {_council_brief(council)}")
        log_i(f"FLOW: {_flow_brief(flow)}")
        
        # إضافة إحصائيات الأداء
        log_i(f"PERFORMANCE: Win Rate: {trade_manager.win_rate:.1f}% | Daily PnL: {trade_manager.daily_profit:.2f}")
        log_i("—"*72)
    except Exception as e:
        log_w(f"SNAPSHOT ERR: {e}")

def _round_amt(q):
    if q is None: return 0.0
    try:
        d = Decimal(str(q))
        if LOT_STEP and isinstance(LOT_STEP,(int,float)) and LOT_STEP>0:
            step = Decimal(str(LOT_STEP))
            d = (d/step).to_integral_value(rounding=ROUND_DOWN)*step
        prec = int(AMT_PREC) if AMT_PREC and AMT_PREC>=0 else 0
        d = d.quantize(Decimal(1).scaleb(-prec), rounding=ROUND_DOWN)
        if LOT_MIN and isinstance(LOT_MIN,(int,float)) and LOT_MIN>0 and d < Decimal(str(LOT_MIN)): return 0.0
        return float(d)
    except (InvalidOperation, ValueError, TypeError):
        return max(0.0, float(q))

def safe_qty(q): 
    q = _round_amt(q)
    if q<=0: log_w(f"qty invalid after normalize → {q}")
    return q

def fmt(v, d=6, na="—"):
    try:
        if v is None or (isinstance(v,float) and (math.isnan(v) or math.isinf(v))): return na
        return f"{float(v):.{d}f}"
    except Exception:
        return na

def with_retry(fn, tries=3, base_wait=0.4):
    global _consec_err
    for i in range(tries):
        try:
            r = fn()
            _consec_err = 0
            return r
        except Exception:
            _consec_err += 1
            if i == tries-1: raise
            time.sleep(base_wait*(2**i) + random.random()*0.25)

def fetch_ohlcv(limit=600):
    rows = with_retry(lambda: ex.fetch_ohlcv(SYMBOL, timeframe=INTERVAL, limit=limit, params={"type":"swap"}))
    return pd.DataFrame(rows, columns=["time","open","high","low","close","volume"])

def price_now():
    try:
        t = with_retry(lambda: ex.fetch_ticker(SYMBOL))
        return t.get("last") or t.get("close")
    except Exception: return None

def balance_usdt():
    if not MODE_LIVE: return 1000.0  # رصيد افتراضي أكبر للتجربة
    try:
        b = with_retry(lambda: ex.fetch_balance(params={"type":"swap"}))
        return b.get("total",{}).get("USDT") or b.get("free",{}).get("USDT")
    except Exception: return None

def orderbook_spread_bps():
    try:
        ob = with_retry(lambda: ex.fetch_order_book(SYMBOL, limit=5))
        bid = ob["bids"][0][0] if ob["bids"] else None
        ask = ob["asks"][0][0] if ob["asks"] else None
        if not (bid and ask): return None
        mid = (bid+ask)/2.0
        return ((ask-bid)/mid)*10000.0
    except Exception:
        return None

def _interval_seconds(iv: str) -> int:
    iv=(iv or "").lower().strip()
    if iv.endswith("m"): return int(float(iv[:-1]))*60
    if iv.endswith("h"): return int(float(iv[:-1]))*3600
    if iv.endswith("d"): return int(float(iv[:-1]))*86400
    return 15*60

def time_to_candle_close(df: pd.DataFrame) -> int:
    tf = _interval_seconds(INTERVAL)
    if len(df) == 0: return tf
    cur_start_ms = int(df["time"].iloc[-1])
    now_ms = int(time.time()*1000)
    next_close_ms = cur_start_ms + tf*1000
    while next_close_ms <= now_ms:
        next_close_ms += tf*1000
    left = max(0, next_close_ms - now_ms)
    return int(left/1000)

def fmt_walls(walls):
    return ", ".join([f"{p:.6f}@{q:.0f}" for p, q in walls]) if walls else "-"

# ========= Bookmap snapshot =========
def bookmap_snapshot(exchange, symbol, depth=BOOKMAP_DEPTH):
    try:
        ob = exchange.fetch_order_book(symbol, depth)
        bids = ob.get("bids", [])[:depth]; asks = ob.get("asks", [])[:depth]
        if not bids or not asks:
            return {"ok": False, "why": "empty"}
        b_sizes = np.array([b[1] for b in bids]); b_prices = np.array([b[0] for b in bids])
        a_sizes = np.array([a[1] for a in asks]); a_prices = np.array([a[0] for a in asks])
        b_idx = b_sizes.argsort()[::-1][:BOOKMAP_TOPWALLS]
        a_idx = a_sizes.argsort()[::-1][:BOOKMAP_TOPWALLS]
        buy_walls = [(float(b_prices[i]), float(b_sizes[i])) for i in b_idx]
        sell_walls = [(float(a_prices[i]), float(a_sizes[i])) for i in a_idx]
        imb = b_sizes.sum() / max(a_sizes.sum(), 1e-12)
        return {"ok": True, "buy_walls": buy_walls, "sell_walls": sell_walls, "imbalance": float(imb)}
    except Exception as e:
        return {"ok": False, "why": f"{e}"}

# ========= Volume flow / Delta & CVD =========
def compute_flow_metrics(df):
    try:
        if len(df) < max(30, FLOW_WINDOW+2):
            return {"ok": False, "why": "short_df"}
        close = df["close"].astype(float).copy()
        vol = df["volume"].astype(float).copy()
        up_mask = close.diff().fillna(0) > 0
        up_vol = (vol * up_mask).astype(float)
        dn_vol = (vol * (~up_mask)).astype(float)
        delta = up_vol - dn_vol
        cvd = delta.cumsum()
        cvd_ma = cvd.rolling(CVD_SMOOTH).mean()
        wnd = delta.tail(FLOW_WINDOW)
        mu = float(wnd.mean()); sd = float(wnd.std() or 1e-12)
        z = float((wnd.iloc[-1] - mu) / sd)
        trend = "up" if (cvd_ma.iloc[-1] - cvd_ma.iloc[-min(CVD_SMOOTH, len(cvd_ma))]) >= 0 else "down"
        return {"ok": True, "delta_last": float(delta.iloc[-1]), "delta_mean": mu, "delta_z": z,
                "cvd_last": float(cvd.iloc[-1]), "cvd_trend": trend, "spike": abs(z) >= FLOW_SPIKE_Z}
    except Exception as e:
        return {"ok": False, "why": str(e)}

# =================== ADVANCED INDICATORS ===================
def compute_advanced_indicators(df):
    """حساب المؤشرات المتقدمة"""
    try:
        close = df['close'].astype(float)
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        volume = df['volume'].astype(float)
        
        # مؤشرات الترند
        sma_20 = talib.SMA(close, timeperiod=20)
        sma_50 = talib.SMA(close, timeperiod=50)
        ema_20 = talib.EMA(close, timeperiod=20)
        
        # مؤشرات الزخم
        rsi = talib.RSI(close, timeperiod=14)
        macd, macd_signal, macd_hist = talib.MACD(close)
        stoch_k, stoch_d = talib.STOCH(high, low, close)
        
        # مؤشرات التقلب
        atr = talib.ATR(high, low, close, timeperiod=14)
        bollinger_upper, bollinger_middle, bollinger_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
        
        # مؤشرات الحجم
        obv = talib.OBV(close, volume)
        
        # مؤشرات الاتجاه
        adx = talib.ADX(high, low, close, timeperiod=14)
        plus_di = talib.PLUS_DI(high, low, close, timeperiod=14)
        minus_di = talib.MINUS_DI(high, low, close, timeperiod=14)
        
        return {
            'sma_20': last_scalar(sma_20),
            'sma_50': last_scalar(sma_50),
            'ema_20': last_scalar(ema_20),
            'rsi': last_scalar(rsi),
            'macd': last_scalar(macd),
            'macd_signal': last_scalar(macd_signal),
            'macd_hist': last_scalar(macd_hist),
            'stoch_k': last_scalar(stoch_k),
            'stoch_d': last_scalar(stoch_d),
            'atr': last_scalar(atr),
            'bollinger_upper': last_scalar(bollinger_upper),
            'bollinger_middle': last_scalar(bollinger_middle),
            'bollinger_lower': last_scalar(bollinger_lower),
            'obv': last_scalar(obv),
            'adx': last_scalar(adx),
            'plus_di': last_scalar(plus_di),
            'minus_di': last_scalar(minus_di),
            'volume': last_scalar(volume)
        }
    except Exception as e:
        log_w(f"Advanced indicators error: {e}")
        return {}

def compute_indicators(df):
    """حساب المؤشرات الأساسية"""
    try:
        close = df['close'].astype(float)
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        
        # ADX و DI
        adx = talib.ADX(high, low, close, timeperiod=ADX_LEN)
        plus_di = talib.PLUS_DI(high, low, close, timeperiod=ADX_LEN)
        minus_di = talib.MINUS_DI(high, low, close, timeperiod=ADX_LEN)
        di_spread = plus_di - minus_di
        
        # RSI
        rsi = talib.RSI(close, timeperiod=RSI_LEN)
        rsi_ma = talib.SMA(rsi, timeperiod=RSI_MA_LEN)
        
        # ATR
        atr = talib.ATR(high, low, close, timeperiod=ATR_LEN)
        
        return {
            'adx': last_scalar(adx),
            'plus_di': last_scalar(plus_di),
            'minus_di': last_scalar(minus_di),
            'di_spread': last_scalar(di_spread),
            'rsi': last_scalar(rsi),
            'rsi_ma': last_scalar(rsi_ma),
            'atr': last_scalar(atr)
        }
    except Exception as e:
        log_w(f"Basic indicators error: {e}")
        return {}

def compute_candles(df):
    """تحليل الشموع"""
    try:
        if len(df) < 3:
            return {"score_buy": 0, "score_sell": 0, "pattern": "none"}
        
        open_p = df['open'].astype(float)
        close = df['close'].astype(float)
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        
        score_buy = 0
        score_sell = 0
        pattern = "none"
        
        # شمعة الشراء القوية
        if close.iloc[-1] > open_p.iloc[-1] and (close.iloc[-1] - open_p.iloc[-1]) > (high.iloc[-1] - low.iloc[-1]) * 0.6:
            score_buy += 2
            pattern = "bullish_engulfing"
        
        # شمعة البيع القوية
        if close.iloc[-1] < open_p.iloc[-1] and (open_p.iloc[-1] - close.iloc[-1]) > (high.iloc[-1] - low.iloc[-1]) * 0.6:
            score_sell += 2
            pattern = "bearish_engulfing"
        
        return {
            "score_buy": score_buy,
            "score_sell": score_sell,
            "pattern": pattern
        }
    except Exception as e:
        log_w(f"Candles analysis error: {e}")
        return {"score_buy": 0, "score_sell": 0, "pattern": "none"}

def golden_zone_check(df, indicators):
    """فحص المناطق الذهبية"""
    try:
        if len(df) < 50:
            return {"ok": False, "score": 0, "zone": {}}
        
        return {"ok": False, "score": 0, "zone": {}}
    except Exception as e:
        log_w(f"Golden zone check error: {e}")
        return {"ok": False, "score": 0, "zone": {}}

# =================== ULTRA INTELLIGENT COUNCIL AI ===================
def ultra_intelligent_council_ai(df):
    """
    مجلس الإدارة الذكي الفائق - يدمج 25 استراتيجية متقدمة لاتخاذ أفضل القرارات
    """
    try:
        if len(df) < 100:
            return {"b": 0, "s": 0, "score_b": 0.0, "score_s": 0.0, "confidence": 0.0, "logs": [], "indicators": {}}
        
        # تحليل السوق المتقدم
        market_phase = market_analyzer.detect_market_phase(df)
        support_resistance = market_analyzer.calculate_support_resistance(df)
        volatility_regime, volatility_ratio = market_analyzer.analyze_volatility_regime(df)
        
        # المؤشرات المتقدمة
        advanced_indicators = compute_advanced_indicators(df)
        basic_indicators = compute_indicators(df)
        
        # دمج المؤشرات
        indicators = {**basic_indicators, **advanced_indicators}
        
        # تحليل الشموع المتقدم
        candles = compute_candles(df)
        
        # التحليل الفني المتقدم
        golden_zone = golden_zone_check(df, indicators)
        flow_metrics = compute_flow_metrics(df)
        orderbook = bookmap_snapshot(ex, SYMBOL)
        
        votes_b = 0
        votes_s = 0
        score_b = 0.0
        score_s = 0.0
        logs = []
        confidence_factors = []
        
        current_price = float(df['close'].iloc[-1])
        
        # ===== 1. تحليل مرحلة السوق =====
        if market_phase == "strong_bull":
            score_b += WEIGHT_MARKET_STRUCTURE * 2.5
            votes_b += 3
            logs.append("📈 مرحلة صاعدة قوية")
            confidence_factors.append(1.8)
        elif market_phase == "bull":
            score_b += WEIGHT_MARKET_STRUCTURE * 1.5
            votes_b += 2
            logs.append("📈 مرحلة صاعدة")
            confidence_factors.append(1.3)
        elif market_phase == "bear":
            score_s += WEIGHT_MARKET_STRUCTURE * 1.5
            votes_s += 2
            logs.append("📉 مرحلة هابطة")
            confidence_factors.append(1.3)
        elif market_phase == "strong_bear":
            score_s += WEIGHT_MARKET_STRUCTURE * 2.5
            votes_s += 3
            logs.append("📉 مرحلة هابطة قوية")
            confidence_factors.append(1.8)
        
        # ===== 2. تحليل الدعم والمقاومة =====
        support_levels = support_resistance.get('support_levels', [])
        resistance_levels = support_resistance.get('resistance_levels', [])
        current_position = support_resistance.get('current_position', 0.5)
        
        if support_levels and current_price <= support_levels[-1] * 1.005:  # قريب من الدعم
            score_b += WEIGHT_MARKET_STRUCTURE * 2.0
            votes_b += 2
            logs.append("🛡️ قريب من دعم قوي")
            confidence_factors.append(1.5)
        
        if resistance_levels and current_price >= resistance_levels[0] * 0.995:  # قريب من المقاومة
            score_s += WEIGHT_MARKET_STRUCTURE * 2.0
            votes_s += 2
            logs.append("🚧 قريب من مقاومة قوية")
            confidence_factors.append(1.5)
        
        # ===== 3. تحليل التقلب =====
        if volatility_regime == "low":
            # في فترات التقلب المنخفض، نبحث عن اختراقات
            if indicators.get('adx', 0) > 25:
                if indicators.get('plus_di', 0) > indicators.get('minus_di', 0):
                    score_b += WEIGHT_VOLATILITY * 1.5
                    votes_b += 2
                    logs.append("💎 اختراق في تقلب منخفض")
                else:
                    score_s += WEIGHT_VOLATILITY * 1.5
                    votes_s += 2
                    logs.append("💎 اختراق في تقلب منخفض")
        elif volatility_regime == "high":
            # في فترات التقلب العالي، نكون أكثر حذراً
            score_b *= 0.8
            score_s *= 0.8
            logs.append("⚡ تقلب عالي - تخفيض ثقة")
        
        # ===== 4. المؤشرات المتقدمة =====
        # RSI مع مستويات متقدمة
        rsi = indicators.get('rsi', 50)
        if rsi < 30:
            score_b += WEIGHT_RSI * 2.0
            votes_b += 2
            logs.append("📊 RSI في منطقة شراء قوية")
        elif rsi > 70:
            score_s += WEIGHT_RSI * 2.0
            votes_s += 2
            logs.append("📊 RSI في منطقة بيع قوية")
        elif 40 < rsi < 60:
            # RSI محايد - نبحث عن إشارات أخرى
            logs.append("📊 RSI محايد")
        
        # MACD
        macd = indicators.get('macd', 0)
        macd_signal = indicators.get('macd_signal', 0)
        if macd > macd_signal and indicators.get('macd_hist', 0) > 0:
            score_b += WEIGHT_MACD * 1.8
            votes_b += 2
            logs.append("📈 MACD صاعد قوي")
        elif macd < macd_signal and indicators.get('macd_hist', 0) < 0:
            score_s += WEIGHT_MACD * 1.8
            votes_s += 2
            logs.append("📉 MACD هابط قوي")
        
        # ستوكاستك
        stoch_k = indicators.get('stoch_k', 50)
        stoch_d = indicators.get('stoch_d', 50)
        if stoch_k < 20 and stoch_k > stoch_d:
            score_b += WEIGHT_MOMENTUM * 1.5
            votes_b += 1
            logs.append("🎯 ستوكاستك في منطقة شراء")
        elif stoch_k > 80 and stoch_k < stoch_d:
            score_s += WEIGHT_MOMENTUM * 1.5
            votes_s += 1
            logs.append("🎯 ستوكاستك في منطقة بيع")
        
        # ===== 5. بولنجر باندز =====
        bb_upper = indicators.get('bollinger_upper', current_price)
        bb_lower = indicators.get('bollinger_lower', current_price)
        
        if current_price <= bb_lower:
            score_b += WEIGHT_VOLATILITY * 1.8
            votes_b += 2
            logs.append("📏 سعر عند النطاق السفلي - شراء")
        elif current_price >= bb_upper:
            score_s += WEIGHT_VOLATILITY * 1.8
            votes_s += 2
            logs.append("📏 سعر عند النطاق العلوي - بيع")
        
        # ===== 6. ADX والاتجاه =====
        adx = indicators.get('adx', 0)
        plus_di = indicators.get('plus_di', 0)
        minus_di = indicators.get('minus_di', 0)
        
        if adx > 25:  # ترند قوي
            if plus_di > minus_di:
                score_b += WEIGHT_ADX * 2.5
                votes_b += 3
                logs.append(f"🎯 ترند صاعد قوي (ADX: {adx:.1f})")
                confidence_factors.append(1.8)
            else:
                score_s += WEIGHT_ADX * 2.5
                votes_s += 3
                logs.append(f"🎯 ترند هابط قوي (ADX: {adx:.1f})")
                confidence_factors.append(1.8)
        
        # ===== 7. المناطق الذهبية =====
        if golden_zone and golden_zone.get('ok'):
            gz_score = golden_zone.get('score', 0)
            zone_type = golden_zone.get('zone', {}).get('type', '')
            
            if zone_type == 'golden_bottom' and gz_score >= 7.0:
                score_b += WEIGHT_GOLDEN * 3.0
                votes_b += 4
                logs.append(f"🏆 منطقة ذهبية صاعدة (قوة: {gz_score:.1f})")
                confidence_factors.append(2.0)
            elif zone_type == 'golden_top' and gz_score >= 7.0:
                score_s += WEIGHT_GOLDEN * 3.0
                votes_s += 4
                logs.append(f"🏆 منطقة ذهبية هابطة (قوة: {gz_score:.1f})")
                confidence_factors.append(2.0)
        
        # ===== 8. تحليل الشموع =====
        if candles.get('score_buy', 0) > 2.0:
            score_b += WEIGHT_CANDLES * 1.8
            votes_b += 2
            logs.append(f"🕯️ تشكيل شموع شرائية قوية ({candles.get('pattern', '')})")
        
        if candles.get('score_sell', 0) > 2.0:
            score_s += WEIGHT_CANDLES * 1.8
            votes_s += 2
            logs.append(f"🕯️ تشكيل شموع بيعية قوية ({candles.get('pattern', '')})")
        
        # ===== 9. تحليل التدفق =====
        if flow_metrics.get('ok'):
            delta_z = flow_metrics.get('delta_z', 0)
            cvd_trend = flow_metrics.get('cvd_trend', '')
            
            if delta_z > 2.5 and cvd_trend == 'up':
                score_b += WEIGHT_FLOW * 2.2
                votes_b += 3
                logs.append(f"🌊 تدفق شرائي قوي جداً (z: {delta_z:.2f})")
                confidence_factors.append(1.7)
            elif delta_z < -2.5 and cvd_trend == 'down':
                score_s += WEIGHT_FLOW * 2.2
                votes_s += 3
                logs.append(f"🌊 تدفق بيعي قوي جداً (z: {delta_z:.2f})")
                confidence_factors.append(1.7)
        
        # ===== 10. تحليل الكتاب =====
        if orderbook.get('ok'):
            imbalance = orderbook.get('imbalance', 1.0)
            if imbalance > 2.0:
                score_b += WEIGHT_SENTIMENT * 1.5
                votes_b += 2
                logs.append(f"📚 تضارب قوي لصالح المشترين (imb: {imbalance:.2f})")
            elif imbalance < 0.5:
                score_s += WEIGHT_SENTIMENT * 1.5
                votes_s += 2
                logs.append(f"📚 تضارب قوي لصالح البائعين (imb: {imbalance:.2f})")
        
        # ===== 11. تحليل الحجم =====
        volume = indicators.get('volume', 0)
        volume_ma = df['volume'].rolling(20).mean().iloc[-1] if len(df) >= 20 else volume
        
        if volume > volume_ma * 1.5:
            # حجم عالي - نبحث عن اتجاه الحركة
            if current_price > float(df['open'].iloc[-1]):
                score_b += WEIGHT_VOLUME * 1.5
                votes_b += 2
                logs.append("📊 حجم عالي مع حركة صاعدة")
            else:
                score_s += WEIGHT_VOLUME * 1.5
                votes_s += 2
                logs.append("📊 حجم عالي مع حركة هابطة")
        
        # ===== 12. تطبيق عوامل الثقة =====
        if confidence_factors:
            confidence_multiplier = sum(confidence_factors) / len(confidence_factors)
            score_b *= confidence_multiplier
            score_s *= confidence_multiplier
        
        # ===== 13. مراعاة أداء التداول السابق =====
        if trade_manager.consecutive_losses >= 2:
            score_b *= 0.7
            score_s *= 0.7
            logs.append("⚠️ خسائر متتالية - تخفيض ثقة")
        
        if trade_manager.consecutive_wins >= 3:
            score_b *= 1.2
            score_s *= 1.2
            logs.append("🎯 أرباح متتالية - زيادة ثقة")
        
        # ===== 14. حساب الثقة النهائية =====
        total_score = score_b + score_s
        max_possible_score = 35.0  # أقصى درجة ممكنة
        
        confidence = min(1.0, total_score / max_possible_score)
        
        # ===== 15. تطبيق الحد الأدنى للثقة =====
        min_confidence = 0.65
        if confidence < min_confidence:
            score_b *= 0.5
            score_s *= 0.5
            logs.append(f"🛡️ ثقة منخفضة ({confidence:.2f} < {min_confidence}) - تخفيض")
        
        return {
            "b": votes_b,
            "s": votes_s,
            "score_b": round(score_b, 2),
            "score_s": round(score_s, 2),
            "confidence": round(confidence, 2),
            "logs": logs,
            "market_phase": market_phase,
            "volatility_regime": volatility_regime,
            "support_resistance": support_resistance,
            "indicators": indicators
        }
        
    except Exception as e:
        log_e(f"Ultra intelligent council error: {e}")
        return {"b": 0, "s": 0, "score_b": 0.0, "score_s": 0.0, "confidence": 0.0, "logs": [f"Error: {e}"], "indicators": {}}

# =================== SMART TRADEPLAN FUNCTIONS ===================
def determine_trend_class(df, indicators):
    """تحديد تصنيف الترند"""
    try:
        adx = indicators.get('adx', 0)
        di_spread = abs(indicators.get('plus_di', 0) - indicators.get('minus_di', 0))
        
        # تحليل متعدد الأطر الزمنية
        close = df['close'].astype(float)
        sma_20 = talib.SMA(close, 20)
        sma_50 = talib.SMA(close, 50)
        
        price_above_sma20 = close.iloc[-1] > sma_20.iloc[-1] if len(sma_20) > 0 else False
        price_above_sma50 = close.iloc[-1] > sma_50.iloc[-1] if len(sma_50) > 0 else False
        
        # شروط الترند الكبير
        if (adx > 25 and di_spread > 10 and 
            ((price_above_sma20 and price_above_sma50) or 
             (not price_above_sma20 and not price_above_sma50))):
            return "large"
        
        return "mid"
        
    except Exception as e:
        log_w(f"Trend class determination error: {e}")
        return "mid"

def analyze_entry_reasons(df, indicators, side, current_price):
    """تحليل أسباب الدخول"""
    reasons = {
        "liquidity": None,
        "structure": None,
        "zone": None,
        "confirmation": None
    }
    
    try:
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        close = df['close'].astype(float)
        open_p = df['open'].astype(float)
        
        # 1. تحليل السيولة
        recent_high = high.tail(10).max()
        recent_low = low.tail(10).min()
        
        if side == "sell" and abs(current_price - recent_high) / recent_high * 100 < 0.5:
            reasons["liquidity"] = "sweep_high"
        elif side == "buy" and abs(current_price - recent_low) / recent_low * 100 < 0.5:
            reasons["liquidity"] = "sweep_low"
        
        # 2. تحليل الهيكل
        adx = indicators.get('adx', 0)
        plus_di = indicators.get('plus_di', 0)
        minus_di = indicators.get('minus_di', 0)
        
        if side == "buy" and plus_di > minus_di and adx > 20:
            reasons["structure"] = "BOS_up"
        elif side == "sell" and minus_di > plus_di and adx > 20:
            reasons["structure"] = "BOS_down"
        
        # 3. تحليل المنطقة
        rsi = indicators.get('rsi', 50)
        if side == "buy" and rsi < 35:
            reasons["zone"] = "oversold"
        elif side == "sell" and rsi > 65:
            reasons["zone"] = "overbought"
        
        # 4. تأكيد الشمعة
        candle_size = abs(close.iloc[-1] - open_p.iloc[-1])
        avg_candle = abs(close - open_p).rolling(5).mean().iloc[-1] if len(df) >= 5 else candle_size
        
        if candle_size > avg_candle * 1.5:
            if side == "buy" and close.iloc[-1] > open_p.iloc[-1]:
                reasons["confirmation"] = "bullish_engulfing"
            elif side == "sell" and close.iloc[-1] < open_p.iloc[-1]:
                reasons["confirmation"] = "bearish_engulfing"
        
        return reasons
        
    except Exception as e:
        log_w(f"Entry reasons analysis error: {e}")
        return reasons

def validate_entry_reasons(entry_reasons):
    """التحقق من أسباب الدخول"""
    # يجب أن يكون هناك على الأقل سببين قويين
    valid_reasons = [v for v in entry_reasons.values() if v is not None]
    return len(valid_reasons) >= 2

def calculate_liquidity_levels(df, side, current_price):
    """حساب مستويات السيولة"""
    try:
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        
        if side == "buy":
            # للشراء: نقاط السيولة عند المقاومة القريبة
            recent_highs = high.tail(20).nlargest(3).tolist()
            return sorted(recent_highs)
        else:
            # للبيع: نقاط السيولة عند الدعم القريب
            recent_lows = low.tail(20).nsmallest(3).tolist()
            return sorted(recent_lows, reverse=True)
            
    except Exception as e:
        log_w(f"Liquidity levels calculation error: {e}")
        return []

def calculate_smart_sl(df, side, entry_price, liquidity_levels):
    """حساب وقف خسارة ذكي"""
    try:
        atr = safe_get(compute_indicators(df), 'atr', 0.001)
        
        if side == "buy":
            # للشراء: SL تحت أقرب دعم أو ATR
            recent_low = df['low'].astype(float).tail(10).min()
            sl_candidate1 = recent_low - (atr * 0.5)
            sl_candidate2 = entry_price - (atr * 2.0)
            return min(sl_candidate1, sl_candidate2)
        else:
            # للبيع: SL فوق أقرب مقاومة أو ATR
            recent_high = df['high'].astype(float).tail(10).max()
            sl_candidate1 = recent_high + (atr * 0.5)
            sl_candidate2 = entry_price + (atr * 2.0)
            return max(sl_candidate1, sl_candidate2)
            
    except Exception as e:
        log_w(f"Smart SL calculation error: {e}")
        return entry_price * 0.98 if side == "buy" else entry_price * 1.02

def calculate_tp_targets(df, side, entry_price, sl_price, trend_class, liquidity_levels):
    """حساب أهداف الربح الذكية"""
    try:
        atr = safe_get(compute_indicators(df), 'atr', 0.001)
        risk = abs(entry_price - sl_price)
        
        targets = []
        
        if trend_class == "large":
            # أهداف متعددة للترند الكبير
            rr_ratios = [1.5, 2.5, 4.0]
            for rr in rr_ratios:
                tp = entry_price + (risk * rr) if side == "buy" else entry_price - (risk * rr)
                
                # ضبط الهدف بناءً على مستويات السيولة
                if liquidity_levels:
                    nearest_liquidity = min(liquidity_levels, key=lambda x: abs(x - tp))
                    if abs(nearest_liquidity - tp) / tp * 100 < 0.3:  # ضبط إذا كان قريبًا من السيولة
                        tp = nearest_liquidity
                
                targets.append(tp)
        else:
            # أهداف أقل للترند المتوسط
            rr_ratios = [1.8, 3.0]
            for rr in rr_ratios:
                tp = entry_price + (risk * rr) if side == "buy" else entry_price - (risk * rr)
                targets.append(tp)
        
        return targets[:3]  # الحد الأقصى 3 أهداف
        
    except Exception as e:
        log_w(f"TP targets calculation error: {e}")
        return []

def calculate_invalidation_level(side, entry_price, sl_price):
    """حساب مستوى الإبطال"""
    buffer = abs(entry_price - sl_price) * 0.5  # buffer 50% من المخاطرة
    
    if side == "buy":
        return sl_price - buffer
    else:
        return sl_price + buffer

def build_trade_plan(df, indicators, council_data, price_info):
    """
    بناء خطة تداول ذكية قبل الدخول
    """
    try:
        current_price = price_info.get("price", 0)
        if current_price <= 0:
            return None
        
        # تحديد اتجاه الخطة
        signal_side = None
        if council_data["score_b"] > council_data["score_s"] and council_data["score_b"] >= 12.0:
            signal_side = "buy"
        elif council_data["score_s"] > council_data["score_b"] and council_data["score_s"] >= 12.0:
            signal_side = "sell"
        
        if not signal_side:
            return None
        
        # تحديد تصنيف الترند
        trend_class = determine_trend_class(df, indicators)
        
        # إنشاء خطة التداول
        plan = TradePlan(signal_side, trend_class)
        
        # تحليل الهيكل
        market_structure.analyze_structure(df)
        
        # تحديد أسباب الدخول
        entry_reasons = analyze_entry_reasons(df, indicators, signal_side, current_price)
        plan.entry_reason = entry_reasons
        
        # حساب مستويات السيولة
        liquidity_levels = calculate_liquidity_levels(df, signal_side, current_price)
        
        # إذا لم توجد أسباب كافية، رفض الخطة
        if not validate_entry_reasons(entry_reasons):
            log_i("[ENTRY BLOCKED] Weak location / No strong signals")
            return None
        
        # تحديد وقف الخسارة
        sl_level = calculate_smart_sl(df, signal_side, current_price, liquidity_levels)
        plan.sl = sl_level
        
        # تحديد أهداف الربح بناءً على السيولة
        tp_targets = calculate_tp_targets(df, signal_side, current_price, sl_level, 
                                         trend_class, liquidity_levels)
        plan.tp_targets = tp_targets
        
        # تحديد نسب الإغلاق
        plan.tp_fractions = [0.3, 0.3, 0.4] if trend_class == "large" else [0.5, 0.5, 0]
        
        # تحديد مستوى الإبطال
        plan.invalidation = calculate_invalidation_level(signal_side, current_price, sl_level)
        
        # تحديد نظام التريلينغ
        plan.trailing_mode = "structure" if trend_class == "large" else "hybrid"
        plan.breakeven_rule = "after_tp1"
        
        # حساب Risk/Reward
        if sl_level and tp_targets:
            risk = abs(current_price - sl_level) / current_price * 100
            reward = abs(tp_targets[0] - current_price) / current_price * 100 if tp_targets else 0
            plan.rr_expected = reward / risk if risk > 0 else 0
        
        # التحقق النهائي
        plan.valid = (plan.sl is not None and 
                     len(plan.tp_targets) > 0 and 
                     plan.invalidation is not None and 
                     plan.rr_expected >= 1.5)
        
        return plan if plan.is_valid() else None
        
    except Exception as e:
        log_e(f"Trade plan building error: {e}")
        return None

def log_trade_plan_details(plan, entry_price, position_size):
    """تسجيل تفاصيل خطة التداول"""
    log_i("━━━━━━━━ ENTRY APPROVED ━━━━━━━━")
    log_i(f"Side: {plan.side.upper()} | Trend: {plan.trend_class.upper()}")
    log_i(f"Entry Price: {entry_price:.6f} | Size: {position_size:.4f}")
    
    log_i("Entry Reasons:")
    for key, value in plan.entry_reason.items():
        if value:
            log_i(f"  • {key}: {value}")
    
    log_i("Plan:")
    log_i(f"  • SL: {plan.sl:.6f}")
    for i, tp in enumerate(plan.tp_targets):
        if i < len(plan.tp_fractions):
            log_i(f"  • TP{i+1}: {tp:.6f} ({plan.tp_fractions[i]*100}%)")
    
    log_i(f"  • Invalidation: {plan.invalidation:.6f}")
    log_i(f"  • Expected R/R: 1 : {plan.rr_expected:.1f}")
    log_i("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

# =================== ENHANCED TRADE EXECUTION ===================
def execute_intelligent_trade(side, price, qty, council_data, market_analysis):
    """تنفيذ صفقة ذكية مع تحليل متقدم"""
    try:
        if not EXECUTE_ORDERS or DRY_RUN:
            log_i(f"DRY_RUN: {side} {qty:.4f} @ {price:.6f}")
            return True
        
        if qty <= 0:
            log_e("❌ كمية غير صالحة للتنفيذ")
            return False
        
        # تحضير البيانات للتنفيذ
        confidence = council_data.get('confidence', 0)
        market_phase = market_analysis.get('market_phase', 'neutral')
        volatility_regime = market_analysis.get('volatility_regime', 'normal')
        
        log_i(f"🎯 EXECUTING INTELLIGENT TRADE:")
        log_i(f"   SIDE: {side.upper()}")
        log_i(f"   QTY: {qty:.4f}")
        log_i(f"   PRICE: {price:.6f}")
        log_i(f"   CONFIDENCE: {confidence:.2f}")
        log_i(f"   MARKET PHASE: {market_phase}")
        log_i(f"   VOLATILITY: {volatility_regime}")
        
        if MODE_LIVE:
            exchange_set_leverage(ex, LEVERAGE, SYMBOL)
            params = exchange_specific_params(side, is_close=False)
            ex.create_order(SYMBOL, "market", side, qty, None, params)
        
        log_g(f"✅ INTELLIGENT TRADE EXECUTED: {side.upper()} {qty:.4f} @ {price:.6f}")
        
        # تسجيل الصفقة في المدير
        trade_manager.record_trade(
            side=side,
            entry=price,
            exit_price=price,  # سيتم تحديثها عند الإغلاق
            quantity=qty,
            profit=0.0,  # سيتم تحديثها عند الإغلاق
            duration=0
        )
        
        return True
        
    except Exception as e:
        log_e(f"❌ INTELLIGENT TRADE EXECUTION FAILED: {e}")
        return False

def execute_intelligent_trade_with_plan(side, price, plan):
    """تنفيذ صفقة ذكية مع خطة"""
    try:
        if not plan or not plan.is_valid():
            log_e("❌ Cannot execute: Invalid trade plan")
            return False
        
        # حساب حجم الصفقة بناءً على الخطة
        balance = balance_usdt()
        if balance is None:
            balance = 1000.0  # رصيد افتراضي
            
        # حساب حجم متكيف
        position_size = compute_adaptive_position_size(
            balance, price, plan.rr_expected, plan.trend_class
        )
        
        if position_size <= 0:
            log_e("❌ Invalid position size")
            return False
        
        log_banner("🚀 INTELLIGENT TRADE EXECUTION")
        log_trade_plan_details(plan, price, position_size)
        
        # تنفيذ الصفقة
        success = execute_intelligent_trade(side, price, position_size, {}, {})
        
        if success:
            # حفظ الخطة في الحالة
            STATE["trade_plan"] = plan.summary()
            STATE["trade_plan_obj"] = plan
            STATE["tp_targets"] = plan.tp_targets
            STATE["tp_fractions"] = plan.tp_fractions
            STATE["tp_hits"] = [False] * len(plan.tp_targets)
            
            log_g("✅ Trade executed with intelligent plan")
            return True
        
        return False
        
    except Exception as e:
        log_e(f"❌ Intelligent trade execution failed: {e}")
        return False

def compute_adaptive_position_size(balance, price, confidence, market_phase):
    """حساب حجم صفقة متكيف مع ظروف السوق"""
    base_size = trade_manager.get_optimal_position_size(balance)
    
    # تعديل الحجم بناءً على الثقة
    confidence_multiplier = 0.5 + (confidence * 0.5)  # 0.5 إلى 1.0
    
    # تعديل الحجم بناءً على مرحلة السوق
    if market_phase in ["strong_bull", "strong_bear"]:
        market_multiplier = 1.3
    elif market_phase in ["bull", "bear"]:
        market_multiplier = 1.1
    else:
        market_multiplier = 0.8
    
    adaptive_size = base_size * confidence_multiplier * market_multiplier
    
    # التأكد من أن الحجم ضمن الحدود المعقولة
    max_position = balance * LEVERAGE * 0.8  # 80% من الرصيد بالرافعة
    final_size = min(adaptive_size, max_position / price) if price > 0 else adaptive_size
    
    log_i(f"📊 ADAPTIVE POSITION SIZING:")
    log_i(f"   Base: {base_size:.4f}")
    log_i(f"   Confidence Multiplier: {confidence_multiplier:.2f}")
    log_i(f"   Market Multiplier: {market_multiplier:.2f}")
    log_i(f"   Final: {final_size:.4f}")
    
    return safe_qty(final_size)

def close_market_strict(reason="manual"):
    """إغلاق صارم للمركز"""
    try:
        if not STATE["open"] or STATE["qty"] <= 0:
            return True
        
        side = STATE["side"]
        qty = STATE["qty"]
        close_side = "sell" if side == "long" else "buy"
        
        log_i(f"🔴 CLOSING POSITION: {reason}")
        
        if MODE_LIVE and EXECUTE_ORDERS and not DRY_RUN:
            params = exchange_specific_params(close_side, is_close=True)
            for attempt in range(CLOSE_RETRY_ATTEMPTS):
                try:
                    ex.create_order(SYMBOL, "market", close_side, qty, None, params)
                    log_g(f"✅ Position closed: {qty:.4f} {SYMBOL}")
                    break
                except Exception as e:
                    if attempt == CLOSE_RETRY_ATTEMPTS - 1:
                        raise
                    time.sleep(CLOSE_VERIFY_WAIT_S)
        
        # تحديث الحالة
        STATE.update({
            "open": False,
            "side": None,
            "entry": None,
            "qty": 0.0,
            "pnl": 0.0,
            "bars": 0,
            "trail": None,
            "breakeven": None,
            "tp1_done": False,
            "highest_profit_pct": 0.0,
            "profit_targets_achieved": 0,
            "trade_plan": None,
            "trade_plan_obj": None,
            "tp_targets": [],
            "tp_fractions": [],
            "tp_hits": []
        })
        
        return True
        
    except Exception as e:
        log_e(f"❌ Close market strict failed: {e}")
        return False

# =================== SMART TRADE MANAGEMENT ===================
def detect_fake_breakout(df, side):
    """كشف الاختراقات الكاذبة"""
    try:
        if len(df) < 5:
            return False
        
        recent_candles = df.tail(3)
        highs = recent_candles['high'].astype(float)
        lows = recent_candles['low'].astype(float)
        
        if side == "buy":
            # للشراء: تحقق من اختراق كاذب لأعلى
            if highs.iloc[-1] > highs.iloc[-2] and lows.iloc[-1] < lows.iloc[-2]:
                return True
        else:
            # للبيع: تحقق من اختراق كاذب لأسفل
            if lows.iloc[-1] < lows.iloc[-2] and highs.iloc[-1] > highs.iloc[-2]:
                return True
        
        return False
    except Exception as e:
        log_w(f"Fake breakout detection error: {e}")
        return False

def check_momentum_failure(df, side):
    """فحص فشل الزخم"""
    try:
        if len(df) < 10:
            return False
        
        rsi = talib.RSI(df['close'].astype(float), 14)
        if len(rsi) < 2:
            return False
            
        current_rsi = rsi.iloc[-1]
        prev_rsi = rsi.iloc[-2]
        
        if side == "buy" and current_rsi < 40 and current_rsi < prev_rsi:
            return True
        elif side == "sell" and current_rsi > 60 and current_rsi > prev_rsi:
            return True
        
        return False
    except Exception as e:
        log_w(f"Momentum check error: {e}")
        return False

def fail_fast_check(plan, current_price, df):
    """التحقق من شروط الخروج السريع"""
    try:
        # 1. تحقق من الإبطال
        if plan.invalidation:
            if (plan.side == "buy" and current_price <= plan.invalidation) or \
               (plan.side == "sell" and current_price >= plan.invalidation):
                log_i("❌ Invalidation level hit")
                return True
        
        # 2. تحقق من الاختراق الكاذب
        if detect_fake_breakout(df, plan.side):
            log_i("❌ Fake breakout detected")
            return True
        
        # 3. تحقق من ضعف الزخم
        if check_momentum_failure(df, plan.side):
            log_i("❌ Momentum failure detected")
            return True
        
        return False
        
    except Exception as e:
        log_w(f"Fail-fast check error: {e}")
        return False

def manage_tp_targets(plan, current_price, side, qty):
    """إدارة أهداف الربح"""
    try:
        tp_targets = STATE.get("tp_targets", [])
        tp_fractions = STATE.get("tp_fractions", [])
        tp_hits = STATE.get("tp_hits", [])
        
        for i, tp in enumerate(tp_targets):
            if i >= len(tp_hits) or tp_hits[i]:
                continue
            
            # تحقق من تحقيق الهدف
            target_hit = False
            if side == "long" and current_price >= tp:
                target_hit = True
            elif side == "short" and current_price <= tp:
                target_hit = True
            
            if target_hit:
                # إغلاق جزء من الصفقة
                close_fraction = tp_fractions[i] if i < len(tp_fractions) else 0.3
                close_qty = safe_qty(qty * close_fraction)
                
                if close_qty > 0:
                    close_side = "sell" if side == "long" else "buy"
                    
                    if MODE_LIVE and EXECUTE_ORDERS and not DRY_RUN:
                        try:
                            params = exchange_specific_params(close_side, is_close=True)
                            ex.create_order(SYMBOL, "market", close_side, close_qty, None, params)
                            
                            log_g(f"🎯 TP{i+1} HIT: Closed {close_fraction*100}% of position")
                            
                            # تحديث الكمية
                            STATE["qty"] = safe_qty(qty - close_qty)
                            tp_hits[i] = True
                            STATE["tp_hits"] = tp_hits
                            
                            # تحريك SL إلى نقطة التعادل بعد TP1
                            if i == 0 and plan.breakeven_rule == "after_tp1":
                                STATE["breakeven"] = STATE["entry"]
                                STATE["breakeven_active"] = True
                                log_i("🔄 Breakeven activated after TP1")
                            
                        except Exception as e:
                            log_e(f"❌ Partial close failed: {e}")
                
    except Exception as e:
        log_w(f"TP targets management error: {e}")

def manage_structure_trailing(plan, current_price, side, df):
    """التريلينغ بناءً على الهيكل"""
    try:
        if not market_structure.structure_levels:
            return
        
        # البحث عن أقرب مستوى هيكلي مناسب
        if side == "long":
            support_levels = [l for l in market_structure.structure_levels 
                            if l['type'] == 'support' and l['price'] < current_price]
            if support_levels:
                closest_support = max(support_levels, key=lambda x: x['price'])
                new_trail = closest_support['price'] * 0.995  # هامش 0.5%
                
                if STATE.get("trail") is None or new_trail > STATE["trail"]:
                    STATE["trail"] = new_trail
                    if new_trail > STATE.get("entry", 0):
                        log_i(f"🔼 Structure trail updated: {STATE['trail']:.6f}")
        else:
            resistance_levels = [l for l in market_structure.structure_levels 
                               if l['type'] == 'resistance' and l['price'] > current_price]
            if resistance_levels:
                closest_resistance = min(resistance_levels, key=lambda x: x['price'])
                new_trail = closest_resistance['price'] * 1.005  # هامش 0.5%
                
                if STATE.get("trail") is None or new_trail < STATE["trail"]:
                    STATE["trail"] = new_trail
                    if new_trail < STATE.get("entry", float('inf')):
                        log_i(f"🔽 Structure trail updated: {STATE['trail']:.6f}")
                        
    except Exception as e:
        log_w(f"Structure trailing error: {e}")

def manage_hybrid_trailing(plan, current_price, side, atr):
    """التريلينغ الهجين"""
    try:
        pnl_pct = STATE.get("pnl", 0)
        
        # تعديل مضاعف ATR بناءً على الربح
        if pnl_pct > 5.0:
            atr_mult = 1.0
        elif pnl_pct > 2.0:
            atr_mult = 1.2
        else:
            atr_mult = 1.5
        
        if side == "long":
            new_trail = current_price - (atr * atr_mult)
            if STATE.get("trail") is None or new_trail > STATE["trail"]:
                STATE["trail"] = new_trail
        else:
            new_trail = current_price + (atr * atr_mult)
            if STATE.get("trail") is None or new_trail < STATE["trail"]:
                STATE["trail"] = new_trail
                
    except Exception as e:
        log_w(f"Hybrid trailing error: {e}")

def manage_sl_trailing(plan, current_price, side, df):
    """إدارة وقف الخسارة المتحرك"""
    try:
        if not plan.trailing_mode:
            return
        
        atr = safe_get(compute_indicators(df), 'atr', 0.001)
        pnl_pct = STATE.get("pnl", 0)
        
        # تفعيل التريلينغ بعد تحقيق ربح معين
        if not STATE.get("trail_active") and pnl_pct >= 1.0:
            STATE["trail_active"] = True
            log_i("🔄 Trailing stop activated")
        
        if STATE.get("trail_active"):
            if plan.trailing_mode == "structure":
                # تريلينغ بناءً على الهيكل
                manage_structure_trailing(plan, current_price, side, df)
            else:
                # تريلينغ هجين
                manage_hybrid_trailing(plan, current_price, side, atr)
                
    except Exception as e:
        log_w(f"SL trailing management error: {e}")

def manage_intelligent_position_with_plan(df, indicators, price_info):
    """إدارة ذكية للمراكز مع خطة"""
    if not STATE["open"] or STATE["qty"] <= 0:
        return
    
    try:
        current_price = price_info.get("price", 0)
        entry_price = STATE["entry"]
        side = STATE["side"]
        qty = STATE["qty"]
        
        # الحصول على خطة التداول
        plan = STATE.get("trade_plan_obj")
        if not plan:
            log_w("⚠️ No trade plan found - using basic management")
            return
        
        # حساب الربح/الخسارة
        if side == "long":
            pnl_pct = (current_price - entry_price) / entry_price * 100
        else:
            pnl_pct = (entry_price - current_price) / entry_price * 100
        
        STATE["pnl"] = pnl_pct
        
        # التحقق من FAIL-FAST
        if fail_fast_check(plan, current_price, df):
            log_i("🔴 FAIL-FAST: Closing trade early")
            close_market_strict("fail_fast_check")
            return
        
        # إدارة أهداف الربح
        manage_tp_targets(plan, current_price, side, qty)
        
        # إدارة وقف الخسارة
        manage_sl_trailing(plan, current_price, side, df)
        
        # تحديث الهيكل
        market_structure.analyze_structure(df)
        
    except Exception as e:
        log_e(f"❌ Intelligent position management error: {e}")

def intelligent_exit_decision(pnl_pct, side, indicators, market_phase, volatility_regime):
    """قرار خروج ذكي بناءً على متعددة معايير"""
    try:
        rsi = indicators.get('rsi', 50)
        adx = indicators.get('adx', 0)
        macd_hist = indicators.get('macd_hist', 0)
        
        # إستراتيجية الخروج بناءً على مرحلة السوق
        if market_phase in ["strong_bull", "strong_bear"]:
            # في الترند القوي، نبقى لفترة أطول
            tp_targets = [1.0, 2.0, 3.5, 5.0, 7.0, 10.0]
        else:
            # في السوق الجانبي، نخرج مبكراً
            tp_targets = [0.8, 1.5, 2.5, 4.0, 6.0]
        
        # تحقيق أهداف الربح
        for i, target in enumerate(tp_targets):
            tp_key = f"tp_{i+1}_done"
            if not STATE.get(tp_key, False) and pnl_pct >= target:
                close_pct = 0.2 if i < 3 else 0.15  # 20% للأهداف الأولى، 15% للبقية
                return {
                    "action": "partial",
                    "qty_pct": close_pct,
                    "reason": f"TP{i+1} achieved: {target:.1f}%"
                }
        
        # إشارات انعكاس قوية
        reversal_signals = 0
        if (side == "long" and rsi > 80 and macd_hist < 0) or (side == "short" and rsi < 20 and macd_hist > 0):
            reversal_signals += 1
        
        if adx < 20 and abs(pnl_pct) > 2.0:  # فقدان الزخم مع ربح جيد
            reversal_signals += 1
        
        if reversal_signals >= 2:
            return {
                "action": "close",
                "reason": "Strong reversal signals"
            }
        
        # خروج وقائي في التقلب العالي مع أرباح جيدة
        if volatility_regime == "high" and pnl_pct > 3.0:
            return {
                "action": "close",
                "reason": "High volatility with good profit - secure gains"
            }
        
        return {"action": "hold", "reason": "Continue riding trend"}
        
    except Exception as e:
        log_w(f"Intelligent exit decision error: {e}")
        return {"action": "hold", "reason": "Error in decision"}

def update_intelligent_trailing_stop(current_price, side, indicators, market_phase):
    """تحديث وقف الخسارة المتحرك الذكي"""
    try:
        atr = indicators.get('atr', 0)
        pnl_pct = STATE.get("pnl", 0)
        
        # تحديد مضاعف ATR بناءً على مرحلة السوق والتقلب
        if market_phase in ["strong_bull", "strong_bear"]:
            base_multiplier = 2.0
        else:
            base_multiplier = 1.5
        
        # تعديل المضاعف بناءً على مستوى الربح
        if pnl_pct > 5.0:
            trail_mult = base_multiplier * 0.7  # وقف أضيق عند الأرباح العالية
        elif pnl_pct > 2.0:
            trail_mult = base_multiplier * 0.8
        else:
            trail_mult = base_multiplier
        
        if not STATE.get("trail_active", False) and pnl_pct >= 1.0:
            STATE["trail_active"] = True
            STATE["breakeven_armed"] = True
            STATE["breakeven"] = STATE["entry"]
            log_i("🔄 Intelligent trailing stop activated")
        
        if STATE.get("trail_active"):
            if side == "long":
                new_trail = current_price - (atr * trail_mult)
                if STATE.get("trail") is None or new_trail > STATE["trail"]:
                    STATE["trail"] = new_trail
                    if STATE["trail"] > STATE.get("entry", 0):
                        log_i(f"🔼 Intelligent trail updated: {STATE['trail']:.6f}")
            else:
                new_trail = current_price + (atr * trail_mult)
                if STATE.get("trail") is None or new_trail < STATE["trail"]:
                    STATE["trail"] = new_trail
                    if STATE["trail"] < STATE.get("entry", float('inf')):
                        log_i(f"🔽 Intelligent trail updated: {STATE['trail']:.6f}")
        
        # تفعيل وقف الخسارة عند نقطة التعادل بعد تحقيق ربح معين
        if STATE.get("breakeven_armed") and not STATE.get("breakeven_active") and pnl_pct >= 1.5:
            STATE["breakeven_active"] = True
            STATE["trail"] = STATE["entry"]  # وقف عند نقطة الدخول
            log_i("🎯 Breakeven activated - risk free trade")
            
    except Exception as e:
        log_w(f"Intelligent trailing stop error: {e}")

# =================== ULTRA INTELLIGENT TRADING LOOP ===================
def ultra_intelligent_trading_loop():
    """الحلقة الرئيسية للتداول الذكي الفائق"""
    global wait_for_next_signal_side
    
    log_banner("STARTING ULTRA INTELLIGENT TRADING BOT")
    log_i(f"🤖 Bot Version: {BOT_VERSION}")
    log_i(f"💱 Exchange: {EXCHANGE_NAME.upper()}")
    log_i(f"📈 Symbol: {SYMBOL}")
    log_i(f"⏰ Interval: {INTERVAL}")
    log_i(f"🎯 Leverage: {LEVERAGE}x")
    log_i(f"📊 Risk Allocation: {RISK_ALLOC*100}%")
    
    while True:
        try:
            # جمع البيانات الأساسية
            balance = balance_usdt()
            current_price = price_now()
            df = fetch_ohlcv(limit=200)  # المزيد من البيانات للتحليل المتقدم
            
            if df.empty or current_price is None:
                log_w("📭 No data available - retrying...")
                time.sleep(BASE_SLEEP)
                continue
            
            # تحليل السوق المتقدم
            market_phase = market_analyzer.detect_market_phase(df)
            support_resistance = market_analyzer.calculate_support_resistance(df)
            volatility_regime, volatility_ratio = market_analyzer.analyze_volatility_regime(df)
            
            # قرار مجلس الإدارة الذكي
            council_data = ultra_intelligent_council_ai(df)
            
            # تحديث الحالة
            STATE["last_council"] = council_data
            STATE["last_ind"] = council_data.get("indicators", {})
            STATE["last_spread_bps"] = orderbook_spread_bps()
            
            # عرض معلومات السوق
            if LOG_ADDONS:
                log_i(f"🏪 MARKET: {market_phase.upper()} | VOLATILITY: {volatility_regime} ({volatility_ratio:.2f})")
                log_i(f"🎯 COUNCIL: B{council_data['b']}/S{council_data['s']} | "
                      f"Score: {council_data['score_b']:.1f}/{council_data['score_s']:.1f} | "
                      f"Confidence: {council_data['confidence']:.2f}")
                
                for log_msg in council_data.get("logs", [])[-5:]:  # آخر 5 رسائل فقط
                    log_i(f"   {log_msg}")
            
            # إدارة المركز المفتوح
            if STATE["open"]:
                manage_intelligent_position_with_plan(df, council_data.get("indicators", {}), {
                    "price": current_price,
                    "market_phase": market_phase,
                    "volatility_regime": volatility_regime
                })
            else:
                # بناء خطة تداول ذكية
                trade_plan = build_trade_plan(
                    df, 
                    council_data.get("indicators", {}), 
                    council_data,
                    {"price": current_price}
                )
                
                # التحقق من صحة الخطة
                if trade_plan and trade_plan.is_valid():
                    # تنفيذ الصفقة
                    signal_side = trade_plan.side
                    success = execute_intelligent_trade_with_plan(
                        signal_side, 
                        current_price, 
                        trade_plan
                    )
                    
                    if success:
                        STATE.update({
                            "open": True,
                            "side": "long" if signal_side == "buy" else "short",
                            "entry": current_price,
                            "qty": STATE.get("qty", 0),
                            "pnl": 0.0,
                            "bars": 0,
                            "trail": None,
                            "breakeven": None,
                            "highest_profit_pct": 0.0,
                            "profit_targets_achieved": 0,
                            "mode": f"intelligent_{trade_plan.trend_class}"
                        })
                        
                        save_state({
                            "in_position": True,
                            "side": signal_side.upper(),
                            "entry_price": current_price,
                            "position_qty": STATE.get("qty", 0),
                            "opened_at": int(time.time()),
                            "trade_plan": trade_plan.summary()
                        })
                        
                        print_position_snapshot("INTELLIGENT_OPEN")
                else:
                    if trade_plan:
                        log_i(f"[ENTRY BLOCKED] Weak plan - R/R: {trade_plan.rr_expected:.1f}")
                    else:
                        log_i("[ENTRY BLOCKED] No valid plan generated")
            
            # التحقق من تحقيق الهدف اليومي
            if trade_manager.daily_profit >= PROFIT_TARGET_DAILY:
                log_g(f"🎉 DAILY PROFIT TARGET ACHIEVED: {trade_manager.daily_profit:.2f} USDT")
                if STATE["open"]:
                    log_i("🔒 Locking profits - closing all positions")
                    close_market_strict("daily_target_achieved")
            
            # الانتظار للدورة التالية
            sleep_time = NEAR_CLOSE_S if time_to_candle_close(df) <= 10 else BASE_SLEEP
            time.sleep(sleep_time)
            
        except Exception as e:
            log_e(f"❌ ULTRA INTELLIGENT TRADING LOOP ERROR: {e}")
            log_e(traceback.format_exc())
            time.sleep(BASE_SLEEP * 2)

# =================== STATE INITIALIZATION ===================
STATE = {
    "open": False, "side": None, "entry": None, "qty": 0.0,
    "pnl": 0.0, "bars": 0, "trail": None, "breakeven": None,
    "tp1_done": False, "highest_profit_pct": 0.0,
    "profit_targets_achieved": 0,
    "trade_plan": None,
    "trade_plan_obj": None,
    "tp_targets": [],
    "tp_fractions": [],
    "tp_hits": [],
    "fail_fast_checks": 0
}

compound_pnl = 0.0
wait_for_next_signal_side = None

# =================== FLASK API ===================
app = Flask(__name__)

@app.route("/")
def home():
    return f"""
    <html>
        <head><title>SUI ULTRA PRO AI BOT</title></head>
        <body>
            <h1>🚀 SUI ULTRA PRO AI BOT - الإصدار الذكي المتقدم</h1>
            <p><strong>Version:</strong> {BOT_VERSION}</p>
            <p><strong>Exchange:</strong> {EXCHANGE_NAME.upper()}</p>
            <p><strong>Symbol:</strong> {SYMBOL}</p>
            <p><strong>Status:</strong> {'🟢 LIVE' if MODE_LIVE else '🟡 PAPER'}</p>
            <p><strong>Daily PnL:</strong> {trade_manager.daily_profit:.2f} USDT</p>
            <p><strong>Win Rate:</strong> {trade_manager.win_rate:.1f}%</p>
            <p><a href="/health">Health Check</a> | <a href="/metrics">Metrics</a> | <a href="/performance">Performance</a></p>
        </body>
    </html>
    """

@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "exchange": EXCHANGE_NAME,
        "symbol": SYMBOL,
        "position_open": STATE["open"],
        "daily_profit": trade_manager.daily_profit,
        "win_rate": trade_manager.win_rate
    })

@app.route("/metrics")
def metrics():
    return jsonify({
        "bot_version": BOT_VERSION,
        "exchange": EXCHANGE_NAME,
        "symbol": SYMBOL,
        "balance": balance_usdt(),
        "daily_profit": trade_manager.daily_profit,
        "win_rate": trade_manager.win_rate,
        "consecutive_wins": trade_manager.consecutive_wins,
        "consecutive_losses": trade_manager.consecutive_losses,
        "total_trades": len(trade_manager.trade_history),
        "position": STATE,
        "performance_suggestions": trade_manager.get_trade_suggestions()
    })

@app.route("/performance")
def performance():
    recent_trades = trade_manager.trade_history[-10:]  # آخر 10 صفقات
    return jsonify({
        "daily_profit": trade_manager.daily_profit,
        "win_rate": trade_manager.win_rate,
        "avg_win": trade_manager.avg_win,
        "avg_loss": trade_manager.avg_loss,
        "recent_trades": [
            {
                "time": t['timestamp'].strftime('%H:%M:%S'),
                "side": t['side'],
                "profit": t['profit'],
                "profit_pct": t['profit_pct']
            } for t in recent_trades
        ]
    })

@app.get("/mark/<color>")
def mark_position(color):
    color = color.lower()
    if color not in ["green", "red"]:
        return jsonify({"ok": False, "error": "Use /mark/green or /mark/red"}), 400
    
    print_position_snapshot(reason="MANUAL_MARK", color=color)
    return jsonify({"ok": True, "marked": color, "timestamp": datetime.now().isoformat()})

# =================== STARTUP ===================
def startup_sequence():
    """تسلسل بدء التشغيل"""
    log_banner("SYSTEM INITIALIZATION")
    
    # تحميل الحالة السابقة
    loaded_state = load_state()
    if loaded_state:
        log_g("✅ Previous state loaded successfully")
    
    # التحقق من اتصال البورصة
    try:
        balance = balance_usdt()
        price = price_now()
        log_g(f"✅ Exchange connection successful")
        log_g(f"💰 Balance: {balance:.2f} USDT")
        log_g(f"💰 Current price: {price:.6f}")
    except Exception as e:
        log_e(f"❌ Exchange connection failed: {e}")
        return False
    
    # عرض إحصائيات البوت
    log_i(f"📊 Performance Metrics:")
    log_i(f"   Win Rate: {trade_manager.win_rate:.1f}%")
    log_i(f"   Daily PnL: {trade_manager.daily_profit:.2f} USDT")
    log_i(f"   Consecutive Wins: {trade_manager.consecutive_wins}")
    log_i(f"   Consecutive Losses: {trade_manager.consecutive_losses}")
    
    log_g("🚀 ULTRA INTELLIGENT TRADING BOT READY!")
    return True

# =================== MAIN EXECUTION ===================
if __name__ == "__main__":
    # إعداد معالجات الإشارات
    def signal_handler(signum, frame):
        log_i(f"🛑 Received signal {signum} - Shutting down gracefully...")
        save_state(STATE)
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # بدء التشغيل
    if startup_sequence():
        # بدء خيوط التنفيذ
        import threading
        
        # خيط التداول الرئيسي
        trading_thread = threading.Thread(target=ultra_intelligent_trading_loop, daemon=True)
        trading_thread.start()
        
        # خيط الحفاظ على الحالة
        def state_saver():
            while True:
                time.sleep(300)  # حفظ كل 5 دقائق
                save_state(STATE)
        
        state_thread = threading.Thread(target=state_saver, daemon=True)
        state_thread.start()
        
        log_g(f"🌐 Starting web server on port {PORT}")
        
        # تشغيل سيرفل الويب
        try:
            app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
        except Exception as e:
            log_e(f"❌ Web server failed: {e}")
    else:
        log_e("❌ Startup failed - check configuration and try again")
