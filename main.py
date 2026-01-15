# -*- coding: utf-8 -*-
"""
SUI ULTRA PRO AI BOT v9.1 - النسخة الذكية المحسنة مع حماية من الخسائر
• نظام مؤشرات بديل باستخدام pandas-ta بدلاً من TA-Lib
• متوافق مع Render وغيرها من المنصات السحابية
"""

import os, time, math, random, signal, sys, traceback, logging, json
from logging.handlers import RotatingFileHandler
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import ccxt
from flask import Flask, jsonify
from decimal import Decimal, ROUND_DOWN, InvalidOperation
import pandas_ta as ta  # استبدال talib بـ pandas_ta
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
BOT_VERSION = f"SUI ULTRA PRO AI v9.1 — {EXCHANGE_NAME.upper()}"
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

# ==== Enhanced Protection Settings ====
CONFIRMATION_TIMEFRAMES = ["15m", "5m", "1h"]
MIN_CONFIRMATIONS = 2
DEAD_ZONE_PCT = 0.15
COOLDOWN_AFTER_EXIT = 600
MIN_HOLD_TIME = 300

# =================== MARKET SPECS ===================
MARKET = {}
AMT_PREC = 0
LOT_STEP = None
LOT_MIN  = None

# =================== STATE INITIALIZATION ===================
STATE = {
    "open": False, "side": None, "entry": None, "qty": 0.0,
    "pnl": 0.0, "bars": 0, "trail": None, "breakeven": None,
    "tp1_done": False, "highest_profit_pct": 0.0,
    "profit_targets_achieved": 0,
    "last_entry_price": None,
    "current_price": 0.0,
    "market_phase": "neutral",
    "volatility_regime": "normal",
    "entry_time": 0,
    "minimum_hold_until": 0
}

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

# =================== ADVANCED MARKET ANALYSIS ===================
class AdvancedMarketAnalyzer:
    def __init__(self):
        self.market_phases = []
        
    def detect_market_phase(self, df):
        """اكتشاف مرحلة السوق الحالية"""
        try:
            close = df['close'].astype(float)
            
            # حساب المتوسطات المتحركة باستخدام pandas
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
            log_w(f"detect_market_phase error: {e}")
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
                'resistance_levels': below_resistance.tolist()
            }
        except Exception as e:
            log_w(f"calculate_support_resistance error: {e}")
            return {'support_levels': [], 'resistance_levels': []}
    
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
            log_w(f"analyze_volatility_regime error: {e}")
            return "normal", 1.0

# =================== PANDAS-TA INDICATORS ===================
def compute_advanced_indicators(df):
    """حساب المؤشرات المتقدمة باستخدام pandas-ta"""
    try:
        close = df['close'].astype(float)
        high = df['high'].astype(float)
        low = df['low'].astype(float)
        volume = df['volume'].astype(float)
        
        # مؤشرات الترند باستخدام pandas-ta
        sma_20 = ta.sma(close, length=20)
        sma_50 = ta.sma(close, length=50)
        ema_20 = ta.ema(close, length=20)
        
        # مؤشرات الزخم
        rsi = ta.rsi(close, length=14)
        
        # MACD
        macd_result = ta.macd(close)
        macd = macd_result['MACD_12_26_9'] if macd_result is not None else None
        macd_signal = macd_result['MACDs_12_26_9'] if macd_result is not None else None
        macd_hist = macd_result['MACDh_12_26_9'] if macd_result is not None else None
        
        # ستوكاستك
        stoch_result = ta.stoch(high, low, close)
        stoch_k = stoch_result['STOCHk_14_3_3'] if stoch_result is not None else None
        stoch_d = stoch_result['STOCHd_14_3_3'] if stoch_result is not None else None
        
        # مؤشرات التقلب
        atr = ta.atr(high, low, close, length=14)
        
        # بولنجر باندز
        bb_result = ta.bbands(close, length=20, std=2)
        bollinger_upper = bb_result['BBU_20_2.0'] if bb_result is not None else None
        bollinger_middle = bb_result['BBM_20_2.0'] if bb_result is not None else None
        bollinger_lower = bb_result['BBL_20_2.0'] if bb_result is not None else None
        
        # مؤشرات الحجم
        obv = ta.obv(close, volume)
        
        # مؤشرات الاتجاه
        adx_result = ta.adx(high, low, close, length=14)
        adx = adx_result['ADX_14'] if adx_result is not None else None
        plus_di = adx_result['DMP_14'] if adx_result is not None else None
        minus_di = adx_result['DMN_14'] if adx_result is not None else None
        
        def safe_last(x):
            """استخراج القيمة الأخيرة بأمان"""
            try:
                if x is None or pd.isna(x.iloc[-1]):
                    return 0.0
                return float(x.iloc[-1])
            except:
                return 0.0
        
        return {
            'sma_20': safe_last(sma_20),
            'sma_50': safe_last(sma_50),
            'ema_20': safe_last(ema_20),
            'rsi': safe_last(rsi),
            'macd': safe_last(macd),
            'macd_signal': safe_last(macd_signal),
            'macd_hist': safe_last(macd_hist),
            'stoch_k': safe_last(stoch_k),
            'stoch_d': safe_last(stoch_d),
            'atr': safe_last(atr),
            'bollinger_upper': safe_last(bollinger_upper),
            'bollinger_middle': safe_last(bollinger_middle),
            'bollinger_lower': safe_last(bollinger_lower),
            'obv': safe_last(obv),
            'adx': safe_last(adx),
            'plus_di': safe_last(plus_di),
            'minus_di': safe_last(minus_di),
            'volume': float(volume.iloc[-1]) if len(volume) > 0 else 0.0
        }
    except Exception as e:
        log_w(f"Advanced indicators error: {e}")
        return {}

# =================== TRADE MANAGER ===================
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
            
        self.calculate_performance_metrics()
        
    def calculate_performance_metrics(self):
        """حساب مقاييس الأداء"""
        if not self.trade_history:
            return
            
        wins = [t for t in self.trade_history if t['profit'] > 0]
        losses = [t for t in self.trade_history if t['profit'] <= 0]
        
        self.win_rate = len(wins) / len(self.trade_history) * 100 if self.trade_history else 0
        
        if wins:
            self.avg_win = sum(t['profit'] for t in wins) / len(wins)
        if losses:
            self.avg_loss = abs(sum(t['profit'] for t in losses) / len(losses))
    
    def get_optimal_position_size(self, balance, risk_per_trade=0.02):
        """حساب حجم الصفقة الأمثل بناءً على الأداء"""
        base_size = balance * risk_per_trade
        
        if self.consecutive_wins >= 3:
            size_multiplier = min(2.0, 1.0 + (self.consecutive_wins * 0.1))
        elif self.consecutive_losses >= 2:
            size_multiplier = max(0.5, 1.0 - (self.consecutive_losses * 0.2))
        else:
            size_multiplier = 1.0
            
        return base_size * size_multiplier

# =================== HELPER FUNCTIONS ===================
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

def fetch_ohlcv(limit=200):
    try:
        rows = ex.fetch_ohlcv(SYMBOL, timeframe=INTERVAL, limit=limit, params={"type":"swap"})
        return pd.DataFrame(rows, columns=["time","open","high","low","close","volume"])
    except Exception as e:
        log_w(f"fetch_ohlcv error: {e}")
        return pd.DataFrame()

def price_now():
    try:
        t = ex.fetch_ticker(SYMBOL)
        return t.get("last") or t.get("close")
    except Exception: 
        return None

def balance_usdt():
    if not MODE_LIVE: return 1000.0
    try:
        b = ex.fetch_balance(params={"type":"swap"})
        return b.get("total",{}).get("USDT") or b.get("free",{}).get("USDT")
    except Exception: 
        return None

def orderbook_spread_bps():
    try:
        ob = ex.fetch_order_book(SYMBOL, limit=5)
        bid = ob["bids"][0][0] if ob["bids"] else None
        ask = ob["asks"][0][0] if ob["asks"] else None
        if not (bid and ask): return None
        mid = (bid+ask)/2.0
        return ((ask-bid)/mid)*10000.0
    except Exception:
        return None

# =================== LIQUIDITY ENGINE ===================
class LiquidityEngine:
    def __init__(self):
        self.swings = []
        
    def detect_liquidity_sweep(self, df, side="buy"):
        """اكتشاف مسح سيولة حقيقي"""
        if len(df) < 20:
            return False
            
        highs = df['high'].astype(float).tail(20)
        lows = df['low'].astype(float).tail(20)
        
        if side == "buy":
            recent_low = lows.min()
            idx_low = lows.idxmin()
            
            if idx_low >= 15:
                prev_lows = lows.iloc[idx_low-5:idx_low]
                if recent_low < prev_lows.min():
                    candle_at_low = df.iloc[idx_low]
                    wick_size = candle_at_low['high'] - candle_at_low['low']
                    body_size = abs(candle_at_low['close'] - candle_at_low['open'])
                    
                    if wick_size > body_size * 2:
                        close_position = (candle_at_low['close'] - candle_at_low['low']) / wick_size
                        if close_position > 0.7:
                            next_candles = df.iloc[idx_low+1:idx_low+4] if idx_low+4 < len(df) else []
                            if len(next_candles) >= 2:
                                if all(c['close'] > c['open'] for _, c in next_candles.head(2).iterrows()):
                                    return True
        else:
            recent_high = highs.max()
            idx_high = highs.idxmax()
            
            if idx_high >= 15:
                prev_highs = highs.iloc[idx_high-5:idx_high]
                if recent_high > prev_highs.max():
                    candle_at_high = df.iloc[idx_high]
                    wick_size = candle_at_high['high'] - candle_at_high['low']
                    body_size = abs(candle_at_high['close'] - candle_at_high['open'])
                    
                    if wick_size > body_size * 2:
                        close_position = (candle_at_high['high'] - candle_at_high['close']) / wick_size
                        if close_position > 0.7:
                            next_candles = df.iloc[idx_high+1:idx_high+4] if idx_high+4 < len(df) else []
                            if len(next_candles) >= 2:
                                if all(c['close'] < c['open'] for _, c in next_candles.head(2).iterrows()):
                                    return True
        return False

# =================== ANTI-REVERSAL GUARD ===================
class AntiReversalGuard:
    def __init__(self):
        self.last_exit_time = None
        self.last_exit_side = None
        
    def can_enter(self, side, current_time=None):
        """التحقق إذا مسموح بالدخول"""
        if current_time is None:
            current_time = time.time()
            
        # فترة التبريد بعد الخروج
        if self.last_exit_time:
            time_since_exit = current_time - self.last_exit_time
            if time_since_exit < COOLDOWN_AFTER_EXIT:
                return False, f"في فترة تبريد ({int(time_since_exit)}s/{COOLDOWN_AFTER_EXIT}s)"
                
        # منع العكس المباشر
        if self.last_exit_side and side != self.last_exit_side:
            return False, "منع عكس بدون كسر هيكل"
                    
        # التحقق من مدة البقاء الأدنى للصفقة الحالية
        if STATE.get("open"):
            entry_time = STATE.get("entry_time", 0)
            time_in_trade = current_time - entry_time
            if time_in_trade < MIN_HOLD_TIME:
                return False, f"أقل من مدة البقاء الأدنى ({int(time_in_trade)}s/{MIN_HOLD_TIME}s)"
                
        return True, "مسموح"
    
    def record_exit(self, side):
        """تسجيل خروج"""
        self.last_exit_time = time.time()
        self.last_exit_side = side
        
    def record_entry(self, side):
        """تسجيل دخول"""
        STATE["entry_time"] = time.time()
        STATE["minimum_hold_until"] = time.time() + MIN_HOLD_TIME

# =================== MAIN TRADING LOGIC ===================
def simplified_council_ai(df):
    """نسخة مبسطة من مجلس الذكاء الاصطناعي"""
    try:
        if len(df) < 50:
            return {"score_b": 0.0, "score_s": 0.0, "confidence": 0.0, "logs": []}
        
        # الحصول على المؤشرات
        indicators = compute_advanced_indicators(df)
        
        # تحليل السوق
        market_phase = market_analyzer.detect_market_phase(df)
        volatility_regime, volatility_ratio = market_analyzer.analyze_volatility_regime(df)
        
        score_b = 0.0
        score_s = 0.0
        logs = []
        
        current_price = float(df['close'].iloc[-1])
        
        # 1. تحليل مرحلة السوق
        if market_phase == "strong_bull":
            score_b += 2.5
            logs.append("📈 مرحلة صاعدة قوية")
        elif market_phase == "bull":
            score_b += 1.5
            logs.append("📈 مرحلة صاعدة")
        elif market_phase == "bear":
            score_s += 1.5
            logs.append("📉 مرحلة هابطة")
        elif market_phase == "strong_bear":
            score_s += 2.5
            logs.append("📉 مرحلة هابطة قوية")
        
        # 2. تحليل RSI
        rsi = indicators.get('rsi', 50)
        if rsi < 30:
            score_b += 2.0
            logs.append("📊 RSI في منطقة شراء قوية")
        elif rsi > 70:
            score_s += 2.0
            logs.append("📊 RSI في منطقة بيع قوية")
        
        # 3. تحليل MACD
        macd_hist = indicators.get('macd_hist', 0)
        if macd_hist > 0:
            score_b += 1.5
            logs.append("📈 MACD صاعد")
        elif macd_hist < 0:
            score_s += 1.5
            logs.append("📉 MACD هابط")
        
        # 4. تحليل ADX
        adx = indicators.get('adx', 0)
        plus_di = indicators.get('plus_di', 0)
        minus_di = indicators.get('minus_di', 0)
        
        if adx > 25:
            if plus_di > minus_di:
                score_b += 2.0
                logs.append(f"🎯 ترند صاعد قوي (ADX: {adx:.1f})")
            else:
                score_s += 2.0
                logs.append(f"🎯 ترند هابط قوي (ADX: {adx:.1f})")
        
        # 5. تحليل بولنجر باندز
        bb_lower = indicators.get('bollinger_lower', current_price)
        bb_upper = indicators.get('bollinger_upper', current_price)
        
        if current_price <= bb_lower * 1.01:
            score_b += 1.8
            logs.append("📏 سعر قرب النطاق السفلي - شراء")
        elif current_price >= bb_upper * 0.99:
            score_s += 1.8
            logs.append("📏 سعر قرب النطاق العلوي - بيع")
        
        # حساب الثقة
        total_score = score_b + score_s
        confidence = min(1.0, total_score / 10.0)
        
        return {
            "score_b": round(score_b, 2),
            "score_s": round(score_s, 2),
            "confidence": round(confidence, 2),
            "logs": logs[-5:],  # آخر 5 رسائل فقط
            "market_phase": market_phase,
            "volatility_regime": volatility_regime,
            "indicators": indicators
        }
        
    except Exception as e:
        log_w(f"Council AI error: {e}")
        return {"score_b": 0.0, "score_s": 0.0, "confidence": 0.0, "logs": []}

def execute_trade(side, price, qty):
    """تنفيذ صفقة"""
    try:
        if DRY_RUN:
            log_i(f"DRY_RUN: {side} {qty:.4f} @ {price:.6f}")
            return True
        
        if not EXECUTE_ORDERS:
            log_i(f"EXECUTION DISABLED: {side} {qty:.4f} @ {price:.6f}")
            return True
        
        if qty <= 0:
            log_e("❌ كمية غير صالحة للتنفيذ")
            return False
        
        log_i(f"🎯 EXECUTING TRADE: {side.upper()} {qty:.4f} @ {price:.6f}")
        
        if MODE_LIVE:
            exchange_set_leverage(ex, LEVERAGE, SYMBOL)
            params = exchange_specific_params(side, is_close=False)
            ex.create_order(SYMBOL, "market", side, qty, None, params)
        
        log_g(f"✅ TRADE EXECUTED: {side.upper()} {qty:.4f} @ {price:.6f}")
        
        # تسجيل الصفقة
        trade_manager.record_trade(
            side=side,
            entry=price,
            exit_price=price,
            quantity=qty,
            profit=0.0,
            duration=0
        )
        
        return True
        
    except Exception as e:
        log_e(f"❌ TRADE EXECUTION FAILED: {e}")
        return False

def close_position(reason="manual_close"):
    """إغلاق المركز الحالي"""
    try:
        if not STATE["open"] or STATE["qty"] <= 0:
            return True
            
        side = STATE["side"]
        qty = STATE["qty"]
        close_side = "sell" if side == "long" else "buy"
        
        log_i(f"🔴 CLOSING POSITION: {side} {qty:.4f} - Reason: {reason}")
        
        if MODE_LIVE and EXECUTE_ORDERS and not DRY_RUN:
            params = exchange_specific_params(close_side, is_close=True)
            ex.create_order(SYMBOL, "market", close_side, qty, None, params)
        
        # تحديث الحالة
        STATE["open"] = False
        STATE["qty"] = 0.0
        
        # تسجيل الخروج في Anti-Reversal
        anti_reversal.record_exit(side)
        
        log_g(f"✅ POSITION CLOSED: {side} {qty:.4f}")
        return True
        
    except Exception as e:
        log_e(f"❌ CLOSE POSITION FAILED: {e}")
        return False

def compute_position_size(balance, price, confidence):
    """حساب حجم الصفقة"""
    base_size = trade_manager.get_optimal_position_size(balance)
    
    # تعديل الحجم بناءً على الثقة
    confidence_multiplier = 0.5 + (confidence * 0.5)
    
    # تعديل الحجم بناءً على مرحلة السوق
    market_phase = STATE.get("market_phase", "neutral")
    if market_phase in ["strong_bull", "strong_bear"]:
        market_multiplier = 1.3
    elif market_phase in ["bull", "bear"]:
        market_multiplier = 1.1
    else:
        market_multiplier = 0.8
    
    adaptive_size = base_size * confidence_multiplier * market_multiplier
    
    # التأكد من أن الحجم ضمن الحدود المعقولة
    max_position = balance * LEVERAGE * 0.8
    final_size = min(adaptive_size, max_position / price) if price > 0 else adaptive_size
    
    return safe_qty(final_size)

def manage_position(df, current_price):
    """إدارة المركز المفتوح"""
    if not STATE["open"] or STATE["qty"] <= 0:
        return
    
    try:
        entry_price = STATE["entry"]
        side = STATE["side"]
        
        # حساب الربح/الخسارة
        if side == "long":
            pnl_pct = (current_price - entry_price) / entry_price * 100
        else:
            pnl_pct = (entry_price - current_price) / entry_price * 100
        
        STATE["pnl"] = pnl_pct
        
        # إستراتيجية الخروج المبسطة
        indicators = compute_advanced_indicators(df)
        rsi = indicators.get('rsi', 50)
        
        # خروج عند تحقيق هدف الربح
        if pnl_pct >= 2.0:
            log_i(f"🎯 TARGET REACHED: {pnl_pct:.2f}% - Closing position")
            close_position(f"target_reached_{pnl_pct:.1f}%")
            return
        
        # خروج عند فقدان الزخم
        if side == "long" and rsi > 80:
            log_i(f"⚠️ OVERBOUGHT: RSI={rsi:.1f} - Closing long position")
            close_position(f"overbought_rsi_{rsi:.1f}")
            return
        
        if side == "short" and rsi < 20:
            log_i(f"⚠️ OVERSOLD: RSI={rsi:.1f} - Closing short position")
            close_position(f"oversold_rsi_{rsi:.1f}")
            return
        
        # خروج وقائي بعد وقت طويل
        entry_time = STATE.get("entry_time", 0)
        time_in_trade = time.time() - entry_time
        if time_in_trade > 3600:  # ساعة واحدة
            log_i(f"⏰ TIME EXIT: In trade for {int(time_in_trade/60)}min")
            close_position("time_exit_1h")
            return
            
    except Exception as e:
        log_w(f"Manage position error: {e}")

# =================== MAIN TRADING LOOP ===================
def main_trading_loop():
    """الحلقة الرئيسية للتداول"""
    log_banner("STARTING SIMPLIFIED SUI TRADING BOT v9.1")
    log_i(f"🤖 Bot Version: {BOT_VERSION}")
    log_i(f"💱 Exchange: {EXCHANGE_NAME.upper()}")
    log_i(f"📈 Symbol: {SYMBOL}")
    log_i(f"⏰ Interval: {INTERVAL}")
    log_i(f"🎯 Leverage: {LEVERAGE}x")
    
    # تحميل مواصفات السوق
    load_market_specs()
    
    # تهيئة المكونات
    global market_analyzer, trade_manager, liquidity_engine, anti_reversal
    market_analyzer = AdvancedMarketAnalyzer()
    trade_manager = SmartTradeManager()
    liquidity_engine = LiquidityEngine()
    anti_reversal = AntiReversalGuard()
    
    while True:
        try:
            # جمع البيانات
            df = fetch_ohlcv(limit=100)
            current_price = price_now()
            
            if df.empty or current_price is None:
                log_w("📭 No data available - retrying...")
                time.sleep(BASE_SLEEP)
                continue
            
            STATE["current_price"] = current_price
            
            # إدارة المركز المفتوح
            manage_position(df, current_price)
            
            # فتح صفقات جديدة فقط إذا لم يكن هناك مركز مفتوح
            if not STATE["open"]:
                # قرار مجلس الذكاء الاصطناعي
                council_data = simplified_council_ai(df)
                
                # تحديث الحالة
                STATE["last_council"] = council_data
                STATE["market_phase"] = council_data.get("market_phase", "neutral")
                STATE["volatility_regime"] = council_data.get("volatility_regime", "normal")
                
                # عرض معلومات السوق
                if LOG_ADDONS:
                    log_i(f"🏪 MARKET: {STATE['market_phase'].upper()} | VOLATILITY: {STATE['volatility_regime']}")
                    log_i(f"🎯 COUNCIL: Score B={council_data['score_b']:.1f} | S={council_data['score_s']:.1f} | Conf={council_data['confidence']:.2f}")
                    
                    for log_msg in council_data.get("logs", []):
                        log_i(f"   {log_msg}")
                
                # تحديد اتجاه التداول
                signal_side = None
                if council_data["score_b"] > council_data["score_s"] and council_data["score_b"] >= 8.0:
                    signal_side = "buy"
                elif council_data["score_s"] > council_data["score_b"] and council_data["score_s"] >= 8.0:
                    signal_side = "sell"
                
                # فتح الصفقة إذا كان هناك إشارة قوية
                if signal_side and council_data["confidence"] >= 0.6:
                    # التحقق من Anti-Reversal
                    can_enter, reason = anti_reversal.can_enter(signal_side)
                    if not can_enter:
                        log_i(f"⛔ Anti-Reversal: {reason}")
                        time.sleep(BASE_SLEEP)
                        continue
                    
                    # التحقق من السيولة
                    if not liquidity_engine.detect_liquidity_sweep(df, signal_side):
                        log_i("⛔ Liquidity: لم يتم اكتشاف مسح سيولة حقيقي")
                        time.sleep(BASE_SLEEP)
                        continue
                    
                    # حساب حجم الصفقة
                    balance = balance_usdt()
                    position_size = compute_position_size(balance, current_price, council_data["confidence"])
                    
                    if position_size > 0:
                        log_i(f"🎯 TRADE SIGNAL: {signal_side.upper()} | Size: {position_size:.4f} | Price: {current_price:.6f}")
                        
                        # تنفيذ الصفقة
                        success = execute_trade(signal_side, current_price, position_size)
                        
                        if success:
                            STATE.update({
                                "open": True,
                                "side": "long" if signal_side == "buy" else "short",
                                "entry": current_price,
                                "last_entry_price": current_price,
                                "qty": position_size,
                                "pnl": 0.0,
                                "entry_time": time.time(),
                                "minimum_hold_until": time.time() + MIN_HOLD_TIME
                            })
                            
                            # تسجيل الدخول في Anti-Reversal
                            anti_reversal.record_entry(signal_side)
                            
                            log_i(f"✅ POSITION OPENED: {signal_side.upper()} {position_size:.4f} @ {current_price:.6f}")
            
            # الانتظار للدورة التالية
            time.sleep(BASE_SLEEP)
            
        except Exception as e:
            log_e(f"❌ TRADING LOOP ERROR: {e}")
            log_e(traceback.format_exc())
            time.sleep(BASE_SLEEP * 2)

# =================== FLASK API ===================
app = Flask(__name__)

@app.route("/")
def home():
    return f"""
    <html>
        <head><title>SUI ULTRA PRO AI BOT v9.1</title></head>
        <body style="font-family: Arial, sans-serif; padding: 20px; background-color: #f0f0f0;">
            <div style="max-width: 800px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
                <h1 style="color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px;">🚀 SUI ULTRA PRO AI BOT v9.1</h1>
                <div style="margin-top: 20px;">
                    <p><strong>Version:</strong> {BOT_VERSION}</p>
                    <p><strong>Exchange:</strong> {EXCHANGE_NAME.upper()}</p>
                    <p><strong>Symbol:</strong> {SYMBOL}</p>
                    <p><strong>Status:</strong> <span style="color: {'green' if MODE_LIVE else 'orange'}">{'🟢 LIVE' if MODE_LIVE else '🟡 PAPER'}</span></p>
                    <p><strong>Daily PnL:</strong> <span style="color: {'green' if trade_manager.daily_profit >= 0 else 'red'}">{trade_manager.daily_profit:.2f} USDT</span></p>
                    <p><strong>Win Rate:</strong> {trade_manager.win_rate:.1f}%</p>
                    <p><strong>Open Position:</strong> {'🟢 YES' if STATE['open'] else '🔴 NO'} {STATE['side'] if STATE['open'] else ''}</p>
                    <div style="margin-top: 30px; padding: 15px; background-color: #f9f9f9; border-radius: 5px;">
                        <p><strong>Quick Links:</strong></p>
                        <p><a href="/health" style="color: #4CAF50; text-decoration: none;">🩺 Health Check</a></p>
                        <p><a href="/metrics" style="color: #4CAF50; text-decoration: none;">📊 Metrics</a></p>
                        <p><a href="/performance" style="color: #4CAF50; text-decoration: none;">📈 Performance</a></p>
                        <p><a href="/close" style="color: #ff4444; text-decoration: none;">🔴 Close Position</a></p>
                    </div>
                </div>
            </div>
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
        "win_rate": trade_manager.win_rate,
        "consecutive_wins": trade_manager.consecutive_wins,
        "consecutive_losses": trade_manager.consecutive_losses
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
        "position": STATE
    })

@app.route("/performance")
def performance():
    recent_trades = trade_manager.trade_history[-10:]
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

@app.route("/close")
def close_position_route():
    """إغلاق المركز عبر API"""
    success = close_position("api_request")
    return jsonify({
        "success": success,
        "message": "Position closed" if success else "Failed to close position",
        "timestamp": datetime.now().isoformat()
    })

# =================== STARTUP ===================
def setup_logging():
    """إعداد التسجيل"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
    logger.addHandler(ch)
    
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    
    log_i("🔄 Logging setup complete")

def startup():
    """بدء التشغيل"""
    log_banner("SYSTEM INITIALIZATION v9.1")
    
    # إعداد التسجيل
    setup_logging()
    
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
    
    # عرض إعدادات الحماية
    log_i(f"🛡️ Protection Settings:")
    log_i(f"   Cooldown after exit: {COOLDOWN_AFTER_EXIT}s")
    log_i(f"   Minimum hold time: {MIN_HOLD_TIME}s")
    log_i(f"   Dead zone: {DEAD_ZONE_PCT}%")
    
    log_g("🚀 SIMPLIFIED TRADING BOT READY!")
    return True

# =================== MAIN EXECUTION ===================
if __name__ == "__main__":
    # إعداد معالجات الإشارات
    def signal_handler(signum, frame):
        log_i(f"🛑 Received signal {signum} - Shutting down gracefully...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # بدء التشغيل
    if startup():
        # بدء خيوط التنفيذ
        import threading
        
        # خيط التداول الرئيسي
        trading_thread = threading.Thread(target=main_trading_loop, daemon=True)
        trading_thread.start()
        
        log_g(f"🌐 Starting web server on port {PORT}")
        
        # تشغيل سيرفل الويب
        try:
            app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
        except Exception as e:
            log_e(f"❌ Web server failed: {e}")
    else:
        log_e("❌ Startup failed - check configuration and try again")
