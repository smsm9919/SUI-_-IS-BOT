# -*- coding: utf-8 -*-
"""
SUI ULTRA PRO AI BOT v9.2 - النسخة المبسطة التي تعمل على Render
• نظام مؤشرات مبنى يدوياً بدون TA-Lib أو pandas-ta
• متوافق تماماً مع Render وغيرها من المنصات السحابية
"""

import os, time, math, random, signal, sys, traceback, logging, json
from logging.handlers import RotatingFileHandler
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import ccxt
from flask import Flask, jsonify
from decimal import Decimal, ROUND_DOWN, InvalidOperation

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
BOT_VERSION = f"SUI ULTRA PRO AI v9.2 — {EXCHANGE_NAME.upper()}"
print("🚀 Booting:", BOT_VERSION, flush=True)

STATE_PATH = "./bot_state.json"
RESUME_ON_RESTART = True
RESUME_LOOKBACK_SECS = 60 * 60

# =================== SETTINGS ===================
SYMBOL     = os.getenv("SYMBOL", "SUI/USDT:USDT")
INTERVAL   = os.getenv("INTERVAL", "15m")
LEVERAGE   = int(os.getenv("LEVERAGE", 10))
RISK_ALLOC = float(os.getenv("RISK_ALLOC", 0.60))
POSITION_MODE = os.getenv("POSITION_MODE", "oneway")

# Dynamic TP / trail - Optimized for SUI
TP1_PCT_BASE       = 0.45
TP1_CLOSE_FRAC     = 0.50
BREAKEVEN_AFTER    = 0.30
TRAIL_ACTIVATE_PCT = 1.20
ATR_TRAIL_MULT     = 1.8

# Pacing
BASE_SLEEP   = 5
NEAR_CLOSE_S = 1

# ==== Enhanced Protection Settings ====
CONFIRMATION_TIMEFRAMES = ["15m", "5m", "1h"]
MIN_CONFIRMATIONS = 2
DEAD_ZONE_PCT = 0.15
COOLDOWN_AFTER_EXIT = 600
MIN_HOLD_TIME = 300

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

# =================== MANUAL TECHNICAL INDICATORS ===================
class ManualIndicators:
    """مؤشرات تقنية مبنية يدوياً بدون مكتبات خارجية"""
    
    @staticmethod
    def sma(data, period):
        """المتوسط المتحرك البسيط"""
        if len(data) < period:
            return None
        return data.rolling(window=period).mean()
    
    @staticmethod
    def ema(data, period):
        """المتوسط المتحرك الأسي"""
        if len(data) < period:
            return None
        return data.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def rsi(data, period=14):
        """مؤشر القوة النسبية"""
        if len(data) < period + 1:
            return None
        
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def macd(data, fast=12, slow=26, signal=9):
        """مؤشر MACD"""
        if len(data) < slow:
            return None, None, None
        
        ema_fast = data.ewm(span=fast, adjust=False).mean()
        ema_slow = data.ewm(span=slow, adjust=False).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(data, period=20, std_dev=2):
        """نطاقات بولنجر"""
        if len(data) < period:
            return None, None, None
        
        sma = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        middle_band = sma
        lower_band = sma - (std * std_dev)
        
        return upper_band, middle_band, lower_band
    
    @staticmethod
    def atr(high, low, close, period=14):
        """متوسط المدى الحقيقي"""
        if len(high) < period + 1:
            return None
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    @staticmethod
    def stoch(high, low, close, k_period=14, d_period=3):
        """مؤشر ستوكاستك"""
        if len(high) < k_period:
            return None, None
        
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k_line = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_line = k_line.rolling(window=d_period).mean()
        
        return k_line, d_line

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

def fetch_ohlcv(limit=100):
    try:
        rows = ex.fetch_ohlcv(SYMBOL, timeframe=INTERVAL, limit=limit, params={"type":"swap"})
        if rows:
            df = pd.DataFrame(rows, columns=["time","open","high","low","close","volume"])
            # تحويل الأعمدة إلى أرقام
            for col in ["open","high","low","close","volume"]:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            return df
        return pd.DataFrame()
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

# =================== MARKET ANALYZER ===================
class MarketAnalyzer:
    def __init__(self):
        self.indicators = ManualIndicators()
        
    def detect_market_phase(self, df):
        """اكتشاف مرحلة السوق الحالية"""
        try:
            close = df['close'].astype(float)
            
            # حساب المتوسطات المتحركة
            sma_20 = self.indicators.sma(close, 20)
            sma_50 = self.indicators.sma(close, 50)
            sma_200 = self.indicators.sma(close, 200)
            
            if sma_20 is None or sma_50 is None or sma_200 is None:
                return "neutral"
            
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
    
    def analyze_volatility(self, df):
        """تحليل التقلب"""
        try:
            high = df['high'].astype(float)
            low = df['low'].astype(float)
            close = df['close'].astype(float)
            
            true_range = np.maximum(high - low, 
                                  np.maximum(abs(high - close.shift(1)), 
                                           abs(low - close.shift(1))))
            atr = true_range.rolling(14).mean()
            current_atr = atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else 0
            avg_atr = atr.mean() if len(atr) > 0 else 1
            
            volatility_ratio = current_atr / avg_atr if avg_atr > 0 else 1.0
            
            if volatility_ratio > 1.5:
                return "high", volatility_ratio
            elif volatility_ratio < 0.7:
                return "low", volatility_ratio
            else:
                return "normal", volatility_ratio
                
        except Exception as e:
            log_w(f"analyze_volatility error: {e}")
            return "normal", 1.0
    
    def compute_indicators(self, df):
        """حساب جميع المؤشرات"""
        try:
            close = df['close'].astype(float)
            high = df['high'].astype(float)
            low = df['low'].astype(float)
            
            # RSI
            rsi = self.indicators.rsi(close, 14)
            rsi_value = rsi.iloc[-1] if rsi is not None and len(rsi) > 0 else 50
            
            # MACD
            macd_line, signal_line, histogram = self.indicators.macd(close)
            macd_value = macd_line.iloc[-1] if macd_line is not None and len(macd_line) > 0 else 0
            macd_hist = histogram.iloc[-1] if histogram is not None and len(histogram) > 0 else 0
            
            # بولنجر باندز
            bb_upper, bb_middle, bb_lower = self.indicators.bollinger_bands(close, 20, 2)
            bb_upper_val = bb_upper.iloc[-1] if bb_upper is not None and len(bb_upper) > 0 else close.iloc[-1]
            bb_lower_val = bb_lower.iloc[-1] if bb_lower is not None and len(bb_lower) > 0 else close.iloc[-1]
            
            # ATR
            atr = self.indicators.atr(high, low, close, 14)
            atr_value = atr.iloc[-1] if atr is not None and len(atr) > 0 else 0
            
            # ستوكاستك
            stoch_k, stoch_d = self.indicators.stoch(high, low, close, 14, 3)
            stoch_k_value = stoch_k.iloc[-1] if stoch_k is not None and len(stoch_k) > 0 else 50
            stoch_d_value = stoch_d.iloc[-1] if stoch_d is not None and len(stoch_d) > 0 else 50
            
            # المتوسطات المتحركة
            sma_20 = self.indicators.sma(close, 20)
            sma_20_value = sma_20.iloc[-1] if sma_20 is not None and len(sma_20) > 0 else close.iloc[-1]
            
            sma_50 = self.indicators.sma(close, 50)
            sma_50_value = sma_50.iloc[-1] if sma_50 is not None and len(sma_50) > 0 else close.iloc[-1]
            
            ema_20 = self.indicators.ema(close, 20)
            ema_20_value = ema_20.iloc[-1] if ema_20 is not None and len(ema_20) > 0 else close.iloc[-1]
            
            return {
                'rsi': round(rsi_value, 2),
                'macd': round(macd_value, 4),
                'macd_hist': round(macd_hist, 4),
                'bollinger_upper': round(bb_upper_val, 4),
                'bollinger_lower': round(bb_lower_val, 4),
                'atr': round(atr_value, 4),
                'stoch_k': round(stoch_k_value, 2),
                'stoch_d': round(stoch_d_value, 2),
                'sma_20': round(sma_20_value, 4),
                'sma_50': round(sma_50_value, 4),
                'ema_20': round(ema_20_value, 4),
                'current_price': round(close.iloc[-1], 4)
            }
            
        except Exception as e:
            log_w(f"compute_indicators error: {e}")
            return {}

# =================== TRADE MANAGER ===================
class TradeManager:
    def __init__(self):
        self.trade_history = []
        self.daily_profit = 0.0
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.win_rate = 0.0
        
    def record_trade(self, side, entry, exit_price, quantity, profit):
        """تسجيل الصفقة"""
        trade = {
            'timestamp': datetime.now(),
            'side': side,
            'entry': entry,
            'exit': exit_price,
            'quantity': quantity,
            'profit': profit,
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
            
        # تحديث نسبة النجاح
        wins = sum(1 for t in self.trade_history if t['profit'] > 0)
        self.win_rate = (wins / len(self.trade_history)) * 100 if self.trade_history else 0
    
    def get_position_size(self, balance, risk_per_trade=0.02):
        """حساب حجم الصفقة"""
        base_size = balance * risk_per_trade
        
        # تعديل الحجم بناءً على الأداء
        if self.consecutive_wins >= 3:
            size_multiplier = min(2.0, 1.0 + (self.consecutive_wins * 0.1))
        elif self.consecutive_losses >= 2:
            size_multiplier = max(0.5, 1.0 - (self.consecutive_losses * 0.2))
        else:
            size_multiplier = 1.0
            
        return base_size * size_multiplier

# =================== LIQUIDITY ENGINE ===================
class LiquidityEngine:
    def detect_support_resistance(self, df, window=20):
        """اكتشاف مستويات الدعم والمقاومة"""
        try:
            high = df['high'].astype(float)
            low = df['low'].astype(float)
            close = df['close'].astype(float)
            
            # استخدام أعلى قمة وأقل قاع كمرجع
            resistance = high.rolling(window=window).max()
            support = low.rolling(window=window).min()
            
            current_price = close.iloc[-1]
            
            # العثور على أقرب مستويات الدعم والمقاومة
            support_levels = []
            resistance_levels = []
            
            for i in range(1, min(window, len(support))):
                support_val = support.iloc[-i]
                if support_val < current_price:
                    support_levels.append(support_val)
                    if len(support_levels) >= 3:
                        break
            
            for i in range(1, min(window, len(resistance))):
                resistance_val = resistance.iloc[-i]
                if resistance_val > current_price:
                    resistance_levels.append(resistance_val)
                    if len(resistance_levels) >= 3:
                        break
            
            return {
                'support_levels': support_levels,
                'resistance_levels': resistance_levels
            }
            
        except Exception as e:
            log_w(f"detect_support_resistance error: {e}")
            return {'support_levels': [], 'resistance_levels': []}
    
    def detect_pin_bar(self, df, side="buy"):
        """اكتشاف شمعة الدبوس (Pin Bar)"""
        try:
            if len(df) < 2:
                return False
                
            current_candle = df.iloc[-1]
            prev_candle = df.iloc[-2] if len(df) > 1 else current_candle
            
            open_price = current_candle['open']
            close_price = current_candle['close']
            high_price = current_candle['high']
            low_price = current_candle['low']
            
            body_size = abs(close_price - open_price)
            total_range = high_price - low_price
            
            if total_range == 0:
                return False
            
            # حساب نسبة الجسم إلى المدى الكلي
            body_ratio = body_size / total_range
            
            # شمعة الدبوس لها جسم صغير وذيل طويل
            if body_ratio < 0.3:
                upper_wick = high_price - max(open_price, close_price)
                lower_wick = min(open_price, close_price) - low_price
                
                if side == "buy":
                    # دبوس صاعد: ذيل سفلي طويل
                    return lower_wick > (body_size * 2) and lower_wick > upper_wick
                else:
                    # دبوس هابط: ذيل علوي طويل
                    return upper_wick > (body_size * 2) and upper_wick > lower_wick
            
            return False
            
        except Exception as e:
            log_w(f"detect_pin_bar error: {e}")
            return False

# =================== ANTI-REVERSAL GUARD ===================
class AntiReversalGuard:
    def __init__(self):
        self.last_exit_time = None
        self.last_exit_side = None
        
    def can_enter(self, side):
        """التحقق إذا مسموح بالدخول"""
        current_time = time.time()
        
        # فترة التبريد بعد الخروج
        if self.last_exit_time:
            time_since_exit = current_time - self.last_exit_time
            if time_since_exit < COOLDOWN_AFTER_EXIT:
                return False, f"في فترة تبريد ({int(time_since_exit)}s/{COOLDOWN_AFTER_EXIT}s)"
        
        # منع العكس المباشر
        if self.last_exit_side and side != self.last_exit_side:
            return False, "منع عكس بدون كسر هيكل"
        
        # التحقق من مدة البقاء الأدنى
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

# =================== TRADING LOGIC ===================
def analyze_market(df):
    """تحليل السوق واتخاذ القرار"""
    try:
        if len(df) < 50:
            return {"score_buy": 0, "score_sell": 0, "decision": "HOLD", "reasons": []}
        
        analyzer = MarketAnalyzer()
        liquidity = LiquidityEngine()
        
        indicators = analyzer.compute_indicators(df)
        market_phase = analyzer.detect_market_phase(df)
        volatility, vol_ratio = analyzer.analyze_volatility(df)
        support_resistance = liquidity.detect_support_resistance(df)
        
        score_buy = 0
        score_sell = 0
        reasons = []
        
        current_price = indicators.get('current_price', 0)
        
        # 1. تحليل مرحلة السوق
        if market_phase == "strong_bull":
            score_buy += 2
            reasons.append("📈 مرحلة صاعدة قوية")
        elif market_phase == "bull":
            score_buy += 1
            reasons.append("📈 مرحلة صاعدة")
        elif market_phase == "bear":
            score_sell += 1
            reasons.append("📉 مرحلة هابطة")
        elif market_phase == "strong_bear":
            score_sell += 2
            reasons.append("📉 مرحلة هابطة قوية")
        
        # 2. تحليل RSI
        rsi = indicators.get('rsi', 50)
        if rsi < 30:
            score_buy += 2
            reasons.append(f"📊 RSI منخفض ({rsi:.1f})")
        elif rsi > 70:
            score_sell += 2
            reasons.append(f"📊 RSI مرتفع ({rsi:.1f})")
        
        # 3. تحليل MACD
        macd_hist = indicators.get('macd_hist', 0)
        if macd_hist > 0:
            score_buy += 1
            reasons.append("📈 MACD إيجابي")
        elif macd_hist < 0:
            score_sell += 1
            reasons.append("📉 MACD سلبي")
        
        # 4. تحليل بولنجر باندز
        bb_upper = indicators.get('bollinger_upper', current_price)
        bb_lower = indicators.get('bollinger_lower', current_price)
        
        if current_price <= bb_lower * 1.01:  # قرب النطاق السفلي
            score_buy += 2
            reasons.append("📏 قرب النطاق السفلي")
        elif current_price >= bb_upper * 0.99:  # قرب النطاق العلوي
            score_sell += 2
            reasons.append("📏 قرب النطاق العلوي")
        
        # 5. تحليل ستوكاستك
        stoch_k = indicators.get('stoch_k', 50)
        if stoch_k < 20:
            score_buy += 1
            reasons.append(f"🎯 ستوكاستك منخفض ({stoch_k:.1f})")
        elif stoch_k > 80:
            score_sell += 1
            reasons.append(f"🎯 ستوكاستك مرتفع ({stoch_k:.1f})")
        
        # 6. تحليل الدعم والمقاومة
        support_levels = support_resistance.get('support_levels', [])
        resistance_levels = support_resistance.get('resistance_levels', [])
        
        if support_levels and current_price <= support_levels[0] * 1.005:
            score_buy += 1
            reasons.append("🛡️ قرب مستوى دعم")
        
        if resistance_levels and current_price >= resistance_levels[0] * 0.995:
            score_sell += 1
            reasons.append("🚧 قرب مستوى مقاومة")
        
        # 7. تحليل شموع الدبوس
        if liquidity.detect_pin_bar(df, "buy"):
            score_buy += 2
            reasons.append("📍 دبوس صاعد")
        
        if liquidity.detect_pin_bar(df, "sell"):
            score_sell += 2
            reasons.append("📍 دبوس هابط")
        
        # اتخاذ القرار
        decision = "HOLD"
        if score_buy >= 6 and score_buy > score_sell:
            decision = "BUY"
        elif score_sell >= 6 and score_sell > score_buy:
            decision = "SELL"
        
        return {
            "score_buy": score_buy,
            "score_sell": score_sell,
            "decision": decision,
            "reasons": reasons,
            "indicators": indicators,
            "market_phase": market_phase,
            "volatility": volatility
        }
        
    except Exception as e:
        log_w(f"analyze_market error: {e}")
        return {"score_buy": 0, "score_sell": 0, "decision": "HOLD", "reasons": []}

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
        
        log_i(f"🎯 EXECUTING: {side.upper()} {qty:.4f} @ {price:.6f}")
        
        if MODE_LIVE:
            exchange_set_leverage(ex, LEVERAGE, SYMBOL)
            params = exchange_specific_params(side, is_close=False)
            ex.create_order(SYMBOL, "market", side, qty, None, params)
        
        log_g(f"✅ EXECUTED: {side.upper()} {qty:.4f} @ {price:.6f}")
        return True
        
    except Exception as e:
        log_e(f"❌ TRADE FAILED: {e}")
        return False

def close_position(reason="manual"):
    """إغلاق المركز"""
    try:
        if not STATE["open"] or STATE["qty"] <= 0:
            return True
            
        side = STATE["side"]
        qty = STATE["qty"]
        close_side = "sell" if side == "long" else "buy"
        
        log_i(f"🔴 CLOSING: {side} {qty:.4f} - Reason: {reason}")
        
        if MODE_LIVE and EXECUTE_ORDERS and not DRY_RUN:
            params = exchange_specific_params(close_side, is_close=True)
            ex.create_order(SYMBOL, "market", close_side, qty, None, params)
        
        # تحديث الحالة
        STATE["open"] = False
        STATE["qty"] = 0.0
        
        # تسجيل الخروج
        anti_reversal.record_exit(side)
        
        log_g(f"✅ CLOSED: {side} {qty:.4f}")
        return True
        
    except Exception as e:
        log_e(f"❌ CLOSE FAILED: {e}")
        return False

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
        
        analyzer = MarketAnalyzer()
        indicators = analyzer.compute_indicators(df)
        rsi = indicators.get('rsi', 50)
        
        # قواعد الخروج
        exit_reason = None
        
        # هدف الربح
        if pnl_pct >= 2.0:
            exit_reason = f"هدف ربح {pnl_pct:.1f}%"
        
        # وقف الخسارة
        elif pnl_pct <= -1.5:
            exit_reason = f"وقف خسارة {pnl_pct:.1f}%"
        
        # RSI متطرف
        elif side == "long" and rsi > 80:
            exit_reason = f"RSI مرتفع {rsi:.1f}"
        elif side == "short" and rsi < 20:
            exit_reason = f"RSI منخفض {rsi:.1f}"
        
        # وقت طويل في الصفقة
        entry_time = STATE.get("entry_time", 0)
        time_in_trade = time.time() - entry_time
        if time_in_trade > 3600:  # ساعة واحدة
            exit_reason = f"وقت طويل ({int(time_in_trade/60)} دقيقة)"
        
        # تنفيذ الخروج إذا كان هناك سبب
        if exit_reason:
            log_i(f"⚠️ EXIT SIGNAL: {exit_reason}")
            close_position(exit_reason)
            
    except Exception as e:
        log_w(f"manage_position error: {e}")

def compute_position_size(balance, price, confidence=0.5):
    """حساب حجم الصفقة"""
    # حجم أساسي 2% من الرصيد
    base_size = balance * 0.02
    
    # تعديل بناءً على الثقة
    size_multiplier = 0.5 + (confidence * 0.5)
    
    # الحد الأقصى 80% من الرصيد بالرافعة
    max_position = balance * LEVERAGE * 0.8
    final_size = min(base_size * size_multiplier, max_position / price) if price > 0 else base_size
    
    return safe_qty(final_size)

# =================== MAIN TRADING LOOP ===================
def trading_loop():
    """الحلقة الرئيسية للتداول"""
    log_banner("🚀 STARTING SUI TRADING BOT v9.2")
    log_i(f"🤖 Version: {BOT_VERSION}")
    log_i(f"💱 Exchange: {EXCHANGE_NAME.upper()}")
    log_i(f"📈 Symbol: {SYMBOL}")
    log_i(f"⏰ Interval: {INTERVAL}")
    log_i(f"🎯 Leverage: {LEVERAGE}x")
    
    # تحميل مواصفات السوق
    load_market_specs()
    
    # تهيئة المكونات
    global market_analyzer, trade_manager, anti_reversal
    market_analyzer = MarketAnalyzer()
    trade_manager = TradeManager()
    anti_reversal = AntiReversalGuard()
    
    while True:
        try:
            # جمع البيانات
            df = fetch_ohlcv(limit=100)
            current_price = price_now()
            
            if df.empty or current_price is None:
                log_w("📭 No data - retrying...")
                time.sleep(BASE_SLEEP)
                continue
            
            STATE["current_price"] = current_price
            
            # إدارة المركز المفتوح
            manage_position(df, current_price)
            
            # فتح صفقات جديدة فقط إذا لم يكن هناك مركز مفتوح
            if not STATE["open"]:
                # تحليل السوق
                analysis = analyze_market(df)
                
                # تحديث الحالة
                STATE["last_analysis"] = analysis
                STATE["market_phase"] = analysis.get("market_phase", "neutral")
                
                # عرض النتائج
                log_i(f"📊 ANALYSIS: Buy={analysis['score_buy']} | Sell={analysis['score_sell']} | Decision={analysis['decision']}")
                
                for reason in analysis.get("reasons", []):
                    log_i(f"   {reason}")
                
                # اتخاذ قرار التداول
                decision = analysis["decision"]
                if decision in ["BUY", "SELL"]:
                    side = "buy" if decision == "BUY" else "sell"
                    
                    # التحقق من الحماية
                    can_enter, reason = anti_reversal.can_enter(side)
                    if not can_enter:
                        log_i(f"⛔ Protection: {reason}")
                        time.sleep(BASE_SLEEP)
                        continue
                    
                    # حساب حجم الصفقة
                    balance = balance_usdt()
                    if balance is None or balance <= 0:
                        log_w("💰 No balance available")
                        time.sleep(BASE_SLEEP)
                        continue
                    
                    confidence = max(analysis["score_buy"], analysis["score_sell"]) / 10.0
                    position_size = compute_position_size(balance, current_price, confidence)
                    
                    if position_size > 0:
                        log_i(f"🎯 SIGNAL: {side.upper()} | Size: {position_size:.4f} | Price: {current_price:.6f}")
                        
                        # تنفيذ الصفقة
                        success = execute_trade(side, current_price, position_size)
                        
                        if success:
                            STATE.update({
                                "open": True,
                                "side": "long" if side == "buy" else "short",
                                "entry": current_price,
                                "last_entry_price": current_price,
                                "qty": position_size,
                                "pnl": 0.0,
                                "entry_time": time.time(),
                                "minimum_hold_until": time.time() + MIN_HOLD_TIME
                            })
                            
                            # تسجيل الدخول
                            anti_reversal.record_entry(side)
                            
                            log_i(f"✅ POSITION OPENED: {side.upper()} {position_size:.4f} @ {current_price:.6f}")
            
            # الانتظار للدورة التالية
            time.sleep(BASE_SLEEP)
            
        except Exception as e:
            log_e(f"❌ LOOP ERROR: {e}")
            time.sleep(BASE_SLEEP * 2)

# =================== FLASK API ===================
app = Flask(__name__)

@app.route("/")
def home():
    return """
    <html>
        <head>
            <title>SUI Trading Bot v9.2</title>
            <meta charset="utf-8">
            <style>
                body { font-family: Arial, sans-serif; padding: 20px; background: #f5f5f5; }
                .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
                h1 { color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }
                .status { padding: 15px; margin: 15px 0; border-radius: 5px; }
                .live { background: #d4edda; color: #155724; }
                .paper { background: #fff3cd; color: #856404; }
                .btn { display: inline-block; padding: 10px 20px; background: #4CAF50; color: white; text-decoration: none; border-radius: 5px; margin: 5px; }
                .btn-danger { background: #dc3545; }
                .info { background: #d1ecf1; color: #0c5460; padding: 10px; border-radius: 5px; margin: 10px 0; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🚀 SUI Trading Bot v9.2</h1>
                <div class="status">
                    <p><strong>Status:</strong> <span class="live">🟢 RUNNING</span></p>
                    <p><strong>Exchange:</strong> """ + EXCHANGE_NAME.upper() + """</p>
                    <p><strong>Symbol:</strong> """ + SYMBOL + """</p>
                    <p><strong>Mode:</strong> """ + ("🟢 LIVE" if MODE_LIVE else "🟡 PAPER") + """</p>
                </div>
                <div class="info">
                    <p><strong>Open Position:</strong> """ + ("🟢 YES (" + STATE.get("side", "") + ")" if STATE["open"] else "🔴 NO") + """</p>
                    <p><strong>Daily PnL:</strong> """ + f"{trade_manager.daily_profit:.2f} USDT" + """</p>
                    <p><strong>Win Rate:</strong> """ + f"{trade_manager.win_rate:.1f}%" + """</p>
                </div>
                <div>
                    <a href="/health" class="btn">🩺 Health Check</a>
                    <a href="/metrics" class="btn">📊 Metrics</a>
                    <a href="/performance" class="btn">📈 Performance</a>
                    <a href="/close" class="btn btn-danger" onclick="return confirm('Are you sure?')">🔴 Close Position</a>
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
        "balance": balance_usdt(),
        "server_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

@app.route("/metrics")
def metrics():
    return jsonify({
        "bot_version": BOT_VERSION,
        "exchange": EXCHANGE_NAME,
        "symbol": SYMBOL,
        "balance": balance_usdt(),
        "position": STATE,
        "performance": {
            "daily_profit": trade_manager.daily_profit,
            "win_rate": trade_manager.win_rate,
            "consecutive_wins": trade_manager.consecutive_wins,
            "consecutive_losses": trade_manager.consecutive_losses,
            "total_trades": len(trade_manager.trade_history)
        }
    })

@app.route("/performance")
def performance():
    recent_trades = trade_manager.trade_history[-5:]
    trades_data = []
    
    for trade in recent_trades:
        trades_data.append({
            "time": trade['timestamp'].strftime('%H:%M:%S'),
            "side": trade['side'],
            "profit": trade['profit'],
            "profit_pct": trade['profit_pct']
        })
    
    return jsonify({
        "daily_profit": trade_manager.daily_profit,
        "win_rate": trade_manager.win_rate,
        "recent_trades": trades_data
    })

@app.route("/close")
def close_position_route():
    success = close_position("api_request")
    return jsonify({
        "success": success,
        "message": "Position closed" if success else "No position to close",
        "timestamp": datetime.now().isoformat()
    })

# =================== STARTUP ===================
def setup_logging():
    """إعداد نظام التسجيل"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # إزالة أي معالجات موجودة
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # معالج للطباعة على الشاشة
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(ch)
    
    # إخفاء رسائل Flask و ccxt المزعجة
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    logging.getLogger('ccxt').setLevel(logging.WARNING)
    
    log_i("🔄 Logging system ready")

def startup():
    """بدء التشغيل"""
    log_banner("SYSTEM INITIALIZATION v9.2")
    
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
    log_i(f"   Daily PnL: {trade_manager.daily_profit:.2f} USDT")
    log_i(f"   Win Rate: {trade_manager.win_rate:.1f}%")
    log_i(f"   Consecutive Wins: {trade_manager.consecutive_wins}")
    log_i(f"   Consecutive Losses: {trade_manager.consecutive_losses}")
    
    # عرض إعدادات الحماية
    log_i(f"🛡️ Protection Settings:")
    log_i(f"   Cooldown after exit: {COOLDOWN_AFTER_EXIT}s")
    log_i(f"   Minimum hold time: {MIN_HOLD_TIME}s")
    log_i(f"   Dead zone: {DEAD_ZONE_PCT}%")
    
    log_g("🚀 Trading Bot is READY!")
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
        trading_thread = threading.Thread(target=trading_loop, daemon=True)
        trading_thread.start()
        
        log_g(f"🌐 Starting web server on port {PORT}")
        
        # تشغيل سيرفل الويب
        try:
            app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
        except Exception as e:
            log_e(f"❌ Web server failed: {e}")
    else:
        log_e("❌ Startup failed - check configuration and try again")
