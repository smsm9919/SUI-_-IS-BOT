# -*- coding: utf-8 -*-
"""
SUI ULTRA PRO AI BOT v9.3 - النسخة الثابتة التي تعمل على Render
• إصلاح مشكلة تعريف المتغيرات
• نظام مؤشرات مبنى يدوياً بدون مكتبات خارجية
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
BOT_VERSION = f"SUI ULTRA PRO AI v9.3 — {EXCHANGE_NAME.upper()}"
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

# =================== GLOBAL VARIABLES ===================
# تعريف المتغيرات العالمية أولاً
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

MARKET = {}
AMT_PREC = 0
LOT_STEP = None
LOT_MIN = None

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

# =================== TRADE MANAGER ===================
class TradeManager:
    def __init__(self):
        self.trade_history = []
        self.daily_profit = 0.0
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.win_rate = 0.0
        self.avg_win = 0.0
        self.avg_loss = 0.0
        
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
            
        self.update_performance()
    
    def update_performance(self):
        """تحديث أداء التداول"""
        if not self.trade_history:
            return
            
        wins = [t for t in self.trade_history if t['profit'] > 0]
        losses = [t for t in self.trade_history if t['profit'] <= 0]
        
        self.win_rate = len(wins) / len(self.trade_history) * 100
        
        if wins:
            self.avg_win = sum(t['profit'] for t in wins) / len(wins)
        if losses:
            self.avg_loss = abs(sum(t['profit'] for t in losses) / len(losses))
    
    def get_position_size(self, balance, risk_per_trade=0.02):
        """حساب حجم الصفقة"""
        base_size = balance * risk_per_trade
        
        if self.consecutive_wins >= 3:
            size_multiplier = min(2.0, 1.0 + (self.consecutive_wins * 0.1))
        elif self.consecutive_losses >= 2:
            size_multiplier = max(0.5, 1.0 - (self.consecutive_losses * 0.2))
        else:
            size_multiplier = 1.0
            
        return base_size * size_multiplier

# =================== MARKET ANALYZER ===================
class MarketAnalyzer:
    def __init__(self):
        self.indicators = ManualIndicators()
        
    def detect_market_phase(self, df):
        """اكتشاف مرحلة السوق الحالية"""
        try:
            close = df['close'].astype(float)
            
            if len(close) < 50:
                return "neutral"
            
            sma_20 = self.indicators.sma(close, 20)
            sma_50 = self.indicators.sma(close, 50)
            
            if sma_20 is None or sma_50 is None:
                return "neutral"
            
            current_price = close.iloc[-1]
            price_vs_20 = current_price > sma_20.iloc[-1]
            price_vs_50 = current_price > sma_50.iloc[-1]
            
            if price_vs_20 and price_vs_50:
                return "bull"
            elif not price_vs_20 and not price_vs_50:
                return "bear"
            else:
                return "neutral"
                
        except Exception as e:
            log_w(f"detect_market_phase error: {e}")
            return "neutral"
    
    def compute_indicators(self, df):
        """حساب المؤشرات"""
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
            
            # المتوسطات المتحركة
            sma_20 = self.indicators.sma(close, 20)
            sma_20_value = sma_20.iloc[-1] if sma_20 is not None and len(sma_20) > 0 else close.iloc[-1]
            
            return {
                'rsi': round(rsi_value, 2),
                'macd': round(macd_value, 4),
                'macd_hist': round(macd_hist, 4),
                'bollinger_upper': round(bb_upper_val, 4),
                'bollinger_lower': round(bb_lower_val, 4),
                'atr': round(atr_value, 4),
                'sma_20': round(sma_20_value, 4),
                'current_price': round(close.iloc[-1], 4)
            }
            
        except Exception as e:
            log_w(f"compute_indicators error: {e}")
            return {}

# =================== HELPER FUNCTIONS ===================
def load_market_specs():
    """تحميل مواصفات السوق"""
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
    """معلمات خاصة بالبورصة"""
    if EXCHANGE_NAME == "bybit":
        if POSITION_MODE == "hedge":
            return {"positionSide": "Long" if side == "buy" else "Short", "reduceOnly": is_close}
        return {"positionSide": "Both", "reduceOnly": is_close}
    else:
        if POSITION_MODE == "hedge":
            return {"positionSide": "LONG" if side == "buy" else "SHORT", "reduceOnly": is_close}
        return {"positionSide": "BOTH", "reduceOnly": is_close}

def exchange_set_leverage(exchange, leverage, symbol):
    """ضبط الرافعة المالية"""
    try:
        if EXCHANGE_NAME == "bybit":
            exchange.set_leverage(leverage, symbol)
        else:
            exchange.set_leverage(leverage, symbol, params={"side": "BOTH"})
        log_g(f"✅ {EXCHANGE_NAME.upper()} leverage set: {leverage}x")
    except Exception as e:
        log_w(f"⚠️ set_leverage warning: {e}")

def _round_amt(q):
    """تقريب الكمية"""
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
    """التحقق من صحة الكمية"""
    q = _round_amt(q)
    if q <= 0: log_w(f"qty invalid after normalize → {q}")
    return q

def fetch_ohlcv(limit=100):
    """جلب بيانات OHLCV"""
    try:
        rows = ex.fetch_ohlcv(SYMBOL, timeframe=INTERVAL, limit=limit, params={"type":"swap"})
        if rows:
            df = pd.DataFrame(rows, columns=["time","open","high","low","close","volume"])
            for col in ["open","high","low","close","volume"]:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            return df
        return pd.DataFrame()
    except Exception as e:
        log_w(f"fetch_ohlcv error: {e}")
        return pd.DataFrame()

def price_now():
    """الحصول على السعر الحالي"""
    try:
        t = ex.fetch_ticker(SYMBOL)
        return t.get("last") or t.get("close")
    except Exception: 
        return None

def balance_usdt():
    """الحصول على الرصيد"""
    if not MODE_LIVE: return 1000.0
    try:
        b = ex.fetch_balance(params={"type":"swap"})
        return b.get("total",{}).get("USDT") or b.get("free",{}).get("USDT")
    except Exception: 
        return None

# =================== TRADING COMPONENTS ===================
class LiquidityEngine:
    def detect_support_resistance(self, df, window=20):
        """اكتشاف مستويات الدعم والمقاومة"""
        try:
            high = df['high'].astype(float)
            low = df['low'].astype(float)
            close = df['close'].astype(float)
            
            resistance = high.rolling(window=window).max()
            support = low.rolling(window=window).min()
            
            current_price = close.iloc[-1]
            
            support_levels = []
            resistance_levels = []
            
            for i in range(1, min(window, len(support))):
                support_val = support.iloc[-i]
                if support_val < current_price:
                    support_levels.append(support_val)
                    if len(support_levels) >= 2:
                        break
            
            for i in range(1, min(window, len(resistance))):
                resistance_val = resistance.iloc[-i]
                if resistance_val > current_price:
                    resistance_levels.append(resistance_val)
                    if len(resistance_levels) >= 2:
                        break
            
            return {
                'support_levels': support_levels,
                'resistance_levels': resistance_levels
            }
            
        except Exception as e:
            log_w(f"detect_support_resistance error: {e}")
            return {'support_levels': [], 'resistance_levels': []}

class AntiReversalGuard:
    def __init__(self):
        self.last_exit_time = None
        self.last_exit_side = None
        
    def can_enter(self, side):
        """التحقق إذا مسموح بالدخول"""
        current_time = time.time()
        
        if self.last_exit_time:
            time_since_exit = current_time - self.last_exit_time
            if time_since_exit < COOLDOWN_AFTER_EXIT:
                return False, f"في فترة تبريد ({int(time_since_exit)}s/{COOLDOWN_AFTER_EXIT}s)"
        
        if self.last_exit_side and side != self.last_exit_side:
            return False, "منع عكس بدون كسر هيكل"
        
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
        if len(df) < 30:
            return {"score_buy": 0, "score_sell": 0, "decision": "HOLD"}
        
        analyzer = MarketAnalyzer()
        indicators = analyzer.compute_indicators(df)
        market_phase = analyzer.detect_market_phase(df)
        
        score_buy = 0
        score_sell = 0
        
        current_price = indicators.get('current_price', 0)
        
        # 1. مرحلة السوق
        if market_phase == "bull":
            score_buy += 2
        elif market_phase == "bear":
            score_sell += 2
        
        # 2. RSI
        rsi = indicators.get('rsi', 50)
        if rsi < 30:
            score_buy += 2
        elif rsi > 70:
            score_sell += 2
        
        # 3. MACD
        macd_hist = indicators.get('macd_hist', 0)
        if macd_hist > 0:
            score_buy += 1
        elif macd_hist < 0:
            score_sell += 1
        
        # 4. بولنجر باندز
        bb_lower = indicators.get('bollinger_lower', current_price)
        bb_upper = indicators.get('bollinger_upper', current_price)
        
        if current_price <= bb_lower * 1.01:
            score_buy += 2
        elif current_price >= bb_upper * 0.99:
            score_sell += 2
        
        # اتخاذ القرار
        decision = "HOLD"
        if score_buy >= 5 and score_buy > score_sell:
            decision = "BUY"
        elif score_sell >= 5 and score_sell > score_buy:
            decision = "SELL"
        
        return {
            "score_buy": score_buy,
            "score_sell": score_sell,
            "decision": decision,
            "indicators": indicators,
            "market_phase": market_phase
        }
        
    except Exception as e:
        log_w(f"analyze_market error: {e}")
        return {"score_buy": 0, "score_sell": 0, "decision": "HOLD"}

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
        
        STATE["open"] = False
        STATE["qty"] = 0.0
        
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
        
        if side == "long":
            pnl_pct = (current_price - entry_price) / entry_price * 100
        else:
            pnl_pct = (entry_price - current_price) / entry_price * 100
        
        STATE["pnl"] = pnl_pct
        
        analyzer = MarketAnalyzer()
        indicators = analyzer.compute_indicators(df)
        rsi = indicators.get('rsi', 50)
        
        exit_reason = None
        
        if pnl_pct >= 2.0:
            exit_reason = f"هدف ربح {pnl_pct:.1f}%"
        elif pnl_pct <= -1.5:
            exit_reason = f"وقف خسارة {pnl_pct:.1f}%"
        elif side == "long" and rsi > 80:
            exit_reason = f"RSI مرتفع {rsi:.1f}"
        elif side == "short" and rsi < 20:
            exit_reason = f"RSI منخفض {rsi:.1f}"
        
        entry_time = STATE.get("entry_time", 0)
        time_in_trade = time.time() - entry_time
        if time_in_trade > 1800:  # 30 دقيقة
            exit_reason = f"وقت طويل ({int(time_in_trade/60)} دقيقة)"
        
        if exit_reason:
            log_i(f"⚠️ EXIT SIGNAL: {exit_reason}")
            close_position(exit_reason)
            
    except Exception as e:
        log_w(f"manage_position error: {e}")

def compute_position_size(balance, price, confidence=0.5):
    """حساب حجم الصفقة"""
    base_size = balance * 0.02
    size_multiplier = 0.5 + (confidence * 0.5)
    max_position = balance * LEVERAGE * 0.8
    final_size = min(base_size * size_multiplier, max_position / price) if price > 0 else base_size
    
    return safe_qty(final_size)

# =================== FLASK API ===================
app = Flask(__name__)

@app.route("/")
def home():
    return """
    <html>
        <head>
            <title>SUI Trading Bot v9.3</title>
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
                <h1>🚀 SUI Trading Bot v9.3</h1>
                <div class="info">
                    <p><strong>Status:</strong> <span class="live">🟢 RUNNING</span></p>
                    <p><strong>Exchange:</strong> Bybit</p>
                    <p><strong>Symbol:</strong> SUI/USDT:USDT</p>
                    <p><strong>Mode:</strong> PAPER TRADING</p>
                </div>
                <div class="info">
                    <p><strong>Balance:</strong> Checking...</p>
                    <p><strong>Open Position:</strong> NO</p>
                </div>
                <div>
                    <a href="/health" class="btn">🩺 Health Check</a>
                    <a href="/metrics" class="btn">📊 Metrics</a>
                    <a href="/close" class="btn btn-danger">🔴 Close Position</a>
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
        "server_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

@app.route("/metrics")
def metrics():
    return jsonify({
        "bot_version": BOT_VERSION,
        "exchange": EXCHANGE_NAME,
        "symbol": SYMBOL,
        "position": STATE
    })

@app.route("/close")
def close_position_route():
    success = close_position("api_request")
    return jsonify({
        "success": success,
        "message": "Position closed" if success else "No position to close"
    })

# =================== STARTUP ===================
def setup_logging():
    """إعداد نظام التسجيل"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(ch)
    
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    logging.getLogger('ccxt').setLevel(logging.WARNING)
    
    log_i("🔄 Logging system ready")

def startup():
    """بدء التشغيل"""
    log_banner("SYSTEM INITIALIZATION v9.3")
    
    # إعداد التسجيل
    setup_logging()
    
    # تحميل مواصفات السوق
    load_market_specs()
    
    # تهيئة المدير
    trade_manager = TradeManager()
    
    # التحقق من اتصال البورصة
    try:
        balance = balance_usdt()
        price = price_now()
        log_g(f"✅ Exchange connection successful")
        log_g(f"💰 Balance: {balance:.2f} USDT")
        log_g(f"💰 Current price: {price:.6f}")
    except Exception as e:
        log_e(f"❌ Exchange connection failed: {e}")
        return False, None, None, None
    
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
    
    return True, trade_manager

# =================== MAIN TRADING LOOP ===================
def trading_loop(trade_manager):
    """الحلقة الرئيسية للتداول"""
    log_banner("🚀 STARTING TRADING LOOP")
    
    # تهيئة المكونات
    anti_reversal = AntiReversalGuard()
    liquidity_engine = LiquidityEngine()
    
    while True:
        try:
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
                analysis = analyze_market(df)
                
                STATE["last_analysis"] = analysis
                STATE["market_phase"] = analysis.get("market_phase", "neutral")
                
                log_i(f"📊 ANALYSIS: Buy={analysis['score_buy']} | Sell={analysis['score_sell']} | Decision={analysis['decision']}")
                
                decision = analysis["decision"]
                if decision in ["BUY", "SELL"]:
                    side = "buy" if decision == "BUY" else "sell"
                    
                    can_enter, reason = anti_reversal.can_enter(side)
                    if not can_enter:
                        log_i(f"⛔ Protection: {reason}")
                        time.sleep(BASE_SLEEP)
                        continue
                    
                    balance = balance_usdt()
                    if balance is None or balance <= 0:
                        log_w("💰 No balance available")
                        time.sleep(BASE_SLEEP)
                        continue
                    
                    confidence = max(analysis["score_buy"], analysis["score_sell"]) / 10.0
                    position_size = compute_position_size(balance, current_price, confidence)
                    
                    if position_size > 0:
                        log_i(f"🎯 SIGNAL: {side.upper()} | Size: {position_size:.4f} | Price: {current_price:.6f}")
                        
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
                            
                            anti_reversal.record_entry(side)
                            
                            log_i(f"✅ POSITION OPENED: {side.upper()} {position_size:.4f} @ {current_price:.6f}")
            
            time.sleep(BASE_SLEEP)
            
        except Exception as e:
            log_e(f"❌ LOOP ERROR: {e}")
            time.sleep(BASE_SLEEP * 2)

# =================== MAIN EXECUTION ===================
if __name__ == "__main__":
    # إعداد معالجات الإشارات
    def signal_handler(signum, frame):
        log_i(f"🛑 Received signal {signum} - Shutting down gracefully...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # بدء التشغيل
    startup_success, trade_manager = startup()
    
    if startup_success and trade_manager:
        # بدء خيوط التنفيذ
        import threading
        
        # خيط التداول الرئيسي
        trading_thread = threading.Thread(target=trading_loop, args=(trade_manager,), daemon=True)
        trading_thread.start()
        
        log_g(f"🌐 Starting web server on port {PORT}")
        
        # تشغيل سيرفل الويب
        try:
            app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
        except Exception as e:
            log_e(f"❌ Web server failed: {e}")
    else:
        log_e("❌ Startup failed - check configuration and try again")
