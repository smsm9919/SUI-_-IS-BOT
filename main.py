# -*- coding: utf-8 -*-
"""
SUI ULTRA PRO AI BOT - الإصدار الذكي المتقدم المتكامل (SNIPER HYBRID)
• نظام Sniper Hybrid المتكامل مع Smart Money Concepts
• مجلس الإدارة الفائق الذكي مع 25 استراتيجية متقدمة
• نظام ركوب الترند الذكي المحترف لتحقيق أقصى ربح متتالي
• التركيز على جودة الصفقات وليس الكمية
• Multi-Exchange Support: BingX & Bybit
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
BOT_VERSION = f"SUI ULTRA PRO SNIPER HYBRID v10.0 — {EXCHANGE_NAME.upper()}"
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

# =================== HYBRID SNIPER SETTINGS ===================
SYMBOL     = os.getenv("SYMBOL", "SUI/USDT:USDT")
INTERVAL   = os.getenv("INTERVAL", "15m")
LEVERAGE   = int(os.getenv("LEVERAGE", 10))
RISK_ALLOC = float(os.getenv("RISK_ALLOC", 0.60))
POSITION_MODE = os.getenv("POSITION_MODE", "oneway")

# ==== SNIPER MODE SETTINGS ====
SNIPER_MODE = True  # تفعيل وضع Sniper
MAX_DAILY_TRADES = 3  # أقصى 3 صفقات يوميًا
COOLDOWN_AFTER_CLOSE = 300  # 5 دقائق بعد الإغلاق

# ==== SNIPER GATES ====
MAX_SPREAD_BPS = 6.0  # Gate 0: Spread
MIN_ADX = 20          # Gate 1: ADX (لا Chop)
MIN_CONFIDENCE = 0.65 # Gate 2: ثقة مجلس الإدارة

# ==== SMC CONTEXT SETTINGS ====
LIQUIDITY_POOL_RANGE = 0.005  # 0.5% للنطاق
SWEEP_RETRACE_THRESHOLD = 0.618  # فيبوناتشي للـ Sweep
DISPLACEMENT_ATR_MULT = 1.5      # شمعة الإزاحة

# ==== PRICE ACTION ENTRY ====
MIN_WICK_RATIO = 0.35      # نسبة الذيل للـ Rejection
MIN_RISK_REWARD = 1.5      # أقل R:R مقبول
FVG_RETEST_BUFFER = 0.001  # 0.1% للـ FVG Retest

# ==== TREND FILTER ====
USE_CHANDELIER_EXIT = True
USE_ADX_FILTER = True

# ==== TP ENGINE (LIQUIDITY-DRIVEN) ====
TP1_PARTIAL_CLOSE = 0.3    # 30% في TP1
BE_ACTIVATE_AT = 0.5       # تفعيل Breakeven عند 0.5% ربح
TRAIL_START_AT = 1.0       # بدء التريل عند 1% ربح
ATR_TRAIL_MULT = 2.0       # مضاعف ATR للتريل

# ==== DEFENSE SYSTEM ====
ANTI_REVERSAL_TIMEOUT = 3  # 3 شمعات للانتظار قبل الإغلاق الكامل
TIGHTEN_TRAIL_AT = 2.0     # تشديد التريل عند 2% ربح
PARTIAL_CLOSE_ON_WEAKNESS = 0.2  # إغلاق 20% عند إشارات الضعف

# ==== DYNAMIC SETTINGS ====
TP1_PCT_BASE       = 0.45
TP1_CLOSE_FRAC     = 0.50
BREAKEVEN_AFTER    = 0.30
TRAIL_ACTIVATE_PCT = 1.20
ATR_TRAIL_MULTIPLIER = 1.8

TREND_TPS       = [0.50, 1.00, 1.80, 2.50, 3.50, 5.00, 7.00]
TREND_TP_FRACS  = [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.10]

# ==== INTELLIGENT COUNCIL ENHANCEMENTS =====
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

# =================== PROFIT ACCUMULATION SYSTEM ===================
COMPOUND_PROFIT_REINVEST = True
PROFIT_REINVEST_RATIO = 0.4  # 40% من الأرباح يعاد استثمارها
MIN_COMPOUND_BALANCE = 50.0
PROFIT_TARGET_DAILY = 5.0  # هدف ربح يومي 5%

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

# =================== HYBRID SNIPER SYSTEMS ===================
class SniperGates:
    """نظام البوابات للتحقق من شروط التداول"""
    
    @staticmethod
    def check_all_gates(df, spread_bps, daily_trades, last_close_time, consecutive_losses):
        """التحقق من جميع البوابات قبل التفكير في أي صفقة"""
        
        # Gate 0: Spread Check
        if spread_bps and spread_bps > MAX_SPREAD_BPS:
            return False, f"Spread too high: {spread_bps:.1f} bps"
        
        # Gate 1: Daily Trade Limit
        if daily_trades >= MAX_DAILY_TRADES:
            return False, "Daily trade limit reached"
        
        # Gate 2: Cooldown Period
        if time.time() - last_close_time < COOLDOWN_AFTER_CLOSE:
            remaining = COOLDOWN_AFTER_CLOSE - (time.time() - last_close_time)
            return False, f"In cooldown: {int(remaining)}s remaining"
        
        # Gate 3: ADX Filter (No Chop)
        if USE_ADX_FILTER and len(df) >= 14:
            adx = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
            if adx.iloc[-1] < MIN_ADX:
                return False, f"ADX too low: {adx.iloc[-1]:.1f}"
        
        # Gate 4: Consecutive Losses Protection
        if consecutive_losses >= 3:
            return False, f"Too many consecutive losses: {consecutive_losses}"
        
        return True, "All gates passed"

class SMCContextAnalyzer:
    """محلل Smart Money Concepts"""
    
    @staticmethod
    def find_liquidity_pools(df, window=20):
        """العثور على برك السيولة"""
        highs = df['high'].rolling(window).max()
        lows = df['low'].rolling(window).min()
        
        current_high = df['high'].iloc[-1]
        current_low = df['low'].iloc[-1]
        
        # البحث عن Equal Highs/Lows
        equal_highs = []
        equal_lows = []
        
        for i in range(1, len(df)-5):
            if abs(df['high'].iloc[i] - current_high) / current_high < LIQUIDITY_POOL_RANGE:
                equal_highs.append(df['high'].iloc[i])
            if abs(df['low'].iloc[i] - current_low) / current_low < LIQUIDITY_POOL_RANGE:
                equal_lows.append(df['low'].iloc[i])
        
        return {
            'equal_highs': equal_highs[-3:] if equal_highs else [],
            'equal_lows': equal_lows[-3:] if equal_lows else [],
            'recent_high': highs.iloc[-1],
            'recent_low': lows.iloc[-1]
        }
    
    @staticmethod
    def detect_sweep(df):
        """كشف عمليات Sweep"""
        if len(df) < 10:
            return None
        
        last_candle = df.iloc[-1]
        prev_candle = df.iloc[-2]
        
        # Bullish Sweep (سحب قيعان)
        if (last_candle['low'] < prev_candle['low'] and 
            last_candle['close'] > prev_candle['close'] and
            last_candle['close'] > (last_candle['open'] + last_candle['low']) / 2):
            
            retrace_level = prev_candle['low'] + (prev_candle['high'] - prev_candle['low']) * SWEEP_RETRACE_THRESHOLD
            
            return {
                'type': 'bullish_sweep',
                'sweep_low': last_candle['low'],
                'retrace_target': retrace_level,
                'strength': abs(last_candle['close'] - last_candle['low']) / (prev_candle['high'] - prev_candle['low'])
            }
        
        # Bearish Sweep (سحب قمم)
        elif (last_candle['high'] > prev_candle['high'] and 
              last_candle['close'] < prev_candle['close'] and
              last_candle['close'] < (last_candle['open'] + last_candle['high']) / 2):
            
            retrace_level = prev_candle['high'] - (prev_candle['high'] - prev_candle['low']) * SWEEP_RETRACE_THRESHOLD
            
            return {
                'type': 'bearish_sweep',
                'sweep_high': last_candle['high'],
                'retrace_target': retrace_level,
                'strength': abs(last_candle['close'] - last_candle['high']) / (prev_candle['high'] - prev_candle['low'])
            }
        
        return None
    
    @staticmethod
    def detect_choch_bos(df):
        """كسر الهيكل وتغيير النية"""
        if len(df) < 10:
            return None
        
        # CHoCH (Change of Character)
        recent_high = df['high'].iloc[-5:].max()
        recent_low = df['low'].iloc[-5:].min()
        
        # BOS (Break of Structure)
        prev_swing_high = df['high'].iloc[-10:-5].max()
        prev_swing_low = df['low'].iloc[-10:-5].min()
        
        current_price = df['close'].iloc[-1]
        
        bullish_choch = (current_price > prev_swing_high and 
                        df['close'].iloc[-2] < prev_swing_high)
        
        bearish_choch = (current_price < prev_swing_low and 
                        df['close'].iloc[-2] > prev_swing_low)
        
        bullish_bos = (current_price > recent_high and 
                      df['high'].iloc[-2] < recent_high)
        
        bearish_bos = (current_price < recent_low and 
                      df['low'].iloc[-2] > recent_low)
        
        result = {}
        if bullish_choch:
            result['bullish_choch'] = True
        if bearish_choch:
            result['bearish_choch'] = True
        if bullish_bos:
            result['bullish_bos'] = True
        if bearish_bos:
            result['bearish_bos'] = True
        
        return result if result else None

class PriceActionEntry:
    """محرك الدخول بناءً على حركة السعر"""
    
    @staticmethod
    def detect_rejection(df):
        """كشف شمعات الرفض"""
        last_candle = df.iloc[-1]
        
        # حساب نسب الذيل
        upper_wick = last_candle['high'] - max(last_candle['open'], last_candle['close'])
        lower_wick = min(last_candle['open'], last_candle['close']) - last_candle['low']
        body = abs(last_candle['close'] - last_candle['open'])
        
        if body == 0:
            return None
        
        upper_wick_ratio = upper_wick / body
        lower_wick_ratio = lower_wick / body
        
        # Bearish Rejection (ذيل علوي كبير)
        if upper_wick_ratio >= MIN_WICK_RATIO and last_candle['close'] < last_candle['open']:
            return {
                'type': 'bearish_rejection',
                'wick_ratio': upper_wick_ratio,
                'rejection_level': last_candle['high']
            }
        
        # Bullish Rejection (ذيل سفلي كبير)
        elif lower_wick_ratio >= MIN_WICK_RATIO and last_candle['close'] > last_candle['open']:
            return {
                'type': 'bullish_rejection',
                'wick_ratio': lower_wick_ratio,
                'rejection_level': last_candle['low']
            }
        
        return None
    
    @staticmethod
    def detect_engulfing(df):
        """كشف شمعات الـ Engulfing"""
        if len(df) < 2:
            return None
        
        last = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Bullish Engulfing
        if (last['close'] > last['open'] and 
            prev['close'] < prev['open'] and
            last['open'] < prev['close'] and 
            last['close'] > prev['open']):
            
            volume_spike = last['volume'] > prev['volume'] * 1.5
            
            return {
                'type': 'bullish_engulfing',
                'strength': (last['close'] - last['open']) / (prev['open'] - prev['close']),
                'volume_spike': volume_spike
            }
        
        # Bearish Engulfing
        elif (last['close'] < last['open'] and 
              prev['close'] > prev['open'] and
              last['open'] > prev['close'] and 
              last['close'] < prev['open']):
            
            volume_spike = last['volume'] > prev['volume'] * 1.5
            
            return {
                'type': 'bearish_engulfing',
                'strength': (last['open'] - last['close']) / (prev['close'] - prev['open']),
                'volume_spike': volume_spike
            }
        
        return None

class TrendRegimeFilter:
    """فلتر الترند والنظام السعري"""
    
    @staticmethod
    def get_chandelier_exit(df, period=22, multiplier=3):
        """Chandelier Exit لتحديد النظام"""
        if len(df) < period:
            return 0
            
        high = df['high'].rolling(period).max()
        low = df['low'].rolling(period).min()
        
        atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=period)
        
        long_stop = high - atr * multiplier
        short_stop = low + atr * multiplier
        
        current_price = df['close'].iloc[-1]
        
        if current_price > long_stop.iloc[-1]:
            return 1  # نظام صاعد
        elif current_price < short_stop.iloc[-1]:
            return -1  # نظام هابط
        else:
            return 0  # نظام جانبي
    
    @staticmethod
    def get_trend_direction(df):
        """تحديد اتجاه الترند"""
        if len(df) < 14:
            return 0
            
        # استخدام ADX و DI
        adx = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        plus_di = talib.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
        minus_di = talib.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
        
        current_adx = adx.iloc[-1]
        current_plus_di = plus_di.iloc[-1]
        current_minus_di = minus_di.iloc[-1]
        
        if current_adx < MIN_ADX:
            return 0  # لا ترند واضح
        
        if current_plus_di > current_minus_di and (current_plus_di - current_minus_di) > 10:
            return 1  # ترند صاعد
        
        elif current_minus_di > current_plus_di and (current_minus_di - current_plus_di) > 10:
            return -1  # ترند هابط
        
        return 0

class SniperRiskEngine:
    """محرك إدارة المخاطر المتقدم"""
    
    @staticmethod
    def calculate_position_size(balance, entry_price, stop_loss_price, risk_percent=0.02):
        """حساب حجم الصفقة"""
        if entry_price <= 0 or stop_loss_price <= 0:
            return 0
        
        risk_amount = balance * risk_percent
        price_risk = abs(entry_price - stop_loss_price)
        
        if price_risk == 0:
            return 0
        
        position_size = risk_amount / price_risk
        
        return position_size
    
    @staticmethod
    def calculate_stop_loss(df, side, entry_price):
        """حساب وقف الخسارة"""
        if len(df) < 14:
            return None
            
        atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14).iloc[-1]
        
        if side == "buy":
            # SL خلف آخر Sweep سفلي
            recent_lows = df['low'].iloc[-10:].values
            if len(recent_lows) > 0:
                last_sweep_low = min(recent_lows)
                sl_candidate = last_sweep_low - atr * 0.5
                
                # استخدام الأكثر تحفظاً
                sl_price = min(sl_candidate, entry_price * 0.99)  # أقصى خسارة 1%
                
                # التأكد من R:R مناسب
                if (entry_price - sl_price) / entry_price < 0.002:  # أقل من 0.2%
                    return None
                
                return sl_price
        
        else:  # sell
            # SL خلف آخر Sweep علوي
            recent_highs = df['high'].iloc[-10:].values
            if len(recent_highs) > 0:
                last_sweep_high = max(recent_highs)
                sl_candidate = last_sweep_high + atr * 0.5
                
                # استخدام الأكثر تحفظاً
                sl_price = max(sl_candidate, entry_price * 1.01)  # أقصى خسارة 1%
                
                # التأكد من R:R مناسب
                if (sl_price - entry_price) / entry_price < 0.002:  # أقل من 0.2%
                    return None
                
                return sl_price
        
        return None

class LiquidityTPEngine:
    """محرك أهداف الربح المعتمد على السيولة"""
    
    @staticmethod
    def calculate_take_profits(df, side, entry_price, sl_price):
        """حساب أهداف الربح"""
        if side == "buy":
            risk = entry_price - sl_price
            
            # TP1: أقرب مستوى سيولة (Equal Highs)
            smc = SMCContextAnalyzer()
            liquidity = smc.find_liquidity_pools(df)
            
            if liquidity['equal_highs']:
                tp1 = min(liquidity['equal_highs'])
            else:
                tp1 = entry_price + risk * MIN_RISK_REWARD
            
            # TP2: مستوى سيولة أعلى أو R:R 2:1
            tp2 = entry_price + risk * 2
            
            return tp1, tp2
        
        else:  # sell
            risk = sl_price - entry_price
            
            # TP1: أقرب مستوى سيولة (Equal Lows)
            smc = SMCContextAnalyzer()
            liquidity = smc.find_liquidity_pools(df)
            
            if liquidity['equal_lows']:
                tp1 = max(liquidity['equal_lows'])
            else:
                tp1 = entry_price - risk * MIN_RISK_REWARD
            
            # TP2: مستوى سيولة أدنى أو R:R 2:1
            tp2 = entry_price - risk * 2
            
            return tp1, tp2

class DefenseSystem:
    """نظام الدفاع ضد الانعكاسات"""
    
    @staticmethod
    def check_weakness_signals(df, side):
        """فحص إشارات الضعف"""
        if len(df) < 5:
            return []
            
        signals = []
        
        # 1. شمعة عكسية كبيرة
        last_candle = df.iloc[-1]
        if side == "buy" and last_candle['close'] < last_candle['open']:
            body_size = abs(last_candle['close'] - last_candle['open'])
            avg_body = abs(df['close'] - df['open']).rolling(5).mean().iloc[-1]
            
            if body_size > avg_body * 1.5:
                signals.append("large_bearish_candle")
        
        elif side == "sell" and last_candle['close'] > last_candle['open']:
            body_size = abs(last_candle['close'] - last_candle['open'])
            avg_body = abs(df['close'] - df['open']).rolling(5).mean().iloc[-1]
            
            if body_size > avg_body * 1.5:
                signals.append("large_bullish_candle")
        
        # 2. ضعف ADX/DI
        if len(df) >= 14:
            adx = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14).iloc[-1]
            if adx < MIN_ADX:
                signals.append("weak_adx")
        
        return signals

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
        self.daily_trades = 0
        self.last_trade_time = 0
        self.last_close_time = 0
        
    def reset_daily_stats(self):
        """إعادة تعيين الإحصائيات اليومية"""
        today = datetime.now().date()
        if today != getattr(self, '_last_reset_date', None):
            self.daily_profit = 0.0
            self.daily_trades = 0
            self._last_reset_date = today
    
    def can_trade_today(self):
        """التحقق من إمكانية التداول اليوم"""
        self.reset_daily_stats()
        return self.daily_trades < MAX_DAILY_TRADES
    
    def in_cooldown(self):
        """التحقق من فترة الانتظار"""
        return time.time() - self.last_close_time < COOLDOWN_AFTER_CLOSE
    
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
        self.daily_trades += 1
        self.last_trade_time = time.time()
        
        if profit > 0:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            
        # تحديث إحصائيات الأداء
        self.calculate_performance_metrics()
        
    def record_trade_close(self):
        """تسجيل إغلاق الصفقة"""
        self.last_close_time = time.time()
        
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
            
    def get_trade_suggestions(self):
        """الحصول على اقتراحات تداول ذكية بناءً على الأداء"""
        suggestions = []
        
        if self.consecutive_losses >= 3:
            suggestions.append("REDUCE_SIZE: خسائر متتالية - تقليل حجم الصفقة")
            
        if self.win_rate < 40 and len(self.trade_history) > 10:
            suggestions.append("REVIEW_STRATEGY: نسبة نجاح منخفضة - مراجعة الاستراتيجية")
            
        if self.avg_loss > self.avg_win * 1.5 and self.avg_loss > 0:
            suggestions.append("ADJUST_STOP_LOSS: متوسط الخسارة أكبر من متوسط الربح - تعديل وقف الخسارة")
            
        if self.daily_trades >= MAX_DAILY_TRADES:
            suggestions.append("DAILY_LIMIT: وصلت للحد اليومي للصفقات")
            
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

# إنشاء مدير الصفقات الذكي
trade_manager = SmartTradeManager()
market_analyzer = AdvancedMarketAnalyzer()

# =================== PROFESSIONAL LOGGING ===================
def log_i(msg): 
    print(f"ℹ️ {datetime.now().strftime('%H:%M:%S')} {msg}", flush=True)

def log_g(msg): 
    print(f"✅ {datetime.now().strftime('%H:%M:%S')} {msg}", flush=True)

def log_w(msg): 
    print(f"🟨 {datetime.now().strftime('%H:%M:%S')} {msg}", flush=True)

def log_e(msg): 
    print(f"❌ {datetime.now().strftime('%H:%M:%S')} {msg}", flush=True)

def log_sniper(action, details=""):
    """تسجيل خاص لعمليات Sniper"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"🎯 [{timestamp}] {action} {details}", flush=True)

def log_banner(text): 
    print(f"\n{'—'*12} {text} {'—'*12}\n", flush=True)

def save_state(state: dict):
    try:
        state["ts"] = int(time.time())
        state["trade_stats"] = {
            "daily_profit": trade_manager.daily_profit,
            "consecutive_wins": trade_manager.consecutive_wins,
            "consecutive_losses": trade_manager.consecutive_losses,
            "win_rate": trade_manager.win_rate,
            "daily_trades": trade_manager.daily_trades,
            "last_close_time": trade_manager.last_close_time
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
            trade_manager.daily_trades = state["trade_stats"].get("daily_trades", 0)
            trade_manager.last_close_time = state["trade_stats"].get("last_close_time", 0)
            
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

def print_position_snapshot(reason="OPEN", color=None):
    try:
        side   = STATE.get("side")
        open_f = STATE.get("open",False)
        qty    = STATE.get("qty"); px = STATE.get("entry")
        mode   = STATE.get("mode","sniper")
        lev    = globals().get("LEVERAGE",0)
        tp1    = STATE.get("tp1", 0)
        tp2    = STATE.get("tp2", 0)
        sl     = STATE.get("sl", 0)
        bal    = balance_usdt()
        spread = STATE.get("last_spread_bps")

        if color is None:
            icon = "🟢" if side=="buy" else "🔴"
        else:
            icon = "🟢" if str(color).lower()=="green" else "🔴"

        log_i(f"{icon} {reason} — SNIPER POSITION SNAPSHOT")
        log_i(f"SIDE: {side} | QTY: {_fmt(qty)} | ENTRY: {_fmt(px)}")
        log_i(f"LEV: {lev}× | MODE: {mode} | OPEN: {open_f}")
        log_i(f"SL: {_fmt(sl)} | TP1: {_fmt(tp1)} | TP2: {_fmt(tp2)}")
        log_i(f"SPREAD: {_fmt(spread,2)} bps")
        
        # إضافة إحصائيات الأداء
        log_i(f"PERFORMANCE: Win Rate: {trade_manager.win_rate:.1f}% | Daily PnL: {trade_manager.daily_profit:.2f}")
        log_i(f"DAILY TRADES: {trade_manager.daily_trades}/{MAX_DAILY_TRADES}")
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

def fetch_ohlcv(limit=200):
    rows = with_retry(lambda: ex.fetch_ohlcv(SYMBOL, timeframe=INTERVAL, limit=limit, params={"type":"swap"}))
    df = pd.DataFrame(rows, columns=["time","open","high","low","close","volume"])
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    return df

def price_now():
    try:
        t = with_retry(lambda: ex.fetch_ticker(SYMBOL))
        return t.get("last") or t.get("close")
    except Exception: return None

def balance_usdt():
    if not MODE_LIVE: return 1000.0
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
        bollinger_upper, bollinger_middle, bollinger_lower = talib.BBANDS(close, timeperiod=20)
        
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

# =================== ULTRA INTELLIGENT COUNCIL AI ===================
def ultra_intelligent_council_ai(df):
    """مجلس الإدارة الذكي الفائق - يدمج SMC مع المؤشرات المتقدمة"""
    try:
        if len(df) < 100:
            return {"b": 0, "s": 0, "score_b": 0.0, "score_s": 0.0, "confidence": 0.0, "logs": []}
        
        # تحليل السوق المتقدم
        market_phase = market_analyzer.detect_market_phase(df)
        support_resistance = market_analyzer.calculate_support_resistance(df)
        volatility_regime, volatility_ratio = market_analyzer.analyze_volatility_regime(df)
        
        # المؤشرات المتقدمة
        advanced_indicators = compute_advanced_indicators(df)
        
        # تحليل SMC
        smc = SMCContextAnalyzer()
        liquidity = smc.find_liquidity_pools(df)
        sweep = smc.detect_sweep(df)
        structure = smc.detect_choch_bos(df)
        
        # تحليل حركة السعر
        pa = PriceActionEntry()
        rejection = pa.detect_rejection(df)
        engulfing = pa.detect_engulfing(df)
        
        # فلتر الترند
        trend_filter = TrendRegimeFilter()
        chandelier_dir = trend_filter.get_chandelier_exit(df)
        trend_dir = trend_filter.get_trend_direction(df)
        
        votes_b = 0
        votes_s = 0
        score_b = 0.0
        score_s = 0.0
        logs = []
        confidence_factors = []
        
        current_price = float(df['close'].iloc[-1])
        
        # ===== 1. تحليل SMC =====
        if sweep:
            if sweep['type'] == 'bullish_sweep':
                score_b += WEIGHT_MARKET_STRUCTURE * 2.5
                votes_b += 3
                logs.append(f"📈 Bullish Sweep detected (strength: {sweep['strength']:.2f})")
                confidence_factors.append(1.8)
            elif sweep['type'] == 'bearish_sweep':
                score_s += WEIGHT_MARKET_STRUCTURE * 2.5
                votes_s += 3
                logs.append(f"📉 Bearish Sweep detected (strength: {sweep['strength']:.2f})")
                confidence_factors.append(1.8)
        
        if structure:
            if structure.get('bullish_choch') or structure.get('bullish_bos'):
                score_b += WEIGHT_BREAKOUT * 2.0
                votes_b += 2
                logs.append("🚀 Bullish structure break")
            if structure.get('bearish_choch') or structure.get('bearish_bos'):
                score_s += WEIGHT_BREAKOUT * 2.0
                votes_s += 2
                logs.append("🚀 Bearish structure break")
        
        # ===== 2. تحليل حركة السعر =====
        if rejection:
            if rejection['type'] == 'bullish_rejection' and rejection['wick_ratio'] >= MIN_WICK_RATIO:
                score_b += WEIGHT_CANDLES * 1.8
                votes_b += 2
                logs.append(f"🕯️ Bullish rejection (wick ratio: {rejection['wick_ratio']:.2f})")
            elif rejection['type'] == 'bearish_rejection' and rejection['wick_ratio'] >= MIN_WICK_RATIO:
                score_s += WEIGHT_CANDLES * 1.8
                votes_s += 2
                logs.append(f"🕯️ Bearish rejection (wick ratio: {rejection['wick_ratio']:.2f})")
        
        if engulfing:
            if engulfing['type'] == 'bullish_engulfing':
                score_b += WEIGHT_CANDLES * 2.0
                votes_b += 2
                if engulfing['volume_spike']:
                    score_b += WEIGHT_VOLUME * 0.5
                    logs.append("📊 Bullish engulfing with volume spike")
                else:
                    logs.append("📊 Bullish engulfing")
            elif engulfing['type'] == 'bearish_engulfing':
                score_s += WEIGHT_CANDLES * 2.0
                votes_s += 2
                if engulfing['volume_spike']:
                    score_s += WEIGHT_VOLUME * 0.5
                    logs.append("📊 Bearish engulfing with volume spike")
                else:
                    logs.append("📊 Bearish engulfing")
        
        # ===== 3. فلتر الترند =====
        if USE_CHANDELIER_EXIT:
            if chandelier_dir == 1:
                score_b += WEIGHT_EARLY_TREND * 1.5
                votes_b += 2
                logs.append("📈 Chandelier Exit: Bullish regime")
            elif chandelier_dir == -1:
                score_s += WEIGHT_EARLY_TREND * 1.5
                votes_s += 2
                logs.append("📉 Chandelier Exit: Bearish regime")
        
        if trend_dir == 1:
            score_b += WEIGHT_ADX * 2.0
            votes_b += 2
            logs.append(f"🎯 Strong uptrend (ADX: {advanced_indicators.get('adx', 0):.1f})")
            confidence_factors.append(1.5)
        elif trend_dir == -1:
            score_s += WEIGHT_ADX * 2.0
            votes_s += 2
            logs.append(f"🎯 Strong downtrend (ADX: {advanced_indicators.get('adx', 0):.1f})")
            confidence_factors.append(1.5)
        
        # ===== 4. المؤشرات التقنية =====
        rsi = advanced_indicators.get('rsi', 50)
        if rsi < 30:
            score_b += WEIGHT_RSI * 2.0
            votes_b += 2
            logs.append("📊 RSI oversold")
        elif rsi > 70:
            score_s += WEIGHT_RSI * 2.0
            votes_s += 2
            logs.append("📊 RSI overbought")
        
        macd_hist = advanced_indicators.get('macd_hist', 0)
        if macd_hist > 0:
            score_b += WEIGHT_MACD * 1.5
            votes_b += 1
        elif macd_hist < 0:
            score_s += WEIGHT_MACD * 1.5
            votes_s += 1
        
        # ===== 5. السيولة =====
        support_levels = support_resistance.get('support_levels', [])
        resistance_levels = support_resistance.get('resistance_levels', [])
        
        if support_levels and current_price <= support_levels[-1] * 1.005:
            score_b += WEIGHT_MARKET_STRUCTURE * 1.5
            votes_b += 1
            logs.append("🛡️ Near strong support")
        
        if resistance_levels and current_price >= resistance_levels[0] * 0.995:
            score_s += WEIGHT_MARKET_STRUCTURE * 1.5
            votes_s += 1
            logs.append("🚧 Near strong resistance")
        
        # ===== 6. التقلب =====
        if volatility_regime == "high":
            # في التقلب العالي، نكون أكثر حذراً
            score_b *= 0.8
            score_s *= 0.8
            logs.append("⚡ High volatility - reducing confidence")
        
        # ===== 7. أداء التداول السابق =====
        if trade_manager.consecutive_losses >= 2:
            score_b *= 0.7
            score_s *= 0.7
            logs.append("⚠️ Consecutive losses - reducing confidence")
        
        if trade_manager.consecutive_wins >= 3:
            score_b *= 1.2
            score_s *= 1.2
            logs.append("🎯 Consecutive wins - increasing confidence")
        
        # ===== 8. حساب الثقة =====
        if confidence_factors:
            confidence_multiplier = sum(confidence_factors) / len(confidence_factors)
            score_b *= confidence_multiplier
            score_s *= confidence_multiplier
        
        total_score = score_b + score_s
        max_possible_score = 25.0
        
        confidence = min(1.0, total_score / max_possible_score)
        
        # الحد الأدنى للثقة
        if confidence < MIN_CONFIDENCE:
            score_b *= 0.5
            score_s *= 0.5
            logs.append(f"🛡️ Low confidence ({confidence:.2f} < {MIN_CONFIDENCE}) - reducing")
        
        return {
            "b": votes_b,
            "s": votes_s,
            "score_b": round(score_b, 2),
            "score_s": round(score_s, 2),
            "confidence": round(confidence, 2),
            "logs": logs[-10:],  # آخر 10 رسائل فقط
            "market_phase": market_phase,
            "volatility_regime": volatility_regime,
            "support_resistance": support_resistance,
            "indicators": advanced_indicators,
            "smc_analysis": {
                "liquidity": liquidity,
                "sweep": sweep,
                "structure": structure
            }
        }
        
    except Exception as e:
        log_e(f"Ultra intelligent council error: {e}")
        return {"b": 0, "s": 0, "score_b": 0.0, "score_s": 0.0, "confidence": 0.0, "logs": [f"Error: {e}"]}

# =================== HYBRID SNIPER EXECUTION ===================
def execute_hybrid_trade(side, price, qty, council_data, sniper_context):
    """تنفيذ صفقة هجينة تجمع بين Sniper وCouncil AI"""
    try:
        if not EXECUTE_ORDERS or DRY_RUN:
            log_sniper("DRY_RUN", f"{side} {qty:.4f} @ {price:.6f}")
            return True
        
        if qty <= 0:
            log_e("❌ كمية غير صالحة للتنفيذ")
            return False
        
        confidence = council_data.get('confidence', 0)
        market_phase = council_data.get('market_phase', 'neutral')
        
        log_sniper("EXECUTING_HYBRID_TRADE", 
                  f"{side.upper()} {qty:.4f} @ {price:.6f}")
        log_i(f"   CONFIDENCE: {confidence:.2f}")
        log_i(f"   MARKET PHASE: {market_phase}")
        log_i(f"   SNIPER CONTEXT: {sniper_context}")
        
        if MODE_LIVE:
            exchange_set_leverage(ex, LEVERAGE, SYMBOL)
            params = exchange_specific_params(side, is_close=False)
            ex.create_order(SYMBOL, "market", side, qty, None, params)
        
        log_g(f"✅ HYBRID TRADE EXECUTED: {side.upper()} {qty:.4f} @ {price:.6f}")
        
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
        log_e(f"❌ HYBRID TRADE EXECUTION FAILED: {e}")
        return False

def close_position_strict(reason=""):
    """إغلاق صارم للمركز"""
    if not STATE.get("open") or STATE.get("qty", 0) <= 0:
        return False
    
    close_side = "sell" if STATE["side"] == "buy" else "buy"
    qty = STATE["qty"]
    
    log_sniper("CLOSING_POSITION", 
              f"{STATE['side']} @ {STATE['entry']:.6f}, Reason: {reason}")
    
    try:
        if MODE_LIVE and EXECUTE_ORDERS and not DRY_RUN:
            params = exchange_specific_params(close_side, is_close=True)
            ex.create_order(SYMBOL, "market", close_side, qty, None, params)
        
        # حساب الربح
        current_price = price_now()
        if current_price:
            if STATE["side"] == "buy":
                profit = (current_price - STATE["entry"]) * STATE["qty"]
            else:
                profit = (STATE["entry"] - current_price) * STATE["qty"]
            
            trade_manager.record_trade(
                side=STATE["side"],
                entry=STATE["entry"],
                exit_price=current_price,
                quantity=STATE["qty"],
                profit=profit,
                duration=(datetime.now() - STATE.get("opened_at", datetime.now())).total_seconds() / 60
            )
            
            log_g(f"Position closed. Profit: {profit:.4f} USDT")
        else:
            trade_manager.record_trade_close()
        
        # تحديث الحالة
        STATE.update({
            "open": False,
            "side": None,
            "entry": None,
            "qty": 0.0,
            "sl": None,
            "tp1": None,
            "tp2": None,
            "trail_price": None,
            "breakeven_activated": False,
            "partial_closed": False,
            "bars_in_trade": 0
        })
        
        return True
        
    except Exception as e:
        log_e(f"❌ CLOSE POSITION FAILED: {e}")
        return False

def partial_close_position(qty_pct, reason=""):
    """إغلاق جزئي للمركز"""
    if not STATE.get("open") or STATE.get("qty", 0) <= 0:
        return False
    
    close_qty = STATE["qty"] * qty_pct
    close_side = "sell" if STATE["side"] == "buy" else "buy"
    
    log_sniper("PARTIAL_CLOSE", 
              f"{qty_pct*100}% of {STATE['qty']:.4f}, Reason: {reason}")
    
    try:
        if MODE_LIVE and EXECUTE_ORDERS and not DRY_RUN:
            params = exchange_specific_params(close_side, is_close=True)
            ex.create_order(SYMBOL, "market", close_side, close_qty, None, params)
        
        STATE["qty"] = safe_qty(STATE["qty"] - close_qty)
        STATE["partial_closed"] = True
        
        log_g(f"Partial close successful. Remaining qty: {STATE['qty']:.4f}")
        return True
        
    except Exception as e:
        log_e(f"❌ PARTIAL CLOSE FAILED: {e}")
        return False

# =================== HYBRID TRADING LOGIC ===================
def hybrid_sniper_cycle():
    """دورة التداول الهجينة التي تجمع بين Sniper وCouncil AI"""
    
    # 0. التحقق من البوابات
    if STATE.get("open", False):
        return
    
    df = fetch_ohlcv(limit=150)
    if df.empty or len(df) < 100:
        return
    
    current_price = price_now()
    if not current_price:
        return
    
    # حساب Spread
    spread_bps = orderbook_spread_bps()
    
    # التحقق من البوابات
    gates_passed, gate_message = SniperGates.check_all_gates(
        df, spread_bps, 
        trade_manager.daily_trades,
        trade_manager.last_close_time,
        trade_manager.consecutive_losses
    )
    
    if not gates_passed:
        if gate_message and "cooldown" not in gate_message:
            log_sniper("GATE_FAILED", gate_message)
        return
    
    # 1. مجلس الإدارة الذكي
    council_data = ultra_intelligent_council_ai(df)
    STATE["last_council"] = council_data
    STATE["last_ind"] = council_data.get("indicators", {})
    STATE["last_spread_bps"] = spread_bps
    
    # 2. تحليل SMC
    smc = SMCContextAnalyzer()
    context_signals = []
    
    sweep = smc.detect_sweep(df)
    if sweep:
        context_signals.append(sweep)
    
    structure = smc.detect_choch_bos(df)
    if structure:
        context_signals.append(structure)
    
    # 3. تحليل حركة السعر
    pa = PriceActionEntry()
    entry_signals = []
    
    rejection = pa.detect_rejection(df)
    if rejection:
        entry_signals.append(rejection)
    
    engulfing = pa.detect_engulfing(df)
    if engulfing:
        entry_signals.append(engulfing)
    
    # 4. فلتر الترند
    trend_filter = TrendRegimeFilter()
    chandelier_dir = trend_filter.get_chandelier_exit(df)
    trend_dir = trend_filter.get_trend_direction(df)
    
    # 5. تحديد اتجاه الصفقة
    trade_side = None
    trade_reason = []
    
    # تحليل إشارات مجلس الإدارة
    if council_data["score_b"] > council_data["score_s"] and council_data["score_b"] >= 15.0:
        if council_data["confidence"] >= MIN_CONFIDENCE:
            # فحص توافق SMC
            for signal in context_signals:
                if isinstance(signal, dict) and 'bullish' in str(signal.get('type', '')):
                    trade_side = "buy"
                    trade_reason.append(f"Council: {council_data['score_b']:.1f} + SMC: {signal.get('type')}")
                    break
    
    elif council_data["score_s"] > council_data["score_b"] and council_data["score_s"] >= 15.0:
        if council_data["confidence"] >= MIN_CONFIDENCE:
            # فحص توافق SMC
            for signal in context_signals:
                if isinstance(signal, dict) and 'bearish' in str(signal.get('type', '')):
                    trade_side = "sell"
                    trade_reason.append(f"Council: {council_data['score_s']:.1f} + SMC: {signal.get('type')}")
                    break
    
    # إذا لم يكن هناك توافق بين Council وSMC، نبحث عن إشارات حركة سعر قوية
    if not trade_side and entry_signals:
        for signal in entry_signals:
            if signal['type'] == 'bullish_rejection' and signal['wick_ratio'] >= MIN_WICK_RATIO * 1.5:
                if trend_dir >= 0:  # صاعد أو جانبي
                    trade_side = "buy"
                    trade_reason.append(f"Strong bullish rejection (wick: {signal['wick_ratio']:.2f})")
                    break
            elif signal['type'] == 'bearish_rejection' and signal['wick_ratio'] >= MIN_WICK_RATIO * 1.5:
                if trend_dir <= 0:  # هابط أو جانبي
                    trade_side = "sell"
                    trade_reason.append(f"Strong bearish rejection (wick: {signal['wick_ratio']:.2f})")
                    break
    
    if not trade_side:
        return
    
    # 6. Risk Engine
    risk_engine = SniperRiskEngine()
    
    # حساب وقف الخسارة
    sl_price = risk_engine.calculate_stop_loss(df, trade_side, current_price)
    if not sl_price:
        log_sniper("SL_REJECTED", "Stop loss too tight or invalid")
        return
    
    # حساب R:R
    if trade_side == "buy":
        risk_pct = (current_price - sl_price) / current_price * 100
    else:
        risk_pct = (sl_price - current_price) / current_price * 100
    
    if risk_pct > 2.0:
        log_sniper("RISK_TOO_HIGH", f"Risk: {risk_pct:.2f}%")
        return
    
    # حساب حجم الصفقة
    balance = balance_usdt()
    if not balance:
        return
    
    position_size = risk_engine.calculate_position_size(balance, current_price, sl_price, RISK_ALLOC)
    if position_size <= 0:
        return
    
    # 7. TP Engine
    tp_engine = LiquidityTPEngine()
    tp1, tp2 = tp_engine.calculate_take_profits(df, trade_side, current_price, sl_price)
    
    # 8. فتح الصفقة الهجينة
    log_sniper("OPENING_HYBRID_TRADE", 
              f"{trade_side.upper()} @ {current_price:.6f}, "
              f"SL: {sl_price:.6f}, TP1: {tp1:.6f}, TP2: {tp2:.6f}, "
              f"Size: {position_size:.4f}, Risk: {risk_pct:.2f}%, "
              f"Reason: {', '.join(trade_reason)}")
    
    success = execute_hybrid_trade(trade_side, current_price, position_size, council_data, {
        "context_signals": [s.get('type') for s in context_signals if isinstance(s, dict)],
        "entry_signals": [s.get('type') for s in entry_signals if isinstance(s, dict)],
        "trend_filter": {"chandelier": chandelier_dir, "trend": trend_dir}
    })
    
    if success:
        STATE.update({
            "open": True,
            "side": trade_side,
            "entry": current_price,
            "qty": position_size,
            "sl": sl_price,
            "tp1": tp1,
            "tp2": tp2,
            "trail_price": None,
            "breakeven_activated": False,
            "partial_closed": False,
            "bars_in_trade": 0,
            "highest_profit_pct": 0.0,
            "mode": "hybrid_sniper",
            "opened_at": datetime.now()
        })
        
        print_position_snapshot("HYBRID_OPEN")
        
        save_state({
            "in_position": True,
            "side": trade_side.upper(),
            "entry_price": current_price,
            "position_qty": position_size,
            "stop_loss": sl_price,
            "take_profit_1": tp1,
            "take_profit_2": tp2,
            "opened_at": int(time.time()),
            "mode": "hybrid_sniper"
        })

# =================== HYBRID POSITION MANAGEMENT ===================
def manage_hybrid_position():
    """إدارة المراكز الهجينة"""
    if not STATE.get("open", False):
        return
    
    df = fetch_ohlcv(limit=50)
    if df.empty:
        return
    
    current_price = price_now()
    if not current_price:
        return
    
    # زيادة عدد الشمعات في الصفقة
    STATE["bars_in_trade"] = STATE.get("bars_in_trade", 0) + 1
    
    # حساب الربح/الخسارة
    if STATE["side"] == "buy":
        pnl_pct = (current_price - STATE["entry"]) / STATE["entry"] * 100
        move_from_sl = (current_price - STATE["sl"]) / (STATE["entry"] - STATE["sl"]) * 100
    else:
        pnl_pct = (STATE["entry"] - current_price) / STATE["entry"] * 100
        move_from_sl = (STATE["sl"] - current_price) / (STATE["sl"] - STATE["entry"]) * 100
    
    STATE["pnl"] = pnl_pct
    
    if pnl_pct > STATE.get("highest_profit_pct", 0):
        STATE["highest_profit_pct"] = pnl_pct
    
    # 1. تحقيق TP1 (إغلاق جزئي + تفعيل Breakeven)
    if not STATE.get("partial_closed", False):
        if (STATE["side"] == "buy" and current_price >= STATE["tp1"]) or \
           (STATE["side"] == "sell" and current_price <= STATE["tp1"]):
            
            log_sniper("TP1_HIT", f"Price: {current_price:.6f}, TP1: {STATE['tp1']:.6f}")
            
            # إغلاق جزئي
            partial_close_position(TP1_PARTIAL_CLOSE, "TP1 achieved")
            
            # تفعيل Breakeven
            STATE["breakeven_activated"] = True
            STATE["trail_price"] = STATE["entry"]
    
    # 2. تفعيل Breakeven عند مستوى معين
    elif not STATE.get("breakeven_activated", False) and pnl_pct >= BE_ACTIVATE_AT:
        STATE["breakeven_activated"] = True
        STATE["trail_price"] = STATE["entry"]
        log_sniper("BREAKEVEN_ACTIVATED", f"PNL: {pnl_pct:.2f}%")
    
    # 3. بدء التريل عند مستوى معين
    elif pnl_pct >= TRAIL_START_AT:
        atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14).iloc[-1]
        
        if STATE["side"] == "buy":
            new_trail = current_price - atr * ATR_TRAIL_MULT
            if STATE.get("trail_price") is None or new_trail > STATE["trail_price"]:
                STATE["trail_price"] = new_trail
                log_sniper("TRAIL_UPDATED", f"New trail: {new_trail:.6f}")
        
        else:  # sell
            new_trail = current_price + atr * ATR_TRAIL_MULT
            if STATE.get("trail_price") is None or new_trail < STATE["trail_price"]:
                STATE["trail_price"] = new_trail
                log_sniper("TRAIL_UPDATED", f"New trail: {new_trail:.6f}")
    
    # 4. تشديد التريل عند أرباح عالية
    if pnl_pct >= TIGHTEN_TRAIL_AT and STATE.get("trail_price"):
        atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14).iloc[-1]
        
        if STATE["side"] == "buy":
            tighter_trail = current_price - atr * (ATR_TRAIL_MULT * 0.7)
            if tighter_trail > STATE["trail_price"]:
                STATE["trail_price"] = tighter_trail
                log_sniper("TRAIL_TIGHTENED", f"Tight trail: {tighter_trail:.6f}")
        
        else:
            tighter_trail = current_price + atr * (ATR_TRAIL_MULT * 0.7)
            if tighter_trail < STATE["trail_price"]:
                STATE["trail_price"] = tighter_trail
                log_sniper("TRAIL_TIGHTENED", f"Tight trail: {tighter_trail:.6f}")
    
    # 5. Defense System
    defense = DefenseSystem()
    weakness_signals = defense.check_weakness_signals(df, STATE["side"])
    
    if weakness_signals and STATE["bars_in_trade"] > ANTI_REVERSAL_TIMEOUT:
        log_sniper("WEAKNESS_DETECTED", f"Signals: {', '.join(weakness_signals)}")
        
        # إغلاق جزئي للدفاع
        partial_close_position(PARTIAL_CLOSE_ON_WEAKNESS, f"Weakness: {weakness_signals[0]}")
    
    # 6. التحقق من وقف الخسارة المتحرك
    if STATE.get("trail_price"):
        if STATE["side"] == "buy" and current_price <= STATE["trail_price"]:
            close_position_strict(f"Trailing stop hit: {current_price:.6f} <= {STATE['trail_price']:.6f}")
        
        elif STATE["side"] == "sell" and current_price >= STATE["trail_price"]:
            close_position_strict(f"Trailing stop hit: {current_price:.6f} >= {STATE['trail_price']:.6f}")
    
    # 7. التحقق من وقف الخسارة الثابت
    if (STATE["side"] == "buy" and current_price <= STATE["sl"]) or \
       (STATE["side"] == "sell" and current_price >= STATE["sl"]):
        close_position_strict("Stop loss hit")
    
    # 8. الحد الزمني للصفقة
    if STATE["bars_in_trade"] > 50:
        close_position_strict("Maximum time in trade reached")
    
    # 9. تحقيق الهدف اليومي
    if trade_manager.daily_profit >= PROFIT_TARGET_DAILY:
        close_position_strict("Daily profit target achieved")

# =================== MAIN HYBRID LOOP ===================
def hybrid_main_loop():
    """الحلقة الرئيسية الهجينة"""
    
    log_banner("STARTING HYBRID SNIPER TRADING BOT")
    log_i(f"🤖 Bot Version: {BOT_VERSION}")
    log_i(f"💱 Exchange: {EXCHANGE_NAME.upper()}")
    log_i(f"📈 Symbol: {SYMBOL}")
    log_i(f"⏰ Interval: {INTERVAL}")
    log_i(f"🎯 Leverage: {LEVERAGE}x")
    log_i(f"📊 Risk Allocation: {RISK_ALLOC*100}%")
    log_i(f"🎯 Max Daily Trades: {MAX_DAILY_TRADES}")
    log_i(f"🛡️ Sniper Mode: {'ACTIVE' if SNIPER_MODE else 'INACTIVE'}")
    
    while True:
        try:
            # إدارة المركز المفتوح
            if STATE.get("open", False):
                manage_hybrid_position()
            
            # البحث عن فرص جديدة
            else:
                hybrid_sniper_cycle()
            
            # الانتظار بين الدورات
            time.sleep(5)
            
        except Exception as e:
            log_e(f"❌ HYBRID MAIN LOOP ERROR: {e}")
            log_e(traceback.format_exc())
            time.sleep(30)

# =================== STATE INITIALIZATION ===================
STATE = {
    "open": False, "side": None, "entry": None, "qty": 0.0,
    "pnl": 0.0, "bars": 0, "trail": None, "breakeven": None,
    "tp1_done": False, "highest_profit_pct": 0.0,
    "profit_targets_achieved": 0, "sl": None, "tp1": None, "tp2": None,
    "trail_price": None, "breakeven_activated": False,
    "partial_closed": False, "bars_in_trade": 0, "mode": "hybrid",
    "opened_at": None
}

# =================== FLASK API ===================
app = Flask(__name__)

@app.route("/")
def home():
    return f"""
    <html>
        <head><title>SUI ULTRA PRO HYBRID SNIPER</title></head>
        <body>
            <h1>🎯 SUI ULTRA PRO HYBRID SNIPER BOT</h1>
            <p><strong>Version:</strong> {BOT_VERSION}</p>
            <p><strong>Exchange:</strong> {EXCHANGE_NAME.upper()}</p>
            <p><strong>Symbol:</strong> {SYMBOL}</p>
            <p><strong>Status:</strong> {'🟢 LIVE' if MODE_LIVE else '🟡 PAPER'}</p>
            <p><strong>Daily PnL:</strong> {trade_manager.daily_profit:.2f} USDT</p>
            <p><strong>Win Rate:</strong> {trade_manager.win_rate:.1f}%</p>
            <p><strong>Daily Trades:</strong> {trade_manager.daily_trades}/{MAX_DAILY_TRADES}</p>
            <p><strong>Position:</strong> {'🟢 OPEN' if STATE['open'] else '🔴 CLOSED'}</p>
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
        "win_rate": trade_manager.win_rate,
        "daily_trades": f"{trade_manager.daily_trades}/{MAX_DAILY_TRADES}",
        "consecutive_losses": trade_manager.consecutive_losses
    })

@app.route("/metrics")
def metrics():
    current_price = price_now()
    balance = balance_usdt()
    
    metrics_data = {
        "bot_version": BOT_VERSION,
        "exchange": EXCHANGE_NAME,
        "symbol": SYMBOL,
        "current_price": current_price,
        "balance": balance,
        "daily_profit": trade_manager.daily_profit,
        "win_rate": trade_manager.win_rate,
        "daily_trades": trade_manager.daily_trades,
        "max_daily_trades": MAX_DAILY_TRADES,
        "consecutive_wins": trade_manager.consecutive_wins,
        "consecutive_losses": trade_manager.consecutive_losses,
        "total_trades": len(trade_manager.trade_history),
        "position": STATE,
        "performance_suggestions": trade_manager.get_trade_suggestions(),
        "sniper_mode": SNIPER_MODE
    }
    
    if STATE["open"] and current_price:
        if STATE["side"] == "buy":
            pnl_pct = (current_price - STATE["entry"]) / STATE["entry"] * 100
        else:
            pnl_pct = (STATE["entry"] - current_price) / STATE["entry"] * 100
        
        metrics_data["position"]["current_pnl_pct"] = pnl_pct
    
    return jsonify(metrics_data)

@app.route("/performance")
def performance():
    recent_trades = trade_manager.trade_history[-10:] if trade_manager.trade_history else []
    return jsonify({
        "daily_profit": trade_manager.daily_profit,
        "win_rate": trade_manager.win_rate,
        "avg_win": trade_manager.avg_win,
        "avg_loss": trade_manager.avg_loss,
        "daily_trades": trade_manager.daily_trades,
        "max_daily_trades": MAX_DAILY_TRADES,
        "recent_trades": [
            {
                "time": t['timestamp'].strftime('%H:%M:%S') if isinstance(t['timestamp'], datetime) else t['timestamp'],
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
    log_i(f"   Daily Trades: {trade_manager.daily_trades}/{MAX_DAILY_TRADES}")
    log_i(f"   Consecutive Wins: {trade_manager.consecutive_wins}")
    log_i(f"   Consecutive Losses: {trade_manager.consecutive_losses}")
    
    log_g("🚀 HYBRID SNIPER TRADING BOT READY!")
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
        trading_thread = threading.Thread(target=hybrid_main_loop, daemon=True)
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
