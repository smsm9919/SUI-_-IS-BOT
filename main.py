# -*- coding: utf-8 -*-
"""
SUI ULTRA PRO AI BOT - الإصدار الذكي المتقدم المتكامل مع Smart Money Engine
• مجلس الإدارة الفائق الذكي مع 15 استراتيجية متقدمة  
• نظام ركوب الترند الذكي المحترف لتحقيق أقصى ربح متتالي
• السكالب الفائق الذكي بأهداف متعددة محسوبة
• إدارة صفقات ذكية متكيفة مع قوة الترند
• نظام Footprint + Diagonal Order-Flow المتقدم
• Multi-Exchange Support: BingX & Bybit
• HQ Trading Intelligence Patch - مناطق ذهبية + SMC + OB/FVG
• SMART PROFIT AI - نظام جني الأرباح الذكي المتقدم
• TP PROFILE SYSTEM - نظام جني الأرباح الذكي (1→2→3 مرات)
• COUNCIL STRONG ENTRY - دخول ذكي من مجلس الإدارة في المناطق القوية
• SMART MONEY ENGINE - نظام SMC محترف لاكتشاف السيولة والانفجارات
"""

import os, time, math, random, signal, sys, traceback, logging, json
from logging.handlers import RotatingFileHandler
from datetime import datetime
import pandas as pd
import numpy as np
import ccxt
from flask import Flask, jsonify
from decimal import Decimal, ROUND_DOWN, InvalidOperation
from collections import deque, defaultdict
import statistics
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple

try:
    from termcolor import colored
except Exception:
    def colored(t,*a,**k): return t

# ============================================
#  SMART MONEY ENGINE - نظام SMC محترف
# ============================================

@dataclass
class MarketState:
    regime: str            # TREND / RANGE / NO_TRADE / CHOP
    trend_strength: float  # ADX-like value
    direction: str         # BULL / BEAR / NONE

@dataclass
class LiquidityState:
    swept_high: bool
    swept_low: bool
    sweep_price: float
    sweep_type: str  # BUY_SWEEP / SELL_SWEEP

@dataclass
class StructureState:
    bos: bool              # Break of Structure
    choch: bool            # Change of Character
    direction: str         # BULL / BEAR / NONE
    confirmation: bool     # هل هناك تأكيد

@dataclass
class ZoneAnalysis:
    order_block: bool
    fvg: bool
    zone_type: str  # BULLISH / BEARISH / NONE
    price_in_zone: bool

@dataclass
class ExplosionState:
    detected: bool
    type: str      # EXPLOSION_UP / EXPLOSION_DOWN / COLLAPSE / NORMAL
    confidence: float

@dataclass
class TradingDecision:
    allow_entry: bool
    side: str      # BUY / SELL / NONE
    reason: str
    confidence: float
    trade_type: str  # SCALP / MID_TREND / LARGE_TREND

class SmartMoneyEngine:
    """المحرك الرئيسي لتحليل Smart Money Concepts"""
    
    def __init__(self, candles: List[Dict], volume: List[float], atr: float):
        """
        candles: قائمة من الشموع [{'open', 'high', 'low', 'close', 'volume'}]
        volume: قائمة بالأحجام
        atr: قيمة ATR الحالية
        """
        self.candles = candles
        self.close = np.array([c.get('close', 0) for c in candles])
        self.high = np.array([c.get('high', 0) for c in candles])
        self.low = np.array([c.get('low', 0) for c in candles])
        self.open = np.array([c.get('open', 0) for c in candles])
        self.volume = np.array(volume)
        self.atr = atr
        
    def analyze_market_regime(self) -> MarketState:
        """تحليل حالة السوق العامة"""
        if len(self.close) < 20:
            return MarketState("NO_DATA", 0.0, "NONE")
        
        # حساب قوة الترند (بديل ADX)
        price_changes = np.abs(np.diff(self.close[-20:]))
        trend_strength = np.mean(price_changes) / np.mean(self.close[-20:]) * 100
        
        # تحديد الاتجاه
        price_trend = "BULL" if self.close[-1] > self.close[-10] else "BEAR"
        
        # تصنيف حالة السوق
        if trend_strength < 0.1:
            regime = "CHOP"
        elif trend_strength < 0.3:
            regime = "RANGE"
        else:
            regime = "TREND"
            
        return MarketState(regime, trend_strength, price_trend)
    
    def analyze_liquidity(self) -> LiquidityState:
        """تحليل السيولة - أهم جزء في SMC"""
        if len(self.high) < 15:
            return LiquidityState(False, False, 0.0, "NONE")
        
        # البحث عن قمم وقيعان حديثة
        recent_highs = self.high[-15:-1]
        recent_lows = self.low[-15:-1]
        
        current_high = self.high[-1]
        current_low = self.low[-1]
        current_close = self.close[-1]
        
        max_recent_high = np.max(recent_highs)
        min_recent_low = np.min(recent_lows)
        
        # كشف سحب السيولة (Sweep)
        swept_high = (current_high > max_recent_high and 
                     current_close < max_recent_high * 0.998)  # إغلاق تحت القمة
        
        swept_low = (current_low < min_recent_low and 
                    current_close > min_recent_low * 1.002)   # إغلاق فوق القاع
        
        sweep_price = max_recent_high if swept_high else (min_recent_low if swept_low else 0.0)
        sweep_type = "SELL_SWEEP" if swept_high else ("BUY_SWEEP" if swept_low else "NONE")
        
        return LiquidityState(swept_high, swept_low, sweep_price, sweep_type)
    
    def analyze_structure(self) -> StructureState:
        """تحليل هيكل السوق (BOS / CHoCH)"""
        if len(self.high) < 20:
            return StructureState(False, False, "NONE", False)
        
        # مستويات الهيكل
        swing_highs = []
        swing_lows = []
        
        # اكتشاف القمم والقيعان (تبسيط)
        for i in range(5, len(self.high)-5):
            if self.high[i] == np.max(self.high[i-5:i+6]):
                swing_highs.append((i, self.high[i]))
            if self.low[i] == np.min(self.low[i-5:i+6]):
                swing_lows.append((i, self.low[i]))
        
        # تحليل BOS (Break of Structure)
        last_swing_high = swing_highs[-1][1] if swing_highs else 0
        last_swing_low = swing_lows[-1][1] if swing_lows else 0
        
        bos_bull = self.close[-1] > last_swing_high and last_swing_high > 0
        bos_bear = self.close[-1] < last_swing_low and last_swing_low > 0
        
        # تحليل CHoCH (Change of Character)
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            prev_swing_high = swing_highs[-2][1]
            prev_swing_low = swing_lows[-2][1]
            
            choch_bull = (self.close[-1] > prev_swing_high and 
                         self.close[-2] < prev_swing_low)
            choch_bear = (self.close[-1] < prev_swing_low and 
                         self.close[-2] > prev_swing_high)
        else:
            choch_bull = choch_bear = False
        
        direction = "BULL" if bos_bull or choch_bull else ("BEAR" if bos_bear or choch_bear else "NONE")
        confirmation = (bos_bull or bos_bear) and abs(self.close[-1] - self.close[-2]) > self.atr * 0.5
        
        return StructureState(bos_bull or bos_bear, choch_bull or choch_bear, direction, confirmation)
    
    def detect_explosion_collapse(self) -> ExplosionState:
        """كشف الانفجارات والانهيارات"""
        if len(self.close) < 3:
            return ExplosionState(False, "NORMAL", 0.0)
        
        current_candle = self.candles[-1]
        prev_candle = self.candles[-2]
        
        body_current = abs(current_candle['close'] - current_candle['open'])
        body_prev = abs(prev_candle['close'] - prev_candle['open'])
        range_current = current_candle['high'] - current_candle['low']
        
        # حجم الشمعة
        volume_current = self.volume[-1]
        volume_avg = np.mean(self.volume[-20:]) if len(self.volume) >= 20 else volume_current
        
        # نسب مهمة
        body_ratio = body_current / (range_current + 0.0001)
        volume_ratio = volume_current / (volume_avg + 0.0001)
        
        # الانفجار الصاعد
        if (body_current > body_prev * 2 and 
            volume_ratio > 1.8 and 
            body_ratio > 0.7 and
            current_candle['close'] > current_candle['open']):
            return ExplosionState(True, "EXPLOSION_UP", min(0.9, volume_ratio / 3))
        
        # الانفجار الهابط
        if (body_current > body_prev * 2 and 
            volume_ratio > 1.8 and 
            body_ratio > 0.7 and
            current_candle['close'] < current_candle['open']):
            return ExplosionState(True, "EXPLOSION_DOWN", min(0.9, volume_ratio / 3))
        
        # الانهيار (Collapse) - شمعة دوجي كبيرة مع حجم عالي
        if (body_ratio < 0.3 and 
            volume_ratio > 2.0 and
            range_current > self.atr * 1.5):
            return ExplosionState(True, "COLLAPSE", 0.7)
        
        return ExplosionState(False, "NORMAL", 0.3)
    
    def analyze_zones(self) -> ZoneAnalysis:
        """تحليل المناطق (Order Blocks / FVG)"""
        if len(self.candles) < 10:
            return ZoneAnalysis(False, False, "NONE", False)
        
        current_price = self.close[-1]
        order_block = False
        fvg = False
        zone_type = "NONE"
        price_in_zone = False
        
        # البحث عن Order Blocks (تبسيط)
        for i in range(-8, -2):
            if i + 2 >= len(self.candles):
                continue
                
            candle = self.candles[i]
            next_candle = self.candles[i+1]
            
            # Bullish OB: شمعة هابطة يليها شمعة صاعدة قوية
            if (candle['close'] < candle['open'] and 
                next_candle['close'] > next_candle['open'] and
                abs(next_candle['close'] - next_candle['open']) > self.atr * 0.8):
                
                ob_high = candle['open']
                ob_low = candle['close']
                
                if ob_low <= current_price <= ob_high:
                    order_block = True
                    zone_type = "BULLISH"
                    price_in_zone = True
                    break
            
            # Bearish OB: شمعة صاعدة يليها شمعة هابطة قوية
            elif (candle['close'] > candle['open'] and 
                  next_candle['close'] < next_candle['open'] and
                  abs(next_candle['close'] - next_candle['open']) > self.atr * 0.8):
                
                ob_high = candle['close']
                ob_low = candle['open']
                
                if ob_low <= current_price <= ob_high:
                    order_block = True
                    zone_type = "BEARISH"
                    price_in_zone = True
                    break
        
        # البحث عن FVG (تبسيط)
        if len(self.candles) >= 5:
            # Bullish FVG: فجوة صاعدة
            if (self.high[-4] < self.low[-2] and 
                self.low[-2] > self.high[-4]):
                fvg_high = self.low[-2]
                fvg_low = self.high[-4]
                
                if fvg_low <= current_price <= fvg_high:
                    fvg = True
                    zone_type = "BULLISH" if not order_block else zone_type
                    price_in_zone = True
            
            # Bearish FVG: فجوة هابطة
            elif (self.low[-4] > self.high[-2] and 
                  self.high[-2] < self.low[-4]):
                fvg_high = self.low[-4]
                fvg_low = self.high[-2]
                
                if fvg_low <= current_price <= fvg_high:
                    fvg = True
                    zone_type = "BEARISH" if not order_block else zone_type
                    price_in_zone = True
        
        return ZoneAnalysis(order_block, fvg, zone_type, price_in_zone)
    
    def detect_fake_breakout(self) -> Tuple[bool, str]:
        """كشف الاختراقات الوهمية"""
        if len(self.high) < 10:
            return False, "NO_DATA"
        
        # اختراق قمة وهمي
        if (self.high[-1] > np.max(self.high[-10:-1]) and 
            self.close[-1] < self.high[-2] and
            self.volume[-1] > np.mean(self.volume[-10:]) * 1.5):
            
            # تحليل الwick
            upper_wick = self.high[-1] - max(self.close[-1], self.open[-1])
            candle_range = self.high[-1] - self.low[-1]
            
            if upper_wick / (candle_range + 0.0001) > 0.6:
                return True, "FAKE_UP_BREAKOUT"
        
        # اختراق قاع وهمي
        if (self.low[-1] < np.min(self.low[-10:-1]) and 
            self.close[-1] > self.low[-2] and
            self.volume[-1] > np.mean(self.volume[-10:]) * 1.5):
            
            lower_wick = min(self.close[-1], self.open[-1]) - self.low[-1]
            candle_range = self.high[-1] - self.low[-1]
            
            if lower_wick / (candle_range + 0.0001) > 0.6:
                return True, "FAKE_DOWN_BREAKOUT"
        
        return False, "GENUINE"
    
    def make_decision(self, adx_value: float, rsi_value: float) -> TradingDecision:
        """اتخاذ القرار النهائي للتداول"""
        # تحليل جميع المكونات
        market = self.analyze_market_regime()
        liquidity = self.analyze_liquidity()
        structure = self.analyze_structure()
        explosion = self.detect_explosion_collapse()
        zones = self.analyze_zones()
        fake_breakout, fake_type = self.detect_fake_breakout()
        
        # منطق القرار
        reasons = []
        confidence = 0.5  # ثقة افتراضية
        side = "NONE"
        trade_type = "SCALP"
        
        # === فلترات أمان ===
        # 1. منع التداول في السوق المتذبذب
        if market.regime == "CHOP" and adx_value < 20:
            return TradingDecision(False, "NONE", "CHOP_MARKET", 0.1, "NONE")
        
        # 2. منع الاختراقات الوهمية
        if fake_breakout:
            return TradingDecision(False, "NONE", f"FAKE_BREAKOUT: {fake_type}", 0.1, "NONE")
        
        # 3. فلتر RSI متطرف
        if rsi_value > 75 or rsi_value < 25:
            reasons.append(f"RSI_EXTREME({rsi_value:.1f})")
            confidence *= 0.7
        
        # === تحديد نوع الترند ===
        if adx_value > 30 and market.trend_strength > 0.4:
            trade_type = "LARGE_TREND"
            confidence *= 1.3
        elif adx_value > 20:
            trade_type = "MID_TREND"
            confidence *= 1.1
        
        # === منطق الشراء ===
        buy_score = 0
        buy_reasons = []
        
        # 1. سحب سيولة شرائية
        if liquidity.sweep_type == "BUY_SWEEP":
            buy_score += 3
            buy_reasons.append("BUY_SWEEP")
        
        # 2. هيكل صاعد
        if structure.direction == "BULL" and structure.confirmation:
            buy_score += 2
            buy_reasons.append("BULL_STRUCTURE")
        
        # 3. انفجار صاعد
        if explosion.type == "EXPLOSION_UP":
            buy_score += 2
            buy_reasons.append("EXPLOSION_UP")
        
        # 4. منطقة شرائية
        if zones.zone_type == "BULLISH" and zones.price_in_zone:
            buy_score += 1
            buy_reasons.append("BULL_ZONE")
        
        # 5. تأكيد حجم
        if self.volume[-1] > np.mean(self.volume[-20:]) * 1.3:
            buy_score += 1
            buy_reasons.append("VOLUME_CONFIRM")
        
        # === منطق البيع ===
        sell_score = 0
        sell_reasons = []
        
        # 1. سحب سيولة بيعية
        if liquidity.sweep_type == "SELL_SWEEP":
            sell_score += 3
            sell_reasons.append("SELL_SWEEP")
        
        # 2. هيكل هابط
        if structure.direction == "BEAR" and structure.confirmation:
            sell_score += 2
            sell_reasons.append("BEAR_STRUCTURE")
        
        # 3. انفجار هابط
        if explosion.type == "EXPLOSION_DOWN":
            sell_score += 2
            sell_reasons.append("EXPLOSION_DOWN")
        
        # 4. منطقة بيعية
        if zones.zone_type == "BEARISH" and zones.price_in_zone:
            sell_score += 1
            sell_reasons.append("BEAR_ZONE")
        
        # 5. تأكيد حجم
        if self.volume[-1] > np.mean(self.volume[-20:]) * 1.3:
            sell_score += 1
            sell_reasons.append("VOLUME_CONFIRM")
        
        # === اتخاذ القرار النهائي ===
        min_score = 4 if trade_type == "LARGE_TREND" else 3
        
        if buy_score >= min_score and buy_score > sell_score:
            side = "BUY"
            reasons = buy_reasons
            confidence = min(0.95, confidence * (1 + buy_score * 0.1))
            return TradingDecision(True, side, " | ".join(reasons), confidence, trade_type)
        
        elif sell_score >= min_score and sell_score > buy_score:
            side = "SELL"
            reasons = sell_reasons
            confidence = min(0.95, confidence * (1 + sell_score * 0.1))
            return TradingDecision(True, side, " | ".join(reasons), confidence, trade_type)
        
        # لا توجد إشارة قوية
        reason_text = "NO_STRONG_SIGNAL"
        if reasons:
            reason_text += " | " + " | ".join(reasons)
        
        return TradingDecision(False, "NONE", reason_text, max(0.2, confidence * 0.7), "NONE")

# ============================================
#  TREND CLASSIFIER ENGINE - تصنيف الترند
# ============================================

class TrendClassifierEngine:
    """محرك تصنيف الترند (MID vs LARGE)"""
    
    @staticmethod
    def classify_trend(adx: float, di_plus: float, di_minus: float, 
                      candles: List[Dict], volume: List[float]) -> Dict:
        """
        تصنيف الترند إلى: LARGE / MID / CHOP / NO_TREND
        """
        if len(candles) < 30:
            return {"type": "NO_DATA", "strength": 0, "confidence": 0}
        
        close = np.array([c['close'] for c in candles])
        high = np.array([c['high'] for c in candles])
        low = np.array([c['low'] for c in candles])
        volume_arr = np.array(volume)
        
        # 1. قوة ADX
        adx_strength = adx
        
        # 2. انتشار DI
        di_spread = abs(di_plus - di_minus)
        
        # 3. توسع الهيكل
        recent_range = np.max(high[-5:]) - np.min(low[-5:])
        avg_range = np.mean([high[i] - low[i] for i in range(-20, 0) if i < len(high)])
        structure_expansion = recent_range / (avg_range + 0.0001)
        
        # 4. توسع الحجم
        recent_volume = np.mean(volume_arr[-5:]) if len(volume_arr) >= 5 else volume_arr[-1]
        avg_volume = np.mean(volume_arr[-20:]) if len(volume_arr) >= 20 else recent_volume
        volume_expansion = recent_volume / (avg_volume + 0.0001)
        
        # 5. استمرارية الاتجاه
        price_trend = "UP" if close[-1] > close[-10] else "DOWN"
        trend_consistency = 0
        
        if price_trend == "UP":
            up_candles = sum(close[i] > close[i-1] for i in range(-9, 0))
            trend_consistency = up_candles / 9
        else:
            down_candles = sum(close[i] < close[i-1] for i in range(-9, 0))
            trend_consistency = down_candles / 9
        
        # التصنيف النهائي
        score = 0
        
        # 🔥 LARGE TREND شروط قاسية
        large_trend_conditions = (
            adx_strength > 30 and
            di_spread > 15 and
            structure_expansion > 1.4 and
            volume_expansion > 1.5 and
            trend_consistency > 0.7
        )
        
        # ⚡ MID TREND شروط متوسطة
        mid_trend_conditions = (
            adx_strength > 20 and
            di_spread > 8 and
            structure_expansion > 1.2
        )
        
        if large_trend_conditions:
            trend_type = "LARGE"
            strength = min(10, adx_strength / 3 + di_spread / 5 + structure_expansion * 2)
            confidence = min(0.95, 0.6 + (trend_consistency * 0.3))
            
        elif mid_trend_conditions:
            trend_type = "MID"
            strength = min(7, adx_strength / 4 + di_spread / 8 + structure_expansion * 1.5)
            confidence = min(0.85, 0.5 + (trend_consistency * 0.25))
            
        elif adx_strength < 15:
            trend_type = "CHOP"
            strength = max(1, adx_strength / 2)
            confidence = 0.7
            
        else:
            trend_type = "NO_TREND"
            strength = adx_strength / 3
            confidence = 0.4
        
        return {
            "type": trend_type,
            "strength": round(strength, 2),
            "confidence": round(confidence, 2),
            "direction": price_trend,
            "adx": adx_strength,
            "di_spread": di_spread,
            "structure_expansion": round(structure_expansion, 2),
            "volume_expansion": round(volume_expansion, 2),
            "consistency": round(trend_consistency, 2)
        }

# ============================================
#  INTELLIGENT TRAILING STOP ENGINE
# ============================================

class IntelligentTrailingEngine:
    """محرك وقف الخسارة المتحرك الذكي"""
    
    def __init__(self, side: str, entry_price: float):
        self.side = side.upper()  # BUY / SELL
        self.entry_price = entry_price
        self.trailing_stop = None
        self.breakeven_price = entry_price
        self.highest_profit = 0
        self.tightened = False
        
    def update(self, current_price: float, candles: List[Dict], atr: float, 
              trend_strength: str = "MID") -> Tuple[float, str]:
        """
        تحديث الوقف المتحرك
        Returns: (new_stop_price, action)
        """
        if self.side == "BUY":
            return self._update_buy(current_price, candles, atr, trend_strength)
        else:
            return self._update_sell(current_price, candles, atr, trend_strength)
    
    def _update_buy(self, current_price: float, candles: List[Dict], 
                   atr: float, trend_strength: str) -> Tuple[float, str]:
        """تحديث وقف الشراء"""
        # حساب الربح الحالي
        profit_pct = ((current_price - self.entry_price) / self.entry_price) * 100
        
        # تحديث أعلى ربح
        if profit_pct > self.highest_profit:
            self.highest_profit = profit_pct
        
        # 1. تفعيل الوقف المتحرك بعد تحقيق ربح معين
        if profit_pct >= 0.8 and self.trailing_stop is None:
            self.trailing_stop = self.entry_price - (atr * 1.5)
            return self.trailing_stop, "TRAIL_ACTIVATED"
        
        # 2. تحريك الوقف عند تحقيق ربح أكبر
        if self.trailing_stop is not None:
            # حساب أعلى قاع حديث
            recent_lows = [c['low'] for c in candles[-5:]]
            if recent_lows:
                recent_low = min(recent_lows)
                
                # حساب وقف جديد
                if trend_strength == "LARGE":
                    new_stop = recent_low - (atr * 1.0)
                elif trend_strength == "MID":
                    new_stop = recent_low - (atr * 1.2)
                else:
                    new_stop = recent_low - (atr * 1.5)
                
                # تحريك الوقف للأعلى فقط (لا رجوع)
                if new_stop > self.trailing_stop:
                    self.trailing_stop = new_stop
                    return self.trailing_stop, "TRAIL_UPDATED"
        
        # 3. تفعيل نقطة التعادل بعد تحقيق ربح جيد
        if profit_pct >= 1.5 and not self.tightened:
            self.breakeven_price = self.entry_price * 1.005  # +0.5%
            self.tightened = True
            return self.breakeven_price, "BREAKEVEN_ACTIVATED"
        
        return self.trailing_stop or self.entry_price, "HOLD"
    
    def _update_sell(self, current_price: float, candles: List[Dict], 
                    atr: float, trend_strength: str) -> Tuple[float, str]:
        """تحديث وقف البيع"""
        # حساب الربح الحالي
        profit_pct = ((self.entry_price - current_price) / self.entry_price) * 100
        
        # تحديث أعلى ربح
        if profit_pct > self.highest_profit:
            self.highest_profit = profit_pct
        
        # 1. تفعيل الوقف المتحرك بعد تحقيق ربح معين
        if profit_pct >= 0.8 and self.trailing_stop is None:
            self.trailing_stop = self.entry_price + (atr * 1.5)
            return self.trailing_stop, "TRAIL_ACTIVATED"
        
        # 2. تحريك الوقف عند تحقيق ربح أكبر
        if self.trailing_stop is not None:
            # حساب أقل قمة حديثة
            recent_highs = [c['high'] for c in candles[-5:]]
            if recent_highs:
                recent_high = max(recent_highs)
                
                # حساب وقف جديد
                if trend_strength == "LARGE":
                    new_stop = recent_high + (atr * 1.0)
                elif trend_strength == "MID":
                    new_stop = recent_high + (atr * 1.2)
                else:
                    new_stop = recent_high + (atr * 1.5)
                
                # تحريك الوقف للأسفل فقط (لا رجوع)
                if new_stop < self.trailing_stop:
                    self.trailing_stop = new_stop
                    return self.trailing_stop, "TRAIL_UPDATED"
        
        # 3. تفعيل نقطة التعادل بعد تحقيق ربح جيد
        if profit_pct >= 1.5 and not self.tightened:
            self.breakeven_price = self.entry_price * 0.995  # -0.5%
            self.tightened = True
            return self.breakeven_price, "BREAKEVEN_ACTIVATED"
        
        return self.trailing_stop or self.entry_price, "HOLD"
    
    def should_close(self, current_price: float) -> Tuple[bool, str]:
        """التحقق إذا كان يجب إغلاق الصفقة"""
        if self.trailing_stop is None:
            return False, "NO_TRAIL"
        
        if self.side == "BUY":
            if current_price <= self.trailing_stop:
                return True, f"TRAIL_STOP_HIT: {current_price} <= {self.trailing_stop}"
        else:
            if current_price >= self.trailing_stop:
                return True, f"TRAIL_STOP_HIT: {current_price} >= {self.trailing_stop}"
        
        return False, "HOLD"

# ============================================
#  DECISION MATRIX ENGINE - مصفوفة القرار
# ============================================

class DecisionMatrixEngine:
    """محرك مصفوفة القرار النهائي"""
    
    def __init__(self):
        self.last_trade_time = 0
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.cooldown_until = 0
        self.trade_history = []
        
    def evaluate(self, signals: Dict, position_open: bool = False) -> Dict:
        """
        تقييم جميع الإشارات واتخاذ القرار النهائي
        
        signals يجب أن تحتوي على:
        - smart_money_decision: من SmartMoneyEngine
        - trend_classification: من TrendClassifierEngine
        - council_signal: إشارة المجلس الحالية
        - rf_signal: إشارة Range Filter
        - market_conditions: ظروف السوق
        """
        # فحص التبريد
        current_time = time.time()
        if current_time < self.cooldown_until:
            return {
                "action": "COOLDOWN",
                "reason": f"Cooldown for {int(self.cooldown_until - current_time)}s",
                "confidence": 0.0
            }
        
        # إذا كانت هناك صفقة مفتوحة
        if position_open:
            return {
                "action": "MANAGE",
                "reason": "Position already open",
                "confidence": 0.0
            }
        
        # استخراج الإشارات
        sm_decision = signals.get("smart_money_decision")
        trend_info = signals.get("trend_classification", {})
        council_signal = signals.get("council_signal", {})
        rf_signal = signals.get("rf_signal", {})
        market_cond = signals.get("market_conditions", {})
        
        # فلترات أمان
        filters_passed, filter_reason = self._apply_filters(trend_info, market_cond)
        if not filters_passed:
            return {
                "action": "REJECT",
                "reason": filter_reason,
                "confidence": 0.0
            }
        
        # تجميع الأصوات
        votes = self._collect_votes(sm_decision, council_signal, rf_signal, trend_info)
        
        # اتخاذ القرار النهائي
        decision = self._make_final_decision(votes, trend_info)
        
        # تسجيل القرار
        if decision["action"] in ["BUY", "SELL"]:
            self.last_trade_time = current_time
            
        return decision
    
    def _apply_filters(self, trend_info: Dict, market_cond: Dict) -> Tuple[bool, str]:
        """تطبيق الفلترات الأمنية"""
        # 1. فلتر حالة السوق
        if trend_info.get("type") == "CHOP":
            return False, "CHOP_MARKET"
        
        # 2. فلتر قوة الترند
        if trend_info.get("strength", 0) < 1 and trend_info.get("type") != "LARGE":
            return False, "WEAK_TREND"
        
        # 3. فلتر التذبذب
        spread_bps = market_cond.get("spread_bps", 0)
        if spread_bps > 10:  # انتشار كبير
            return False, f"HIGH_SPREAD: {spread_bps}bps"
        
        # 4. فلتر التوقيت
        current_hour = datetime.utcnow().hour
        if current_hour in [0, 1, 2, 3]:  *سوق هادئ*
            return False, "LOW_LIQUIDITY_HOURS"
        
        return True, "ALL_FILTERS_PASSED"
    
    def _collect_votes(self, sm_decision, council_signal, rf_signal, trend_info) -> Dict:
        """جمع الأصوات من جميع المصادر"""
        votes = {
            "BUY": 0,
            "SELL": 0,
            "CONFIDENCE_BUY": 0.0,
            "CONFIDENCE_SELL": 0.0,
            "REASONS": []
        }
        
        # 1. أصوات Smart Money Engine
        if sm_decision and sm_decision.allow_entry:
            if sm_decision.side == "BUY":
                votes["BUY"] += 3
                votes["CONFIDENCE_BUY"] += sm_decision.confidence
                votes["REASONS"].append(f"SM: {sm_decision.reason}")
            elif sm_decision.side == "SELL":
                votes["SELL"] += 3
                votes["CONFIDENCE_SELL"] += sm_decision.confidence
                votes["REASONS"].append(f"SM: {sm_decision.reason}")
        
        # 2. أصوات المجلس
        if council_signal:
            score_b = council_signal.get("score_b", 0)
            score_s = council_signal.get("score_s", 0)
            
            if score_b > score_s * 1.2:  *تفوق واضح*
                votes["BUY"] += 2
                votes["CONFIDENCE_BUY"] += min(0.8, score_b / 50)
                votes["REASONS"].append(f"COUNCIL_BUY: {score_b:.1f}")
            elif score_s > score_b * 1.2:
                votes["SELL"] += 2
                votes["CONFIDENCE_SELL"] += min(0.8, score_s / 50)
                votes["REASONS"].append(f"COUNCIL_SELL: {score_s:.1f}")
        
        # 3. أصوات Range Filter
        if rf_signal:
            if rf_signal.get("long", False):
                votes["BUY"] += 1
                votes["REASONS"].append("RF_BUY")
            elif rf_signal.get("short", False):
                votes["SELL"] += 1
                votes["REASONS"].append("RF_SELL")
        
        # 4. وزن حسب نوع الترند
        trend_type = trend_info.get("type", "MID")
        if trend_type == "LARGE":
            # زيادة وزن Smart Money في الترند الكبير
            votes["BUY"] = int(votes["BUY"] * 1.2)
            votes["SELL"] = int(votes["SELL"] * 1.2)
            votes["REASONS"].append("LARGE_TREND_BOOST")
        
        return votes
    
    def _make_final_decision(self, votes: Dict, trend_info: Dict) -> Dict:
        """اتخاذ القرار النهائي بناءً على الأصوات"""
        buy_votes = votes["BUY"]
        sell_votes = votes["SELL"]
        confidence_buy = votes["CONFIDENCE_BUY"]
        confidence_sell = votes["CONFIDENCE_SELL"]
        
        # تحديد الحد الأدنى للأصوات حسب نوع الترند
        trend_type = trend_info.get("type", "MID")
        min_votes = 4 if trend_type == "LARGE" else 3
        
        # قرار الشراء
        if buy_votes >= min_votes and buy_votes > sell_votes:
            confidence = min(0.95, confidence_buy * (1 + buy_votes * 0.05))
            return {
                "action": "BUY",
                "reason": " | ".join(votes["REASONS"]),
                "confidence": round(confidence, 2),
                "votes": {"BUY": buy_votes, "SELL": sell_votes},
                "trade_type": "LARGE_TREND" if trend_type == "LARGE" else "MID_TREND"
            }
        
        # قرار البيع
        elif sell_votes >= min_votes and sell_votes > buy_votes:
            confidence = min(0.95, confidence_sell * (1 + sell_votes * 0.05))
            return {
                "action": "SELL",
                "reason": " | ".join(votes["REASONS"]),
                "confidence": round(confidence, 2),
                "votes": {"BUY": buy_votes, "SELL": sell_votes},
                "trade_type": "LARGE_TREND" if trend_type == "LARGE" else "MID_TREND"
            }
        
        # لا توجد إشارة قوية
        return {
            "action": "WAIT",
            "reason": f"INSUFFICIENT_VOTES (BUY:{buy_votes}, SELL:{sell_votes}, MIN:{min_votes})",
            "confidence": 0.0,
            "votes": {"BUY": buy_votes, "SELL": sell_votes}
        }
    
    def record_trade_result(self, is_win: bool):
        """تسجيل نتيجة الصفقة"""
        if is_win:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
            
            # تفعيل التبريد بعد 3 أرباح متتالية
            if self.consecutive_wins >= 3:
                self.cooldown_until = time.time() + 300  # 5 دقائق
                self.consecutive_wins = 0
                logging.info("🎯 3 consecutive wins → 5min cooldown activated")
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            
            # تفعيل التبريد بعد خسارتين متتاليتين
            if self.consecutive_losses >= 2:
                self.cooldown_until = time.time() + 600  # 10 دقائق
                self.consecutive_losses = 0
                logging.info("⚠️ 2 consecutive losses → 10min cooldown activated")

# ============================================
#  INTEGRATION WITH EXISTING CODE
# ============================================

# تهيئة المحركات الجديدة
smart_money_engine = None
trend_classifier = TrendClassifierEngine()
decision_matrix = DecisionMatrixEngine()
trailing_engine = None

def integrate_smart_money_analysis(df: pd.DataFrame, ind: Dict, council_data: Dict, 
                                  rf_signal: Dict) -> Dict:
    """دمج تحليل Smart Money مع النظام الحالي"""
    global smart_money_engine
    
    try:
        # تحويل DataFrame إلى قائمة
        candles = []
        for i in range(len(df)):
            candles.append({
                'open': float(df['open'].iloc[i]),
                'high': float(df['high'].iloc[i]),
                'low': float(df['low'].iloc[i]),
                'close': float(df['close'].iloc[i]),
                'volume': float(df['volume'].iloc[i])
            })
        
        # بيانات إضافية
        volumes = [c['volume'] for c in candles]
        atr_value = safe_get(ind, 'atr', 0.001)
        adx_value = safe_get(ind, 'adx', 0)
        rsi_value = safe_get(ind, 'rsi', 50)
        di_plus = safe_get(ind, 'plus_di', 0)
        di_minus = safe_get(ind, 'minus_di', 0)
        
        # 1. تصنيف الترند
        trend_info = trend_classifier.classify_trend(
            adx_value, di_plus, di_minus, candles, volumes
        )
        
        # 2. تحليل Smart Money
        smart_money_engine = SmartMoneyEngine(candles, volumes, atr_value)
        sm_decision = smart_money_engine.make_decision(adx_value, rsi_value)
        
        # 3. جمع إشارة المجلس
        council_signal = {
            'score_b': council_data.get('score_b', 0),
            'score_s': council_data.get('score_s', 0),
            'b': council_data.get('b', 0),
            's': council_data.get('s', 0)
        }
        
        # 4. تحليل ظروف السوق
        market_conditions = {
            'spread_bps': STATE.get('last_spread_bps', 0),
            'volatility': atr_value / (df['close'].iloc[-1] if len(df) > 0 else 1) * 100
        }
        
        # 5. اتخاذ القرار النهائي
        signals = {
            "smart_money_decision": sm_decision,
            "trend_classification": trend_info,
            "council_signal": council_signal,
            "rf_signal": rf_signal,
            "market_conditions": market_conditions
        }
        
        final_decision = decision_matrix.evaluate(signals, STATE.get("open", False))
        
        # إضافة معلومات Smart Money إلى اللوج
        if sm_decision.allow_entry:
            log_i(f"🧠 SMART MONEY → {sm_decision.side} | Confidence: {sm_decision.confidence:.2f} | Type: {sm_decision.trade_type}")
            log_i(f"   Reason: {sm_decision.reason}")
        
        log_i(f"📊 TREND CLASSIFIER → {trend_info.get('type', 'UNKNOWN')} | Strength: {trend_info.get('strength', 0):.1f}")
        
        return {
            "final_decision": final_decision,
            "smart_money": sm_decision,
            "trend_info": trend_info,
            "signals": signals
        }
        
    except Exception as e:
        log_w(f"Smart Money integration error: {e}")
        return {
            "final_decision": {"action": "ERROR", "reason": str(e), "confidence": 0.0},
            "smart_money": None,
            "trend_info": {"type": "ERROR", "strength": 0},
            "signals": {}
        }

def execute_smart_money_trade(decision: Dict, price: float, balance: float) -> bool:
    """تنفيذ صفقة بناءً على قرار Smart Money"""
    if not decision or decision.get("action") not in ["BUY", "SELL"]:
        return False
    
    action = decision["action"]
    confidence = decision.get("confidence", 0.0)
    trade_type = decision.get("trade_type", "MID_TREND")
    reason = decision.get("reason", "")
    
    # حساب الحجم بناءً على الثقة ونوع الصفقة
    base_qty = compute_size(balance, price)
    
    # تعديل الحجم بناءً على الثقة
    if confidence > 0.8:
        qty = base_qty * 1.2
    elif confidence > 0.6:
        qty = base_qty
    else:
        qty = base_qty * 0.7
    
    # تعديل إضافي بناءً على نوع الترند
    if trade_type == "LARGE_TREND":
        qty *= 1.3  *زيادة الحجم في الترند الكبير*
    elif trade_type == "SCALP":
        qty *= 0.7  *تقليل الحجم في السكالب*
    
    qty = safe_qty(qty)
    
    # تنفيذ الصفقة
    log_g(f"🚀 SMART MONEY EXECUTION → {action} | Confidence: {confidence:.2f} | Type: {trade_type}")
    log_g(f"   Qty: {qty:.4f} | Price: {price:.6f}")
    log_g(f"   Reason: {reason}")
    
    success = open_market_enhanced(action.lower(), qty, price)
    
    if success:
        # تسجيل نوع الصفقة في STATE
        STATE["trade_type"] = trade_type
        STATE["entry_confidence"] = confidence
        STATE["entry_reason"] = reason
        
        # بدء نظام الوقف المتحرك الذكي
        global trailing_engine
        trailing_engine = IntelligentTrailingEngine(action, price)
        
        return True
    
    return False

def manage_smart_money_position(df: pd.DataFrame, ind: Dict, current_price: float):
    """إدارة الصفقة باستخدام Smart Money Engine"""
    global trailing_engine
    
    if not STATE.get("open") or STATE["qty"] <= 0 or trailing_engine is None:
        return
    
    # تحديث الوقف المتحرك
    try:
        candles = []
        for i in range(len(df)):
            candles.append({
                'open': float(df['open'].iloc[i]),
                'high': float(df['high'].iloc[i]),
                'low': float(df['low'].iloc[i]),
                'close': float(df['close'].iloc[i]),
                'volume': float(df['volume'].iloc[i])
            })
        
        atr_value = safe_get(ind, 'atr', 0.001)
        trade_type = STATE.get("trade_type", "MID_TREND")
        
        new_stop, action = trailing_engine.update(
            current_price, 
            candles, 
            atr_value, 
            "LARGE" if trade_type == "LARGE_TREND" else "MID"
        )
        
        # التحقق إذا كان يجب الإغلاق
        should_close, close_reason = trailing_engine.should_close(current_price)
        
        if should_close:
            log_w(f"🛑 SMART TRAILING STOP: {close_reason}")
            close_market_strict(f"Smart Trailing Stop: {close_reason}")
            
            # تسجيل نتيجة الصفقة
            profit_pct = ((current_price - STATE["entry"]) / STATE["entry"]) * 100
            if STATE["side"] == "short":
                profit_pct = -profit_pct
            
            is_win = profit_pct > 0
            decision_matrix.record_trade_result(is_win)
            
        elif action != "HOLD":
            log_i(f"📌 TRAILING UPDATE: {action} | New Stop: {new_stop:.6f}")
            
    except Exception as e:
        log_w(f"Smart position management error: {e}")

# ============================================
#  MODIFIED TRADE LOOP WITH SMART MONEY INTEGRATION
# ============================================

def trade_loop_with_smart_money():
    """الدورة الرئيسية مع تكامل Smart Money"""
    global wait_for_next_signal_side, compound_pnl
    
    log_i("🚀 STARTING SMART MONEY HYBRID ENGINE")
    
    while True:
        try:
            # جمع البيانات الأساسية
            bal = balance_usdt()
            px = price_now()
            df = fetch_ohlcv()
            
            if df.empty:
                time.sleep(BASE_SLEEP)
                continue
            
            # المؤشرات التقليدية
            info = rf_signal_live(df)
            ind = compute_indicators(df)
            spread_bps = orderbook_spread_bps()
            
            # إشارة المجلس
            council_data = council_votes_pro_enhanced(df)
            
            # ============================================
            #  SMART MONEY ANALYSIS BLOCK
            # ============================================
            smart_analysis = integrate_smart_money_analysis(df, ind, council_data, info)
            
            final_decision = smart_analysis.get("final_decision", {})
            trend_info = smart_analysis.get("trend_info", {})
            
            # ============================================
            #  EXECUTION LOGIC
            # ============================================
            
            # إذا كانت هناك صفقة مفتوحة
            if STATE.get("open"):
                # إدارة الصفقة الحالية مع Smart Money
                manage_smart_money_position(df, ind, px or info.get("price", 0))
                
                # تطبيق نظام جني الأرباح الذكي
                apply_smart_profit_strategy()
                
            # إذا لم تكن هناك صفقة مفتوحة
            else:
                # اتخاذ قرار الدخول
                if final_decision.get("action") in ["BUY", "SELL"]:
                    
                    # التحقق من بوابة الانتظار
                    allow_wait, wait_reason = wait_gate_allow(df, info)
                    
                    if not allow_wait and wait_for_next_signal_side:
                        log_i(f"⏳ Waiting for opposite RF: {wait_for_next_signal_side}")
                    else:
                        # تنفيذ الصفقة الذكية
                        success = execute_smart_money_trade(
                            final_decision, 
                            px or info.get("price", 0), 
                            bal or 100.0
                        )
                        
                        if success:
                            wait_for_next_signal_side = None
            
            # ============================================
            #  LOGGING AND MONITORING
            # ============================================
            
            # تحديث اللوج مع معلومات Smart Money
            if LOG_ADDONS:
                # معلومات Smart Money
                sm_decision = smart_analysis.get("smart_money")
                if sm_decision:
                    sm_status = f"SM: {sm_decision.side if sm_decision.allow_entry else 'NONE'}"
                    sm_conf = f"({sm_decision.confidence:.2f})"
                else:
                    sm_status = "SM: N/A"
                    sm_conf = ""
                
                # معلومات الترند
                trend_type = trend_info.get("type", "N/A")
                trend_str = trend_info.get("strength", 0)
                
                print(f"🧠 SMART ENGINE | Decision: {final_decision.get('action', 'N/A')} "
                      f"| {sm_status}{sm_conf} | Trend: {trend_type}({trend_str:.1f})", flush=True)
            
            # اللوج التقليدي
            if LOG_LEGACY:
                pretty_snapshot(bal, {"price": px or info.get("price", 0), **info}, 
                              ind, spread_bps, "", df)
            
            # النوم حتى التكرار التالي
            sleep_s = NEAR_CLOSE_S if time_to_candle_close(df) <= 10 else BASE_SLEEP
            time.sleep(sleep_s)
            
        except Exception as e:
            log_e(f"Smart money loop error: {e}\n{traceback.format_exc()}")
            time.sleep(BASE_SLEEP)

# ============================================
#  UPDATE MAIN EXECUTION
# ============================================

# استبدال الدورة الرئيسية بالإصدار الذكي
trade_loop = trade_loop_with_smart_money

# ============================================
#  REST OF THE ORIGINAL CODE REMAINS THE SAME
# ============================================

# [يتبع باقي الكود الأصلي بدون تغيير...]
# جميع الدوال والمتغيرات الأصلية تبقى كما هي
# فقط تمت إضافة الأنظمة الذكية الجديدة

# =================== INITIALIZATION ===================
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
BOT_VERSION = f"SUI ULTRA PRO AI v8.0 — {EXCHANGE_NAME.upper()} - SMART MONEY HYBRID ENGINE"
print("🚀 Booting:", BOT_VERSION, flush=True)

STATE_PATH = "./bot_state.json"
RESUME_ON_RESTART = True
RESUME_LOOKBACK_SECS = 60 * 60

# ... [rest of the original initialization code] ...

# =================== ENHANCED LOGGING ===================
def log_i(msg): 
    print(f"ℹ️ {msg}", flush=True)
    logging.info(msg)

def log_g(msg): 
    print(f"✅ {msg}", flush=True)
    logging.info(msg)

def log_w(msg): 
    print(f"🟨 {msg}", flush=True)
    logging.warning(msg)

def log_e(msg): 
    print(f"❌ {msg}", flush=True)
    logging.error(msg)

def log_smart_money(msg):
    """لوج خاص بـ Smart Money Engine"""
    print(f"🧠 {msg}", flush=True)
    logging.info(f"SMART_MONEY: {msg}")

# ... [rest of the original code remains exactly the same] ...

# =================== MAIN EXECUTION ===================
if __name__ == "__main__":
    log_i("🚀 SUI ULTRA PRO AI BOT STARTED WITH SMART MONEY ENGINE")
    log_i("🎯 FEATURES: Smart Money Concepts + Trend Classification + Intelligent Trailing")
    log_i("💡 STRATEGY: Liquidity Sweeps + Structure Analysis + Explosion Detection")
    
    # بدء الأنظمة
    import threading
    threading.Thread(target=keepalive_loop, daemon=True).start()
    threading.Thread(target=trade_loop, daemon=True).start()
    
    # بدء الخادم
    app.run(host="0.0.0.0", port=PORT, debug=False)
