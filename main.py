# -*- coding: utf-8 -*-
"""
SUI ULTRA PRO AI BOT - الإصدار الذكي المتكامل مع نظام إدارة الصفقات المتقدم
• نظام TradePlan الذكي (خطة صفقة قبل الدخول)
• نظام إدارة المراحل الذكي مع تصنيف MID/LARGE
• ANSI Logger احترافي بألوان وملفات
• نظام Fail-Fast للخروج السريع
• ذكاء السيولة والهيكل
• محرك الانفجار وإعادة الدخول الذكية
• نظام Confidence Scoring
• واجهة Flask API للتتبع والمراقبة
"""

import os, time, math, random, signal, sys, traceback, logging, json
from logging.handlers import RotatingFileHandler
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import ccxt
from flask import Flask, jsonify, render_template_string
from decimal import Decimal, ROUND_DOWN, InvalidOperation
from collections import deque, defaultdict
import statistics
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Any
from enum import Enum

# ============================================
#  ANSI LOGGER ENGINE - نظام تسجيل موحد بألوان
# ============================================

class C:
    """ألوان ANSI للكونسول"""
    RESET = "\033[0m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    GRAY = "\033[90m"
    LIGHT_RED = "\033[91m"
    LIGHT_GREEN = "\033[92m"
    LIGHT_YELLOW = "\033[93m"
    LIGHT_BLUE = "\033[94m"
    LIGHT_CYAN = "\033[96m"
    LIGHT_WHITE = "\033[97m"

LEVEL_COLOR = {
    "DEBUG": C.GRAY,
    "INFO": C.GREEN,
    "WARN": C.YELLOW,
    "ERROR": C.RED
}

def setup_logger(name="SUI_BOT", log_dir="logs", file_name="sui_bot.log", max_mb=10, backup_count=5):
    """إعداد نظام تسجيل الملفات مع Rotation"""
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # معالج الكونسول
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # معالج الملفات مع Rotation
    fh = RotatingFileHandler(
        os.path.join(log_dir, file_name),
        maxBytes=max_mb * 1024 * 1024,
        backupCount=backup_count,
        encoding="utf-8"
    )
    fh.setLevel(logging.DEBUG)

    # تنسيق اللوج
    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(section)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    ch.setFormatter(fmt)
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

# إنشاء اللوجر الأساسي
ansi_logger = setup_logger()

def slog(section: str, message: str, level: str = "INFO", confidence: Optional[int] = None):
    """
    تسجيل رسالة مع ألوان ANSI وتصنيف
    
    Args:
        section: القسم (ENTRY, EXIT, LIQUIDITY, etc.)
        message: الرسالة
        level: المستوى (INFO, WARN, ERROR, DEBUG)
        confidence: درجة الثقة (0-10)
    """
    level = level.upper()
    color = LEVEL_COLOR.get(level, C.RESET)
    
    # إضافة درجة الثقة إذا كانت موجودة
    conf_txt = f" | Confidence: {confidence}/10" if confidence is not None else ""
    msg = f"{message}{conf_txt}"
    
    # بيانات إضافية للتنسيق
    extra = {"section": section}
    
    # إضافة الألوان للكونسول
    colored_msg = f"{color}{msg}{C.RESET}"
    
    # التسجيل حسب المستوى
    if level == "DEBUG":
        ansi_logger.debug(colored_msg, extra=extra)
    elif level == "INFO":
        ansi_logger.info(colored_msg, extra=extra)
    elif level == "WARN":
        ansi_logger.warning(colored_msg, extra=extra)
    elif level == "ERROR":
        ansi_logger.error(colored_msg, extra=extra)

# ============================================
#  CONFIDENCE ENGINE - محرك حساب الثقة
# ============================================

class ConfidenceEngine:
    """محرك حساب درجة الثقة في الصفقات"""
    
    def score(self, market: Dict, plan: Dict) -> int:
        """
        حساب درجة الثقة من 0 إلى 10
        
        Args:
            market: بيانات السوق
            plan: خطة الصفقة
            
        Returns:
            درجة الثقة (0-10)
        """
        score = 0
        
        # 1. حدث السيولة (0-3 نقطة)
        if market.get("liquidity_sweep"):
            score += 3
        elif market.get("liquidity_tap"):
            score += 1
        
        # 2. محاذاة الهيكل (0-3 نقطة)
        structure = market.get("structure", {})
        if structure.get("type") in ["BOS_UP", "BOS_DOWN"]:
            score += 3
        elif structure.get("type") == "CHoCH":
            score += 1
        
        # 3. تأكيد الحجم (0-2 نقطة)
        if market.get("volume_spike"):
            score += 2
        
        # 4. الزخم (0-2 نقطة)
        momentum = market.get("momentum", {})
        if momentum.get("direction") in ["BULLISH", "BEARISH"] and momentum.get("score", 0) > 0.5:
            score += 2
        
        # 5. نوع الترند (0-2 نقطة)
        trend = market.get("trend", {})
        if trend.get("strength", 0) > 2.0:
            score += 2
        elif trend.get("strength", 0) > 1.0:
            score += 1
        
        # الحد الأقصى 10 نقاط
        return min(score, 10)
    
    def get_confidence_level(self, score: int) -> str:
        """الحصول على مستوى الثقة نصياً"""
        if score >= 9:
            return "VERY_HIGH"
        elif score >= 7:
            return "HIGH"
        elif score >= 5:
            return "MEDIUM"
        elif score >= 3:
            return "LOW"
        else:
            return "VERY_LOW"

# ============================================
#  MARKET ANALYZER - محلل السوق الكامل
# ============================================

class MarketAnalyzer:
    """محلل السوق المتقدم مع كل الدوال المطلوبة"""
    
    def __init__(self):
        self.market_states = deque(maxlen=100)
        
    def analyze_market(self, df: pd.DataFrame, timeframe: str = "15m") -> Dict[str, Any]:
        """
        تحليل شامل للسوق
        """
        if df.empty or len(df) < 20:
            slog("ERROR", "Insufficient data for market analysis", level="ERROR")
            return {"error": "Insufficient data"}
        
        try:
            # تحليل الاتجاه
            trend = self._analyze_trend(df)
            
            # تحليل الهيكل
            structure = self._analyze_structure(df)
            
            # تحليل السيولة
            liquidity = self._analyze_liquidity(df)
            
            # تحليل الزخم
            momentum = self._analyze_momentum(df)
            
            # تحليل الحجم
            volume_profile = self._analyze_volume(df)
            
            # سبب التحليل
            reason = self._generate_analysis_reason(trend, structure, liquidity)
            
            # حفظ حالة السوق
            market_state = {
                'timestamp': datetime.now().isoformat(),
                'trend': trend,
                'structure': structure,
                'liquidity': liquidity,
                'momentum': momentum,
                'timeframe': timeframe
            }
            self.market_states.append(market_state)
            
            # لوج حالة السوق
            slog("MARKET",
                f"TF={timeframe} | Trend={trend['direction']} | Structure={structure['type']} | Liquidity={liquidity['level']}",
                level="INFO")
            
            return {
                'trend': trend,
                'structure': structure,
                'liquidity': liquidity,
                'momentum': momentum,
                'volume': volume_profile,
                'reason': reason,
                'timestamp': datetime.now().isoformat(),
                'timeframe': timeframe
            }
            
        except Exception as e:
            slog("ERROR", f"Market analysis error: {str(e)}", level="ERROR")
            return {"error": str(e)}
    
    def _analyze_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """تحليل الاتجاه"""
        if len(df) < 21:
            return {"direction": "NEUTRAL", "strength": 0, "confirmed": False}
        
        # حساب المتوسطات المتحركة
        closes = df['close'].astype(float).values
        sma_short = self._calculate_sma(closes, 9)
        sma_long = self._calculate_sma(closes, 21)
        
        # تحديد الاتجاه
        if sma_short > sma_long:
            direction = "BULL"
            strength = ((sma_short - sma_long) / sma_long) * 100
        elif sma_short < sma_long:
            direction = "BEAR"
            strength = ((sma_long - sma_short) / sma_short) * 100
        else:
            direction = "SIDEWAYS"
            strength = 0
        
        # تحليل تأكيد الاتجاه
        confirmed = abs(strength) > 1.0
        
        return {
            'direction': direction,
            'strength': abs(strength),
            'sma_short': sma_short,
            'sma_long': sma_long,
            'confirmed': confirmed
        }
    
    def _analyze_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """تحليل الهيكل السعري"""
        if len(df) < 10:
            return {"type": "NO_CLEAR_STRUCTURE", "key_level": None}
        
        highs = df['high'].astype(float).values
        lows = df['low'].astype(float).values
        
        # البحث عن Higher Highs و Lower Lows
        recent_highs = highs[-5:]
        recent_lows = lows[-5:]
        
        # تحليل الهيكل البسيط
        if len(recent_highs) >= 2 and len(recent_lows) >= 2:
            # Higher Highs و Higher Lows
            if (recent_highs[-1] > recent_highs[-2] and 
                recent_lows[-1] > recent_lows[-2]):
                return {"type": "BOS_UP", "key_level": recent_lows[-1]}
            
            # Lower Highs و Lower Lows
            elif (recent_highs[-1] < recent_highs[-2] and 
                  recent_lows[-1] < recent_lows[-2]):
                return {"type": "BOS_DOWN", "key_level": recent_highs[-1]}
            
            # الهيكل الجانبي
            else:
                return {"type": "CONSOLIDATION", "key_level": (max(highs[-10:]) + min(lows[-10:])) / 2}
        
        return {"type": "NO_CLEAR_STRUCTURE", "key_level": None}
    
    def _analyze_liquidity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """تحليل السيولة"""
        if len(df) < 10:
            return {"level": "UNKNOWN", "volume_ratio": 1.0}
        
        # تحليل الحجم
        volumes = df['volume'].astype(float).values
        current_volume = volumes[-1]
        avg_volume = np.mean(volumes[-10:])
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # تحديد مستوى السيولة
        if volume_ratio > 1.5:
            level = "HIGH"
        elif volume_ratio > 1.2:
            level = "MEDIUM_HIGH"
        elif volume_ratio > 0.8:
            level = "MEDIUM"
        elif volume_ratio > 0.5:
            level = "LOW"
        else:
            level = "VERY_LOW"
        
        return {
            'level': level,
            'volume_ratio': volume_ratio,
            'current_volume': current_volume,
            'avg_volume': avg_volume
        }
    
    def _analyze_momentum(self, df: pd.DataFrame) -> Dict[str, Any]:
        """تحليل الزخم"""
        if len(df) < 14:
            return {"score": 0, "direction": "NEUTRAL"}
        
        # حساب معدل التغير
        closes = df['close'].astype(float).values
        roc = ((closes[-1] - closes[-5]) / closes[-5]) * 100 if closes[-5] > 0 else 0
        
        # تحديد اتجاه الزخم
        if roc > 2.0:
            direction = "STRONG_BULLISH"
            score = min(abs(roc) / 10, 1.0)
        elif roc > 0.5:
            direction = "BULLISH"
            score = min(abs(roc) / 10, 1.0)
        elif roc < -2.0:
            direction = "STRONG_BEARISH"
            score = min(abs(roc) / 10, 1.0)
        elif roc < -0.5:
            direction = "BEARISH"
            score = min(abs(roc) / 10, 1.0)
        else:
            direction = "NEUTRAL"
            score = 0
        
        return {
            'score': score,
            'direction': direction,
            'roc': roc
        }
    
    def _analyze_volume(self, df: pd.DataFrame) -> Dict[str, Any]:
        """تحليل الحجم"""
        if len(df) < 10:
            return {"profile": "UNKNOWN", "trend": "UNKNOWN"}
        
        volumes = df['volume'].astype(float).values
        
        # اتجاه الحجم
        recent_volumes = volumes[-5:]
        volume_trend = "INCREASING" if recent_volumes[-1] > recent_volumes[0] else "DECREASING" if recent_volumes[-1] < recent_volumes[0] else "STABLE"
        
        # تحديد ملف الحجم
        avg_volume = np.mean(volumes[-10:])
        current_volume = volumes[-1]
        
        if current_volume > avg_volume * 1.5:
            profile = "HIGH_ACCUMULATION" if volume_trend == "INCREASING" else "HIGH_DISTRIBUTION"
        elif current_volume > avg_volume * 1.2:
            profile = "MODERATE_ACCUMULATION" if volume_trend == "INCREASING" else "MODERATE_DISTRIBUTION"
        elif current_volume > avg_volume * 0.8:
            profile = "NORMAL"
        else:
            profile = "LOW_PARTICIPATION"
        
        return {
            'profile': profile,
            'trend': volume_trend,
            'current': current_volume,
            'average': avg_volume,
            'ratio': current_volume / avg_volume if avg_volume > 0 else 1
        }
    
    def _generate_analysis_reason(self, trend: Dict, structure: Dict, liquidity: Dict) -> str:
        """توليد سبب التحليل"""
        reasons = []
        
        if trend['confirmed']:
            reasons.append(f"Trend: {trend['direction']} (Strength: {trend['strength']:.1f})")
        
        if structure['type'] != "NO_CLEAR_STRUCTURE":
            reasons.append(f"Structure: {structure['type']}")
        
        if liquidity['level'] in ["HIGH", "MEDIUM_HIGH"]:
            reasons.append(f"Liquidity: {liquidity['level']}")
        
        if reasons:
            return " | ".join(reasons)
        return "No clear market signals"
    
    def _calculate_sma(self, prices: np.ndarray, period: int) -> float:
        """حساب المتوسط المتحرك البسيط"""
        if len(prices) < period:
            return float(prices[-1]) if len(prices) > 0 else 0
        return float(np.mean(prices[-period:]))
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """حساب مؤشر RSI"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        
        if down == 0:
            return 100.0
        
        rs = up / down
        rsi = 100.0 - (100.0 / (1.0 + rs))
        
        return rsi

# ============================================
#  EXPLOSION & RE-ENTRY ENGINE - كشف الانفجار وإعادة الدخول
# ============================================

class ExplosionReEntryEngine:
    """محرك كشف الانفجار الحقيقي وإعادة الدخول الذكية"""
    
    def __init__(self):
        self.last_closed_trade = None
        self.last_reentry_time = 0
        self.reentry_cooldown = 300  # 5 دقائق بين إعادة الدخول
        self.explosion_detected = False
        
    def detect_explosion(self, df: pd.DataFrame) -> Tuple[bool, Dict]:
        """
        كشف انفجار حقيقي (للباي أو السيل)
        
        Returns:
            (تم الكشف, تفاصيل الانفجار)
        """
        if len(df) < 20:
            return False, {}
        
        try:
            # حساب ATR الحالي والمتوسط
            atr_now = self._calculate_atr(df, 14)
            atr_ma = df['close'].rolling(14).apply(lambda x: self._calculate_atr(df.loc[x.index], 14)).mean()
            
            # تحليل الحجم
            volume_now = df['volume'].iloc[-1]
            volume_ma = df['volume'].rolling(14).mean().iloc[-1]
            
            # تحليل الشمعة الأخيرة
            latest = df.iloc[-1]
            high = float(latest['high'])
            low = float(latest['low'])
            open_price = float(latest['open'])
            close = float(latest['close'])
            
            body_size = abs(close - open_price)
            candle_range = high - low
            
            if candle_range == 0:
                return False, {}
            
            body_ratio = body_size / candle_range
            
            # حساب نسب الـ Wicks
            upper_wick = (high - max(open_price, close)) / candle_range
            lower_wick = (min(open_price, close) - low) / candle_range
            wick_ratio = max(upper_wick, lower_wick)
            
            # التحقق من الخروج عن النطاق
            close_outside_range = (close > df['high'].iloc[-5:-1].max() or 
                                  close < df['low'].iloc[-5:-1].min())
            
            # شروط الانفجار الحقيقي
            atr_burst = atr_now > atr_ma * 1.8
            volume_burst = volume_now > volume_ma * 1.5
            clean_candle = body_ratio > 0.65
            no_rejection = wick_ratio < 0.25
            
            if atr_burst and volume_burst and clean_candle and no_rejection and close_outside_range:
                explosion_details = {
                    'atr_ratio': atr_now / atr_ma,
                    'volume_ratio': volume_now / volume_ma,
                    'body_ratio': body_ratio,
                    'wick_ratio': wick_ratio,
                    'direction': 'BULL' if close > open_price else 'BEAR',
                    'price': close
                }
                
                slog("EXPLOSION", 
                    f"Detected! ATR: {atr_now/atr_ma:.2f}x | Volume: {volume_now/volume_ma:.2f}x | Direction: {'BULL' if close > open_price else 'BEAR'}",
                    level="INFO")
                
                self.explosion_detected = True
                return True, explosion_details
            
            return False, {}
            
        except Exception as e:
            slog("ERROR", f"Explosion detection failed: {str(e)}", level="ERROR")
            return False, {}
    
    def detect_breakdown(self, df: pd.DataFrame, trade_plan) -> Tuple[bool, str]:
        """
        كشف انهيار عنيف ضد الصفقة الحالية
        
        Returns:
            (تم الكشف, سبب الانهيار)
        """
        if len(df) < 10 or not trade_plan:
            return False, "Insufficient data"
        
        try:
            # حساب ATR
            atr_now = self._calculate_atr(df, 14)
            atr_ma = df['close'].rolling(14).apply(lambda x: self._calculate_atr(df.loc[x.index], 14)).mean()
            
            # تحديد اتجاه الشمعة الأخيرة
            latest = df.iloc[-1]
            direction = "BUY" if latest['close'] > latest['open'] else "SELL"
            opposite_direction = direction != trade_plan.side
            
            # عد الشمعات ضد الصفقة
            closes_against = 0
            lookback = min(5, len(df))
            
            for i in range(-1, -lookback-1, -1):
                candle = df.iloc[i]
                if trade_plan.side == "BUY" and candle['close'] < candle['open']:
                    closes_against += 1
                elif trade_plan.side == "SELL" and candle['close'] > candle['open']:
                    closes_against += 1
            
            # شروط الانهيار العنيف
            violent_move = atr_now > atr_ma * 2.0
            consecutive_against = closes_against >= 3
            
            if violent_move and opposite_direction and consecutive_against:
                reason = f"Violent move against {trade_plan.side}: ATR {atr_now/atr_ma:.2f}x, {closes_against} consecutive against candles"
                slog("BREAKDOWN", reason, level="ERROR")
                return True, reason
            
            return False, "No breakdown detected"
            
        except Exception as e:
            slog("ERROR", f"Breakdown detection failed: {str(e)}", level="ERROR")
            return False, str(e)
    
    def detect_reentry(self, df: pd.DataFrame, last_trade: Dict) -> Tuple[bool, Dict]:
        """
        كشف فرصة ذكية لإعادة الدخول
        
        Returns:
            (تم الكشف, تفاصيل إعادة الدخول)
        """
        if not last_trade:
            return False, {"reason": "No last trade"}
        
        # فحص فترة التبريد
        current_time = time.time()
        if current_time - self.last_reentry_time < self.reentry_cooldown:
            remaining = int(self.reentry_cooldown - (current_time - self.last_reentry_time))
            return False, {"reason": f"Cooldown active: {remaining}s remaining"}
        
        # التأكد من أن الصفقة السابقة أغلقت بشكل جيد
        exit_reason = last_trade.get('exit_reason', '')
        if exit_reason not in ['TP1', 'TP2', 'TP3', 'STRUCTURE_EXIT', 'MANUAL_CLOSE']:
            return False, {"reason": f"Last trade closed badly: {exit_reason}"}
        
        try:
            # تحليل التراجع (Pullback)
            current_price = df['close'].iloc[-1]
            exit_price = last_trade.get('exit_price', current_price)
            
            if exit_price == 0:
                return False, {"reason": "Invalid exit price"}
            
            pullback_depth = abs((current_price - exit_price) / exit_price) * 100
            
            # تحليل الحجم
            volume_now = df['volume'].iloc[-1]
            volume_ma = df['volume'].rolling(14).mean().iloc[-1]
            volume_cooloff = volume_now < volume_ma * 0.8
            
            # التحقق من استمرار الترند
            sma_short = df['close'].rolling(9).mean().iloc[-1]
            sma_long = df['close'].rolling(21).mean().iloc[-1]
            
            if last_trade['side'] == "BUY":
                trend_intact = sma_short > sma_long
            else:
                trend_intact = sma_short < sma_long
            
            # تحقق من إعادة الاختبار (Retest)
            recent_high = df['high'].iloc[-10:-1].max()
            recent_low = df['low'].iloc[-10:-1].min()
            recent_range = recent_high - recent_low
            
            if recent_range == 0:
                return False, {"reason": "Zero price range"}
            
            retest_zone = abs(current_price - exit_price) < recent_range * 0.15
            
            # تحليل شمعة التأكيد
            latest = df.iloc[-1]
            open_price = float(latest['open'])
            close = float(latest['close'])
            high = float(latest['high'])
            low = float(latest['low'])
            
            candle_range = high - low
            if candle_range == 0:
                return False, {"reason": "Zero candle range"}
            
            # شمعة رفض (Rejection)
            if last_trade['side'] == "BUY":
                rejection = (close > open_price and 
                           (close - open_price) / candle_range > 0.6)
            else:
                rejection = (close < open_price and 
                           (open_price - close) / candle_range > 0.6)
            
            # شروط إعادة الدخول
            pullback_ok = pullback_depth < 1.5  # تراجع أقل من 1.5%
            
            if (pullback_ok and volume_cooloff and trend_intact and 
                retest_zone and rejection):
                
                reentry_details = {
                    'side': last_trade['side'],
                    'current_price': current_price,
                    'exit_price': exit_price,
                    'pullback_depth': pullback_depth,
                    'volume_ratio': volume_now / volume_ma,
                    'confidence': 8  # ثقة عالية في إعادة الدخول
                }
                
                slog("RE-ENTRY", 
                    f"Opportunity detected for {last_trade['side']} | Pullback: {pullback_depth:.2f}% | Retest confirmed",
                    level="INFO",
                    confidence=8)
                
                return True, reentry_details
            
            return False, {"reason": "Re-entry conditions not met"}
            
        except Exception as e:
            slog("ERROR", f"Re-entry detection failed: {str(e)}", level="ERROR")
            return False, {"reason": str(e)}
    
    def execute_reentry(self, smart_manager, df: pd.DataFrame, balance: float) -> Tuple[bool, str]:
        """تنفيذ إعادة الدخول إذا تحققت الشروط"""
        if not self.last_closed_trade:
            return False, "No previous trade recorded"
        
        # كشف فرصة إعادة الدخول
        reentry_detected, details = self.detect_reentry(df, self.last_closed_trade)
        
        if not reentry_detected:
            return False, details.get('reason', 'Re-entry not detected')
        
        try:
            side = details['side']
            current_price = details['current_price']
            
            # تحليل السوق المبسط
            market_analysis = {
                'trend': {'direction': side, 'strength': 2.0},
                'liquidity_sweep': True,
                'volume_spike': False,
                'structure': {'type': 'BOS_UP' if side == 'BUY' else 'BOS_DOWN'}
            }
            
            # بناء الخطة
            trade_plan = smart_manager.build_trade_plan(
                side, current_price, market_analysis, df
            )
            
            if not trade_plan:
                return False, "Failed to build trade plan"
            
            # تحديث أسباب الدخول
            trade_plan.entry_reason["mode"] = "SMART_REENTRY"
            trade_plan.entry_reason["original_exit"] = self.last_closed_trade.get('exit_reason', '')
            trade_plan.entry_reason["reentry_confidence"] = details['confidence']
            
            # حساب الثقة النهائية
            confidence_engine = ConfidenceEngine()
            market_data = {
                'liquidity_sweep': True,
                'structure': {'type': 'RETEST'},
                'volume_spike': False,
                'momentum': {'direction': 'BULLISH' if side == 'BUY' else 'BEARISH', 'score': 0.7},
                'trend': {'strength': 2.0}
            }
            
            confidence = confidence_engine.score(market_data, trade_plan.get_summary())
            
            # فحص عتبة الثقة
            if confidence < 7:
                slog("RE-ENTRY", f"Blocked - Low confidence ({confidence}/10)", level="WARN")
                return False, f"Low confidence: {confidence}/10"
            
            # تنفيذ الصفقة
            success = smart_manager.open_trade_with_plan(
                trade_plan, current_price, balance, "Smart Re-Entry"
            )
            
            if success:
                self.last_reentry_time = time.time()
                slog("RE-ENTRY", 
                    f"Executed successfully | Side: {side} | Price: {current_price:.4f} | Confidence: {confidence}/10",
                    level="INFO",
                    confidence=confidence)
                return True, "Re-entry executed"
            else:
                return False, "Trade opening failed"
                
        except Exception as e:
            slog("ERROR", f"Re-entry execution failed: {str(e)}", level="ERROR")
            return False, str(e)
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """حساب Average True Range"""
        try:
            high = df['high'].astype(float)
            low = df['low'].astype(float)
            close = df['close'].astype(float)
            
            # حساب True Range
            tr1 = high - low
            tr2 = (high - close.shift(1)).abs()
            tr3 = (low - close.shift(1)).abs()
            
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=period).mean().iloc[-1]
            
            return atr if not pd.isna(atr) else 0.0
            
        except Exception as e:
            slog("ERROR", f"ATR calculation failed: {str(e)}", level="ERROR")
            return 0.0
    
    def record_closed_trade(self, trade: Dict):
        """تسجيل الصفقة المغلقة للإشارة المرجعية"""
        self.last_closed_trade = trade
        slog("SYSTEM", f"Recorded closed trade: {trade.get('side')} | Exit: {trade.get('exit_reason')}", level="DEBUG")

# ============================================
#  TRADE PLAN - خطة الصفقة الذكية
# ============================================

class TradePlan:
    """خطة الصفقة - العقل الذي يدير الصفقة من البداية للنهاية"""
    
    def __init__(self, side: str, trend_class: str):
        self.side = side.upper()  # BUY / SELL
        self.trend_class = trend_class.upper()  # MID / LARGE
        
        # أسباب الدخول
        self.entry_reason = {
            "liquidity": None,      # sweep_low / sweep_high
            "structure": None,      # BOS / CHoCH / OB
            "zone": None,           # OB / FVG / DEMAND / SUPPLY
            "confirmation": None,   # rejection / engulf / absorption
            "mode": "NORMAL"        # NORMAL / RE-ENTRY
        }
        
        # مستوى الإبطال (حيث تصبح الصفقة خاطئة)
        self.invalidation = None
        self.invalidation_reason = ""
        
        # أهداف السيولة
        self.tp1 = None  # أول سيولة داخلية
        self.tp2 = None  # سيولة متوسطة
        self.tp3 = None  # سيولة رئيسية (للموجات الكبيرة)
        
        # إدارة المخاطرة
        self.sl = None
        self.risk_pct = 0.0
        self.rr_expected = 0.0
        
        # نظام الإدارة
        self.trailing_mode = "STRUCTURE"  # STRUCTURE / HYBRID
        self.breakeven_rule = "AFTER_TP1"
        self.partial_rules = {}
        
        # حالة الخطة
        self.created_at = time.time()
        self.valid = False
        self.reason = ""  # سبب رفض الخطة إذا لم تكن صالحة
        
        # تتبع الأداء
        self.tp1_hit = False
        self.tp2_hit = False
        self.tp3_hit = False
        
    def is_valid(self) -> bool:
        """التحقق من صلاحية الخطة"""
        # يجب أن يكون لدينا سبب دخول واضح
        if not self.entry_reason["liquidity"] or not self.entry_reason["zone"]:
            self.reason = "No valid liquidity event or zone"
            return False
        
        # يجب أن تكون جميع العناصر الأساسية موجودة
        if not all([self.invalidation, self.sl, self.tp1]):
            self.reason = "Missing required fields (invalidation, sl, tp1)"
            return False
        
        # يجب أن تكون نسبة العائد المتوقعة 1:2 على الأقل
        if self.rr_expected < 1.5:
            self.reason = f"Insufficient risk/reward: 1:{self.rr_expected:.1f}"
            return False
        
        self.valid = True
        return True
    
    def calculate_rr_expected(self, entry_price: float) -> float:
        """حساب نسبة العائد المتوقعة"""
        if self.side == "BUY":
            if self.sl and self.tp1:
                risk = entry_price - self.sl
                reward = self.tp1 - entry_price
                if risk > 0:
                    return reward / risk
        else:  # SELL
            if self.sl and self.tp1:
                risk = self.sl - entry_price
                reward = entry_price - self.tp1
                if risk > 0:
                    return reward / risk
        return 0.0
    
    def get_summary(self) -> Dict[str, Any]:
        """الحصول على ملخص الخطة"""
        return {
            "side": self.side,
            "trend_class": self.trend_class,
            "entry_reason": self.entry_reason,
            "invalidation": self.invalidation,
            "sl": self.sl,
            "tp1": self.tp1,
            "tp2": self.tp2,
            "tp3": self.tp3,
            "rr_expected": self.rr_expected,
            "trailing_mode": self.trailing_mode,
            "valid": self.valid,
            "reason": self.reason,
            "created_at": self.created_at
        }

# ============================================
#  TRADE STATE MACHINE - نظام مراحل الصفقة
# ============================================

class TradeState:
    """حالات الصفقة مع أسباب التحول"""
    ENTRY = "ENTRY"          # مرحلة الدخول
    PROTECT = "PROTECT"      # حماية أولية (لا تريل)
    BREAKEVEN = "BREAKEVEN"  # نقطة التعادل
    TRAIL = "TRAIL"         # تريل بالهيكل
    TRIM = "TRIM"          # تقليل مخاطرة
    EXIT = "EXIT"          # خروج نهائي

class TradePhaseEngine:
    """محرك إدارة مراحل الصفقة مع خطة"""
    
    def __init__(self, entry_price: float, side: str, trade_plan: TradePlan):
        self.entry_price = entry_price
        self.side = side.upper()
        self.trade_plan = trade_plan
        self.current_state = TradeState.ENTRY
        self.state_changed_at = time.time()
        self.structure_levels = []
        self.last_stop_loss = trade_plan.sl
        self.trim_count = 0
        self.max_trims = 2
        self.state_log = []
        
        # أهداف الصفقة
        self.targets_hit = {
            'tp1': False,
            'tp2': False,
            'tp3': False
        }
        
        # إعدادات حسب نوع الصفقة
        if trade_plan.trend_class == "MID":
            self.protection_pct = 0.3  # حماية أسرع للموجات المتوسطة
            self.be_pct = 0.2          # نقطة التعادل أسرع
            self.trail_activation_pct = 0.5  # تفعيل التريل عند 0.5%
            self.trim_pct = 0.3        # تقليل أكبر في الترام
        else:  # LARGE
            self.protection_pct = 0.5
            self.be_pct = 0.3
            self.trail_activation_pct = 0.8
            self.trim_pct = 0.2
    
    def update_state(self, new_state: str, reason: str):
        """تحديث حالة الصفقة مع التسجيل"""
        old_state = self.current_state
        self.current_state = new_state
        self.state_changed_at = time.time()
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'old_state': old_state,
            'new_state': new_state,
            'reason': reason
        }
        self.state_log.append(log_entry)
        
        slog("MANAGEMENT", f"State changed: {old_state} → {new_state} | Reason: {reason}", level="INFO")

# ============================================
#  MARKET INTELLIGENCE - ذكاء السوق
# ============================================

class MarketIntelligence:
    """ذكاء السوق - تحليل السيولة والهيكل"""
    
    def __init__(self):
        pass
        
    def detect_liquidity_zones(self, df: pd.DataFrame) -> Dict[str, Any]:
        """كشف مناطق السيولة"""
        if len(df) < 20:
            return {"highs": [], "lows": [], "equal_highs": [], "equal_lows": []}
        
        highs = df['high'].values
        lows = df['low'].values
        
        # كشف القمم والقيعان المتساوية
        equal_highs = []
        equal_lows = []
        
        for i in range(2, len(highs) - 2):
            # قمم متساوية (فارق أقل من 0.1%)
            if abs(highs[i] - highs[i-1]) / highs[i] < 0.001:
                equal_highs.append(highs[i])
            # قيعان متساوية
            if abs(lows[i] - lows[i-1]) / lows[i] < 0.001:
                equal_lows.append(lows[i])
        
        # مناطق السيولة الرئيسية
        liquidity_highs = sorted(set(equal_highs[-5:])) if equal_highs else []
        liquidity_lows = sorted(set(equal_lows[-5:])) if equal_lows else []
        
        return {
            "highs": liquidity_highs,
            "lows": liquidity_lows,
            "equal_highs": equal_highs[-3:] if equal_highs else [],
            "equal_lows": equal_lows[-3:] if equal_lows else [],
            "major_high": max(highs[-10:]) if len(highs) >= 10 else None,
            "major_low": min(lows[-10:]) if len(lows) >= 10 else None
        }

# ============================================
#  EXECUTION GUARD - حماية التنفيذ
# ============================================

class ExecutionGuard:
    """حارس تنفيذ الأوامر مع Bybit"""
    
    def __init__(self, exchange):
        self.exchange = exchange
        self.last_failed_order = None
        self.failure_count = 0
        self.max_failures = 3
        self.cooldown_until = 0

# ============================================
#  SMART TRADE MANAGER - المدير الرئيسي مع نظام Plan
# ============================================

class SmartTradeManager:
    """المدير الذكي للصفقات مع نظام TradePlan"""
    
    def __init__(self, exchange, symbol: str, risk_percent: float = 0.6):
        self.exchange = exchange
        self.symbol = symbol
        self.risk_percent = risk_percent
        
        # الأنظمة الفرعية
        self.execution_guard = ExecutionGuard(exchange)
        self.trade_phase_engine = None
        self.market_intelligence = MarketIntelligence()
        self.explosion_engine = ExplosionReEntryEngine()
        self.confidence_engine = ConfidenceEngine()
        
        self.active_trade = False
        self.current_position = {
            'side': None,
            'entry_price': 0.0,
            'quantity': 0.0,
            'entry_time': None,
            'plan': None
        }
        
        # إحصائيات
        self.trades_history = []
        self.total_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        
    def build_trade_plan(self, side: str, current_price: float, market_analysis: Dict, 
                        df: pd.DataFrame) -> Optional[TradePlan]:
        """
        بناء خطة الصفقة قبل الدخول
        """
        # إنشاء خطة الصفقة
        trend = market_analysis.get('trend', {})
        structure = market_analysis.get('structure', {})
        
        # تحديد فئة الترند
        trend_strength = trend.get('strength', 0)
        structure_type = structure.get('type', '')
        
        if trend_strength > 2.0 and structure_type.startswith('BOS'):
            trend_class = "LARGE"
        else:
            trend_class = "MID"
        
        plan = TradePlan(side=side, trend_class=trend_class)
        
        # إضافة أسباب الدخول
        plan.entry_reason["liquidity"] = "SWEEP"
        plan.entry_reason["zone"] = "DEMAND" if side == "BUY" else "SUPPLY"
        plan.entry_reason["structure"] = structure_type
        plan.entry_reason["confirmation"] = "REJECTION"
        
        # حساب مستويات أساسية (مبسطة للتوضيح)
        if side == "BUY":
            plan.sl = current_price * 0.99
            plan.tp1 = current_price * 1.02
            plan.tp2 = current_price * 1.04
            plan.invalidation = current_price * 0.985
        else:
            plan.sl = current_price * 1.01
            plan.tp1 = current_price * 0.98
            plan.tp2 = current_price * 0.96
            plan.invalidation = current_price * 1.015
        
        plan.rr_expected = plan.calculate_rr_expected(current_price)
        
        # التحقق من صلاحية الخطة
        if plan.is_valid():
            slog("TRADEPLAN", 
                f"Built plan: {side} | Entry: {current_price:.4f} | SL: {plan.sl:.4f} | TP1: {plan.tp1:.4f} | RR: 1:{plan.rr_expected:.1f}",
                level="INFO",
                confidence=self.confidence_engine.score(market_analysis, plan.get_summary()))
            return plan
        else:
            slog("TRADEPLAN", f"Plan rejected: {plan.reason}", level="WARN")
            return None
    
    def open_trade_with_plan(self, plan: TradePlan, current_price: float, balance: float, 
                            reason: str = "") -> bool:
        """فتح صفقة جديدة مع خطة"""
        
        # التحقق من عدم وجود صفقة نشطة
        if self.active_trade:
            slog("SYSTEM", "Cannot open trade: Active trade exists", level="WARN")
            return False
        
        # حساب حجم المركز (مبسط)
        qty = balance * self.risk_percent / current_price
        
        # تحديث المركز الحالي
        self.current_position = {
            'side': plan.side,
            'entry_price': current_price,
            'quantity': qty,
            'entry_time': datetime.now(),
            'plan': plan,
            'reason': reason
        }
        
        # تهيئة نظام إدارة المراحل مع خطة الصفقة
        self.trade_phase_engine = TradePhaseEngine(current_price, plan.side, plan)
        self.active_trade = True
        
        # تسجيل الصفقة
        trade_record = {
            'id': len(self.trades_history) + 1,
            'timestamp': datetime.now().isoformat(),
            'side': plan.side,
            'entry_price': current_price,
            'qty': qty,
            'plan': plan.get_summary(),
            'reason': reason
        }
        self.trades_history.append(trade_record)
        
        # حساب الثقة
        market_data = {
            'liquidity_sweep': True,
            'structure': {'type': 'BOS_UP' if plan.side == 'BUY' else 'BOS_DOWN'},
            'volume_spike': True,
            'momentum': {'direction': 'BULLISH' if plan.side == 'BUY' else 'BEARISH', 'score': 0.8},
            'trend': {'strength': 2.5 if plan.trend_class == 'LARGE' else 1.5}
        }
        
        confidence = self.confidence_engine.score(market_data, plan.get_summary())
        
        # لوج الدخول
        slog("ENTRY", 
            f"{plan.side} | Price: {current_price:.4f} | Qty: {qty:.4f} | SL: {plan.sl:.4f} | TP1: {plan.tp1:.4f}",
            level="INFO",
            confidence=confidence)
        
        # فحص Fail-Fast إذا كانت الثقة منخفضة
        if confidence < 6:
            slog("FAIL-FAST", f"Low confidence entry ({confidence}/10) - Monitoring closely", level="WARN")
        
        slog("SYSTEM", f"Trade opened | {plan.side} @ {current_price:.4f} | RR: 1:{plan.rr_expected:.1f}", level="INFO")
        
        return True
    
    def manage_trade_with_plan(self, current_price: float, df: pd.DataFrame):
        """إدارة الصفقة النشطة مع خطة"""
        if not self.active_trade or self.trade_phase_engine is None:
            # محاولة إعادة الدخول إذا لم توجد صفقة نشطة
            balance = 100.0  # رصيد افتراضي
            reentry_success, reentry_msg = self.explosion_engine.execute_reentry(self, df, balance)
            if reentry_success:
                slog("RE-ENTRY", f"Successful re-entry: {reentry_msg}", level="INFO")
            return
        
        plan = self.current_position['plan']
        
        # 1. فحص الانهيار العنيف
        breakdown_detected, breakdown_reason = self.explosion_engine.detect_breakdown(df, plan)
        if breakdown_detected:
            slog("BREAKDOWN", f"Emergency exit: {breakdown_reason}", level="ERROR")
            self.close_trade(f"VIOLENT BREAKDOWN: {breakdown_reason}", current_price)
            return
        
        # 2. فحص Fail-Fast بناءً على الثقة
        market_data = {
            'liquidity_sweep': True,
            'structure': {'type': 'BOS_UP' if plan.side == 'BUY' else 'BOS_DOWN'},
            'volume_spike': False,
            'momentum': {'direction': 'BULLISH' if plan.side == 'BUY' else 'BEARISH', 'score': 0.3},
            'trend': {'strength': 1.0}
        }
        
        current_confidence = self.confidence_engine.score(market_data, plan.get_summary())
        
        # إذا انخفضت الثقة بشدة أثناء الصفقة
        if current_confidence < 4:
            slog("FAIL-FAST", f"Confidence dropped to {current_confidence}/10 - Early exit", level="ERROR")
            self.close_trade(f"Confidence dropped to {current_confidence}/10", current_price)
            return
        
        # 3. فحص أهداف الصفقة
        self._check_targets(current_price, plan)
        
        # 4. تحديث حالة الصفقة
        profit_pct = ((current_price - self.current_position['entry_price']) / 
                     self.current_position['entry_price'] * 100) if plan.side == "BUY" else (
                     (self.current_position['entry_price'] - current_price) / 
                     self.current_position['entry_price'] * 100)
        
        slog("MANAGEMENT", 
            f"Active: {plan.side} | Price: {current_price:.4f} | PnL: {profit_pct:+.2f}% | State: {self.trade_phase_engine.current_state}",
            level="INFO",
            confidence=current_confidence)
    
    def _check_targets(self, current_price: float, plan: TradePlan):
        """فحص أهداف الصفقة"""
        if plan.side == "BUY":
            if not self.trade_phase_engine.targets_hit['tp1'] and current_price >= plan.tp1:
                slog("TARGET", f"TP1 Hit @ {current_price:.4f}", level="INFO")
                self.trade_phase_engine.targets_hit['tp1'] = True
                
            if (self.trade_phase_engine.targets_hit['tp1'] and 
                not self.trade_phase_engine.targets_hit['tp2'] and 
                current_price >= plan.tp2):
                slog("TARGET", f"TP2 Hit @ {current_price:.4f}", level="INFO")
                self.trade_phase_engine.targets_hit['tp2'] = True
                
        else:  # SELL
            if not self.trade_phase_engine.targets_hit['tp1'] and current_price <= plan.tp1:
                slog("TARGET", f"TP1 Hit @ {current_price:.4f}", level="INFO")
                self.trade_phase_engine.targets_hit['tp1'] = True
                
            if (self.trade_phase_engine.targets_hit['tp1'] and 
                not self.trade_phase_engine.targets_hit['tp2'] and 
                current_price <= plan.tp2):
                slog("TARGET", f"TP2 Hit @ {current_price:.4f}", level="INFO")
                self.trade_phase_engine.targets_hit['tp2'] = True
    
    def close_trade(self, reason: str, exit_price: float):
        """إغلاق الصفقة"""
        if not self.active_trade:
            return
        
        # حساب الربح/الخسارة
        entry_price = self.current_position['entry_price']
        side = self.current_position['side']
        quantity = self.current_position['quantity']
        
        if side == "BUY":
            pnl_pct = ((exit_price - entry_price) / entry_price) * 100
            pnl_usd = (exit_price - entry_price) * quantity
        else:
            pnl_pct = ((entry_price - exit_price) / entry_price) * 100
            pnl_usd = (entry_price - exit_price) * quantity
        
        # تحديث الإحصائيات
        self.total_pnl += pnl_pct
        self.total_trades += 1
        if pnl_pct > 0:
            self.winning_trades += 1
        
        # تسجيل الصفقة المغلقة في محرك إعادة الدخول
        closed_trade = {
            'side': side,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'exit_reason': reason,
            'pnl_pct': pnl_pct,
            'pnl_usd': pnl_usd
        }
        self.explosion_engine.record_closed_trade(closed_trade)
        
        # تحديث سجل الصفقات
        if self.trades_history:
            self.trades_history[-1].update({
                'exit_price': exit_price,
                'exit_reason': reason,
                'pnl_pct': pnl_pct,
                'pnl_usd': pnl_usd,
                'exit_time': datetime.now().isoformat()
            })
        
        # لوج الخروج
        log_level = "WARN" if "FAIL" in reason or "BREAKDOWN" in reason else "INFO"
        slog("EXIT", 
            f"{side} | Exit: {exit_price:.4f} | PnL: {pnl_pct:+.2f}% | Reason: {reason}",
            level=log_level)
        
        # إعادة التعيين
        self.active_trade = False
        self.trade_phase_engine = None
        self.current_position = {
            'side': None,
            'entry_price': 0.0,
            'quantity': 0.0,
            'entry_time': None,
            'plan': None
        }
        
        slog("SYSTEM", f"Trade closed | PnL: {pnl_pct:+.2f}% | Total Trades: {self.total_trades}", level="INFO")
    
    def get_trade_report(self) -> Dict:
        """تقرير عن أداء الصفقات"""
        total_trades = len(self.trades_history)
        winning_trades = len([t for t in self.trades_history if t.get('pnl_pct', 0) > 0])
        losing_trades = total_trades - winning_trades
        
        if total_trades > 0:
            avg_pnl = sum(t.get('pnl_pct', 0) for t in self.trades_history) / total_trades
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            total_pnl_usd = sum(t.get('pnl_usd', 0) for t in self.trades_history)
        else:
            avg_pnl = 0
            win_rate = 0
            total_pnl_usd = 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_pnl': avg_pnl,
            'total_pnl_pct': self.total_pnl,
            'total_pnl_usd': total_pnl_usd,
            'active_trade': self.active_trade,
            'current_state': self.trade_phase_engine.current_state if self.trade_phase_engine else None,
            'recent_trades': self.trades_history[-5:] if self.trades_history else [],
            'current_position': self.current_position if self.active_trade else None
        }

# ============================================
#  SIGNAL GENERATOR - مولد الإشارات
# ============================================

class SignalGenerator:
    """مولد إشارات التداول"""
    
    def __init__(self):
        self.last_signal_time = 0
        self.signal_cooldown = 60
    
    def generate_signal(self, df: pd.DataFrame, market_analysis: Dict) -> Tuple[bool, str, float, str]:
        """
        توليد إشارة تداول
        """
        current_time = time.time()
        
        # فحص فترة التبريد
        if current_time - self.last_signal_time < self.signal_cooldown:
            return False, "", 0.0, f"Signal cooldown: {int(self.signal_cooldown - (current_time - self.last_signal_time))}s"
        
        if df.empty or len(df) < 20:
            return False, "", 0.0, "Insufficient data"
        
        # استخدام تحليل السوق
        trend = market_analysis.get('trend', {})
        structure = market_analysis.get('structure', {})
        
        # إشارة شراء
        if trend.get('direction') == "BULL" and structure.get('type') == "BOS_UP":
            confidence = 8.0
            reason = "Bullish trend with BOS structure"
            self.last_signal_time = current_time
            return True, "buy", confidence, reason
        
        # إشارة بيع
        elif trend.get('direction') == "BEAR" and structure.get('type') == "BOS_DOWN":
            confidence = 8.0
            reason = "Bearish trend with BOS structure"
            self.last_signal_time = current_time
            return True, "sell", confidence, reason
        
        return False, "", 0.0, "No clear signal"

# ============================================
#  MAIN BOT INTEGRATION - التكامل الرئيسي
# ============================================

# إعدادات البوت
EXCHANGE_NAME = os.getenv("EXCHANGE", "bybit").lower()
API_KEY = os.getenv("BYBIT_API_KEY", "")
API_SECRET = os.getenv("BYBIT_API_SECRET", "")
MODE_LIVE = bool(API_KEY and API_SECRET)
PORT = int(os.getenv("PORT", 5000))
DRY_RUN = os.getenv("DRY_RUN", "false").lower() == "true"

BOT_VERSION = "SUI ULTRA PRO AI v9.8 — ANSI LOGGER + CONFIDENCE + EXPLOSION/RE-ENTRY"

# إعدادات التداول
SYMBOL = os.getenv("SYMBOL", "SUI/USDT:USDT")
INTERVAL = os.getenv("INTERVAL", "15m")
RISK_ALLOC = float(os.getenv("RISK_ALLOC", "0.60"))
BASE_SLEEP = int(os.getenv("BASE_SLEEP", "5"))

# تهيئة Exchange
def make_exchange():
    """تهيئة كائن Exchange"""
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

# الدوال المساعدة
def get_balance(exchange) -> float:
    """الحصول على الرصيد"""
    if not MODE_LIVE:
        return 100.0
    try:
        b = exchange.fetch_balance(params={"type":"swap"})
        return b.get("total",{}).get("USDT") or b.get("free",{}).get("USDT", 0.0)
    except Exception as e:
        slog("ERROR", f"Failed to fetch balance: {str(e)}", level="ERROR")
        return None

def get_current_price(exchange, symbol: str) -> float:
    """الحصول على السعر الحالي"""
    try:
        t = exchange.fetch_ticker(symbol)
        return t.get("last") or t.get("close")
    except Exception as e:
        slog("ERROR", f"Failed to fetch price: {str(e)}", level="ERROR")
        return None

def fetch_ohlcv_data(exchange, symbol: str, timeframe: str = "15m", limit: int = 100) -> pd.DataFrame:
    """جلب بيانات OHLCV"""
    try:
        rows = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit, params={"type":"swap"})
        return pd.DataFrame(rows, columns=["timestamp","open","high","low","close","volume"])
    except Exception as e:
        slog("ERROR", f"Failed to fetch OHLCV: {str(e)}", level="ERROR")
        return pd.DataFrame()

def convert_candles_to_dicts(df: pd.DataFrame) -> List[Dict]:
    """تحويل DataFrame إلى قائمة من القواميس"""
    if df.empty:
        return []
    
    candles = []
    for i in range(len(df)):
        candles.append({
            'open': float(df['open'].iloc[i]),
            'high': float(df['high'].iloc[i]),
            'low': float(df['low'].iloc[i]),
            'close': float(df['close'].iloc[i]),
            'volume': float(df['volume'].iloc[i])
        })
    return candles

# ============================================
#  MAIN BOT CLASS - الفئة الرئيسية للبوت
# ============================================

class SUIUltraProBot:
    """الفئة الرئيسية للبوت مع نظام TradePlan"""
    
    def __init__(self):
        self.exchange = None
        self.smart_trade_manager = None
        self.market_analyzer = None
        self.signal_generator = None
        self.running = False
        
    def initialize(self):
        """تهيئة البوت"""
        try:
            slog("SYSTEM", f"🚀 Booting: {BOT_VERSION}", level="INFO")
            
            # تهيئة Exchange
            self.exchange = make_exchange()
            slog("SYSTEM", f"Exchange: {EXCHANGE_NAME.upper()} | Symbol: {SYMBOL}", level="INFO")
            slog("SYSTEM", f"Mode: {'LIVE' if MODE_LIVE else 'PAPER'} | Dry Run: {DRY_RUN}", level="INFO")
            
            # تهيئة الأنظمة
            self.smart_trade_manager = SmartTradeManager(
                exchange=self.exchange,
                symbol=SYMBOL,
                risk_percent=RISK_ALLOC
            )
            
            self.market_analyzer = MarketAnalyzer()
            self.signal_generator = SignalGenerator()
            
            slog("SYSTEM", "Smart Trade System with TradePlan Initialized", level="INFO")
            slog("SYSTEM", f"Symbol: {SYMBOL} | Risk: {RISK_ALLOC*100:.0f}% | Interval: {INTERVAL}", level="INFO")
            
            return True
            
        except Exception as e:
            slog("ERROR", f"Failed to initialize bot: {str(e)}", level="ERROR")
            return False
    
    def run_trade_loop(self):
        """تشغيل حلقة التداول الرئيسية"""
        slog("SYSTEM", "Starting Smart Trade Loop with TradePlan", level="INFO")
        self.running = True
        
        while self.running:
            try:
                # جمع بيانات السوق
                balance = get_balance(self.exchange)
                current_price = get_current_price(self.exchange, SYMBOL)
                df = fetch_ohlcv_data(self.exchange, SYMBOL, INTERVAL)
                
                if df.empty or current_price is None:
                    slog("DEBUG", "Waiting for market data...", level="DEBUG")
                    time.sleep(BASE_SLEEP)
                    continue
                
                # 1. كشف الانفجار
                explosion_detected, explosion_details = self.smart_trade_manager.explosion_engine.detect_explosion(df)
                if explosion_detected:
                    slog("EXPLOSION", f"Market explosion detected! Direction: {explosion_details['direction']}", level="INFO")
                
                # 2. تحليل السوق
                market_analysis = self.market_analyzer.analyze_market(df, INTERVAL)
                
                # 3. إذا كانت هناك صفقة نشطة
                if self.smart_trade_manager.active_trade:
                    # إدارة الصفقة الحالية مع خطة
                    self.smart_trade_manager.manage_trade_with_plan(current_price, df)
                
                else:
                    # 4. توليد إشارة تداول
                    signal, side, confidence, reason = self.signal_generator.generate_signal(df, market_analysis)
                    
                    if signal and balance and balance > 10:
                        # حساب الثقة النهائية
                        confidence_engine = ConfidenceEngine()
                        final_confidence = confidence_engine.score(market_analysis, {})
                        
                        # فحص عتبة الثقة
                        if final_confidence >= 6:
                            # بناء خطة الصفقة
                            trade_plan = self.smart_trade_manager.build_trade_plan(
                                side, current_price, market_analysis, df
                            )
                            
                            if trade_plan and trade_plan.is_valid():
                                # فتح صفقة مع خطة
                                success = self.smart_trade_manager.open_trade_with_plan(
                                    trade_plan, current_price, balance, reason
                                )
                                
                                if success:
                                    slog("SYSTEM", f"Trade opened with plan | {side.upper()} @ {current_price:.4f}", level="INFO")
                            else:
                                if trade_plan:
                                    slog("SYSTEM", f"Entry blocked: {trade_plan.reason}", level="WARN")
                                else:
                                    slog("SYSTEM", f"Entry blocked: No valid trade plan generated", level="WARN")
                        else:
                            slog("FAIL-FAST", f"Entry blocked - Low confidence: {final_confidence}/10", level="WARN")
                
                # النوم حتى التكرار التالي
                time.sleep(BASE_SLEEP)
                
            except KeyboardInterrupt:
                slog("SYSTEM", "Trade loop stopped by user", level="INFO")
                self.running = False
                break
                
            except Exception as e:
                slog("ERROR", f"Trade loop error: {str(e)}", level="ERROR")
                time.sleep(BASE_SLEEP * 2)
    
    def stop(self):
        """إيقاف البوت"""
        self.running = False
        slog("SYSTEM", "Bot stopped", level="INFO")
    
    def get_status_report(self) -> Dict:
        """الحصول على تقرير حالة البوت"""
        trade_report = self.smart_trade_manager.get_trade_report() if self.smart_trade_manager else {}
        
        return {
            'bot_version': BOT_VERSION,
            'exchange': EXCHANGE_NAME.upper(),
            'symbol': SYMBOL,
            'mode': 'LIVE' if MODE_LIVE else 'PAPER',
            'dry_run': DRY_RUN,
            'running': self.running,
            'trade_report': trade_report,
            'timestamp': datetime.now().isoformat()
        }

# ============================================
#  FLASK API SERVER - خادم API
# ============================================

app = Flask(__name__)
bot_instance = None

@app.route('/')
def dashboard():
    """لوحة التحكم الرئيسية"""
    return "<h1>SUI ULTRA PRO AI v9.8 Dashboard</h1><p>ANSI Logger + Confidence + Explosion/Re-Entry Engine</p>"

@app.route('/health')
def health_check():
    """فحص صحة النظام"""
    if bot_instance and bot_instance.running:
        return jsonify({
            'status': 'healthy',
            'bot_version': BOT_VERSION,
            'timestamp': datetime.now().isoformat()
        }), 200
    return jsonify({'status': 'unhealthy', 'error': 'Bot not running'}), 503

@app.route('/api/status')
def api_status():
    """حالة البوت والإحصائيات"""
    if bot_instance:
        return jsonify(bot_instance.get_status_report())
    return jsonify({'error': 'Bot not initialized'}), 500

# ============================================
#  MAIN EXECUTION - التنفيذ الرئيسي
# ============================================

def main():
    """الدالة الرئيسية"""
    global bot_instance
    
    try:
        # طباعة بانر البداية
        print(f"\n{C.LIGHT_CYAN}{'='*80}{C.RESET}")
        print(f"{C.LIGHT_GREEN}{BOT_VERSION}{C.RESET}")
        print(f"{C.LIGHT_CYAN}🔥 ANSI Logger + Confidence Engine + Explosion/Re-Entry System 🔥{C.RESET}")
        print(f"{C.LIGHT_CYAN}{'='*80}{C.RESET}\n")
        
        # إنشاء وتشغيل البوت
        bot_instance = SUIUltraProBot()
        
        if not bot_instance.initialize():
            slog("ERROR", "Failed to initialize bot", level="ERROR")
            return
        
        # بدء حلقة التداول في thread منفصل
        import threading
        trade_thread = threading.Thread(target=bot_instance.run_trade_loop, daemon=True)
        trade_thread.start()
        
        slog("SYSTEM", f"Starting Flask server on port {PORT}", level="INFO")
        
        # تشغيل خادم Flask
        app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
        
    except KeyboardInterrupt:
        slog("SYSTEM", "Bot stopped by user", level="INFO")
    except Exception as e:
        slog("ERROR", f"Fatal error in main: {str(e)}", level="ERROR")
    finally:
        if bot_instance:
            bot_instance.stop()

if __name__ == "__main__":
    main()
