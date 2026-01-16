# -*- coding: utf-8 -*-
"""
SUI ULTRA PRO AI BOT - الإصدار الذكي مع نظام إدارة الصفقات المتقدم
• نظام إدارة المراحل الذكي (Entry → Protect → BE → Trail → Trim → Exit)
• لوج احترافي واضح مع أسباب القرارات
• Structure-Based Trailing (ليس ATR تقليدي)
• حماية تنفيذية من أخطاء Bybit/MinQty
• نظام Trim الذكي لتقليل المخاطرة
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
from typing import Optional, List, Dict, Tuple, Any

try:
    from termcolor import colored
except Exception:
    def colored(t,*a,**k): return t

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
    """محرك إدارة مراحل الصفقة"""
    
    def __init__(self, entry_price: float, side: str, entry_zone: str):
        self.entry_price = entry_price
        self.side = side.upper()  # BUY/SELL
        self.entry_zone = entry_zone
        self.current_state = TradeState.ENTRY
        self.state_changed_at = time.time()
        self.structure_levels = []  # مستويات الهيكل
        self.last_stop_loss = None
        self.trim_count = 0
        self.max_trims = 2
        self.state_log = []
        
        # إعدادات حسب نوع الصفقة
        self.protection_pct = 0.5  # حماية عند 0.5%
        self.be_pct = 0.3         # نقطة التعادل عند 0.3%
        self.trail_activation_pct = 0.8  # تفعيل التريل عند 0.8%
        self.trim_pct = 0.2       # تقليل 20% في كل ترايم
        
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
        
        log_i(f"🔄 STATE CHANGE: {old_state} → {new_state} | Reason: {reason}")
        
    def analyze_structure(self, candles: List[Dict]) -> Dict:
        """تحليل الهيكل السعري الحالي"""
        if len(candles) < 10:
            return {"hh": None, "hl": None, "lh": None, "ll": None, "trend": "UNKNOWN"}
        
        highs = [c['high'] for c in candles[-10:]]
        lows = [c['low'] for c in candles[-10:]]
        closes = [c['close'] for c in candles[-10:]]
        
        # التعرف على القمم والقيعان المحلية
        hh = max(highs[-5:])  # أعلى قمة حديثة
        ll = min(lows[-5:])   # أقل قاع حديث
        
        # تحديد الهيكل
        if self.side == "BUY":
            # في الشراء: نبحث عن Higher Highs و Higher Lows
            recent_highs = sorted(highs[-5:], reverse=True)[:2]
            recent_lows = sorted(lows[-5:])[:2]
            
            hh = max(recent_highs) if recent_highs else None
            hl = min(recent_lows) if len(recent_lows) > 1 else recent_lows[0] if recent_lows else None
            
            structure_info = {
                "hh": hh,
                "hl": hl,
                "lh": None,
                "ll": None,
                "trend": "UP" if closes[-1] > closes[-5] else "CONSOLIDATION"
            }
            
        else:  # SELL
            # في البيع: نبحث عن Lower Highs و Lower Lows
            recent_highs = sorted(highs[-5:])[:2]
            recent_lows = sorted(lows[-5:], reverse=True)[:2]
            
            lh = min(recent_highs) if recent_highs else None
            ll = max(recent_lows) if len(recent_lows) > 1 else recent_lows[0] if recent_lows else None
            
            structure_info = {
                "hh": None,
                "hl": None,
                "lh": lh,
                "ll": ll,
                "trend": "DOWN" if closes[-1] < closes[-5] else "CONSOLIDATION"
            }
        
        return structure_info
    
    def detect_liquidity_event(self, candles: List[Dict]) -> Dict:
        """كشف أحداث السيولة"""
        if len(candles) < 3:
            return {"sweep": False, "tap": False, "type": None}
        
        current = candles[-1]
        prev = candles[-2]
        
        # كشف Sweep
        sweep_up = current['high'] > max([c['high'] for c in candles[-4:-1]]) and current['close'] < prev['close']
        sweep_down = current['low'] < min([c['low'] for c in candles[-4:-1]]) and current['close'] > prev['close']
        
        # كشف Liquidity Tap (تلامس سيولة بدون اختراق)
        tap_up = abs(current['high'] - max([c['high'] for c in candles[-4:-1]])) < (current['high'] * 0.001)
        tap_down = abs(current['low'] - min([c['low'] for c in candles[-4:-1]])) < (current['low'] * 0.001)
        
        return {
            "sweep": sweep_up or sweep_down,
            "tap": tap_up or tap_down,
            "type": "SWEEP_UP" if sweep_up else ("SWEEP_DOWN" if sweep_down else 
                    "TAP_UP" if tap_up else ("TAP_DOWN" if tap_down else None))
        }
    
    def should_move_to_protect(self, current_price: float, candles: List[Dict]) -> Tuple[bool, str]:
        """التحقق من الانتقال لمرحلة الحماية"""
        if self.current_state != TradeState.ENTRY:
            return False, "Already in protection or beyond"
        
        profit_pct = self.calculate_profit_pct(current_price)
        
        # الشرط: ربح 0.5% وهيكل صحيح
        if profit_pct >= self.protection_pct:
            structure = self.analyze_structure(candles)
            
            if self.side == "BUY" and structure['trend'] == "UP":
                return True, f"Profit {profit_pct:.2f}% + Uptrend intact"
            elif self.side == "SELL" and structure['trend'] == "DOWN":
                return True, f"Profit {profit_pct:.2f}% + Downtrend intact"
        
        return False, f"Insufficient profit: {profit_pct:.2f}%"
    
    def should_move_to_breakeven(self, current_price: float, candles: List[Dict]) -> Tuple[bool, str]:
        """التحقق من الانتقال لنقطة التعادل"""
        if self.current_state != TradeState.PROTECT:
            return False, "Not in PROTECT phase"
        
        profit_pct = self.calculate_profit_pct(current_price)
        
        # الشرط: ربح 0.3% وعدم وجود انعكاس
        if profit_pct >= self.be_pct:
            # تحقق من عدم وجود CHoCH ضد الصفقة
            choch = self.detect_choch(candles)
            if not choch['against_trade']:
                return True, f"Profit {profit_pct:.2f}% + No CHoCH against"
        
        return False, f"Waiting for BE conditions"
    
    def should_move_to_trail(self, current_price: float, candles: List[Dict]) -> Tuple[bool, str]:
        """التحقق من الانتقال لمرحلة التريل"""
        if self.current_state not in [TradeState.BREAKEVEN, TradeState.TRAIL, TradeState.TRIM]:
            return False, "Not in BE/TRAIL/TRIM phase"
        
        profit_pct = self.calculate_profit_pct(current_price)
        
        # الشرط: ربح 0.8% وهيكل جديد مؤكد
        if profit_pct >= self.trail_activation_pct:
            structure = self.analyze_structure(candles)
            liq_event = self.detect_liquidity_event(candles)
            
            # في الشراء: تأكيد Higher Low جديد
            if self.side == "BUY" and structure['hl'] and not liq_event['sweep']:
                if self.last_stop_loss is None or structure['hl'] > self.last_stop_loss:
                    return True, f"Profit {profit_pct:.2f}% + New HL confirmed"
            
            # في البيع: تأكيد Lower High جديد
            elif self.side == "SELL" and structure['lh'] and not liq_event['sweep']:
                if self.last_stop_loss is None or structure['lh'] < self.last_stop_loss:
                    return True, f"Profit {profit_pct:.2f}% + New LH confirmed"
        
        return False, f"Trail conditions not met"
    
    def should_trim_position(self, current_price: float, candles: List[Dict]) -> Tuple[bool, str]:
        """التحقق من الحاجة لتقليل المخاطرة"""
        if self.current_state not in [TradeState.TRAIL, TradeState.TRIM]:
            return False, "Not in trail phase"
        
        if self.trim_count >= self.max_trims:
            return False, "Max trims reached"
        
        # أسباب الترايم
        reasons = []
        
        # 1. Wick قوي ضد الاتجاه
        current_candle = candles[-1]
        candle_range = current_candle['high'] - current_candle['low']
        
        if self.side == "BUY":
            upper_wick = current_candle['high'] - max(current_candle['close'], current_candle['open'])
            if upper_wick > candle_range * 0.6:  # wick كبير
                reasons.append("Strong upper wick against")
        else:
            lower_wick = min(current_candle['close'], current_candle['open']) - current_candle['low']
            if lower_wick > candle_range * 0.6:  # wick كبير
                reasons.append("Strong lower wick against")
        
        # 2. ضعف الحجم
        if len(candles) >= 3:
            current_volume = current_candle['volume']
            avg_volume = sum(c['volume'] for c in candles[-4:-1]) / 3
            if current_volume < avg_volume * 0.7:
                reasons.append("Weak volume")
        
        # 3. Liquidity Tap جانبي
        liq_event = self.detect_liquidity_event(candles)
        if liq_event['tap']:
            reasons.append("Liquidity tap detected")
        
        if reasons:
            return True, " | ".join(reasons)
        
        return False, "No trim signals"
    
    def should_exit_trade(self, current_price: float, candles: List[Dict]) -> Tuple[bool, str]:
        """التحقق من الحاجة للخروج الكامل"""
        # 1. CHoCH ضد الصفقة
        choch = self.detect_choch(candles)
        if choch['against_trade'] and choch['confirmed']:
            return True, f"Confirmed CHoCH against trade"
        
        # 2. كسر الهيكل الداعم
        structure = self.analyze_structure(candles)
        profit_pct = self.calculate_profit_pct(current_price)
        
        if self.side == "BUY":
            if structure['trend'] == "DOWN" and profit_pct > 0:
                return True, "Structure broken to downside"
        else:
            if structure['trend'] == "UP" and profit_pct > 0:
                return True, "Structure broken to upside"
        
        # 3. إغلاق شمعة ضد الاتجاه بقوة
        current_candle = candles[-1]
        if self.side == "BUY":
            if current_candle['close'] < current_candle['open'] and (current_candle['open'] - current_candle['close']) > (current_candle['high'] - current_candle['low']) * 0.7:
                return True, "Strong bearish candle"
        else:
            if current_candle['close'] > current_candle['open'] and (current_candle['close'] - current_candle['open']) > (current_candle['high'] - current_candle['low']) * 0.7:
                return True, "Strong bullish candle"
        
        return False, "Trade still valid"
    
    def detect_choch(self, candles: List[Dict]) -> Dict:
        """كشف Change of Character"""
        if len(candles) < 6:
            return {"detected": False, "against_trade": False, "confirmed": False}
        
        # تحليل بسيط لـ CHoCH
        recent_closes = [c['close'] for c in candles[-6:]]
        
        if self.side == "BUY":
            # في الشراء: CHoCH هابط عندما يكسر آخر Low
            recent_lows = [c['low'] for c in candles[-6:-1]]
            last_low = min(recent_lows) if recent_lows else None
            
            if last_low and candles[-1]['close'] < last_low:
                # تأكيد: شمعتين إغلاق تحت last_low
                if len(candles) >= 8 and candles[-2]['close'] < last_low:
                    return {"detected": True, "against_trade": True, "confirmed": True}
                return {"detected": True, "against_trade": True, "confirmed": False}
        
        else:  # SELL
            # في البيع: CHoCH صاعد عندما يكسر آخر High
            recent_highs = [c['high'] for c in candles[-6:-1]]
            last_high = max(recent_highs) if recent_highs else None
            
            if last_high and candles[-1]['close'] > last_high:
                # تأكيد: شمعتين إغلاق فوق last_high
                if len(candles) >= 8 and candles[-2]['close'] > last_high:
                    return {"detected": True, "against_trade": True, "confirmed": True}
                return {"detected": True, "against_trade": True, "confirmed": False}
        
        return {"detected": False, "against_trade": False, "confirmed": False}
    
    def calculate_profit_pct(self, current_price: float) -> float:
        """حساب نسبة الربح/الخسارة"""
        if self.side == "BUY":
            return ((current_price - self.entry_price) / self.entry_price) * 100
        else:
            return ((self.entry_price - current_price) / self.entry_price) * 100
    
    def calculate_stop_loss(self, current_price: float, candles: List[Dict]) -> Tuple[float, str]:
        """حساب وقف الخسارة الحالي"""
        structure = self.analyze_structure(candles)
        reason = ""
        
        if self.current_state == TradeState.ENTRY:
            # في الدخول: وقف خلف منطقة الدخول
            if self.side == "BUY":
                sl = self.entry_price * 0.995  # 0.5% تحت الدخول
                reason = "Initial protection"
            else:
                sl = self.entry_price * 1.005  # 0.5% فوق الدخول
                reason = "Initial protection"
                
        elif self.current_state == TradeState.PROTECT:
            # حماية: عند 0.2% ربح
            if self.side == "BUY":
                sl = self.entry_price * 1.002
                reason = "Protection phase"
            else:
                sl = self.entry_price * 0.998
                reason = "Protection phase"
                
        elif self.current_state == TradeState.BREAKEVEN:
            # نقطة التعادل
            sl = self.entry_price
            reason = "Breakeven activated"
            
        elif self.current_state == TradeState.TRAIL:
            # تريل بالهيكل
            if self.side == "BUY" and structure['hl']:
                sl = structure['hl'] * 0.998  # تحت الـ HL قليلاً
                reason = f"Trailing below HL: {structure['hl']:.4f}"
            elif self.side == "SELL" and structure['lh']:
                sl = structure['lh'] * 1.002  # فوق الـ LH قليلاً
                reason = f"Trailing above LH: {structure['lh']:.4f}"
            else:
                # إذا لم يتكون هيكل بعد
                sl = self.entry_price
                reason = "No structure yet, at breakeven"
                
        elif self.current_state == TradeState.TRIM:
            # بعد الترام: وقف أكثر تحفظاً
            if self.last_stop_loss:
                sl = self.last_stop_loss
                reason = "Maintaining SL after trim"
            else:
                sl = self.entry_price
                reason = "Breakeven after trim"
        else:
            sl = current_price  # في حالة EXIT
            reason = "Exit phase"
        
        self.last_stop_loss = sl
        return sl, reason
    
    def get_trade_summary(self) -> Dict:
        """ملخص حالة الصفقة"""
        return {
            'current_state': self.current_state,
            'state_duration': time.time() - self.state_changed_at,
            'trim_count': self.trim_count,
            'state_history': self.state_log[-5:],  # آخر 5 تغييرات
            'last_stop_loss': self.last_stop_loss,
            'entry_price': self.entry_price,
            'side': self.side,
            'entry_zone': self.entry_zone
        }

# ============================================
#  ADVANCED LOGGER - لوج احترافي
# ============================================

class AdvancedLogger:
    """نظام لوج متقدم مع ألوان وتنسيق"""
    
    COLORS = {
        'MARKET': 'cyan',
        'ENTRY': 'green',
        'EXECUTION': 'yellow',
        'MANAGE': 'magenta',
        'EXIT': 'red',
        'ERROR': 'red',
        'INFO': 'white'
    }
    
    ICONS = {
        'MARKET': '📊',
        'ENTRY': '🎯',
        'EXECUTION': '⚡',
        'MANAGE': '🔄',
        'EXIT': '🚪',
        'ERROR': '❌',
        'INFO': 'ℹ️'
    }
    
    @classmethod
    def log(cls, category: str, message: str, details: Dict = None):
        """تسجيل رسالة مع تنسيق"""
        color = cls.COLORS.get(category, 'white')
        icon = cls.ICONS.get(category, '📝')
        
        timestamp = datetime.now().strftime('%H:%M:%S')
        formatted_msg = f"{icon} [{timestamp}] {category}: {message}"
        
        # إضافة التفاصيل إذا وجدت
        if details:
            details_str = " | ".join([f"{k}: {v}" for k, v in details.items()])
            formatted_msg += f" | {details_str}"
        
        # طباعة ملونة
        try:
            print(colored(formatted_msg, color), flush=True)
        except:
            print(formatted_msg, flush=True)
        
        # تسجيل في ملف اللوج
        logging.info(f"{category}: {message}")
    
    @classmethod
    def log_market(cls, trend: str, structure: str, liquidity: str, timeframe: str = "15m"):
        """لوج حالة السوق"""
        cls.log('MARKET', f"TF={timeframe} | Trend={trend} | Structure={structure} | Liquidity={liquidity}")
    
    @classmethod
    def log_entry(cls, side: str, zone: str, reason: str, confidence: float):
        """لوج الدخول"""
        cls.log('ENTRY', f"{side} | Zone={zone} | Reason={reason} | Conf={confidence:.2f}")
    
    @classmethod
    def log_execution(cls, price: float, qty: float, sl: float, tp_plan: str):
        """لوج التنفيذ"""
        cls.log('EXECUTION', f"Price={price:.4f} | Qty={qty:.2f} | SL={sl:.4f} | Plan={tp_plan}")
    
    @classmethod
    def log_management(cls, phase: str, action: str, reason: str, details: Dict = None):
        """لوج إدارة الصفقة"""
        cls.log('MANAGE', f"Phase={phase} | Action={action} | Reason={reason}", details)
    
    @classmethod
    def log_exit(cls, reason: str, pnl: float, rr: float = None):
        """لوج الخروج"""
        details = {"PnL": f"{pnl:.2f}%"}
        if rr:
            details["RR"] = f"1:{rr:.1f}"
        cls.log('EXIT', f"Reason: {reason}", details)

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
        
    def sanitize_order(self, symbol: str, qty: float) -> Tuple[Optional[float], str]:
        """تنقية وتنظيم الكمية قبل الإرسال"""
        try:
            # جلب معلومات السوق
            market = self.exchange.market(symbol)
            
            # الحد الأدنى للكمية
            min_qty = market['limits']['amount']['min']
            
            # الدقة
            precision = market['precision']['amount']
            
            # التقريب للدقة المطلوبة
            qty = round(qty, precision)
            
            # التحقق من الحد الأدنى
            if qty < min_qty:
                AdvancedLogger.log('ERROR', 
                    f"Quantity {qty} < Minimum {min_qty} → ORDER CANCELLED")
                return None, f"Qty < Min: {qty} < {min_qty}"
            
            # التحقق من الحد الأقصى (إذا موجود)
            if 'max' in market['limits']['amount']:
                max_qty = market['limits']['amount']['max']
                if qty > max_qty:
                    qty = max_qty
                    AdvancedLogger.log('INFO', f"Quantity capped at maximum: {max_qty}")
            
            AdvancedLogger.log('INFO', f"Sanitized Qty: {qty} (Min: {min_qty}, Precision: {precision})")
            return qty, "VALID"
            
        except Exception as e:
            AdvancedLogger.log('ERROR', f"Sanitization error: {str(e)}")
            return None, f"Error: {str(e)}"
    
    def should_allow_order(self) -> Tuple[bool, str]:
        """التحقق إذا كان مسموحاً بإرسال أمر جديد"""
        current_time = time.time()
        
        # فحص التبريد بعد فشل سابق
        if current_time < self.cooldown_until:
            remaining = self.cooldown_until - current_time
            return False, f"In cooldown: {int(remaining)}s remaining"
        
        # فحص عدد الفشل المتتالي
        if self.failure_count >= self.max_failures:
            self.cooldown_until = current_time + 300  # 5 دقائق تبريد
            self.failure_count = 0
            return False, "Max consecutive failures reached, 5min cooldown"
        
        return True, "Allowed"
    
    def record_success(self):
        """تسجيل نجاح الأمر"""
        self.failure_count = 0
        self.last_failed_order = None
    
    def record_failure(self, error: str):
        """تسجيل فشل الأمر"""
        self.failure_count += 1
        self.last_failed_order = {
            'time': time.time(),
            'error': error
        }
        
        # تفعيل التبريد إذا فشلت أمرين متتاليين
        if self.failure_count >= 2:
            self.cooldown_until = time.time() + 60  # 1 دقيقة تبريد
        
        AdvancedLogger.log('ERROR', f"Order failed ({self.failure_count}/{self.max_failures}): {error}")

# ============================================
#  SMART TRADE MANAGER - المدير الرئيسي
# ============================================

class SmartTradeManager:
    """المدير الذكي للصفقات"""
    
    def __init__(self, exchange, symbol: str, risk_percent: float = 0.6):
        self.exchange = exchange
        self.symbol = symbol
        self.risk_percent = risk_percent
        
        # الأنظمة الفرعية
        self.execution_guard = ExecutionGuard(exchange)
        self.trade_phase_engine = None
        self.active_trade = False
        
        # إحصائيات
        self.trades_history = []
        self.total_pnl = 0.0
        
    def calculate_position_size(self, balance: float, entry_price: float, confidence: float = 0.7) -> float:
        """حساب حجم المركز الذكي"""
        # رأس المال المستخدم
        risk_capital = balance * self.risk_percent
        
        # تعديل حسب الثقة
        if confidence > 0.8:
            risk_multiplier = 1.2
        elif confidence > 0.6:
            risk_multiplier = 1.0
        elif confidence > 0.4:
            risk_multiplier = 0.7
        else:
            risk_multiplier = 0.5
        
        adjusted_capital = risk_capital * risk_multiplier
        
        # حساب الكمية
        raw_qty = adjusted_capital / entry_price
        
        # تنقية الكمية
        sanitized_qty, status = self.execution_guard.sanitize_order(self.symbol, raw_qty)
        
        if sanitized_qty is None:
            AdvancedLogger.log('ERROR', f"Position size invalid: {status}")
            return 0.0
        
        AdvancedLogger.log('INFO', 
            f"Position Size: {sanitized_qty:.4f} | "
            f"Capital: ${adjusted_capital:.2f} | "
            f"Confidence: {confidence:.2f}")
        
        return sanitized_qty
    
    def open_trade(self, side: str, entry_price: float, balance: float, 
                   entry_zone: str, confidence: float = 0.7, reason: str = "") -> bool:
        """فتح صفقة جديدة"""
        
        # التحقق من عدم وجود صفقة نشطة
        if self.active_trade:
            AdvancedLogger.log('ERROR', "Cannot open trade: Active trade exists")
            return False
        
        # التحقق من صلاحية التنفيذ
        allow, allow_reason = self.execution_guard.should_allow_order()
        if not allow:
            AdvancedLogger.log('WARNING', f"Order not allowed: {allow_reason}")
            return False
        
        # حساب حجم المركز
        qty = self.calculate_position_size(balance, entry_price, confidence)
        if qty <= 0:
            return False
        
        # تنفيذ الأمر (أو محاكاة)
        success = self.execute_order(side, qty, entry_price, is_open=True)
        
        if success:
            # تهيئة نظام إدارة المراحل
            self.trade_phase_engine = TradePhaseEngine(entry_price, side, entry_zone)
            self.active_trade = True
            
            # تسجيل الصفقة
            trade_record = {
                'id': len(self.trades_history) + 1,
                'timestamp': datetime.now().isoformat(),
                'side': side,
                'entry_price': entry_price,
                'qty': qty,
                'zone': entry_zone,
                'reason': reason,
                'confidence': confidence
            }
            self.trades_history.append(trade_record)
            
            # لوج الدخول
            AdvancedLogger.log_entry(side, entry_zone, reason, confidence)
            AdvancedLogger.log_execution(
                entry_price, qty, 
                self.trade_phase_engine.calculate_stop_loss(entry_price, [])[0],
                "Protect → BE → Trail → Trim"
            )
            
            return True
        
        return False
    
    def manage_trade(self, current_price: float, candles: List[Dict]):
        """إدارة الصفقة النشطة"""
        if not self.active_trade or self.trade_phase_engine is None:
            return
        
        # تحليل المرحلة الحالية واتخاذ القرارات
        self._update_trade_phase(current_price, candles)
        
        # حساب وقف الخسارة الحالي
        sl_price, sl_reason = self.trade_phase_engine.calculate_stop_loss(current_price, candles)
        
        # التحقق من وقف الخسارة
        if self._should_hit_stop_loss(current_price, sl_price):
            self.close_trade(f"Stop Loss: {sl_reason}", current_price)
            return
        
        # التحقق من الخروج
        should_exit, exit_reason = self.trade_phase_engine.should_exit_trade(current_price, candles)
        if should_exit:
            self.close_trade(exit_reason, current_price)
            return
        
        # لوج إدارة الصفقة
        profit_pct = self.trade_phase_engine.calculate_profit_pct(current_price)
        state = self.trade_phase_engine.current_state
        
        AdvancedLogger.log_management(
            state,
            "HOLD",
            f"P&L: {profit_pct:.2f}% | SL: {sl_price:.4f}",
            {
                "State": state,
                "PnL": f"{profit_pct:.2f}%",
                "SL": f"{sl_price:.4f}",
                "SL_Reason": sl_reason
            }
        )
    
    def _update_trade_phase(self, current_price: float, candles: List[Dict]):
        """تحديث مرحلة الصفقة"""
        engine = self.trade_phase_engine
        
        # الانتقال بين المراحل
        if engine.current_state == TradeState.ENTRY:
            should_protect, reason = engine.should_move_to_protect(current_price, candles)
            if should_protect:
                engine.update_state(TradeState.PROTECT, reason)
                AdvancedLogger.log_management("PROTECT", "ACTIVATED", reason)
        
        elif engine.current_state == TradeState.PROTECT:
            should_be, reason = engine.should_move_to_breakeven(current_price, candles)
            if should_be:
                engine.update_state(TradeState.BREAKEVEN, reason)
                AdvancedLogger.log_management("BREAKEVEN", "ACTIVATED", reason)
        
        elif engine.current_state in [TradeState.BREAKEVEN, TradeState.TRAIL, TradeState.TRIM]:
            should_trail, reason = engine.should_move_to_trail(current_price, candles)
            if should_trail:
                engine.update_state(TradeState.TRAIL, reason)
                AdvancedLogger.log_management("TRAIL", "ACTIVATED", reason)
            
            should_trim, trim_reason = engine.should_trim_position(current_price, candles)
            if should_trim:
                # تنفيذ ترام جزئي
                self._execute_trim(current_price, trim_reason)
                engine.trim_count += 1
                engine.update_state(TradeState.TRIM, f"Trim #{engine.trim_count}: {trim_reason}")
    
    def _execute_trim(self, current_price: float, reason: str):
        """تنفيذ تقليل المركز"""
        if self.trade_phase_engine:
            # افتراضي: إغلاق 20% من المركز
            trim_percent = 0.2
            
            # لوج الترام
            AdvancedLogger.log_management(
                "TRIM",
                "EXECUTING",
                f"Closing {trim_percent*100:.0f}%: {reason}",
                {"Trim_Pct": f"{trim_percent*100:.0f}%", "Reason": reason}
            )
            
            # هنا يمكن إضافة تنفيذ الأمر الفعلي
            # close_qty = self.current_qty * trim_percent
            # self.execute_order(opposite_side, close_qty, current_price, is_close=True)
    
    def _should_hit_stop_loss(self, current_price: float, stop_loss: float) -> bool:
        """التحقق من ضرب وقف الخسارة"""
        if self.trade_phase_engine.side == "BUY":
            return current_price <= stop_loss
        else:
            return current_price >= stop_loss
    
    def close_trade(self, reason: str, exit_price: float):
        """إغلاق الصفقة"""
        if not self.active_trade or self.trade_phase_engine is None:
            return
        
        # حساب الربح/الخسارة
        entry_price = self.trade_phase_engine.entry_price
        side = self.trade_phase_engine.side
        
        if side == "BUY":
            pnl_pct = ((exit_price - entry_price) / entry_price) * 100
        else:
            pnl_pct = ((entry_price - exit_price) / entry_price) * 100
        
        # تحديث الإحصائيات
        self.total_pnl += pnl_pct
        
        # لوج الخروج
        AdvancedLogger.log_exit(reason, pnl_pct)
        
        # إغلاق الصفقة (أو محاكاة)
        # self.execute_order(opposite_side, self.current_qty, exit_price, is_close=True)
        
        # إعادة التعيين
        self.active_trade = False
        self.trade_phase_engine = None
        
        # تسجيل النتيجة
        if self.trades_history:
            self.trades_history[-1].update({
                'exit_price': exit_price,
                'exit_reason': reason,
                'pnl_pct': pnl_pct,
                'exit_time': datetime.now().isoformat()
            })
    
    def execute_order(self, side: str, qty: float, price: float, 
                      is_open: bool = True) -> bool:
        """تنفيذ الأمر (محاكاة أو حقيقي)"""
        # هذا مثال للتنفيذ المحاكى
        # في التنفيذ الحقيقي، استخدم exchange.create_order()
        
        if DRY_RUN or not EXECUTE_ORDERS:
            AdvancedLogger.log('EXECUTION', 
                f"DRY RUN: {'OPEN' if is_open else 'CLOSE'} {side.upper()} {qty:.4f} @ {price:.6f}")
            return True
        
        try:
            # تنفيذ حقيقي
            params = {"reduceOnly": not is_open}
            order = self.exchange.create_order(
                self.symbol, 
                "market", 
                side, 
                qty, 
                None, 
                params
            )
            
            AdvancedLogger.log('EXECUTION', 
                f"ORDER FILLED: {'OPEN' if is_open else 'CLOSE'} {side.upper()} {qty:.4f} @ {price:.6f}")
            
            self.execution_guard.record_success()
            return True
            
        except Exception as e:
            error_msg = str(e)
            AdvancedLogger.log('ERROR', f"Order execution failed: {error_msg}")
            self.execution_guard.record_failure(error_msg)
            return False
    
    def get_trade_report(self) -> Dict:
        """تقرير عن أداء الصفقات"""
        total_trades = len(self.trades_history)
        winning_trades = len([t for t in self.trades_history if t.get('pnl_pct', 0) > 0])
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': (winning_trades / total_trades * 100) if total_trades > 0 else 0,
            'total_pnl': self.total_pnl,
            'active_trade': self.active_trade,
            'current_state': self.trade_phase_engine.current_state if self.trade_phase_engine else None,
            'recent_trades': self.trades_history[-3:] if self.trades_history else []
        }

# ============================================
#  INTEGRATION WITH EXISTING BOT
# ============================================

# تهيئة النظام الجديد
smart_trade_manager = None

def initialize_smart_trade_system(exchange, symbol):
    """تهيئة نظام التداول الذكي"""
    global smart_trade_manager
    smart_trade_manager = SmartTradeManager(exchange, symbol, risk_percent=0.6)
    
    AdvancedLogger.log('INFO', "Smart Trade System Initialized")
    AdvancedLogger.log('INFO', f"Symbol: {symbol} | Risk: 60%")

def integrate_smart_trade_loop():
    """الدورة التجارية الرئيسية مع النظام الذكي"""
    global smart_trade_manager
    
    if smart_trade_manager is None:
        initialize_smart_trade_system(ex, SYMBOL)
    
    AdvancedLogger.log('INFO', "Starting Smart Trade Loop")
    
    while True:
        try:
            # جمع بيانات السوق
            balance = balance_usdt()
            current_price = price_now()
            df = fetch_ohlcv()
            
            if df.empty or current_price is None:
                time.sleep(BASE_SLEEP)
                continue
            
            # تحويل البيانات للأنظمة الجديدة
            candles = []
            for i in range(len(df)):
                candles.append({
                    'open': float(df['open'].iloc[i]),
                    'high': float(df['high'].iloc[i]),
                    'low': float(df['low'].iloc[i]),
                    'close': float(df['close'].iloc[i]),
                    'volume': float(df['volume'].iloc[i])
                })
            
            # إذا كانت هناك صفقة نشطة
            if smart_trade_manager.active_trade:
                # إدارة الصفقة الحالية
                smart_trade_manager.manage_trade(current_price, candles[-10:])
            
            else:
                # قرار الدخول (مثال باستخدام إشارة بسيطة)
                # هنا يمكن دمج نظام القرار الحالي
                should_enter, side, confidence, reason = evaluate_entry_signal(df, current_price)
                
                if should_enter:
                    # محاولة فتح صفقة
                    success = smart_trade_manager.open_trade(
                        side=side,
                        entry_price=current_price,
                        balance=balance or 100.0,
                        entry_zone="TEST_ZONE",
                        confidence=confidence,
                        reason=reason
                    )
                    
                    if success:
                        AdvancedLogger.log('INFO', f"Trade opened successfully | {side} @ {current_price:.4f}")
            
            # النوم حتى التكرار التالي
            time.sleep(BASE_SLEEP)
            
        except Exception as e:
            AdvancedLogger.log('ERROR', f"Trade loop error: {str(e)}")
            time.sleep(BASE_SLEEP)

def evaluate_entry_signal(df, current_price) -> Tuple[bool, str, float, str]:
    """تقييم إشارة الدخول (مثال مبسط)"""
    # هذه دالة مثال - يمكن استبدالها بنظام القرار الحالي
    
    if len(df) < 20:
        return False, "", 0.0, "Insufficient data"
    
    # تحليل بسيط
    closes = df['close'].astype(float).tail(10)
    rsi = calculate_rsi(closes)
    
    # إشارة شراء إذا RSI < 30
    if rsi < 30:
        return True, "buy", 0.7, f"Oversold RSI: {rsi:.1f}"
    
    # إشارة بيع إذا RSI > 70
    elif rsi > 70:
        return True, "sell", 0.7, f"Overbought RSI: {rsi:.1f}"
    
    return False, "", 0.0, "No clear signal"

def calculate_rsi(prices, period=14):
    """حساب RSI مبسط"""
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
#  INTEGRATION HELPERS
# ============================================

# الدوال المساعدة للتكامل مع النظام القديم
def log_i(msg):
    AdvancedLogger.log('INFO', msg)

def log_g(msg):
    AdvancedLogger.log('INFO', msg)  # يمكن تغيير الفئة حسب الحاجة

def log_w(msg):
    AdvancedLogger.log('WARNING', msg)

def log_e(msg):
    AdvancedLogger.log('ERROR', msg)

# ============================================
#  ORIGINAL BOT SETTINGS (محفوظة)
# ============================================

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

LOG_LEGACY = False
LOG_ADDONS = True
EXECUTE_ORDERS = True
SHADOW_MODE_DASHBOARD = False
DRY_RUN = False  # يمكن تغييرها لـ True للاختبار

BOT_VERSION = f"SUI ULTRA PRO AI v9.0 — SMART TRADE MANAGEMENT ENGINE"
print("🚀 Booting:", BOT_VERSION, flush=True)

STATE_PATH = "./bot_state.json"
RESUME_ON_RESTART = True
RESUME_LOOKBACK_SECS = 60 * 60

SYMBOL = os.getenv("SYMBOL", "SUI/USDT:USDT")
INTERVAL = os.getenv("INTERVAL", "15m")
LEVERAGE = 10
RISK_ALLOC = 0.60
BASE_SLEEP = 5
NEAR_CLOSE_S = 1

# تهيئة Exchange
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

# الدوال الأساسية (مبسطة للتكامل)
def balance_usdt():
    if not MODE_LIVE:
        return 100.0
    try:
        b = ex.fetch_balance(params={"type":"swap"})
        return b.get("total",{}).get("USDT") or b.get("free",{}).get("USDT")
    except Exception:
        return None

def price_now():
    try:
        t = ex.fetch_ticker(SYMBOL)
        return t.get("last") or t.get("close")
    except Exception:
        return None

def fetch_ohlcv(limit=100):
    try:
        rows = ex.fetch_ohlcv(SYMBOL, timeframe=INTERVAL, limit=limit, params={"type":"swap"})
        return pd.DataFrame(rows, columns=["time","open","high","low","close","volume"])
    except Exception:
        return pd.DataFrame()

# ============================================
#  MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    AdvancedLogger.log('INFO', f"Starting {BOT_VERSION}")
    AdvancedLogger.log('INFO', f"Exchange: {EXCHANGE_NAME.upper()} | Symbol: {SYMBOL}")
    AdvancedLogger.log('INFO', f"Mode: {'LIVE' if MODE_LIVE else 'PAPER'} | Dry Run: {DRY_RUN}")
    
    # بدء نظام التداول الذكي
    initialize_smart_trade_system(ex, SYMBOL)
    
    # بدء الحلقة التجارية
    import threading
    threading.Thread(target=integrate_smart_trade_loop, daemon=True).start()
    
    # بدء خادم API (مبسط)
    from flask import Flask
    app = Flask(__name__)
    
    @app.route('/')
    def home():
        return f"{BOT_VERSION} - Smart Trade Management Active"
    
    @app.route('/health')
    def health():
        report = smart_trade_manager.get_trade_report() if smart_trade_manager else {}
        return jsonify({
            'status': 'running',
            'bot_version': BOT_VERSION,
            'trade_report': report
        })
    
    @app.route('/trade_report')
    def trade_report():
        if smart_trade_manager:
            return jsonify(smart_trade_manager.get_trade_report())
        return jsonify({'error': 'Trade manager not initialized'})
    
    # تشغيل الخادم
    AdvancedLogger.log('INFO', f"Starting Flask server on port {PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
