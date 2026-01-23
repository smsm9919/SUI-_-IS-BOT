# -*- coding: utf-8 -*-
"""
SUI ULTRA PRO AI BOT - الإصدار المحسن مع نظام SMC المتكامل
• نظام SMC كامل: عرض/طلب، BOS/CHoCH، Order Blocks، FVG
• نظام سيولة ذكي ثلاثي الأبعاد
• كشف فخاخ التلاعب السوقي
• مؤشرات انفجار/انهيار سعري
• أنماط دخول متعددة (7 أنماط مختلفة)
• نظام حماية متكامل مع تأكيد دخول محسن
• مدير رصيد محفظة وربح تراكمي مع أيقونات ملونة
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
from typing import Optional, List, Dict, Tuple, Any, Set
from enum import Enum
import warnings
import threading
warnings.filterwarnings('ignore')

# ============================================
#  CONFIGURATION - الإعدادات
# ============================================

PORT = int(os.environ.get("PORT", 5000))
INITIAL_BALANCE = float(os.environ.get("INITIAL_BALANCE", "1000.0"))
RISK_PERCENT = float(os.environ.get("RISK_PERCENT", "0.6"))
SYMBOL = os.environ.get("SYMBOL", "SUI/USDT:USDT")
INTERVAL = os.environ.get("INTERVAL", "15m")

# ============================================
#  ENHANCED CONSOLE LOGGER - اللوجر المحسن
# ============================================

class AdvancedConsoleColors:
    """ألوان كونسول متقدمة"""
    # ألوان أساسية
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    ITALIC = '\033[3m'
    UNDERLINE = '\033[4m'
    BLINK = '\033[5m'
    REVERSE = '\033[7m'
    HIDDEN = '\033[8m'
    
    # ألوان النص
    class FG:
        BLACK = '\033[30m'
        RED = '\033[31m'
        GREEN = '\033[32m'
        YELLOW = '\033[33m'
        BLUE = '\033[34m'
        MAGENTA = '\033[35m'
        CYAN = '\033[36m'
        WHITE = '\033[37m'
        LIGHT_BLACK = '\033[90m'
        LIGHT_RED = '\033[91m'
        LIGHT_GREEN = '\033[92m'
        LIGHT_YELLOW = '\033[93m'
        LIGHT_BLUE = '\033[94m'
        LIGHT_MAGENTA = '\033[95m'
        LIGHT_CYAN = '\033[96m'
        LIGHT_WHITE = '\033[97m'
    
    # ألوان الخلفية
    class BG:
        BLACK = '\033[40m'
        RED = '\033[41m'
        GREEN = '\033[42m'
        YELLOW = '\033[43m'
        BLUE = '\033[44m'
        MAGENTA = '\033[45m'
        CYAN = '\033[46m'
        WHITE = '\033[47m'
        LIGHT_BLACK = '\033[100m'
        LIGHT_RED = '\033[101m'
        LIGHT_GREEN = '\033[102m'
        LIGHT_YELLOW = '\033[103m'
        LIGHT_BLUE = '\033[104m'
        LIGHT_MAGENTA = '\033[105m'
        LIGHT_CYAN = '\033[106m'
        LIGHT_WHITE = '\033[107m'

class SmartMoneyPatterns(Enum):
    """أنماط Smart Money Concepts"""
    BOS = "BREAK_OF_STRUCTURE"  # كسر الهيكل
    CHOCH = "CHANGE_OF_CHARACTER"  # تغيير الهيكل
    OB = "ORDER_BLOCK"  # كتلة أوامر
    FVG = "FAIR_VALUE_GAP"  # فجوة القيمة العادلة
    LIQUIDITY_SWEEP = "LIQUIDITY_SWEEP"  # مسح السيولة
    LIQUIDITY_GRAB = "LIQUIDITY_GRAB"  # خطف السيولة
    MIT = "MARKET_INDUCED_TRAP"  # فخ محرض سوقي
    MSB = "MARKET_STRUCTURE_BREAK"  # كسر هيكل السوق
    EQH = "EQUAL_HIGHS"  # قمم متساوية
    EQL = "EQUAL_LOWS"  # قيعان متساوية
    DBD = "DISPLACEMENT_BREAKDOWN"  # انهيار الإزاحة
    SMS = "SMART_MONEY_SIGNATURE"  # توقيع المال الذكي

class EnhancedProConsoleLogger:
    """
    لوجر محسن مع دعم كامل لـ SMC
    """
    
    # أيقونات وأنماط SMC
    SMC_ICONS = {
        SmartMoneyPatterns.BOS: "🔄",
        SmartMoneyPatterns.CHOCH: "⚡",
        SmartMoneyPatterns.OB: "📦",
        SmartMoneyPatterns.FVG: "⏳",
        SmartMoneyPatterns.LIQUIDITY_SWEEP: "💧",
        SmartMoneyPatterns.LIQUIDITY_GRAB: "🎯",
        SmartMoneyPatterns.MIT: "🕳️",
        SmartMoneyPatterns.MSB: "🏗️",
        SmartMoneyPatterns.EQH: "⏫",
        SmartMoneyPatterns.EQL: "⏬",
        SmartMoneyPatterns.DBD: "💥",
        SmartMoneyPatterns.SMS: "🧠"
    }
    
    def __init__(self, show_timestamp: bool = True):
        self.show_timestamp = show_timestamp
        
        # إعداد نظام الملفات
        self.setup_file_logging()
    
    def setup_file_logging(self):
        """إعداد تسجيل الملفات"""
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        log_file = os.path.join(log_dir, f"sui_smc_bot_{datetime.now().strftime('%Y%m%d')}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                RotatingFileHandler(log_file, maxBytes=10485760, backupCount=10),
                logging.StreamHandler()
            ]
        )
        self.file_logger = logging.getLogger('SUI_SMC_BOT')
    
    def _format_timestamp(self) -> str:
        """تنسيق الطابع الزمني"""
        now = datetime.now()
        return f"{now.hour:02d}:{now.minute:02d}:{now.second:02d}.{now.microsecond // 1000:03d}"
    
    def log_system(self, message: str, level: str = "INFO"):
        """تسجيل رسالة نظام"""
        level_color = {
            "INFO": AdvancedConsoleColors.FG.CYAN,
            "WARNING": AdvancedConsoleColors.FG.YELLOW,
            "ERROR": AdvancedConsoleColors.FG.RED,
            "SUCCESS": AdvancedConsoleColors.FG.GREEN
        }.get(level, AdvancedConsoleColors.FG.CYAN)
        
        level_icon = {
            "INFO": "ℹ️",
            "WARNING": "⚠️",
            "ERROR": "❌",
            "SUCCESS": "✅"
        }.get(level, "ℹ️")
        
        formatted_message = f"{level_color}{level_icon} {message}{AdvancedConsoleColors.RESET}"
        
        if self.show_timestamp:
            timestamp = f"{AdvancedConsoleColors.FG.LIGHT_BLACK}[{self._format_timestamp()}]{AdvancedConsoleColors.RESET}"
            print(f"{timestamp} {formatted_message}")
        else:
            print(formatted_message)
        
        # تسجيل في الملف
        getattr(self.file_logger, level.lower(), self.file_logger.info)(message)
    
    def log_error(self, message: str, error: Exception, context: str = ""):
        """تسجيل خطأ"""
        error_msg = f"ERROR in {context}: {message} - {str(error)}"
        self.log_system(error_msg, "ERROR")
        self.file_logger.error(error_msg, exc_info=True)
    
    def log_balance(self, balance: float, pnl: float, initial_balance: float):
        """تسجيل رصيد المحفظة والربح التراكمي"""
        balance_color = AdvancedConsoleColors.FG.LIGHT_BLUE
        pnl_color = AdvancedConsoleColors.FG.GREEN if pnl >= 0 else AdvancedConsoleColors.FG.RED
        pnl_icon = "📈" if pnl >= 0 else "📉"
        pnl_percent = (pnl / initial_balance) * 100
        
        balance_line = f"{AdvancedConsoleColors.BOLD}💰 BALANCE:{AdvancedConsoleColors.RESET} {balance_color}{balance:,.2f} USDT{AdvancedConsoleColors.RESET}"
        pnl_line = f"{pnl_icon} {AdvancedConsoleColors.BOLD}CUMULATIVE P&L:{AdvancedConsoleColors.RESET} {pnl_color}{pnl:+,.2f} USDT ({pnl_percent:+.2f}%){AdvancedConsoleColors.RESET}"
        
        # خط فاصل
        separator = f"{AdvancedConsoleColors.FG.LIGHT_BLACK}{'─' * 60}{AdvancedConsoleColors.RESET}"
        
        print(f"\n{separator}")
        print(f"{balance_line} | {pnl_line}")
        print(separator)
    
    def log_portfolio_summary(self, total_trades: int, win_rate: float, total_pnl: float, active_trade: bool):
        """تسجيل ملخص المحفظة"""
        win_rate_color = AdvancedConsoleColors.FG.GREEN if win_rate >= 60 else AdvancedConsoleColors.FG.YELLOW if win_rate >= 50 else AdvancedConsoleColors.FG.RED
        pnl_color = AdvancedConsoleColors.FG.GREEN if total_pnl >= 0 else AdvancedConsoleColors.FG.RED
        
        summary = (
            f"{AdvancedConsoleColors.BOLD}📊 PORTFOLIO SUMMARY:{AdvancedConsoleColors.RESET}\n"
            f"  • {AdvancedConsoleColors.FG.CYAN}Trades:{AdvancedConsoleColors.RESET} {total_trades}\n"
            f"  • {AdvancedConsoleColors.FG.MAGENTA}Win Rate:{AdvancedConsoleColors.RESET} {win_rate_color}{win_rate:.1f}%{AdvancedConsoleColors.RESET}\n"
            f"  • {AdvancedConsoleColors.FG.YELLOW}Total P&L:{AdvancedConsoleColors.RESET} {pnl_color}{total_pnl:+,.2f}%{AdvancedConsoleColors.RESET}\n"
            f"  • {AdvancedConsoleColors.FG.LIGHT_BLUE}Active Trade:{AdvancedConsoleColors.RESET} {'✅ Yes' if active_trade else '❌ No'}"
        )
        
        print(f"\n{summary}")
    
    def log_smc_pattern(self, pattern: SmartMoneyPatterns, details: Dict, confidence: float = 0.0):
        """تسجيل نمط SMC"""
        icon = self.SMC_ICONS.get(pattern, "❓")
        
        # تحديد لون الثقة
        if confidence >= 0.8:
            conf_color = AdvancedConsoleColors.FG.LIGHT_GREEN
            conf_icon = "✅"
        elif confidence >= 0.6:
            conf_color = AdvancedConsoleColors.FG.YELLOW
            conf_icon = "⚠️"
        else:
            conf_color = AdvancedConsoleColors.FG.LIGHT_RED
            conf_icon = "❌"
        
        # بناء الرسالة
        message_parts = [
            f"{AdvancedConsoleColors.BOLD}{icon} {pattern.value}{AdvancedConsoleColors.RESET}",
            f"{conf_color}{conf_icon} {confidence:.2f}{AdvancedConsoleColors.RESET}"
        ]
        
        # إضافة التفاصيل
        details_str = " | ".join([f"{k}: {v}" for k, v in details.items()])
        if details_str:
            message_parts.append(f"{AdvancedConsoleColors.DIM}{details_str}{AdvancedConsoleColors.RESET}")
        
        full_message = " | ".join(message_parts)
        
        if self.show_timestamp:
            timestamp = f"{AdvancedConsoleColors.FG.LIGHT_BLACK}[{self._format_timestamp()}]{AdvancedConsoleColors.RESET}"
            print(f"{timestamp} {full_message}")
        else:
            print(full_message)
        
        # تسجيل في الملف
        self.file_logger.info(f"SMC | {pattern.value} | Confidence: {confidence:.2f} | Details: {details}")
    
    def log_liquidity_event(self, event_type: str, zone_type: str, price: float, volume: float, reason: str):
        """تسجيل حدث سيولة"""
        # ألوان حسب نوع الحدث
        if "SWEEP" in event_type:
            color = AdvancedConsoleColors.FG.LIGHT_CYAN
            icon = "💧"
        elif "GRAB" in event_type:
            color = AdvancedConsoleColors.FG.LIGHT_MAGENTA
            icon = "🎯"
        elif "TRAP" in event_type:
            color = AdvancedConsoleColors.FG.RED
            icon = "🕳️"
        else:
            color = AdvancedConsoleColors.FG.YELLOW
            icon = "💎"
        
        message = (
            f"{color}{icon} {event_type} | "
            f"Zone: {zone_type} | "
            f"Price: {price:.4f} | "
            f"Volume: {volume:.0f} | "
            f"{AdvancedConsoleColors.FG.LIGHT_YELLOW}WHY: {reason}{AdvancedConsoleColors.RESET}"
        )
        
        if self.show_timestamp:
            timestamp = f"{AdvancedConsoleColors.FG.LIGHT_BLACK}[{self._format_timestamp()}]{AdvancedConsoleColors.RESET}"
            print(f"{timestamp} {message}")
        else:
            print(message)
        
        self.file_logger.info(f"LIQUIDITY | {event_type} | Zone: {zone_type} | Price: {price:.4f} | Reason: {reason}")
    
    def log_explosion_signal(self, direction: str, intensity: float, trigger: str, target: float, confidence: float):
        """تسجيل إشارة انفجار/انهيار سعري"""
        if direction.upper() == "BULLISH":
            color = AdvancedConsoleColors.FG.GREEN
            icon = "🚀"
            dir_text = "EXPLOSION"
        else:
            color = AdvancedConsoleColors.FG.RED
            icon = "💥"
            dir_text = "COLLAPSE"
        
        # لون الشدة
        if intensity >= 0.8:
            intensity_color = AdvancedConsoleColors.FG.LIGHT_RED
            intensity_icon = "🔥"
        elif intensity >= 0.5:
            intensity_color = AdvancedConsoleColors.FG.YELLOW
            intensity_icon = "⚡"
        else:
            intensity_color = AdvancedConsoleColors.FG.LIGHT_BLUE
            intensity_icon = "✨"
        
        # لون الثقة
        if confidence >= 0.8:
            conf_color = AdvancedConsoleColors.FG.LIGHT_GREEN
        elif confidence >= 0.6:
            conf_color = AdvancedConsoleColors.FG.YELLOW
        else:
            conf_color = AdvancedConsoleColors.FG.LIGHT_RED
        
        message = (
            f"{color}{icon} {dir_text} SIGNAL | "
            f"{intensity_color}Intensity: {intensity_icon}{intensity:.2f}{AdvancedConsoleColors.RESET} | "
            f"Trigger: {trigger} | "
            f"Target: {target:.4f} | "
            f"{conf_color}Confidence: {confidence:.2f}{AdvancedConsoleColors.RESET}"
        )
        
        if self.show_timestamp:
            timestamp = f"{AdvancedConsoleColors.FG.LIGHT_BLACK}[{self._format_timestamp()}]{AdvancedConsoleColors.RESET}"
            print(f"{timestamp} {message}")
        else:
            print(message)
        
        self.file_logger.info(f"EXPLOSION | {direction} | Intensity: {intensity:.2f} | Trigger: {trigger} | Confidence: {confidence:.2f}")

# إنشاء كائن اللوجر المحسن
logger = EnhancedProConsoleLogger(show_timestamp=True)

# ============================================
#  ENHANCED ENTRY VALIDATOR - مدقق الدخول المحسن
# ============================================

class EnhancedEntryValidator:
    """مدقق دخول محسن مع بوابات تأكيد متعددة"""
    
    def __init__(self):
        self.confirmation_rules = {
            'mandatory_gates': [
                'price_structure_alignment',
                'volume_confirmation',
                'market_context_filter',
                'risk_reward_check',
                'pattern_confirmation'
            ],
            'optional_gates': [
                'momentum_confirmation',
                'time_of_day_filter',
                'volatility_check'
            ]
        }
    
    def validate_entry(self, scenario: Dict, candles: List[Dict], 
                      smc_analysis: Dict) -> Tuple[bool, str]:
        """
        التحقق من صحة سيناريو الدخول مع جميع البوابات
        
        Returns:
            (is_valid, reason)
        """
        # 1. بوابة الهيكل السعري
        if not self._check_price_structure(scenario, candles):
            return False, "Price structure not aligned"
        
        # 2. بوابة تأكيد الحجم
        if not self._check_volume_confirmation(scenario, candles):
            return False, "Volume not confirming"
        
        # 3. بوابة سياق السوق
        if not self._check_market_context(scenario, smc_analysis):
            return False, "Market context unfavorable"
        
        # 4. بوابة نسبة المخاطرة/العائد
        if not self._check_risk_reward(scenario):
            return False, "Risk/Reward ratio too low"
        
        # 5. بوابة تأكيد النمط
        if not self._check_pattern_confirmation(scenario, candles):
            return False, "Pattern not confirmed"
        
        return True, "All gates passed"
    
    def _check_price_structure(self, scenario: Dict, candles: List[Dict]) -> bool:
        """التحقق من محاذاة الهيكل السعري"""
        current_price = candles[-1]['close']
        entry_type = scenario['entry_type']
        
        # لمنع الدخول في مناطق معاكسة
        if entry_type == 'BUY':
            # تأكيد أن السعر فوق مستويات دعم مهمة
            recent_lows = [c['low'] for c in candles[-5:]]
            support_level = min(recent_lows)
            
            # الحصول على آخر 3 شمعات قبل الحالية
            last_3_candles = candles[-4:-1] if len(candles) >= 4 else candles[:-1]
            
            # شرط: الشمعة الحالية تغلق فوق قمة الشمعة السابقة
            if len(candles) >= 2:
                if current_price <= candles[-2]['high']:
                    return False
            
            # شرط: وجود شمعة إزاحة (Displacement) قبل الدخول
            displacement_found = False
            for i in range(2, min(6, len(candles))):
                idx = -i
                candle = candles[idx]
                prev_candle = candles[idx-1] if idx-1 >= 0 else None
                
                if prev_candle:
                    # شمعة إزاحة صعودية: شمعة خضراء كبيرة
                    if (candle['close'] > candle['open'] * 1.005 and
                        (candle['close'] - candle['open']) > (candle['high'] - candle['low']) * 0.7):
                        displacement_found = True
                        break
            
            return current_price > support_level and displacement_found
        
        else:  # SELL
            # تأكيد أن السعر تحت مستويات مقاومة مهمة
            recent_highs = [c['high'] for c in candles[-5:]]
            resistance_level = max(recent_highs)
            
            # شرط: الشمعة الحالية تغلق تحت قاع الشمعة السابقة
            if len(candles) >= 2:
                if current_price >= candles[-2]['low']:
                    return False
            
            # شرط: وجود شمعة إزاحة هابطة
            displacement_found = False
            for i in range(2, min(6, len(candles))):
                idx = -i
                candle = candles[idx]
                prev_candle = candles[idx-1] if idx-1 >= 0 else None
                
                if prev_candle:
                    # شمعة إزاحة هابطة: شمعة حمراء كبيرة
                    if (candle['close'] < candle['open'] * 0.995 and
                        (candle['open'] - candle['close']) > (candle['high'] - candle['low']) * 0.7):
                        displacement_found = True
                        break
            
            return current_price < resistance_level and displacement_found
    
    def _check_volume_confirmation(self, scenario: Dict, candles: List[Dict]) -> bool:
        """تأكيد الحجم"""
        if len(candles) < 10:
            return False
        
        current_volume = candles[-1]['volume']
        avg_volume = np.mean([c['volume'] for c in candles[-10:-1]])
        
        entry_type = scenario['entry_type']
        
        # حجم الشمعة الحالية يجب أن يكون أعلى من المتوسط
        if current_volume < avg_volume * 0.8:
            return False
        
        # تحليل نمط الحجم
        volume_pattern = self._analyze_volume_pattern(candles[-5:])
        
        if entry_type == 'BUY':
            # في الشراء: نفضل حجم متزايد في الشمعات الصعودية
            return volume_pattern in ['INCREASING', 'SPIKE']
        else:
            # في البيع: نفضل حجم متزايد في الشمعات الهابطة
            return volume_pattern in ['INCREASING', 'SPIKE']
    
    def _analyze_volume_pattern(self, candles: List[Dict]) -> str:
        """تحليل نمط الحجم"""
        if len(candles) < 3:
            return "UNKNOWN"
        
        volumes = [c['volume'] for c in candles]
        
        if volumes[-1] > max(volumes[:-1]) * 1.5:
            return "SPIKE"
        elif volumes[-1] < min(volumes[:-1]) * 0.7:
            return "DRYUP"
        elif all(v > volumes[i-1] for i, v in enumerate(volumes[1:], 1)):
            return "INCREASING"
        elif all(v < volumes[i-1] for i, v in enumerate(volumes[1:], 1)):
            return "DECREASING"
        else:
            return "CONGESTION"
    
    def _check_market_context(self, scenario: Dict, smc_analysis: Dict) -> bool:
        """فحص سياق السوق"""
        trend_structure = smc_analysis.get('trend_structure', {})
        market_cycle = smc_analysis.get('market_cycle', {})
        entry_type = scenario['entry_type']
        
        trend = trend_structure.get('trend', 'SIDEWAYS')
        cycle = market_cycle.get('cycle', 'UNKNOWN')
        phase = market_cycle.get('phase', 'UNKNOWN')
        
        # 1. تحقق من الاتجاه
        if entry_type == 'BUY' and trend == 'BEARISH':
            # في اتجاه هابط، نشتري فقط في مراكز معينة
            if phase not in ['ACCUMULATION', 'OVERSOLD']:
                return False
        
        elif entry_type == 'SELL' and trend == 'BULLISH':
            # في اتجاه صاعد، نبيع فقط في مراكز معينة
            if phase not in ['DISTRIBUTION', 'OVERBOUGHT']:
                return False
        
        # 2. تحقق من مرحلة الدورة
        if cycle == 'OVERBOUGHT' and entry_type == 'BUY':
            return False
        
        if cycle == 'OVERSOLD' and entry_type == 'SELL':
            return False
        
        return True
    
    def _check_risk_reward(self, scenario: Dict) -> bool:
        """فحص نسبة المخاطرة/العائد"""
        min_rr = 1.5  # أقل نسبة مقبولة 1:1.5
        
        if scenario['risk_reward'] < min_rr:
            return False
        
        # نسبة المخاطرة لا تزيد عن 2%
        risk_pct = abs(scenario['entry_price'] - scenario['stop_loss']) / scenario['entry_price']
        if risk_pct > 0.02:
            return False
        
        return True
    
    def _check_pattern_confirmation(self, scenario: Dict, candles: List[Dict]) -> bool:
        """تأكيد النمط السعري"""
        entry_type = scenario['entry_type']
        scenario_type = scenario['type']
        
        # شروط تأكيد عامة لكل الأنماط
        current_candle = candles[-1]
        prev_candle = candles[-2] if len(candles) >= 2 else None
        
        if not prev_candle:
            return False
        
        if entry_type == 'BUY':
            # تأكيد شراء: شمعة خضراء تغلق فوق فتحها
            if current_candle['close'] <= current_candle['open']:
                return False
            
            # تأكيد إضافي: الإغلاق فوق منتصف مدى الشمعة
            candle_mid = (current_candle['high'] + current_candle['low']) / 2
            if current_candle['close'] < candle_mid:
                return False
            
            # نموذج Engulfing صعودي (اختياري)
            if (current_candle['close'] > prev_candle['open'] and
                current_candle['open'] < prev_candle['close'] and
                prev_candle['close'] < prev_candle['open']):
                return True
            
        else:  # SELL
            # تأكيد بيع: شمعة حمراء تغلق تحت فتحها
            if current_candle['close'] >= current_candle['open']:
                return False
            
            # تأكيد إضافي: الإغلاق تحت منتصف مدى الشمعة
            candle_mid = (current_candle['high'] + current_candle['low']) / 2
            if current_candle['close'] > candle_mid:
                return False
            
            # نموذج Engulfing هابط (اختياري)
            if (current_candle['close'] < prev_candle['open'] and
                current_candle['open'] > prev_candle['close'] and
                prev_candle['close'] > prev_candle['open']):
                return True
        
        # إذا لم يكن هناك نمط engulfing، نكتفي بالشمعة المؤكدة
        return True

# ============================================
#  ADVANCED SMC ANALYZER - محلل SMC المتقدم
# ============================================

class AdvancedSMCAnalyzer:
    """محلل SMC (Smart Money Concepts) المتكامل"""
    
    def __init__(self, logger: EnhancedProConsoleLogger):
        self.logger = logger
        self.supply_zones = []
        self.demand_zones = []
        self.order_blocks = []
        self.fvg_zones = []
        self.liquidity_zones = []
        
    def analyze_candles(self, candles: List[Dict]) -> Dict[str, Any]:
        """
        تحليل شامل للشموع باستخدام مفاهيم SMC
        
        Returns:
            Dict يحتوي على جميع الأنماط المكتشفة
        """
        if len(candles) < 50:
            return {"error": "Insufficient data for SMC analysis"}
        
        try:
            # تحويل القائمة إلى DataFrame للتحليل
            df = pd.DataFrame(candles)
            
            # اكتشاف جميع الأنماط
            analysis_results = {
                "supply_zones": self._detect_supply_zones(candles),
                "demand_zones": self._detect_demand_zones(candles),
                "bos_signals": self._detect_bos_patterns(candles),
                "choch_signals": self._detect_choch_patterns(candles),
                "order_blocks": self._detect_order_blocks(candles),
                "fvg_zones": self._detect_fvg_zones(candles),
                "liquidity_zones": self._detect_liquidity_zones(candles),
                "manipulation_signals": self._detect_manipulation_patterns(candles),
                "explosion_signals": self._detect_explosion_patterns(candles),
                "trend_structure": self._analyze_trend_structure(candles),
                "market_cycle": self._analyze_market_cycle(candles)
            }
            
            # تسجيل الأنماط المهمة
            self._log_significant_patterns(analysis_results, candles)
            
            return analysis_results
            
        except Exception as e:
            self.logger.file_logger.error(f"SMC analysis error: {str(e)}", exc_info=True)
            return {"error": str(e)}
    
    def _detect_supply_zones(self, candles: List[Dict]) -> List[Dict]:
        """اكتشاف مناطق العرض (Supply Zones)"""
        zones = []
        
        for i in range(2, len(candles) - 2):
            current = candles[i]
            prev = candles[i-1]
            prev2 = candles[i-2]
            
            # نموذج العرض: شمعة صعودية قوية ثم انعكاس
            if (prev['close'] > prev['open'] * 1.01 and  # شمعة صعودية قوية
                current['close'] < current['open'] and  # شمعة هابطة
                current['high'] <= prev['high'] and  # عدم كسر قمة السابقة
                current['volume'] > prev['volume'] * 0.8):  # حجم جيد
                
                zone = {
                    'type': 'SUPPLY',
                    'price_level': current['high'],
                    'strength': self._calculate_zone_strength(candles, i, 'supply'),
                    'formation_time': i,
                    'volume_profile': self._analyze_volume_at_level(candles, current['high']),
                    'test_count': 0,
                    'last_test': i
                }
                
                # تجنب المناطق المتشابهة
                if not self._is_duplicate_zone(zones, zone):
                    zones.append(zone)
        
        return sorted(zones, key=lambda x: x['strength'], reverse=True)[:10]  # أهم 10 مناطق
    
    def _detect_demand_zones(self, candles: List[Dict]) -> List[Dict]:
        """اكتشاف مناطق الطلب (Demand Zones)"""
        zones = []
        
        for i in range(2, len(candles) - 2):
            current = candles[i]
            prev = candles[i-1]
            prev2 = candles[i-2]
            
            # نموذج الطلب: شمعة هابطة قوية ثم انعكاس
            if (prev['close'] < prev['open'] * 0.99 and  # شمعة هابطة قوية
                current['close'] > current['open'] and  # شمعة صعودية
                current['low'] >= prev['low'] and  # عدم كسر قاع السابقة
                current['volume'] > prev['volume'] * 0.8):  # حجم جيد
                
                zone = {
                    'type': 'DEMAND',
                    'price_level': current['low'],
                    'strength': self._calculate_zone_strength(candles, i, 'demand'),
                    'formation_time': i,
                    'volume_profile': self._analyze_volume_at_level(candles, current['low']),
                    'test_count': 0,
                    'last_test': i
                }
                
                # تجنب المناطق المتشابهة
                if not self._is_duplicate_zone(zones, zone):
                    zones.append(zone)
        
        return sorted(zones, key=lambda x: x['strength'], reverse=True)[:10]
    
    def _detect_bos_patterns(self, candles: List[Dict]) -> List[Dict]:
        """اكتشاف أنماط كسر الهيكل (BOS)"""
        patterns = []
        
        for i in range(10, len(candles) - 5):
            # البحث عن Higher High في صعودي
            if (candles[i]['high'] > max([c['high'] for c in candles[i-5:i]]) and
                candles[i-1]['high'] < candles[i-2]['high'] and
                candles[i]['close'] > candles[i-1]['close']):
                
                pattern = {
                    'type': 'BOS_BULLISH',
                    'index': i,
                    'price': candles[i]['close'],
                    'previous_structure': self._get_previous_structure(candles, i, 'high'),
                    'volume_confirmation': candles[i]['volume'] > np.mean([c['volume'] for c in candles[i-3:i]]) * 1.2,
                    'strength': self._calculate_bos_strength(candles, i, 'bullish'),
                    'momentum': self._calculate_momentum(candles, i, 3)
                }
                patterns.append(pattern)
            
            # البحث عن Lower Low في هابطي
            elif (candles[i]['low'] < min([c['low'] for c in candles[i-5:i]]) and
                  candles[i-1]['low'] > candles[i-2]['low'] and
                  candles[i]['close'] < candles[i-1]['close']):
                
                pattern = {
                    'type': 'BOS_BEARISH',
                    'index': i,
                    'price': candles[i]['close'],
                    'previous_structure': self._get_previous_structure(candles, i, 'low'),
                    'volume_confirmation': candles[i]['volume'] > np.mean([c['volume'] for c in candles[i-3:i]]) * 1.2,
                    'strength': self._calculate_bos_strength(candles, i, 'bearish'),
                    'momentum': self._calculate_momentum(candles, i, 3)
                }
                patterns.append(pattern)
        
        return patterns
    
    def _detect_choch_patterns(self, candles: List[Dict]) -> List[Dict]:
        """اكتشاف أنماط تغيير الهيكل (CHoCH)"""
        patterns = []
        
        for i in range(15, len(candles) - 10):
            # CHoCH من صعودي إلى هابطي
            if (self._is_uptrend(candles[:i]) and
                candles[i]['low'] < min([c['low'] for c in candles[i-3:i]]) and
                candles[i]['close'] < candles[i-1]['close'] and
                candles[i]['volume'] > candles[i-1]['volume'] * 1.5):
                
                pattern = {
                    'type': 'CHOCH_BEARISH',
                    'index': i,
                    'price': candles[i]['close'],
                    'trend_change': 'UP_TO_DOWN',
                    'volume_spike': candles[i]['volume'] / np.mean([c['volume'] for c in candles[i-5:i]]) if i > 5 else 1,
                    'momentum_shift': self._calculate_momentum_shift(candles, i),
                    'structure_break': True
                }
                patterns.append(pattern)
            
            # CHoCH من هابطي إلى صعودي
            elif (self._is_downtrend(candles[:i]) and
                  candles[i]['high'] > max([c['high'] for c in candles[i-3:i]]) and
                  candles[i]['close'] > candles[i-1]['close'] and
                  candles[i]['volume'] > candles[i-1]['volume'] * 1.5):
                
                pattern = {
                    'type': 'CHOCH_BULLISH',
                    'index': i,
                    'price': candles[i]['close'],
                    'trend_change': 'DOWN_TO_UP',
                    'volume_spike': candles[i]['volume'] / np.mean([c['volume'] for c in candles[i-5:i]]) if i > 5 else 1,
                    'momentum_shift': self._calculate_momentum_shift(candles, i),
                    'structure_break': True
                }
                patterns.append(pattern)
        
        return patterns
    
    def _detect_order_blocks(self, candles: List[Dict]) -> List[Dict]:
        """اكتشاف كتل الأوامر (Order Blocks)"""
        ob_blocks = []
        
        for i in range(3, len(candles) - 2):
            current = candles[i]
            prev = candles[i-1]
            prev2 = candles[i-2]
            
            # Order Block هابط: شمعة صعودية كبيرة تليها شمعة هابطة
            if (prev['close'] > prev['open'] * 1.005 and  # شمعة صعودية
                current['close'] < current['open'] and  # شمعة هابطة
                current['low'] < prev['low'] and  # كسر قاع السابقة
                current['volume'] > prev['volume'] * 0.7):  # حجم معقول
                
                ob = {
                    'type': 'BEARISH_OB',
                    'index': i,
                    'price_range': (prev['low'], prev['high']),
                    'mid_price': (prev['low'] + prev['high']) / 2,
                    'volume': prev['volume'],
                    'strength': self._calculate_ob_strength(candles, i, 'bearish'),
                    'test_count': 0
                }
                ob_blocks.append(ob)
            
            # Order Block صعودي: شمعة هابطة كبيرة تليها شمعة صعودية
            elif (prev['close'] < prev['open'] * 0.995 and  # شمعة هابطة
                  current['close'] > current['open'] and  # شمعة صعودية
                  current['high'] > prev['high'] and  # كسر قمة السابقة
                  current['volume'] > prev['volume'] * 0.7):  # حجم معقول
                
                ob = {
                    'type': 'BULLISH_OB',
                    'index': i,
                    'price_range': (prev['low'], prev['high']),
                    'mid_price': (prev['low'] + prev['high']) / 2,
                    'volume': prev['volume'],
                    'strength': self._calculate_ob_strength(candles, i, 'bullish'),
                    'test_count': 0
                }
                ob_blocks.append(ob)
        
        return sorted(ob_blocks, key=lambda x: x['strength'], reverse=True)[:15]
    
    def _detect_fvg_zones(self, candles: List[Dict]) -> List[Dict]:
        """اكتشاف فجوات القيمة العادلة (Fair Value Gaps)"""
        fvg_list = []
        
        for i in range(1, len(candles) - 1):
            current = candles[i]
            prev = candles[i-1]
            
            # FVG هابطة: قاع السابقة > قمة اللاحقة
            if prev['low'] > current['high']:
                fvg = {
                    'type': 'BEARISH_FVG',
                    'index': i,
                    'gap_range': (current['high'], prev['low']),
                    'gap_size': prev['low'] - current['high'],
                    'volume': (prev['volume'] + current['volume']) / 2,
                    'strength': self._calculate_fvg_strength(candles, i, 'bearish'),
                    'filled': False
                }
                fvg_list.append(fvg)
            
            # FVG صعودية: قمة السابقة < قاع اللاحقة
            elif prev['high'] < current['low']:
                fvg = {
                    'type': 'BULLISH_FVG',
                    'index': i,
                    'gap_range': (prev['high'], current['low']),
                    'gap_size': current['low'] - prev['high'],
                    'volume': (prev['volume'] + current['volume']) / 2,
                    'strength': self._calculate_fvg_strength(candles, i, 'bullish'),
                    'filled': False
                }
                fvg_list.append(fvg)
        
        return sorted(fvg_list, key=lambda x: x['gap_size'], reverse=True)[:10]
    
    def _detect_liquidity_zones(self, candles: List[Dict]) -> List[Dict]:
        """اكتشاف مناطق السيولة الذكية"""
        liquidity_zones = []
        
        # اكتشاف Equal Highs
        for i in range(5, len(candles) - 5):
            highs = [c['high'] for c in candles[i-5:i]]
            if abs(max(highs) - candles[i]['high']) / candles[i]['high'] < 0.001:
                zone = {
                    'type': 'LIQUIDITY_EQUAL_HIGHS',
                    'price': candles[i]['high'],
                    'index': i,
                    'volume_cluster': sum([c['volume'] for c in candles[i-5:i+1]]) / 6,
                    'density': self._calculate_liquidity_density(candles, i, 'high'),
                    'purpose': 'STOP_HUNT' if candles[i]['close'] < candles[i]['open'] else 'LIQUIDITY_POOL'
                }
                liquidity_zones.append(zone)
        
        # اكتشاف Equal Lows
        for i in range(5, len(candles) - 5):
            lows = [c['low'] for c in candles[i-5:i]]
            if abs(min(lows) - candles[i]['low']) / candles[i]['low'] < 0.001:
                zone = {
                    'type': 'LIQUIDITY_EQUAL_LOWS',
                    'price': candles[i]['low'],
                    'index': i,
                    'volume_cluster': sum([c['volume'] for c in candles[i-5:i+1]]) / 6,
                    'density': self._calculate_liquidity_density(candles, i, 'low'),
                    'purpose': 'STOP_HUNT' if candles[i]['close'] > candles[i]['open'] else 'LIQUIDITY_POOL'
                }
                liquidity_zones.append(zone)
        
        # اكتشاف مناطق سحب السيولة
        for i in range(3, len(candles) - 3):
            if self._is_liquidity_sweep(candles, i):
                zone = {
                    'type': 'LIQUIDITY_SWEEP',
                    'price': candles[i]['high'] if candles[i]['close'] < candles[i]['open'] else candles[i]['low'],
                    'index': i,
                    'direction': 'UP' if candles[i]['close'] < candles[i]['open'] else 'DOWN',
                    'volume': candles[i]['volume'],
                    'intensity': candles[i]['volume'] / np.mean([c['volume'] for c in candles[i-3:i]]) if i > 3 else 1
                }
                liquidity_zones.append(zone)
        
        return sorted(liquidity_zones, key=lambda x: x.get('density', 0) or x.get('volume', 0), reverse=True)[:10]
    
    def _detect_manipulation_patterns(self, candles: List[Dict]) -> List[Dict]:
        """اكتشاف أنماط التلاعب السوقي"""
        manipulation_signals = []
        
        for i in range(5, len(candles) - 5):
            # فخاخ الشراء (Buy Traps): كسر وهمي لأعلى ثم انعكاس
            if (candles[i]['high'] > max([c['high'] for c in candles[i-5:i]]) and
                candles[i]['close'] < candles[i]['open'] and
                candles[i+1]['close'] < candles[i+1]['open']):
                
                signal = {
                    'type': 'BUY_TRAP',
                    'index': i,
                    'trap_price': candles[i]['high'],
                    'fakeout_direction': 'UP',
                    'reversal_direction': 'DOWN',
                    'volume_pattern': self._analyze_volume_pattern(candles, i, 3),
                    'confidence': self._calculate_trap_confidence(candles, i, 'buy')
                }
                manipulation_signals.append(signal)
            
            # فخاخ البيع (Sell Traps): كسر وهمي لأسفل ثم انعكاس
            elif (candles[i]['low'] < min([c['low'] for c in candles[i-5:i]]) and
                  candles[i]['close'] > candles[i]['open'] and
                  candles[i+1]['close'] > candles[i+1]['open']):
                
                signal = {
                    'type': 'SELL_TRAP',
                    'index': i,
                    'trap_price': candles[i]['low'],
                    'fakeout_direction': 'DOWN',
                    'reversal_direction': 'UP',
                    'volume_pattern': self._analyze_volume_pattern(candles, i, 3),
                    'confidence': self._calculate_trap_confidence(candles, i, 'sell')
                }
                manipulation_signals.append(signal)
        
        return manipulation_signals
    
    def _detect_explosion_patterns(self, candles: List[Dict]) -> List[Dict]:
        """اكتشاف أنماط الانفجار/الانهيار السعري"""
        explosion_signals = []
        
        for i in range(3, len(candles) - 3):
            current = candles[i]
            prev = candles[i-1]
            prev2 = candles[i-2]
            
            # انفجار صعودي: شمعة كبيرة صعودية مع حجم هائل
            if (current['close'] > current['open'] * 1.02 and  # حركة 2%+
                current['volume'] > np.mean([c['volume'] for c in candles[max(0, i-10):i]]) * 3 and  # حجم 3x متوسط
                current['close'] > max([c['close'] for c in candles[i-3:i]])):  # أعلى إغلاق
                
                signal = {
                    'type': 'BULLISH_EXPLOSION',
                    'index': i,
                    'trigger_price': current['close'],
                    'momentum': (current['close'] - current['open']) / current['open'],
                    'volume_intensity': current['volume'] / np.mean([c['volume'] for c in candles[max(0, i-10):i]]),
                    'target_price': current['close'] * 1.03,  # هدف 3%
                    'stop_loss': current['open'] * 0.99,
                    'confidence': min(0.9, current['volume'] / np.mean([c['volume'] for c in candles[max(0, i-10):i]]) / 3)
                }
                explosion_signals.append(signal)
            
            # انهيار هابط: شمعة كبيرة هابطة مع حجم هائل
            elif (current['close'] < current['open'] * 0.98 and  # حركة 2%-
                  current['volume'] > np.mean([c['volume'] for c in candles[max(0, i-10):i]]) * 3 and
                  current['close'] < min([c['close'] for c in candles[i-3:i]])):  # أقل إغلاق
                
                signal = {
                    'type': 'BEARISH_COLLAPSE',
                    'index': i,
                    'trigger_price': current['close'],
                    'momentum': (current['open'] - current['close']) / current['open'],
                    'volume_intensity': current['volume'] / np.mean([c['volume'] for c in candles[max(0, i-10):i]]),
                    'target_price': current['close'] * 0.97,  # هدف 3%
                    'stop_loss': current['open'] * 1.01,
                    'confidence': min(0.9, current['volume'] / np.mean([c['volume'] for c in candles[max(0, i-10):i]]) / 3)
                }
                explosion_signals.append(signal)
        
        return sorted(explosion_signals, key=lambda x: x['confidence'], reverse=True)[:5]
    
    def _analyze_trend_structure(self, candles: List[Dict]) -> Dict[str, Any]:
        """تحليل هيكل الاتجاه"""
        if len(candles) < 20:
            return {"trend": "UNKNOWN", "strength": 0}
        
        # حساب المتوسطات المتحركة
        closes = [c['close'] for c in candles]
        sma_20 = np.mean(closes[-20:])
        sma_50 = np.mean(closes[-min(50, len(closes)):])
        
        # تحديد الاتجاه
        if sma_20 > sma_50 and closes[-1] > sma_20:
            trend = "BULLISH"
            strength = (sma_20 - sma_50) / sma_50
        elif sma_20 < sma_50 and closes[-1] < sma_20:
            trend = "BEARISH"
            strength = (sma_50 - sma_20) / sma_20
        else:
            trend = "SIDEWAYS"
            strength = 0
        
        # تحليل Higher Highs/Lower Lows
        recent_highs = [c['high'] for c in candles[-10:]]
        recent_lows = [c['low'] for c in candles[-10:]]
        
        hh_count = sum(1 for i in range(1, len(recent_highs)) if recent_highs[i] > recent_highs[i-1])
        ll_count = sum(1 for i in range(1, len(recent_lows)) if recent_lows[i] < recent_lows[i-1])
        
        return {
            "trend": trend,
            "strength": abs(strength),
            "sma_20": sma_20,
            "sma_50": sma_50,
            "hh_count": hh_count,
            "ll_count": ll_count,
            "structure": "HH_HL" if hh_count > ll_count else "LH_LL" if ll_count > hh_count else "RANGING"
        }
    
    def _analyze_market_cycle(self, candles: List[Dict]) -> Dict[str, Any]:
        """تحليل دورة السوق"""
        if len(candles) < 50:
            return {"cycle": "UNKNOWN", "phase": "UNKNOWN"}
        
        # حساب مؤشر RSI
        closes = [c['close'] for c in candles]
        rsi = self._calculate_rsi(closes, 14)
        
        # تحديد مرحلة الدورة
        if rsi > 70:
            cycle = "OVERBOUGHT"
            phase = "DISTRIBUTION"
        elif rsi < 30:
            cycle = "OVERSOLD"
            phase = "ACCUMULATION"
        elif rsi > 50:
            cycle = "BULLISH"
            phase = "MARKUP"
        else:
            cycle = "BEARISH"
            phase = "MARKDOWN"
        
        # تحليل حجم التداول
        volumes = [c['volume'] for c in candles[-20:]]
        volume_trend = "INCREASING" if volumes[-1] > np.mean(volumes[:-1]) else "DECREASING"
        
        return {
            "cycle": cycle,
            "phase": phase,
            "rsi": rsi,
            "volume_trend": volume_trend,
            "volatility": np.std(closes[-20:]) / np.mean(closes[-20:]) if len(closes) >= 20 else 0
        }
    
    # ============================================
    #  HELPER METHODS - الدوال المساعدة
    # ============================================
    
    def _calculate_zone_strength(self, candles: List[Dict], idx: int, zone_type: str) -> float:
        """حساب قوة المنطقة"""
        strength = 0.0
        
        # الأساس: حجم الشمعة
        volume_ratio = candles[idx]['volume'] / np.mean([c['volume'] for c in candles[max(0, idx-10):idx]])
        strength += min(volume_ratio / 3, 0.4)  # 40% للحجم
        
        # المسافة من السعر الحالي
        current_price = candles[-1]['close']
        zone_price = candles[idx]['high'] if zone_type == 'supply' else candles[idx]['low']
        distance_ratio = abs(current_price - zone_price) / current_price
        
        if distance_ratio < 0.02:  # قريبة (2%)
            strength += 0.3
        elif distance_ratio < 0.05:  # متوسطة (5%)
            strength += 0.2
        else:  # بعيدة
            strength += 0.1
        
        # عدد الاختبارات السابقة
        test_count = self._count_zone_tests(candles, idx, zone_type, zone_price)
        strength += min(test_count * 0.1, 0.3)  # 30% للاختبارات
        
        return min(strength, 1.0)
    
    def _is_duplicate_zone(self, zones: List[Dict], new_zone: Dict) -> bool:
        """التحقق إذا كانت المنطقة مكررة"""
        for zone in zones:
            if abs(zone['price_level'] - new_zone['price_level']) / new_zone['price_level'] < 0.002:  # 0.2%
                return True
        return False
    
    def _analyze_volume_at_level(self, candles: List[Dict], price: float) -> Dict[str, Any]:
        """تحليل الحجم عند مستوى سعري"""
        volume_cluster = 0
        touch_count = 0
        
        for candle in candles[-50:]:
            if price >= candle['low'] and price <= candle['high']:
                volume_cluster += candle['volume']
                touch_count += 1
        
        return {
            'total_volume': volume_cluster,
            'touch_count': touch_count,
            'avg_volume_per_touch': volume_cluster / max(1, touch_count)
        }
    
    def _is_uptrend(self, candles: List[Dict]) -> bool:
        """التحقق من الاتجاه الصاعد"""
        if len(candles) < 10:
            return False
        
        highs = [c['high'] for c in candles]
        lows = [c['low'] for c in candles]
        
        # تأكيد Higher Highs و Higher Lows
        hh = all(highs[i] >= highs[i-1] for i in range(1, len(highs)-1))
        hl = all(lows[i] >= lows[i-1] for i in range(1, len(lows)-1))
        
        return hh or hl
    
    def _is_downtrend(self, candles: List[Dict]) -> bool:
        """التحقق من الاتجاه الهابط"""
        if len(candles) < 10:
            return False
        
        highs = [c['high'] for c in candles]
        lows = [c['low'] for c in candles]
        
        # تأكيد Lower Highs و Lower Lows
        lh = all(highs[i] <= highs[i-1] for i in range(1, len(highs)-1))
        ll = all(lows[i] <= lows[i-1] for i in range(1, len(lows)-1))
        
        return lh or ll
    
    def _get_previous_structure(self, candles: List[Dict], idx: int, level_type: str) -> Dict[str, Any]:
        """الحصول على الهيكل السابق"""
        if idx < 10:
            return {"level": 0, "distance": 0}
        
        if level_type == 'high':
            prev_level = max([c['high'] for c in candles[idx-5:idx]])
        else:
            prev_level = min([c['low'] for c in candles[idx-5:idx]])
        
        return {
            "level": prev_level,
            "distance": abs(candles[idx]['close'] - prev_level) / prev_level
        }
    
    def _calculate_bos_strength(self, candles: List[Dict], idx: int, direction: str) -> float:
        """حساب قوة BOS"""
        strength = 0.0
        
        # حجم الشمعة
        volume_ratio = candles[idx]['volume'] / np.mean([c['volume'] for c in candles[max(0, idx-5):idx]])
        strength += min(volume_ratio / 2, 0.4)
        
        # قوة الحركة
        if direction == 'bullish':
            move_strength = (candles[idx]['close'] - candles[idx]['open']) / candles[idx]['open']
        else:
            move_strength = (candles[idx]['open'] - candles[idx]['close']) / candles[idx]['open']
        
        strength += min(move_strength * 10, 0.4)
        
        # موضع في الشمعة
        if direction == 'bullish':
            position = (candles[idx]['close'] - candles[idx]['low']) / (candles[idx]['high'] - candles[idx]['low'])
        else:
            position = (candles[idx]['high'] - candles[idx]['close']) / (candles[idx]['high'] - candles[idx]['low'])
        
        strength += position * 0.2
        
        return min(strength, 1.0)
    
    def _calculate_momentum(self, candles: List[Dict], idx: int, lookback: int) -> float:
        """حساب الزخم"""
        if idx < lookback:
            return 0.0
        
        recent_closes = [c['close'] for c in candles[idx-lookback:idx+1]]
        momentum = (recent_closes[-1] - recent_closes[0]) / recent_closes[0]
        
        return momentum
    
    def _calculate_momentum_shift(self, candles: List[Dict], idx: int) -> float:
        """حساب تحول الزخم"""
        if idx < 10:
            return 0.0
        
        prev_momentum = self._calculate_momentum(candles, idx-5, 3)
        current_momentum = self._calculate_momentum(candles, idx, 3)
        
        return current_momentum - prev_momentum
    
    def _calculate_ob_strength(self, candles: List[Dict], idx: int, ob_type: str) -> float:
        """حساب قوة Order Block"""
        strength = 0.0
        
        # حجم الشمعة
        volume_ratio = candles[idx-1]['volume'] / np.mean([c['volume'] for c in candles[max(0, idx-10):idx]])
        strength += min(volume_ratio / 2, 0.3)
        
        # قوة الشمعة السابقة
        if ob_type == 'bearish':
            candle_strength = (candles[idx-1]['close'] - candles[idx-1]['open']) / candles[idx-1]['open']
        else:
            candle_strength = (candles[idx-1]['open'] - candles[idx-1]['close']) / candles[idx-1]['open']
        
        strength += min(abs(candle_strength) * 20, 0.4)
        
        # رد فعل السعر
        reaction = abs(candles[idx]['close'] - candles[idx-1]['close']) / candles[idx-1]['close']
        strength += min(reaction * 10, 0.3)
        
        return min(strength, 1.0)
    
    def _calculate_fvg_strength(self, candles: List[Dict], idx: int, fvg_type: str) -> float:
        """حساب قوة FVG"""
        strength = 0.0
        
        # حجم الفجوة
        if fvg_type == 'bearish':
            gap_size = candles[idx-1]['low'] - candles[idx]['high']
        else:
            gap_size = candles[idx]['low'] - candles[idx-1]['high']
        
        avg_candle_size = np.mean([c['high'] - c['low'] for c in candles[max(0, idx-10):idx]])
        gap_ratio = gap_size / avg_candle_size
        
        strength += min(gap_ratio / 2, 0.5)
        
        # حجم التداول
        volume_ratio = (candles[idx-1]['volume'] + candles[idx]['volume']) / (2 * np.mean([c['volume'] for c in candles[max(0, idx-10):idx]]))
        strength += min(volume_ratio / 3, 0.5)
        
        return min(strength, 1.0)
    
    def _calculate_liquidity_density(self, candles: List[Dict], idx: int, level_type: str) -> float:
        """حساب كثافة السيولة"""
        density = 0.0
        
        # عدد اللمسات للمستوى
        touch_count = 0
        target_price = candles[idx]['high'] if level_type == 'high' else candles[idx]['low']
        
        for i in range(max(0, idx-20), idx+1):
            if target_price >= candles[i]['low'] and target_price <= candles[i]['high']:
                touch_count += 1
                # إضافة كثافة حسب القرب من المستوى
                if level_type == 'high':
                    proximity = (candles[i]['high'] - target_price) / (candles[i]['high'] - candles[i]['low'])
                else:
                    proximity = (target_price - candles[i]['low']) / (candles[i]['high'] - candles[i]['low'])
                
                density += (1 - proximity) * candles[i]['volume']
        
        # تطبيع الكثافة
        avg_volume = np.mean([c['volume'] for c in candles[max(0, idx-20):idx+1]])
        density = density / (avg_volume * max(1, touch_count))
        
        return min(density, 1.0)
    
    def _is_liquidity_sweep(self, candles: List[Dict], idx: int) -> bool:
        """التحقق من سحب السيولة"""
        if idx < 5:
            return False
        
        current = candles[idx]
        
        # سحب صعودي: قمة جديدة ثم انعكاس
        if (current['high'] > max([c['high'] for c in candles[idx-5:idx]]) and
            current['close'] < current['open'] and
            current['volume'] > np.mean([c['volume'] for c in candles[idx-5:idx]]) * 1.5):
            return True
        
        # سحب هابط: قاع جديد ثم انعكاس
        if (current['low'] < min([c['low'] for c in candles[idx-5:idx]]) and
            current['close'] > current['open'] and
            current['volume'] > np.mean([c['volume'] for c in candles[idx-5:idx]]) * 1.5):
            return True
        
        return False
    
    def _analyze_volume_pattern(self, candles: List[Dict], idx: int, lookback: int) -> str:
        """تحليل نمط الحجم"""
        if idx < lookback:
            return "UNKNOWN"
        
        volumes = [c['volume'] for c in candles[idx-lookback:idx+1]]
        
        if volumes[-1] > max(volumes[:-1]) * 1.5:
            return "SPIKE"
        elif volumes[-1] < min(volumes[:-1]) * 0.7:
            return "DRYUP"
        elif all(v > volumes[i-1] for i, v in enumerate(volumes[1:], 1)):
            return "INCREASING"
        elif all(v < volumes[i-1] for i, v in enumerate(volumes[1:], 1)):
            return "DECREASING"
        else:
            return "CONGESTION"
    
    def _calculate_trap_confidence(self, candles: List[Dict], idx: int, trap_type: str) -> float:
        """حساب ثقة الفخ"""
        confidence = 0.0
        
        # حجم التداول في الكسر الكاذب
        fakeout_volume = candles[idx]['volume']
        avg_volume = np.mean([c['volume'] for c in candles[max(0, idx-10):idx]])
        volume_ratio = fakeout_volume / avg_volume
        
        confidence += min(volume_ratio / 3, 0.4)
        
        # قوة الانعكاس
        if idx < len(candles) - 2:
            reversal_candle = candles[idx+1]
            if trap_type == 'buy':
                reversal_strength = (reversal_candle['close'] - reversal_candle['open']) / reversal_candle['open']
            else:
                reversal_strength = (reversal_candle['open'] - reversal_candle['close']) / reversal_candle['open']
            
            confidence += min(abs(reversal_strength) * 20, 0.4)
        
        # موقع الكسر الكاذب
        if trap_type == 'buy':
            fakeout_position = (candles[idx]['high'] - candles[idx-1]['high']) / candles[idx-1]['high']
        else:
            fakeout_position = (candles[idx-1]['low'] - candles[idx]['low']) / candles[idx-1]['low']
        
        confidence += min(fakeout_position * 100, 0.2)
        
        return min(confidence, 1.0)
    
    def _count_zone_tests(self, candles: List[Dict], zone_idx: int, zone_type: str, zone_price: float) -> int:
        """عد عدد اختبارات المنطقة"""
        test_count = 0
        
        for i in range(zone_idx + 1, min(zone_idx + 50, len(candles))):
            if zone_type == 'supply':
                if candles[i]['high'] >= zone_price * 0.998 and candles[i]['high'] <= zone_price * 1.002:
                    test_count += 1
            else:
                if candles[i]['low'] >= zone_price * 0.998 and candles[i]['low'] <= zone_price * 1.002:
                    test_count += 1
        
        return test_count
    
    def _calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """حساب مؤشر RSI"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices[-period-1:])
        seed = deltas[:period]
        
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        
        if down == 0:
            return 100.0
        
        rs = up / down
        rsi = 100.0 - (100.0 / (1.0 + rs))
        
        return rsi
    
    def _log_significant_patterns(self, analysis_results: Dict, candles: List[Dict]):
        """تسجيل الأنماط المهمة"""
        current_price = candles[-1]['close'] if candles else 0
        
        # تسجيل BOS قوي
        for bos in analysis_results.get('bos_signals', [])[-3:]:  # آخر 3 إشارات
            if bos.get('strength', 0) > 0.7:
                self.logger.log_smc_pattern(
                    SmartMoneyPatterns.BOS,
                    {
                        'type': bos['type'],
                        'price': bos['price'],
                        'strength': f"{bos['strength']:.2f}",
                        'momentum': f"{bos.get('momentum', 0):.3f}"
                    },
                    bos.get('strength', 0)
                )
        
        # تسجيل CHoCH قوي
        for choch in analysis_results.get('choch_signals', [])[-2:]:
            if choch.get('volume_spike', 1) > 2:
                self.logger.log_smc_pattern(
                    SmartMoneyPatterns.CHOCH,
                    {
                        'type': choch['type'],
                        'price': choch['price'],
                        'trend_change': choch['trend_change'],
                        'volume_spike': f"{choch['volume_spike']:.1f}x"
                    },
                    min(choch.get('volume_spike', 1) / 3, 0.9)
                )
        
        # تسجيل Order Blocks قريبة
        for ob in analysis_results.get('order_blocks', []):
            ob_price = ob['mid_price']
            distance_pct = abs(current_price - ob_price) / current_price
            
            if distance_pct < 0.02 and ob.get('strength', 0) > 0.6:  # ضمن 2% وقوة عالية
                self.logger.log_smc_pattern(
                    SmartMoneyPatterns.OB,
                    {
                        'type': ob['type'],
                        'mid_price': f"{ob_price:.4f}",
                        'strength': f"{ob['strength']:.2f}",
                        'distance': f"{distance_pct*100:.1f}%"
                    },
                    ob.get('strength', 0) * (1 - distance_pct * 20)
                )
        
        # تسجيل FVGs غير ممتلئة
        for fvg in analysis_results.get('fvg_zones', []):
            if not fvg.get('filled', False):
                gap_range = fvg.get('gap_range', (0, 0))
                if gap_range[0] < current_price < gap_range[1]:
                    self.logger.log_smc_pattern(
                        SmartMoneyPatterns.FVG,
                        {
                            'type': fvg['type'],
                            'gap_size': f"{fvg.get('gap_size', 0):.4f}",
                            'range': f"{gap_range[0]:.4f}-{gap_range[1]:.4f}",
                            'strength': f"{fvg.get('strength', 0):.2f}"
                        },
                        fvg.get('strength', 0)
                    )
        
        # تسجيل أحداث سيولة قريبة
        for liq in analysis_results.get('liquidity_zones', []):
            liq_price = liq.get('price', 0)
            distance_pct = abs(current_price - liq_price) / current_price
            
            if distance_pct < 0.01:  # ضمن 1%
                event_type = "LIQUIDITY_" + liq.get('type', '').split('_')[-1]
                self.logger.log_liquidity_event(
                    event_type=event_type,
                    zone_type=liq.get('type', 'UNKNOWN'),
                    price=liq_price,
                    volume=liq.get('volume_cluster', 0) or liq.get('volume', 0),
                    reason=f"Density: {liq.get('density', 0):.2f}, Purpose: {liq.get('purpose', 'N/A')}"
                )
        
        # تسجيل إشارات انفجار قوية
        for exp in analysis_results.get('explosion_signals', []):
            if exp.get('confidence', 0) > 0.7:
                self.logger.log_explosion_signal(
                    direction='BULLISH' if exp['type'] == 'BULLISH_EXPLOSION' else 'BEARISH',
                    intensity=exp.get('volume_intensity', 1),
                    trigger=exp.get('type', 'UNKNOWN'),
                    target=exp.get('target_price', 0),
                    confidence=exp.get('confidence', 0)
                )

# ============================================
#  ENTRY SCENARIO ENGINE - محرك سيناريوهات الدخول
# ============================================

class EntryScenario:
    """سيناريوهات الدخول المتعددة مع بوابة تأكيد"""
    
    def __init__(self, smc_analyzer: AdvancedSMCAnalyzer, logger: EnhancedProConsoleLogger):
        self.smc_analyzer = smc_analyzer
        self.logger = logger
        self.entry_validator = EnhancedEntryValidator()  # ✨ المدقق الجديد
        
        # أوزان معدلة
        self.scenario_weights = {
            'bos_breakout': 0.30,
            'choch_reversal': 0.25,
            'ob_retest': 0.18,
            'fvg_fill': 0.12,
            'liquidity_sweep': 0.10,
            'correction_entry': 0.05
        }
    
    def analyze_entry_scenarios(self, candles: List[Dict], smc_analysis: Dict) -> List[Dict]:
        """
        تحليل جميع سيناريوهات الدخول الممكنة مع التصفية
        """
        scenarios = []
        
        # جمع جميع السيناريوهات
        all_scenarios = []
        all_scenarios.extend(self._analyze_bos_entries(candles, smc_analysis))
        all_scenarios.extend(self._analyze_choch_entries(candles, smc_analysis))
        all_scenarios.extend(self._analyze_ob_entries(candles, smc_analysis))
        all_scenarios.extend(self._analyze_fvg_entries(candles, smc_analysis))
        all_scenarios.extend(self._analyze_liquidity_entries(candles, smc_analysis))
        all_scenarios.extend(self._analyze_correction_entries(candles, smc_analysis))
        
        # تصفية السيناريوهات باستخدام المدقق
        for scenario in all_scenarios:
            is_valid, reason = self.entry_validator.validate_entry(scenario, candles, smc_analysis)
            
            if is_valid:
                # زيادة درجة الثقة للسيناريوهات المؤكدة
                scenario['confidence'] *= 1.2
                scenario['validated'] = True
                scenario['validation_reason'] = reason
                scenarios.append(scenario)
            else:
                # تسجيل السيناريوهات المرفوضة للتحليل
                scenario['validated'] = False
                scenario['validation_reason'] = reason
                scenario['total_score'] *= 0.3  # تخفيض شديد للدرجة
        
        # تصنيف السيناريوهات حسب القوة
        ranked_scenarios = sorted(
            [s for s in scenarios if s['validated']], 
            key=lambda x: x['total_score'], 
            reverse=True
        )
        
        # تسجيل أفضل 3 سيناريوهات
        for i, scenario in enumerate(ranked_scenarios[:3]):
            self._log_scenario(scenario, i+1)
        
        return ranked_scenarios[:5]  # أفضل 5 سيناريوهات مؤكدة
    
    def _analyze_bos_entries(self, candles: List[Dict], smc_analysis: Dict) -> List[Dict]:
        """تحليل دخولات كسر الهيكل - معدل"""
        scenarios = []
        current_price = candles[-1]['close']
        
        for bos in smc_analysis.get('bos_signals', []):
            # 🔴 شرط: الإشارة حديثة (آخر 5 شمعات)
            if len(candles) - bos['index'] > 5:
                continue
            
            # 🔴 تعديل: ننتظر إعادة اختبار (Retest) قبل الدخول
            needs_retest = True
            
            # البحث عن إعادة اختبار في الشمعات اللاحقة
            for i in range(bos['index'] + 1, min(bos['index'] + 5, len(candles))):
                test_candle = candles[i]
                
                if bos['type'] == 'BOS_BULLISH':
                    # في BOS صعودي: نبحث عن اختبار للمقاومة المحولة لدعم
                    if test_candle['low'] <= bos['price'] and test_candle['close'] > bos['price']:
                        needs_retest = False
                        current_price = test_candle['close']  # ندخل عند تأكيد الإغلاق فوق
                        break
                
                elif bos['type'] == 'BOS_BEARISH':
                    # في BOS هابط: نبحث عن اختبار للدعم المحول لمقاومة
                    if test_candle['high'] >= bos['price'] and test_candle['close'] < bos['price']:
                        needs_retest = False
                        current_price = test_candle['close']  # ندخل عند تأكيد الإغلاق تحت
                        break
            
            # إذا لم يكن هناك إعادة اختبار واضحة، نتخطى
            if needs_retest:
                continue
            
            entry_type = None
            entry_price = current_price
            stop_loss = 0
            take_profit = 0
            confidence = bos.get('strength', 0) * 1.1  # زيادة الثقة بسبب Retest
            
            if bos['type'] == 'BOS_BULLISH' and bos.get('volume_confirmation', False):
                # دخول شراء من BOS صعودي بعد إعادة اختبار
                entry_type = 'BUY'
                stop_loss = current_price * 0.99  # 1% stop loss
                take_profit = entry_price * 1.02  # هدف 2%
                
                scenario = {
                    'type': 'BOS_BULLISH_BREAKOUT',
                    'entry_type': entry_type,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'confidence': confidence,
                    'risk_reward': (take_profit - entry_price) / (entry_price - stop_loss),
                    'volume_signal': 'CONFIRMED' if bos['volume_confirmation'] else 'WEAK',
                    'momentum': bos.get('momentum', 0),
                    'retest_confirmed': True,
                    'total_score': confidence * self.scenario_weights['bos_breakout'] * (1 + bos.get('momentum', 0) * 10)
                }
                scenarios.append(scenario)
            
            elif bos['type'] == 'BOS_BEARISH' and bos.get('volume_confirmation', False):
                # دخول بيع من BOS هابط بعد إعادة اختبار
                entry_type = 'SELL'
                stop_loss = current_price * 1.01  # 1% stop loss
                take_profit = entry_price * 0.98  # هدف 2%
                
                scenario = {
                    'type': 'BOS_BEARISH_BREAKOUT',
                    'entry_type': entry_type,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'confidence': confidence,
                    'risk_reward': (entry_price - take_profit) / (stop_loss - entry_price),
                    'volume_signal': 'CONFIRMED' if bos['volume_confirmation'] else 'WEAK',
                    'momentum': bos.get('momentum', 0),
                    'retest_confirmed': True,
                    'total_score': confidence * self.scenario_weights['bos_breakout'] * (1 + abs(bos.get('momentum', 0)) * 10)
                }
                scenarios.append(scenario)
        
        return scenarios
    
    def _analyze_choch_entries(self, candles: List[Dict], smc_analysis: Dict) -> List[Dict]:
        """تحليل دخولات انعكاس الهيكل"""
        scenarios = []
        current_price = candles[-1]['close']
        
        for choch in smc_analysis.get('choch_signals', []):
            # 🔴 فقط الإشارات القريبة جداً (آخر 3 شمعات)
            if len(candles) - choch['index'] > 3:
                continue
            
            entry_type = None
            entry_price = current_price
            stop_loss = 0
            take_profit = 0
            
            # حساب الثقة بناءً على حجم spike وتحول الزخم
            confidence = min(choch.get('volume_spike', 1) / 3, 0.9) * (1 + abs(choch.get('momentum_shift', 0)) * 5)
            
            if choch['type'] == 'CHOCH_BULLISH' and choch.get('structure_break', False):
                # دخول شراء من CHoCH صعودي
                entry_type = 'BUY'
                stop_loss = candles[-1]['low'] * 0.99
                take_profit = entry_price * 1.025  # هدف 2.5%
                
                scenario = {
                    'type': 'CHOCH_BULLISH_REVERSAL',
                    'entry_type': entry_type,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'confidence': confidence,
                    'risk_reward': (take_profit - entry_price) / (entry_price - stop_loss),
                    'volume_spike': f"{choch.get('volume_spike', 1):.1f}x",
                    'momentum_shift': choch.get('momentum_shift', 0),
                    'total_score': confidence * self.scenario_weights['choch_reversal']
                }
                scenarios.append(scenario)
            
            elif choch['type'] == 'CHOCH_BEARISH' and choch.get('structure_break', False):
                # دخول بيع من CHoCH هابط
                entry_type = 'SELL'
                stop_loss = candles[-1]['high'] * 1.01
                take_profit = entry_price * 0.975  # هدف 2.5%
                
                scenario = {
                    'type': 'CHOCH_BEARISH_REVERSAL',
                    'entry_type': entry_type,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'confidence': confidence,
                    'risk_reward': (entry_price - take_profit) / (stop_loss - entry_price),
                    'volume_spike': f"{choch.get('volume_spike', 1):.1f}x",
                    'momentum_shift': choch.get('momentum_shift', 0),
                    'total_score': confidence * self.scenario_weights['choch_reversal']
                }
                scenarios.append(scenario)
        
        return scenarios
    
    def _analyze_ob_entries(self, candles: List[Dict], smc_analysis: Dict) -> List[Dict]:
        """تحليل دخولات Order Blocks"""
        scenarios = []
        current_price = candles[-1]['close']
        
        for ob in smc_analysis.get('order_blocks', []):
            ob_range = ob.get('price_range', (0, 0))
            ob_mid = ob.get('mid_price', 0)
            
            # 🔴 فقط الـ Order Blocks القريبة (ضمن 0.5%)
            if abs(current_price - ob_mid) / current_price > 0.005:
                continue
            
            entry_type = None
            entry_price = current_price
            stop_loss = 0
            take_profit = 0
            
            # 🔴 تحديد إذا كان السعر يختبر الـ OB مع تأكيد
            is_testing = ob_range[0] * 0.999 <= current_price <= ob_range[1] * 1.001
            
            # 🔴 تأكيد إضافي: شمعة رد فعل
            if len(candles) >= 2:
                current_candle = candles[-1]
                if ob['type'] == 'BULLISH_OB':
                    # في Bullish OB: ننتظر شمعة خضراء بعد الاختبار
                    if not (current_candle['close'] > current_candle['open']):
                        continue
                else:  # BEARISH_OB
                    # في Bearish OB: ننتظر شمعة حمراء بعد الاختبار
                    if not (current_candle['close'] < current_candle['open']):
                        continue
            
            if ob['type'] == 'BULLISH_OB' and is_testing:
                # دخول شراء من Bullish OB
                entry_type = 'BUY'
                stop_loss = ob_range[0] * 0.995
                take_profit = ob_mid * 1.015  # هدف 1.5% فوق منتصف الـ OB
                
                scenario = {
                    'type': 'BULLISH_OB_RETEST',
                    'entry_type': entry_type,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'confidence': ob.get('strength', 0) * 1.1,
                    'risk_reward': (take_profit - entry_price) / (entry_price - stop_loss),
                    'ob_strength': ob.get('strength', 0),
                    'test_count': ob.get('test_count', 0),
                    'reaction_candle': True,
                    'total_score': ob.get('strength', 0) * self.scenario_weights['ob_retest']
                }
                scenarios.append(scenario)
            
            elif ob['type'] == 'BEARISH_OB' and is_testing:
                # دخول بيع من Bearish OB
                entry_type = 'SELL'
                stop_loss = ob_range[1] * 1.005
                take_profit = ob_mid * 0.985  # هدف 1.5% تحت منتصف الـ OB
                
                scenario = {
                    'type': 'BEARISH_OB_RETEST',
                    'entry_type': entry_type,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'confidence': ob.get('strength', 0) * 1.1,
                    'risk_reward': (entry_price - take_profit) / (stop_loss - entry_price),
                    'ob_strength': ob.get('strength', 0),
                    'test_count': ob.get('test_count', 0),
                    'reaction_candle': True,
                    'total_score': ob.get('strength', 0) * self.scenario_weights['ob_retest']
                }
                scenarios.append(scenario)
        
        return scenarios
    
    def _analyze_fvg_entries(self, candles: List[Dict], smc_analysis: Dict) -> List[Dict]:
        """تحليل دخولات FVG"""
        scenarios = []
        current_price = candles[-1]['close']
        
        for fvg in smc_analysis.get('fvg_zones', []):
            if fvg.get('filled', False):
                continue
            
            fvg_range = fvg.get('gap_range', (0, 0))
            
            # التحقق إذا كان السعر داخل FVG
            if fvg_range[0] < current_price < fvg_range[1]:
                entry_type = None
                entry_price = current_price
                stop_loss = 0
                take_profit = 0
                
                # 🔴 تأكيد: شمعة رد فعل
                if len(candles) >= 2:
                    current_candle = candles[-1]
                    if fvg['type'] == 'BULLISH_FVG':
                        if not (current_candle['close'] > current_candle['open']):
                            continue
                    else:  # BEARISH_FVG
                        if not (current_candle['close'] < current_candle['open']):
                            continue
                
                if fvg['type'] == 'BULLISH_FVG':
                    # دخول شراء من Bullish FVG
                    entry_type = 'BUY'
                    stop_loss = fvg_range[0] * 0.995
                    take_profit = fvg_range[1] * 1.01
                    
                    scenario = {
                        'type': 'BULLISH_FVG_FILL',
                        'entry_type': entry_type,
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'confidence': fvg.get('strength', 0) * 1.1,
                        'risk_reward': (take_profit - entry_price) / (entry_price - stop_loss),
                        'gap_size': f"{fvg.get('gap_size', 0):.4f}",
                        'reaction_candle': True,
                        'total_score': fvg.get('strength', 0) * self.scenario_weights['fvg_fill']
                    }
                    scenarios.append(scenario)
                
                elif fvg['type'] == 'BEARISH_FVG':
                    # دخول بيع من Bearish FVG
                    entry_type = 'SELL'
                    stop_loss = fvg_range[1] * 1.005
                    take_profit = fvg_range[0] * 0.99
                    
                    scenario = {
                        'type': 'BEARISH_FVG_FILL',
                        'entry_type': entry_type,
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'confidence': fvg.get('strength', 0) * 1.1,
                        'risk_reward': (entry_price - take_profit) / (stop_loss - entry_price),
                        'gap_size': f"{fvg.get('gap_size', 0):.4f}",
                        'reaction_candle': True,
                        'total_score': fvg.get('strength', 0) * self.scenario_weights['fvg_fill']
                    }
                    scenarios.append(scenario)
        
        return scenarios
    
    def _analyze_liquidity_entries(self, candles: List[Dict], smc_analysis: Dict) -> List[Dict]:
        """تحليل دخولات بعد سحب السيولة"""
        scenarios = []
        current_price = candles[-1]['close']
        
        for liq in smc_analysis.get('liquidity_zones', []):
            if liq.get('type', '').startswith('LIQUIDITY_SWEEP'):
                entry_type = None
                entry_price = current_price
                stop_loss = 0
                take_profit = 0
                
                # 🔴 شرط: السحب قوي (intensity > 2)
                if liq.get('intensity', 1) <= 2:
                    continue
                
                # 🔴 تأكيد: شمعة رد فعل بعد السحب
                if len(candles) >= 2:
                    current_candle = candles[-1]
                    sweep_direction = liq.get('direction')
                    
                    if sweep_direction == 'UP':  # سحب صعودي
                        if not (current_candle['close'] < current_candle['open']):  # شمعة حمراء
                            continue
                    else:  # سحب هابط
                        if not (current_candle['close'] > current_candle['open']):  # شمعة خضراء
                            continue
                
                # دخول بعد سحب سيولة صعودي (للبيع)
                if liq.get('direction') == 'UP':
                    entry_type = 'SELL'
                    stop_loss = liq['price'] * 1.01
                    take_profit = entry_price * 0.98
                    
                    scenario = {
                        'type': 'POST_SWEEP_SELL',
                        'entry_type': entry_type,
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'confidence': min(liq.get('intensity', 1) / 4, 0.8),
                        'risk_reward': (entry_price - take_profit) / (stop_loss - entry_price),
                        'sweep_intensity': liq.get('intensity', 1),
                        'reaction_candle': True,
                        'total_score': min(liq.get('intensity', 1) / 4, 0.8) * self.scenario_weights['liquidity_sweep']
                    }
                    scenarios.append(scenario)
                
                # دخول بعد سحب سيولة هابط (للشراء)
                elif liq.get('direction') == 'DOWN':
                    entry_type = 'BUY'
                    stop_loss = liq['price'] * 0.99
                    take_profit = entry_price * 1.02
                    
                    scenario = {
                        'type': 'POST_SWEEP_BUY',
                        'entry_type': entry_type,
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'confidence': min(liq.get('intensity', 1) / 4, 0.8),
                        'risk_reward': (take_profit - entry_price) / (entry_price - stop_loss),
                        'sweep_intensity': liq.get('intensity', 1),
                        'reaction_candle': True,
                        'total_score': min(liq.get('intensity', 1) / 4, 0.8) * self.scenario_weights['liquidity_sweep']
                    }
                    scenarios.append(scenario)
        
        return scenarios
    
    def _analyze_correction_entries(self, candles: List[Dict], smc_analysis: Dict) -> List[Dict]:
        """تحليل دخولات من مناطق التصحيح"""
        scenarios = []
        current_price = candles[-1]['close']
        
        # تحليل الاتجاه العام
        trend_structure = smc_analysis.get('trend_structure', {})
        trend = trend_structure.get('trend', 'SIDEWAYS')
        
        if trend == 'BULLISH':
            # في اتجاه صاعد، نبحث عن تصحيح للشراء
            demand_zones = smc_analysis.get('demand_zones', [])
            
            for zone in demand_zones:
                zone_price = zone.get('price_level', 0)
                distance_pct = abs(current_price - zone_price) / current_price
                
                # 🔴 إذا كان السعر قريب من منطقة طلب قوية (ضمن 0.3%)
                if distance_pct < 0.003 and zone.get('strength', 0) > 0.7:
                    # 🔴 تأكيد: شمعة رد فعل صعودية
                    if len(candles) >= 2:
                        current_candle = candles[-1]
                        if not (current_candle['close'] > current_candle['open']):
                            continue
                    
                    entry_type = 'BUY'
                    entry_price = current_price
                    stop_loss = zone_price * 0.995
                    take_profit = entry_price * 1.02  # هدف 2%
                    
                    scenario = {
                        'type': 'BULLISH_CORRECTION_BUY',
                        'entry_type': entry_type,
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'confidence': zone.get('strength', 0) * 0.9,
                        'risk_reward': (take_profit - entry_price) / (entry_price - stop_loss),
                        'zone_strength': zone.get('strength', 0),
                        'test_count': zone.get('test_count', 0),
                        'reaction_candle': True,
                        'total_score': zone.get('strength', 0) * self.scenario_weights['correction_entry']
                    }
                    scenarios.append(scenario)
        
        elif trend == 'BEARISH':
            # في اتجاه هابط، نبحث عن تصحيح للبيع
            supply_zones = smc_analysis.get('supply_zones', [])
            
            for zone in supply_zones:
                zone_price = zone.get('price_level', 0)
                distance_pct = abs(current_price - zone_price) / current_price
                
                # 🔴 إذا كان السعر قريب من منطقة عرض قوية (ضمن 0.3%)
                if distance_pct < 0.003 and zone.get('strength', 0) > 0.7:
                    # 🔴 تأكيد: شمعة رد فعل هابطة
                    if len(candles) >= 2:
                        current_candle = candles[-1]
                        if not (current_candle['close'] < current_candle['open']):
                            continue
                    
                    entry_type = 'SELL'
                    entry_price = current_price
                    stop_loss = zone_price * 1.005
                    take_profit = entry_price * 0.98  # هدف 2%
                    
                    scenario = {
                        'type': 'BEARISH_CORRECTION_SELL',
                        'entry_type': entry_type,
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'take_profit': take_profit,
                        'confidence': zone.get('strength', 0) * 0.9,
                        'risk_reward': (entry_price - take_profit) / (stop_loss - entry_price),
                        'zone_strength': zone.get('strength', 0),
                        'test_count': zone.get('test_count', 0),
                        'reaction_candle': True,
                        'total_score': zone.get('strength', 0) * self.scenario_weights['correction_entry']
                    }
                    scenarios.append(scenario)
        
        return scenarios
    
    def _log_scenario(self, scenario: Dict, rank: int):
        """تسجيل السيناريو"""
        rank_icon = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][min(rank-1, 4)]
        entry_icon = "🟢" if scenario['entry_type'] == 'BUY' else "🔴"
        
        validated_mark = "✅" if scenario.get('validated', False) else "❌"
        
        print(
            f"{AdvancedConsoleColors.FG.CYAN}{rank_icon} "
            f"{AdvancedConsoleColors.BOLD}{scenario['type']}{AdvancedConsoleColors.RESET} "
            f"{validated_mark} | "
            f"{entry_icon} {scenario['entry_type']} @ {scenario['entry_price']:.4f} | "
            f"{AdvancedConsoleColors.FG.YELLOW}SL: {scenario['stop_loss']:.4f}{AdvancedConsoleColors.RESET} | "
            f"{AdvancedConsoleColors.FG.GREEN}TP: {scenario['take_profit']:.4f}{AdvancedConsoleColors.RESET} | "
            f"Conf: {scenario['confidence']:.2f} | "
            f"RR: 1:{scenario['risk_reward']:.1f} | "
            f"Score: {scenario['total_score']:.3f}"
        )

# ============================================
#  ADVANCED TRADE PROTECTOR - حماية الصفقات المتقدمة
# ============================================

class AdvancedTradeProtector:
    """نظام حماية الصفقات المتقدم"""
    
    def __init__(self, logger: EnhancedProConsoleLogger):
        self.logger = logger
        self.protection_rules = self._initialize_protection_rules()
        
    def _initialize_protection_rules(self) -> Dict[str, Any]:
        """تهيئة قواعد الحماية"""
        return {
            'initial_protection': {
                'min_profit_to_protect': 0.003,  # 0.3%
                'protection_distance': 0.002,  # 0.2%
                'max_time_unprotected': 300  # 5 دقائق
            },
            'breakeven_rules': {
                'min_profit_for_be': 0.005,  # 0.5%
                'be_distance': 0.001,  # 0.1%
                'time_to_be': 60  # 1 دقيقة بعد تحقيق الربح
            },
            'trailing_rules': {
                'activation_profit': 0.008,  # 0.8%
                'trailing_distance': 0.004,  # 0.4%
                'dynamic_adjustment': True,
                'use_structure_levels': True
            },
            'emergency_exit': {
                'max_loss': -0.02,  # -2%
                'volume_spike_against': 3.0,  # حجم 3x ضد الصفقة
                'structural_break': True,
                'time_based_exit': 7200  # ساعتان كحد أقصى
            }
        }
    
    def analyze_trade_protection(self, trade_data: Dict, candles: List[Dict], 
                                 smc_analysis: Dict) -> Dict[str, Any]:
        """
        تحليل حماية الصفقة الحالية
        
        Returns:
            Dict: توصيات الحماية والإجراءات
        """
        if not trade_data.get('active', False):
            return {'action': 'NO_ACTION', 'reason': 'No active trade'}
        
        current_price = candles[-1]['close'] if candles else trade_data.get('current_price', 0)
        entry_price = trade_data.get('entry_price', 0)
        side = trade_data.get('side', 'BUY')
        
        # حساب الربح/الخسارة الحالي
        if side == 'BUY':
            current_pnl = (current_price - entry_price) / entry_price
        else:
            current_pnl = (entry_price - current_price) / entry_price
        
        trade_age = time.time() - trade_data.get('entry_time', time.time())
        
        protection_actions = {
            'initial_protection': self._check_initial_protection(current_pnl, trade_age),
            'breakeven': self._check_breakeven(current_pnl, trade_age),
            'trailing_stop': self._check_trailing_stop(current_pnl, candles, smc_analysis, side),
            'emergency_exit': self._check_emergency_exit(current_pnl, candles, smc_analysis, side),
            'partial_exit': self._check_partial_exit(current_pnl, trade_age)
        }
        
        # تحديد الإجراء الأفضل
        recommended_action = self._select_best_action(protection_actions, current_pnl)
        
        return {
            'current_pnl': current_pnl,
            'trade_age': trade_age,
            'protection_analysis': protection_actions,
            'recommended_action': recommended_action,
            'current_price': current_price,
            'side': side
        }
    
    def _check_initial_protection(self, current_pnl: float, trade_age: float) -> Dict[str, Any]:
        """فحص الحماية الأولية"""
        rules = self.protection_rules['initial_protection']
        
        if current_pnl >= rules['min_profit_to_protect']:
            return {
                'action': 'SET_PROTECTION',
                'distance': rules['protection_distance'],
                'reason': f'Initial profit {current_pnl:.2%} achieved'
            }
        elif trade_age > rules['max_time_unprotected']:
            return {
                'action': 'SET_PROTECTION',
                'distance': rules['protection_distance'] * 1.5,
                'reason': f'Max unprotected time reached: {trade_age:.0f}s'
            }
        
        return {'action': 'HOLD', 'reason': 'Waiting for initial profit'}
    
    def _check_breakeven(self, current_pnl: float, trade_age: float) -> Dict[str, Any]:
        """فحص نقطة التعادل"""
        rules = self.protection_rules['breakeven_rules']
        
        if current_pnl >= rules['min_profit_for_be']:
            return {
                'action': 'MOVE_TO_BREAKEVEN',
                'distance': rules['be_distance'],
                'reason': f'Profit {current_pnl:.2%} > {rules["min_profit_for_be"]:.2%}'
            }
        
        return {'action': 'HOLD', 'reason': 'Not enough profit for breakeven'}
    
    def _check_trailing_stop(self, current_pnl: float, candles: List[Dict], 
                            smc_analysis: Dict, side: str) -> Dict[str, Any]:
        """فحص وقف الخسارة المتحرك"""
        rules = self.protection_rules['trailing_rules']
        
        if current_pnl >= rules['activation_profit']:
            # حساب مستوى التريلينج بناءً على هيكل السوق
            trail_level = self._calculate_trail_level(candles, smc_analysis, side, current_pnl)
            
            return {
                'action': 'ACTIVATE_TRAILING',
                'trail_level': trail_level,
                'distance': rules['trailing_distance'],
                'reason': f'Profit {current_pnl:.2%} activated trailing',
                'dynamic': rules['dynamic_adjustment']
            }
        
        return {'action': 'HOLD', 'reason': 'Trailing not activated yet'}
    
    def _check_emergency_exit(self, current_pnl: float, candles: List[Dict],
                             smc_analysis: Dict, side: str) -> Dict[str, Any]:
        """فحص الخروج الطارئ"""
        rules = self.protection_rules['emergency_exit']
        
        # 1. خسارة كبيرة
        if current_pnl <= rules['max_loss']:
            return {
                'action': 'EMERGENCY_EXIT',
                'reason': f'Max loss exceeded: {current_pnl:.2%}'
            }
        
        # 2. حجم كبير ضد الصفقة
        if len(candles) >= 3:
            recent_volume = candles[-1]['volume']
            avg_volume = np.mean([c['volume'] for c in candles[-4:-1]])
            
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
            
            if volume_ratio > rules['volume_spike_against']:
                # تحقق إذا كان الحجم ضد الصفقة
                if (side == 'BUY' and candles[-1]['close'] < candles[-1]['open']) or \
                   (side == 'SELL' and candles[-1]['close'] > candles[-1]['open']):
                    return {
                        'action': 'EMERGENCY_EXIT',
                        'reason': f'Volume spike against position: {volume_ratio:.1f}x'
                    }
        
        # 3. كسر هيكلي ضد الصفقة
        if rules['structural_break']:
            if self._is_structural_break(candles, smc_analysis, side):
                return {
                    'action': 'EMERGENCY_EXIT',
                    'reason': 'Structural break against position'
                }
        
        return {'action': 'HOLD', 'reason': 'No emergency signals'}
    
    def _check_partial_exit(self, current_pnl: float, trade_age: float) -> Dict[str, Any]:
        """فحص الخروج الجزئي"""
        # خروج جزئي عند تحقيق أهداف ربح محددة
        exit_levels = [
            {'profit': 0.01, 'exit_percent': 0.3, 'reason': 'First target reached'},
            {'profit': 0.02, 'exit_percent': 0.3, 'reason': 'Second target reached'},
            {'profit': 0.03, 'exit_percent': 0.4, 'reason': 'Final target reached'}
        ]
        
        for level in exit_levels:
            if current_pnl >= level['profit']:
                return {
                    'action': 'PARTIAL_EXIT',
                    'exit_percent': level['exit_percent'],
                    'reason': level['reason']
                }
        
        return {'action': 'HOLD', 'reason': 'No partial exit targets reached'}
    
    def _calculate_trail_level(self, candles: List[Dict], smc_analysis: Dict, 
                              side: str, current_pnl: float) -> float:
        """حساب مستوى التريلينج"""
        rules = self.protection_rules['trailing_rules']
        
        if rules['use_structure_levels'] and side == 'BUY':
            # في الشراء: استخدام أعلى قاع حديث
            recent_lows = [c['low'] for c in candles[-10:]]
            if recent_lows:
                structure_level = max(recent_lows[-3:])  # آخر 3 قيعان
                return structure_level * 0.998  # تحت المستوى قليلاً
        
        elif rules['use_structure_levels'] and side == 'SELL':
            # في البيع: استخدام أقل قمة حديثة
            recent_highs = [c['high'] for c in candles[-10:]]
            if recent_highs:
                structure_level = min(recent_highs[-3:])  # آخر 3 قمم
                return structure_level * 1.002  # فوق المستوى قليلاً
        
        # الحساب التقليدي إذا لم يكن هناك هيكل واضح
        current_price = candles[-1]['close']
        distance = current_price * rules['trailing_distance']
        
        if side == 'BUY':
            return current_price - distance
        else:
            return current_price + distance
    
    def _is_structural_break(self, candles: List[Dict], smc_analysis: Dict, side: str) -> bool:
        """التحقق من كسر هيكلي ضد الصفقة"""
        bos_signals = smc_analysis.get('bos_signals', [])
        choch_signals = smc_analysis.get('choch_signals', [])
        
        # فحص آخر إشارات BOS
        for bos in bos_signals[-2:]:
            if side == 'BUY' and bos['type'] == 'BOS_BEARISH':
                return True
            elif side == 'SELL' and bos['type'] == 'BOS_BULLISH':
                return True
        
        # فحص آخر إشارات CHoCH
        for choch in choch_signals[-2:]:
            if side == 'BUY' and choch['type'] == 'CHOCH_BEARISH':
                return True
            elif side == 'SELL' and choch['type'] == 'CHOCH_BULLISH':
                return True
        
        return False
    
    def _select_best_action(self, protection_actions: Dict[str, Any], current_pnl: float) -> Dict[str, Any]:
        """اختيار أفضل إجراء حماية"""
        action_priority = {
            'EMERGENCY_EXIT': 100,
            'PARTIAL_EXIT': 80,
            'ACTIVATE_TRAILING': 60,
            'MOVE_TO_BREAKEVEN': 40,
            'SET_PROTECTION': 20,
            'HOLD': 0
        }
        
        # الحصول على أعلى إجراء أولوية
        best_action = None
        best_priority = -1
        
        for action_type, action_data in protection_actions.items():
            action = action_data.get('action', 'HOLD')
            priority = action_priority.get(action, 0)
            
            if priority > best_priority:
                best_priority = priority
                best_action = action_data
        
        # تعديل حسب الربح الحالي
        if best_action and best_action['action'] != 'HOLD' and current_pnl < 0:
            # إذا كانت هناك خسارة، تكون الأولوية أقل للخروج الجزئي
            if 'PARTIAL_EXIT' in best_action['action']:
                best_action['action'] = 'HOLD'
                best_action['reason'] = 'Avoiding partial exit during loss'
        
        return best_action if best_action else {'action': 'HOLD', 'reason': 'No protection needed'}

# ============================================
#  ENHANCED TRADE MANAGER - مدير الصفقات المحسن
# ============================================

class EnhancedTradeManager:
    """مدير الصفقات المحسن مع دعم SMC وإدارة الرصيد"""
    
    def __init__(self, exchange, symbol: str, initial_balance: float = INITIAL_BALANCE,
                 risk_percent: float = RISK_PERCENT, 
                 logger: EnhancedProConsoleLogger = None):
        self.exchange = exchange
        self.symbol = symbol
        self.risk_percent = risk_percent
        self.logger = logger or EnhancedProConsoleLogger()
        
        # 🆕 إدارة الرصيد
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.cumulative_pnl = 0.0
        
        # الأنظمة الفرعية
        self.smc_analyzer = AdvancedSMCAnalyzer(logger)
        self.scenario_engine = EntryScenario(self.smc_analyzer, logger)
        self.trade_protector = AdvancedTradeProtector(logger)
        
        # حالة التداول
        self.active_trade = False
        self.current_trade = None
        self.trades_history = []
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0.0,
            'best_trade': 0.0,
            'worst_trade': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0
        }
    
    def update_balance(self, pnl_usd: float):
        """تحديث الرصيد والربح التراكمي"""
        self.current_balance += pnl_usd
        self.cumulative_pnl += pnl_usd
        
        # تسجيل الرصيد
        self.logger.log_balance(
            balance=self.current_balance,
            pnl=self.cumulative_pnl,
            initial_balance=self.initial_balance
        )
    
    def get_balance_summary(self) -> Dict[str, Any]:
        """الحصول على ملخص الرصيد"""
        pnl_percent = (self.cumulative_pnl / self.initial_balance) * 100 if self.initial_balance > 0 else 0
        
        return {
            'initial_balance': self.initial_balance,
            'current_balance': self.current_balance,
            'cumulative_pnl': self.cumulative_pnl,
            'pnl_percent': pnl_percent,
            'roi_percent': (self.current_balance - self.initial_balance) / self.initial_balance * 100 if self.initial_balance > 0 else 0
        }
    
    def analyze_market_for_entry(self, candles: List[Dict]) -> List[Dict]:
        """
        تحليل السوق للدخول باستخدام SMC
        
        Returns:
            List[Dict]: سيناريوهات الدخول المصنفة
        """
        # تحليل SMC
        smc_analysis = self.smc_analyzer.analyze_candles(candles)
        
        if 'error' in smc_analysis:
            self.logger.file_logger.error(f"SMC analysis failed: {smc_analysis['error']}")
            return []
        
        # توليد سيناريوهات الدخول
        entry_scenarios = self.scenario_engine.analyze_entry_scenarios(candles, smc_analysis)
        
        # تصفية السيناريوهات ذات الثقة العالية
        high_confidence_scenarios = [
            s for s in entry_scenarios 
            if s['confidence'] > 0.7 and s['total_score'] > 0.5
        ]
        
        return high_confidence_scenarios[:5]  # أفضل 5 سيناريوهات
    
    def execute_trade(self, scenario: Dict, current_price: float) -> bool:
        """تنفيذ صفقة بناءً على السيناريو"""
        if self.active_trade:
            self.logger.file_logger.warning("Cannot execute trade: Active trade exists")
            return False
        
        # حساب حجم المركز
        position_size = self._calculate_position_size(self.current_balance, current_price, scenario['confidence'])
        
        if position_size <= 0:
            self.logger.log_system(f"Position size too small: {position_size}", "WARNING")
            return False
        
        # تسجيل الصفقة
        trade_id = len(self.trades_history) + 1
        trade_record = {
            'id': trade_id,
            'timestamp': datetime.now().isoformat(),
            'scenario': scenario['type'],
            'side': scenario['entry_type'],
            'entry_price': current_price,
            'position_size': position_size,
            'position_value': position_size * current_price,
            'stop_loss': scenario['stop_loss'],
            'take_profit': scenario['take_profit'],
            'confidence': scenario['confidence'],
            'risk_reward': scenario['risk_reward'],
            'status': 'ACTIVE'
        }
        
        # تنفيذ الأمر (محاكاة أو حقيقي)
        success = self._place_order(
            side=scenario['entry_type'],
            quantity=position_size,
            price=current_price
        )
        
        if success:
            self.active_trade = True
            self.current_trade = trade_record
            self.current_trade['entry_time'] = time.time()
            self.trades_history.append(self.current_trade)
            
            # تسجيل الدخول
            self._log_trade_entry(scenario, current_price, position_size)
            
            return True
        
        return False
    
    def manage_active_trade(self, candles: List[Dict]):
        """إدارة الصفقة النشطة"""
        if not self.active_trade or not self.current_trade:
            return
        
        current_price = candles[-1]['close'] if candles else 0
        
        # تحليل SMC للوقت الحالي
        smc_analysis = self.smc_analyzer.analyze_candles(candles)
        
        # تحليل الحماية
        trade_data = {
            'active': self.active_trade,
            'side': self.current_trade['side'],
            'entry_price': self.current_trade['entry_price'],
            'entry_time': self.current_trade.get('entry_time', time.time()),
            'current_price': current_price
        }
        
        protection_analysis = self.trade_protector.analyze_trade_protection(
            trade_data, candles, smc_analysis
        )
        
        # تنفيذ إجراءات الحماية
        self._execute_protection_actions(protection_analysis, current_price)
        
        # التحقق من وقف الخسارة وجني الأرباح
        self._check_exit_conditions(current_price, smc_analysis)
    
    def _calculate_position_size(self, balance: float, entry_price: float, confidence: float) -> float:
        """حساب حجم المركز الذكي"""
        # رأس المال المعرض للمخاطرة
        risk_capital = balance * self.risk_percent
        
        # تعديل حسب الثقة
        confidence_multiplier = min(confidence * 1.5, 1.2)  # حتى 1.2x للثقة العالية
        
        adjusted_capital = risk_capital * confidence_multiplier
        
        # حساب الكمية
        raw_qty = adjusted_capital / entry_price
        
        # تقريب للكمية المناسبة
        qty = round(raw_qty, 4)  # تقريب لأربعة منازل عشرية
        
        # الحد الأدنى للكمية
        min_qty = 0.001  # الحد الأدنى للكمية
        if qty < min_qty:
            self.logger.log_system(f"Position size below minimum: {qty} < {min_qty}", "WARNING")
            return 0
        
        return qty
    
    def _place_order(self, side: str, quantity: float, price: float) -> bool:
        """وضع أمر (محاكاة)"""
        # في التنفيذ الحقيقي، استخدم exchange.create_order()
        self.logger.file_logger.info(
            f"ORDER: {side} {quantity:.4f} @ {price:.4f} | "
            f"Scenario: {self.current_trade['scenario'] if self.current_trade else 'N/A'}"
        )
        return True
    
    def _log_trade_entry(self, scenario: Dict, entry_price: float, position_size: float):
        """تسجيل دخول الصفقة"""
        side_icon = "🟢" if scenario['entry_type'] == 'BUY' else "🔴"
        position_value = position_size * entry_price
        risk_percent = abs((scenario['stop_loss'] - entry_price) / entry_price * 100)
        
        print(
            f"\n{AdvancedConsoleColors.BG.LIGHT_BLACK}{'='*80}{AdvancedConsoleColors.RESET}"
        )
        print(
            f"{AdvancedConsoleColors.BOLD}{AdvancedConsoleColors.FG.CYAN}"
            f"🎯 TRADE ENTERED{AdvancedConsoleColors.RESET} | "
            f"{side_icon} {scenario['entry_type']} | "
            f"Price: {entry_price:.4f} | "
            f"Size: {position_size:.4f} | "
            f"Value: {position_value:.2f} USDT"
        )
        print(
            f"{AdvancedConsoleColors.FG.YELLOW}SL: {scenario['stop_loss']:.4f} "
            f"({risk_percent:.2f}%){AdvancedConsoleColors.RESET} | "
            f"{AdvancedConsoleColors.FG.GREEN}TP: {scenario['take_profit']:.4f} "
            f"({abs((scenario['take_profit'] - entry_price)/entry_price*100):.2f}%){AdvancedConsoleColors.RESET}"
        )
        print(
            f"{AdvancedConsoleColors.FG.LIGHT_MAGENTA}Scenario: {scenario['type']}{AdvancedConsoleColors.RESET} | "
            f"Confidence: {scenario['confidence']:.2f} | "
            f"Risk/Reward: 1:{scenario['risk_reward']:.1f}"
        )
        print(
            f"{AdvancedConsoleColors.BG.LIGHT_BLACK}{'='*80}{AdvancedConsoleColors.RESET}\n"
        )
    
    def _execute_protection_actions(self, protection_analysis: Dict, current_price: float):
        """تنفيذ إجراءات الحماية"""
        action = protection_analysis.get('recommended_action', {})
        
        if action['action'] == 'HOLD':
            return
        
        action_type = action['action']
        reason = action.get('reason', 'Unknown')
        
        if action_type == 'EMERGENCY_EXIT':
            self._close_trade(f"EMERGENCY: {reason}", current_price)
        
        elif action_type == 'PARTIAL_EXIT':
            exit_percent = action.get('exit_percent', 0.3)
            self._partial_exit(exit_percent, current_price, reason)
        
        elif action_type in ['SET_PROTECTION', 'MOVE_TO_BREAKEVEN', 'ACTIVATE_TRAILING']:
            # تحديث وقف الخسارة
            new_stop = action.get('trail_level') or action.get('distance')
            if new_stop:
                self._update_stop_loss(new_stop, reason)
    
    def _check_exit_conditions(self, current_price: float, smc_analysis: Dict):
        """التحقق من شروط الخروج"""
        if not self.current_trade:
            return
        
        side = self.current_trade['side']
        entry_price = self.current_trade['entry_price']
        stop_loss = self.current_trade['stop_loss']
        take_profit = self.current_trade['take_profit']
        
        # 1. وقف الخسارة
        if (side == 'BUY' and current_price <= stop_loss) or \
           (side == 'SELL' and current_price >= stop_loss):
            self._close_trade(f"Stop Loss hit @ {current_price:.4f}", current_price)
            return
        
        # 2. جني الأرباح
        if (side == 'BUY' and current_price >= take_profit) or \
           (side == 'SELL' and current_price <= take_profit):
            self._close_trade(f"Take Profit hit @ {current_price:.4f}", current_price)
            return
        
        # 3. إشارات انعكاس SMC قوية ضد الصفقة
        if self._check_smc_reversal_signals(smc_analysis, side):
            self._close_trade("SMC reversal signal against position", current_price)
            return
    
    def _check_smc_reversal_signals(self, smc_analysis: Dict, side: str) -> bool:
        """التحقق من إشارات انعكاس SMC"""
        # فحص CHoCH ضد الصفقة
        for choch in smc_analysis.get('choch_signals', [])[-2:]:
            if side == 'BUY' and choch['type'] == 'CHOCH_BEARISH':
                return True
            elif side == 'SELL' and choch['type'] == 'CHOCH_BULLISH':
                return True
        
        # فحص BOS ضد الصفقة
        for bos in smc_analysis.get('bos_signals', [])[-2:]:
            if side == 'BUY' and bos['type'] == 'BOS_BEARISH':
                return True
            elif side == 'SELL' and bos['type'] == 'BOS_BULLISH':
                return True
        
        return False
    
    def _partial_exit(self, exit_percent: float, exit_price: float, reason: str):
        """تنفيذ خروج جزئي"""
        if not self.current_trade:
            return
        
        # تحديث حجم المركز
        current_size = self.current_trade['position_size']
        exit_size = current_size * exit_percent
        remaining_size = current_size - exit_size
        
        # حساب الربح/الخسارة للجزء المخرج
        entry_price = self.current_trade['entry_price']
        side = self.current_trade['side']
        
        if side == 'BUY':
            partial_pnl = (exit_price - entry_price) * exit_size
        else:
            partial_pnl = (entry_price - exit_price) * exit_size
        
        # تحديث الرصيد
        self.update_balance(partial_pnl)
        
        self.current_trade['position_size'] = remaining_size
        
        # تسجيل الخروج الجزئي
        pnl_color = AdvancedConsoleColors.FG.GREEN if partial_pnl >= 0 else AdvancedConsoleColors.FG.RED
        
        print(
            f"{AdvancedConsoleColors.FG.YELLOW}🔄 PARTIAL EXIT | "
            f"Closed {exit_percent*100:.0f}% ({exit_size:.4f}) @ {exit_price:.4f} | "
            f"PnL: {pnl_color}{partial_pnl:+.2f} USDT{AdvancedConsoleColors.RESET} | "
            f"Reason: {reason}"
        )
    
    def _update_stop_loss(self, new_stop: float, reason: str):
        """تحديث وقف الخسارة"""
        if not self.current_trade:
            return
        
        old_stop = self.current_trade['stop_loss']
        self.current_trade['stop_loss'] = new_stop
        
        # تسجيل تحديث وقف الخسارة
        stop_change = abs(new_stop - old_stop) / old_stop * 100
        
        print(
            f"{AdvancedConsoleColors.FG.CYAN}🛡️ STOP LOSS UPDATED | "
            f"Old: {old_stop:.4f} → New: {new_stop:.4f} ({stop_change:.2f}%) | "
            f"Reason: {reason}{AdvancedConsoleColors.RESET}"
        )
    
    def _close_trade(self, reason: str, exit_price: float):
        """إغلاق الصفقة"""
        if not self.current_trade or not self.active_trade:
            return
        
        trade = self.current_trade
        side = trade['side']
        entry_price = trade['entry_price']
        position_size = trade.get('position_size', 0)
        
        # حساب الربح/الخسارة
        if side == 'BUY':
            pnl_pct = (exit_price - entry_price) / entry_price * 100
            pnl_usd = (exit_price - entry_price) * position_size
        else:
            pnl_pct = (entry_price - exit_price) / entry_price * 100
            pnl_usd = (entry_price - exit_price) * position_size
        
        # تحديث الرصيد
        self.update_balance(pnl_usd)
        
        # تحديث سجل الصفقة
        trade['exit_price'] = exit_price
        trade['exit_reason'] = reason
        trade['pnl_pct'] = pnl_pct
        trade['pnl_usd'] = pnl_usd
        trade['exit_time'] = datetime.now().isoformat()
        trade['status'] = 'CLOSED'
        
        # تحديث المقاييس
        self.performance_metrics['total_trades'] += 1
        self.performance_metrics['total_pnl'] += pnl_pct
        
        if pnl_pct > 0:
            self.performance_metrics['winning_trades'] += 1
            self.performance_metrics['avg_win'] = (
                (self.performance_metrics['avg_win'] * (self.performance_metrics['winning_trades'] - 1) + pnl_pct) /
                self.performance_metrics['winning_trades']
            )
        else:
            self.performance_metrics['avg_loss'] = (
                (self.performance_metrics['avg_loss'] * (self.performance_metrics['total_trades'] - self.performance_metrics['winning_trades'] - 1) + pnl_pct) /
                (self.performance_metrics['total_trades'] - self.performance_metrics['winning_trades'])
            )
        
        if pnl_pct > self.performance_metrics['best_trade']:
            self.performance_metrics['best_trade'] = pnl_pct
        
        if pnl_pct < self.performance_metrics['worst_trade']:
            self.performance_metrics['worst_trade'] = pnl_pct
        
        # تسجيل الخروج
        pnl_color = AdvancedConsoleColors.FG.GREEN if pnl_pct > 0 else AdvancedConsoleColors.FG.RED
        pnl_icon = "💰" if pnl_pct > 0 else "💸"
        
        print(
            f"\n{AdvancedConsoleColors.BG.LIGHT_BLACK}{'='*80}{AdvancedConsoleColors.RESET}"
        )
        print(
            f"{AdvancedConsoleColors.BOLD}{pnl_color}"
            f"{pnl_icon} TRADE CLOSED{AdvancedConsoleColors.RESET} | "
            f"PnL: {pnl_color}{pnl_pct:+.2f}%{AdvancedConsoleColors.RESET} | "
            f"${pnl_usd:+.2f} | "
            f"Reason: {reason}"
        )
        print(
            f"Entry: {entry_price:.4f} → Exit: {exit_price:.4f} | "
            f"Side: {side} | "
            f"Scenario: {trade['scenario']}"
        )
        print(
            f"{AdvancedConsoleColors.BG.LIGHT_BLACK}{'='*80}{AdvancedConsoleColors.RESET}\n"
        )
        
        # إعادة التعيين
        self.active_trade = False
        self.current_trade = None
    
    def get_performance_report(self) -> Dict[str, Any]:
        """الحصول على تقرير الأداء"""
        metrics = self.performance_metrics
        balance_summary = self.get_balance_summary()
        
        if metrics['total_trades'] > 0:
            win_rate = (metrics['winning_trades'] / metrics['total_trades']) * 100
            avg_trade = metrics['total_pnl'] / metrics['total_trades']
        else:
            win_rate = 0.0
            avg_trade = 0.0
        
        return {
            'balance_summary': balance_summary,
            'total_trades': metrics['total_trades'],
            'winning_trades': metrics['winning_trades'],
            'losing_trades': metrics['total_trades'] - metrics['winning_trades'],
            'win_rate': win_rate,
            'total_pnl_pct': metrics['total_pnl'],
            'avg_trade_pct': avg_trade,
            'best_trade_pct': metrics['best_trade'],
            'worst_trade_pct': metrics['worst_trade'],
            'avg_win_pct': metrics['avg_win'],
            'avg_loss_pct': metrics['avg_loss'],
            'active_trade': self.active_trade,
            'recent_trades': self.trades_history[-5:] if self.trades_history else []
        }

# ============================================
#  MAIN ENHANCED BOT - البوت الرئيسي المحسن
# ============================================

class SUIUltraProAIEnhanced:
    """البوت المحسن مع نظام SMC المتكامل وإدارة الرصيد"""
    
    def __init__(self, symbol: str = SYMBOL, interval: str = INTERVAL,
                 initial_balance: float = INITIAL_BALANCE,
                 risk_percent: float = RISK_PERCENT):
        
        self.logger = EnhancedProConsoleLogger()
        self.exchange = None
        self.symbol = symbol
        self.interval = interval
        self.initial_balance = initial_balance
        self.risk_percent = risk_percent
        self.trade_manager = None
        self.running = False
        
        # Flask App
        self.app = Flask(__name__)
        self.setup_flask_routes()
    
    def setup_flask_routes(self):
        """إعداد مسارات Flask"""
        @self.app.route('/')
        def home():
            return jsonify({
                'status': 'running',
                'bot': 'SUI Ultra Pro AI Enhanced',
                'version': '2.0',
                'symbol': self.symbol,
                'interval': self.interval
            })
        
        @self.app.route('/status')
        def status():
            if not self.trade_manager:
                return jsonify({'error': 'Trade manager not initialized'}), 500
            
            report = self.trade_manager.get_performance_report()
            return jsonify(report)
        
        @self.app.route('/balance')
        def balance():
            if not self.trade_manager:
                return jsonify({'error': 'Trade manager not initialized'}), 500
            
            balance_summary = self.trade_manager.get_balance_summary()
            return jsonify(balance_summary)
        
        @self.app.route('/trades')
        def trades():
            if not self.trade_manager:
                return jsonify({'error': 'Trade manager not initialized'}), 500
            
            trades_history = self.trade_manager.trades_history if hasattr(self.trade_manager, 'trades_history') else []
            return jsonify({
                'total_trades': len(trades_history),
                'trades': trades_history[-20:]  # آخر 20 صفقة
            })
    
    def initialize(self) -> bool:
        """تهيئة البوت"""
        try:
            self.logger.log_system("Initializing Enhanced SUI Ultra Pro AI Bot", "INFO")
            self.logger.log_system(f"Symbol: {self.symbol}, Interval: {self.interval}", "INFO")
            self.logger.log_system(f"Initial Balance: {self.initial_balance:.2f} USDT", "INFO")
            self.logger.log_system(f"Risk Percent: {self.risk_percent*100:.1f}%", "INFO")
            
            # تهيئة Exchange (مثال)
            # self.exchange = ccxt.bybit({...})
            
            # تهيئة مدير الصفقات
            self.trade_manager = EnhancedTradeManager(
                exchange=self.exchange,
                symbol=self.symbol,
                initial_balance=self.initial_balance,
                risk_percent=self.risk_percent,
                logger=self.logger
            )
            
            self.logger.log_system("Enhanced bot initialized successfully", "SUCCESS")
            return True
            
        except Exception as e:
            self.logger.log_error("Failed to initialize bot", e, "Initialization")
            return False
    
    def fetch_candles(self, limit: int = 100) -> List[Dict]:
        """جلب بيانات الشموع (محاكاة)"""
        # في التنفيذ الحقيقي، استخدم exchange.fetch_ohlcv()
        # هذا مثال للبيانات المحاكاة
        candles = []
        base_price = 1.0
        
        for i in range(limit):
            open_price = base_price * (1 + random.uniform(-0.01, 0.01))
            close_price = open_price * (1 + random.uniform(-0.02, 0.02))
            high_price = max(open_price, close_price) * (1 + random.uniform(0, 0.01))
            low_price = min(open_price, close_price) * (1 - random.uniform(0, 0.01))
            
            candles.append({
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': random.uniform(1000, 10000)
            })
        
        return candles
    
    def run_trade_loop(self):
        """تشغيل حلقة التداول الرئيسية"""
        self.logger.log_system("Starting Enhanced SUI Ultra Pro AI Bot Trading Loop", "INFO")
        self.running = True
        
        iteration_count = 0
        
        while self.running:
            try:
                iteration_count += 1
                
                # جلب بيانات الشموع
                candles = self.fetch_candles(100)
                
                if not candles or len(candles) < 50:
                    self.logger.log_system("Waiting for more candle data...", "INFO")
                    time.sleep(5)
                    continue
                
                current_price = candles[-1]['close']
                
                # عرض السعر الحالي كل 10 دورات
                if iteration_count % 10 == 0:
                    price_color = AdvancedConsoleColors.FG.GREEN if candles[-1]['close'] > candles[-1]['open'] else AdvancedConsoleColors.FG.RED
                    price_icon = "📈" if candles[-1]['close'] > candles[-1]['open'] else "📉"
                    
                    print(
                        f"{AdvancedConsoleColors.FG.LIGHT_BLACK}[{datetime.now().strftime('%H:%M:%S')}] "
                        f"{price_icon} {self.symbol}: {price_color}{current_price:.4f}{AdvancedConsoleColors.RESET} | "
                        f"Volume: {candles[-1]['volume']:.0f}"
                    )
                
                # إذا كانت هناك صفقة نشطة
                if self.trade_manager.active_trade:
                    # إدارة الصفقة الحالية
                    self.trade_manager.manage_active_trade(candles)
                
                else:
                    # تحليل السوق للدخول
                    entry_scenarios = self.trade_manager.analyze_market_for_entry(candles)
                    
                    if entry_scenarios:
                        # اختيار أفضل سيناريو
                        best_scenario = entry_scenarios[0]
                        
                        # تنفيذ الصفقة إذا كان السيناريو جيد
                        if best_scenario['confidence'] > 0.75:
                            self.logger.log_system(f"Executing trade with confidence: {best_scenario['confidence']:.2f}", "INFO")
                            self.trade_manager.execute_trade(
                                scenario=best_scenario,
                                current_price=current_price
                            )
                
                # عرض أداء المحفظة كل 30 دورة
                if iteration_count % 30 == 0:
                    self._display_portfolio_performance()
                
                time.sleep(10)  # انتظار 10 ثواني بين الدورات
                
            except KeyboardInterrupt:
                self.logger.log_system("Trading loop stopped by user", "INFO")
                self.running = False
                break
                
            except Exception as e:
                self.logger.log_error("Error in trading loop", e, "Trade Loop")
                time.sleep(30)  # انتظار أطول عند الخطأ
    
    def _display_portfolio_performance(self):
        """عرض أداء المحفظة"""
        if not self.trade_manager:
            return
        
        report = self.trade_manager.get_performance_report()
        balance_summary = report['balance_summary']
        
        # خط فاصل
        separator = f"{AdvancedConsoleColors.BG.LIGHT_BLACK}{'='*70}{AdvancedConsoleColors.RESET}"
        
        print(f"\n{separator}")
        print(f"{AdvancedConsoleColors.BOLD}{AdvancedConsoleColors.FG.CYAN}📊 PORTFOLIO PERFORMANCE{AdvancedConsoleColors.RESET}")
        print(f"{AdvancedConsoleColors.BG.LIGHT_BLACK}{'-'*70}{AdvancedConsoleColors.RESET}")
        
        # رصيد المحفظة
        print(
            f"{AdvancedConsoleColors.FG.LIGHT_BLUE}💰 Balance:{AdvancedConsoleColors.RESET} "
            f"{balance_summary['current_balance']:,.2f} USDT | "
            f"{AdvancedConsoleColors.FG.GREEN if balance_summary['pnl_percent'] >= 0 else AdvancedConsoleColors.FG.RED}"
            f"Cumulative P&L: {balance_summary['cumulative_pnl']:+,.2f} USDT ({balance_summary['pnl_percent']:+.2f}%){AdvancedConsoleColors.RESET}"
        )
        
        # أداء التداول
        win_rate_color = AdvancedConsoleColors.FG.GREEN if report['win_rate'] >= 60 else AdvancedConsoleColors.FG.YELLOW if report['win_rate'] >= 50 else AdvancedConsoleColors.FG.RED
        
        print(
            f"{AdvancedConsoleColors.FG.MAGENTA}📈 Trading Performance:{AdvancedConsoleColors.RESET} "
            f"Trades: {report['total_trades']} | "
            f"Win Rate: {win_rate_color}{report['win_rate']:.1f}%{AdvancedConsoleColors.RESET} | "
            f"Total P&L: {AdvancedConsoleColors.FG.GREEN if report['total_pnl_pct'] >= 0 else AdvancedConsoleColors.FG.RED}"
            f"{report['total_pnl_pct']:+.2f}%{AdvancedConsoleColors.RESET}"
        )
        
        # تفاصيل إضافية
        if report['total_trades'] > 0:
            print(
                f"{AdvancedConsoleColors.FG.YELLOW}📊 Details:{AdvancedConsoleColors.RESET} "
                f"Avg Trade: {report['avg_trade_pct']:+.2f}% | "
                f"Best: {AdvancedConsoleColors.FG.GREEN}{report['best_trade_pct']:+.2f}%{AdvancedConsoleColors.RESET} | "
                f"Worst: {AdvancedConsoleColors.FG.RED}{report['worst_trade_pct']:+.2f}%{AdvancedConsoleColors.RESET}"
            )
        
        # الصفقة النشطة
        active_status = f"{AdvancedConsoleColors.FG.GREEN}✅ Active" if report['active_trade'] else f"{AdvancedConsoleColors.FG.RED}❌ Inactive"
        print(f"{AdvancedConsoleColors.FG.CYAN}🔄 Active Trade:{AdvancedConsoleColors.RESET} {active_status}{AdvancedConsoleColors.RESET}")
        
        print(separator)
        
        # تسجيل في اللوجر
        self.logger.log_portfolio_summary(
            total_trades=report['total_trades'],
            win_rate=report['win_rate'],
            total_pnl=report['total_pnl_pct'],
            active_trade=report['active_trade']
        )
    
    def stop(self):
        """إيقاف البوت"""
        self.running = False
        self.logger.log_system("Bot stopped", "INFO")

# ============================================
#  FLASK APP SETUP - إعداد تطبيق Flask
# ============================================

app = Flask(__name__)

# إنشاء كائن البوت
bot_instance = None

@app.route('/')
def home():
    return jsonify({
        'status': 'running',
        'bot': 'SUI Ultra Pro AI Enhanced',
        'version': '2.0',
        'symbol': SYMBOL,
        'interval': INTERVAL,
        'port': PORT
    })

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

# ============================================
#  EXECUTION - التنفيذ
# ============================================

if __name__ == "__main__":
    try:
        logger.log_system("🚀 SUI ULTRA PRO AI - ENHANCED SMC EDITION", "INFO")
        logger.log_system("⚡ Smart Money Concepts Fully Integrated", "INFO")
        logger.log_system("🎯 7 Entry Scenarios | Advanced Protection System | Balance Management", "INFO")
        logger.log_system(f"Initial Balance: {INITIAL_BALANCE:.2f} USDT | Risk: {RISK_PERCENT*100:.1f}%", "INFO")
        logger.log_system("="*60, "INFO")
        
        # إنشاء البوت
        bot_instance = SUIUltraProAIEnhanced(
            symbol=SYMBOL,
            interval=INTERVAL,
            initial_balance=INITIAL_BALANCE,
            risk_percent=RISK_PERCENT
        )
        
        # تهيئة البوت
        if not bot_instance.initialize():
            logger.log_system("Failed to initialize bot. Exiting...", "ERROR")
            sys.exit(1)
        
        # تشغيل حلقة التداول في Thread منفصل
        trade_thread = threading.Thread(
            target=bot_instance.run_trade_loop,
            daemon=True
        )
        trade_thread.start()
        
        logger.log_system(f"Starting Flask server on port {PORT}", "INFO")
        logger.log_system(f"API Endpoints:", "INFO")
        logger.log_system(f"  • http://localhost:{PORT}/ - Home", "INFO")
        logger.log_system(f"  • http://localhost:{PORT}/status - Bot Status", "INFO")
        logger.log_system(f"  • http://localhost:{PORT}/balance - Balance Summary", "INFO")
        logger.log_system(f"  • http://localhost:{PORT}/trades - Recent Trades", "INFO")
        logger.log_system(f"  • http://localhost:{PORT}/health - Health Check", "INFO")
        
        # تشغيل Flask Server
        app.run(
            host="0.0.0.0",
            port=PORT,
            debug=False,
            use_reloader=False
        )
        
    except KeyboardInterrupt:
        logger.log_system("Bot stopped by user", "INFO")
        
    except Exception as e:
        logger.log_error(f"Fatal error in main: {str(e)}", e, "Main Execution")
        
    finally:
        if bot_instance:
            bot_instance.stop()
        logger.log_system("Bot shutdown complete", "INFO")
