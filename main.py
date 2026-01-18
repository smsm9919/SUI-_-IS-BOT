# -*- coding: utf-8 -*-
"""
SUI ULTRA PRO AI BOT - الإصدار الذكي المتكامل مع نظام إدارة الصفقات المتقدم
• نظام إدارة المراحل الذكي (Entry → Protect → BE → Trail → Trim → Exit)
• نظام TradePlan الذكي (قلب البوت الجديد)
• لوج احترافي ذكي مع ألوان وأيقونات وأسباب القرارات
• Structure-Based Trailing (ليس ATR تقليدي)
• حماية تنفيذية من أخطاء Bybit/MinQty
• نظام Trim الذكي لتقليل المخاطرة
• Fail-Fast Logic (قتل الصفقات الغلط بدري)
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
#  COLOR & LOGGING SYSTEM - نظام الألوان والتسجيل
# ============================================

try:
    from termcolor import colored
except Exception:
    def colored(text, color=None, on_color=None, attrs=None):
        return text

class ConsoleColors:
    """ألوان الكونسول للإخراج الجميل"""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    # ألوان النص
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
    BG_BLACK = '\033[40m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'
    BG_WHITE = '\033[47m'

class LogCategory(Enum):
    """فئات اللوج المختلفة"""
    MARKET = "MARKET"
    ENTRY = "ENTRY"
    EXECUTION = "EXECUTION"
    MANAGEMENT = "MANAGEMENT"
    EXIT = "EXIT"
    SYSTEM = "SYSTEM"
    ERROR = "ERROR"
    DEBUG = "DEBUG"
    PLAN = "PLAN"  # إضافة فئة جديدة للخطط

class ProConsoleLogger:
    """
    لوجر احترافي للكونسول مع ألوان وأيقونات وتنسيق متقدم
    """
    
    # تعريف الطبقات والألوان والأيقونات
    LAYER_CONFIG = {
        LogCategory.MARKET: {
            'color': ConsoleColors.CYAN,
            'icon': '📊',
            'emoji': '🌐',
            'width': 10,
            'bg_color': None
        },
        LogCategory.ENTRY: {
            'color': ConsoleColors.GREEN,
            'icon': '🎯',
            'emoji': '⚡',
            'width': 10,
            'bg_color': None
        },
        LogCategory.EXECUTION: {
            'color': ConsoleColors.YELLOW,
            'icon': '⚙️',
            'emoji': '🚀',
            'width': 10,
            'bg_color': None
        },
        LogCategory.MANAGEMENT: {
            'color': ConsoleColors.MAGENTA,
            'icon': '🔄',
            'emoji': '🎮',
            'width': 10,
            'bg_color': None
        },
        LogCategory.EXIT: {
            'color': ConsoleColors.RED,
            'icon': '🚪',
            'emoji': '💰',
            'width': 10,
            'bg_color': None
        },
        LogCategory.SYSTEM: {
            'color': ConsoleColors.LIGHT_WHITE,
            'icon': '⚡',
            'emoji': '🤖',
            'width': 10,
            'bg_color': None
        },
        LogCategory.ERROR: {
            'color': ConsoleColors.RED,
            'icon': '❌',
            'emoji': '🚨',
            'width': 10,
            'bg_color': ConsoleColors.BG_BLACK
        },
        LogCategory.DEBUG: {
            'color': ConsoleColors.LIGHT_BLACK,
            'icon': '🔍',
            'emoji': '🐛',
            'width': 10,
            'bg_color': None
        },
        LogCategory.PLAN: {
            'color': ConsoleColors.LIGHT_MAGENTA,
            'icon': '📋',
            'emoji': '🧠',
            'width': 10,
            'bg_color': None
        }
    }
    
    # رموز الحالة
    STATUS_ICONS = {
        'SUCCESS': '✅',
        'FAILURE': '❌',
        'WARNING': '⚠️',
        'INFO': 'ℹ️',
        'ARROW_UP': '🔼',
        'ARROW_DOWN': '🔽',
        'CHECK': '✔',
        'CROSS': '✘',
        'STAR': '⭐',
        'FIRE': '🔥',
        'ROCKET': '🚀',
        'ALERT': '🚨',
        'UP_TREND': '📈',
        'DOWN_TREND': '📉',
        'CONSOLIDATION': '↔️',
        'PLAN': '📋',
        'TARGET': '🎯',
        'STOP': '🛑',
        'LIQUIDITY': '💧'
    }
    
    def __init__(self, show_timestamp: bool = True, show_emoji: bool = True):
        """
        تهيئة اللوجر
        
        Args:
            show_timestamp: عرض الطابع الزمني
            show_emoji: عرض الإيموجي
        """
        self.show_timestamp = show_timestamp
        self.show_emoji = show_emoji
        
        # إعداد نظام تسجيل الملفات
        self.setup_file_logging()
    
    def setup_file_logging(self):
        """إعداد تسجيل الملفات"""
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        log_file = os.path.join(log_dir, f"sui_bot_{datetime.now().strftime('%Y%m%d')}.log")
        
        # إعداد اللوجر الأساسي
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                RotatingFileHandler(log_file, maxBytes=10485760, backupCount=5),
                logging.StreamHandler()
            ]
        )
        self.file_logger = logging.getLogger('SUI_BOT')
    
    def _format_timestamp(self) -> str:
        """تنسيق الطابع الزمني"""
        now = datetime.now()
        return f"{now.hour:02d}:{now.minute:02d}:{now.second:02d}"
    
    def _get_layer_header(self, category: LogCategory) -> str:
        """إنشاء هيدر الطبقة"""
        if category not in self.LAYER_CONFIG:
            category = LogCategory.SYSTEM
        
        config = self.LAYER_CONFIG[category]
        layer_name = f"[{category.value}]".ljust(config['width'])
        
        # بناء الهيدر مع الألوان والإيموجي
        if self.show_emoji:
            icon = config['emoji']
        else:
            icon = config['icon']
        
        # إضافة اللون
        color_code = config['color']
        bg_code = config['bg_color'] if config['bg_color'] else ""
        
        header = f"{color_code}{bg_code}{icon} {layer_name}{ConsoleColors.RESET}"
        return header
    
    def _format_reason(self, reason: str) -> str:
        """تنسيق سبب القرار (WHY)"""
        return f"{ConsoleColors.LIGHT_YELLOW}WHY: {reason}{ConsoleColors.RESET}"
    
    def _format_metric(self, key: str, value: Any, unit: str = '') -> str:
        """تنسيق القياسات"""
        # تحديد اللون بناءً على القيمة
        if isinstance(value, (int, float)):
            if value > 0:
                color = ConsoleColors.GREEN
            elif value < 0:
                color = ConsoleColors.RED
            else:
                color = ConsoleColors.YELLOW
            value_str = f"{value:+.3f}" if isinstance(value, float) else str(value)
        else:
            color = ConsoleColors.CYAN
            value_str = str(value)
        
        return f"{ConsoleColors.LIGHT_BLACK}{key}={color}{value_str}{ConsoleColors.LIGHT_BLACK}{unit}{ConsoleColors.RESET}"
    
    def log_market(self, 
                  timeframe: str,
                  trend: str,
                  structure: str,
                  liquidity: str,
                  momentum: Optional[float] = None,
                  volume_profile: Optional[str] = None,
                  reason: Optional[str] = None,
                  extra_details: Dict[str, Any] = None):
        """
        طباعة تحليل السوق
        
        Args:
            timeframe: الإطار الزمني
            trend: الاتجاه (BULL/BEAR/SIDEWAYS)
            structure: الهيكل (BOS/CHoCH/OB/DISTRIBUTION)
            liquidity: السيولة (HIGH/LOW)
            momentum: الزخم (اختياري)
            volume_profile: ملف الحجم (اختياري)
            reason: سبب التحليل (اختياري)
            extra_details: تفاصيل إضافية (اختياري)
        """
        header = self._get_layer_header(LogCategory.MARKET)
        
        # بناء الرسالة
        parts = []
        
        # المعلومات الأساسية
        parts.append(f"TF={timeframe}")
        
        # الاتجاه مع أيقونة
        trend_icon = self.STATUS_ICONS['UP_TREND'] if trend == 'BULL' else self.STATUS_ICONS['DOWN_TREND'] if trend == 'BEAR' else self.STATUS_ICONS['CONSOLIDATION']
        parts.append(f"Trend={trend_icon}{trend}")
        parts.append(f"Structure={structure}")
        
        # إضافة الزخم إذا كان موجوداً
        if momentum is not None:
            momentum_icon = self.STATUS_ICONS['ARROW_UP'] if momentum > 0 else self.STATUS_ICONS['ARROW_DOWN']
            parts.append(f"Momentum={momentum_icon}{abs(momentum):.1f}")
        
        # السيولة مع أيقونة
        liquidity_icon = self.STATUS_ICONS['CHECK'] if 'HIGH' in liquidity.upper() else self.STATUS_ICONS['CROSS']
        parts.append(f"Liquidity={liquidity} {liquidity_icon}")
        
        # ملف الحجم إذا كان موجوداً
        if volume_profile:
            parts.append(f"Volume={volume_profile}")
        
        # تفاصيل إضافية
        if extra_details:
            for key, value in extra_details.items():
                parts.append(f"{key}={value}")
        
        # تجميع الأجزاء
        message = f"{ConsoleColors.LIGHT_WHITE} | ".join(parts) + ConsoleColors.RESET
        
        # الطباعة
        if self.show_timestamp:
            print(f"{ConsoleColors.LIGHT_BLACK}[{self._format_timestamp()}]{ConsoleColors.RESET} {header} {message}")
        else:
            print(f"{header} {message}")
        
        # سبب التحليل إذا كان موجوداً
        if reason:
            why_text = self._format_reason(reason)
            indent = " " * (len(self._format_timestamp()) + 3 if self.show_timestamp else 0)
            print(f"{indent}{why_text}")
        
        # تسجيل في الملف
        self.file_logger.info(f"MARKET | TF={timeframe} | Trend={trend} | Structure={structure} | Liquidity={liquidity}")
    
    def log_entry(self,
                 side: str,
                 zone_type: str,
                 candle_pattern: str,
                 confidence: float,
                 reason: Optional[str] = None,
                 priority: Optional[str] = None,
                 entry_price: Optional[float] = None):
        """
        طباعة نقطة الدخول
        
        Args:
            side: الجانب (BUY/SELL)
            zone_type: نوع المنطقة (DEMAND/SUPPLY/OB/LQ)
            candle_pattern: نمط الشمعة (Rejection/Absorption/PinBar/Engulfing)
            confidence: الثقة (0-1)
            reason: سبب الدخول (اختياري)
            priority: الأولوية (HIGH/MEDIUM/LOW) (اختياري)
            entry_price: سعر الدخول (اختياري)
        """
        header = self._get_layer_header(LogCategory.ENTRY)
        
        # تحديد لون الجانب
        if side.upper() == 'BUY':
            side_color = ConsoleColors.GREEN
            side_icon = '🟢'
        else:
            side_color = ConsoleColors.RED
            side_icon = '🔴'
        
        # تحديد أيقونة الثقة
        if confidence >= 0.8:
            conf_icon = self.STATUS_ICONS['STAR']
            conf_color = ConsoleColors.GREEN
        elif confidence >= 0.6:
            conf_icon = self.STATUS_ICONS['CHECK']
            conf_color = ConsoleColors.YELLOW
        else:
            conf_icon = self.STATUS_ICONS['WARNING']
            conf_color = ConsoleColors.RED
        
        # بناء الرسالة
        parts = []
        parts.append(f"Side={side_color}{side_icon}{side}{ConsoleColors.RESET}")
        parts.append(f"Zone={zone_type}")
        parts.append(f"Candle={candle_pattern}")
        parts.append(f"Conf={conf_color}{conf_icon}{confidence:.2f}{ConsoleColors.RESET}")
        
        # إضافة سعر الدخول إذا كان موجوداً
        if entry_price is not None:
            parts.append(f"Price={entry_price:.4f}")
        
        # إضافة الأولوية إذا كانت موجودة
        if priority:
            priority_color = ConsoleColors.RED if priority == 'HIGH' else ConsoleColors.YELLOW if priority == 'MEDIUM' else ConsoleColors.GREEN
            parts.append(f"Priority={priority_color}{priority}{ConsoleColors.RESET}")
        
        message = f"{ConsoleColors.LIGHT_WHITE} | ".join(parts) + ConsoleColors.RESET
        
        # الطباعة
        if self.show_timestamp:
            print(f"{ConsoleColors.LIGHT_BLACK}[{self._format_timestamp()}]{ConsoleColors.RESET} {header} {message}")
        else:
            print(f"{header} {message}")
        
        # سبب الدخول
        if reason:
            why_text = self._format_reason(reason)
            indent = " " * (len(self._format_timestamp()) + 3 if self.show_timestamp else 0)
            print(f"{indent}{why_text}")
        
        # تسجيل في الملف
        self.file_logger.info(f"ENTRY | Side={side} | Zone={zone_type} | Confidence={confidence:.2f}")
    
    def log_execution(self,
                     price: float,
                     quantity: float,
                     stop_loss: float,
                     sl_reason: str,
                     order_type: str = "MARKET",
                     exchange: str = "BYBIT",
                     position_value: Optional[float] = None):
        """
        طباعة تنفيذ الأمر
        
        Args:
            price: سعر الدخول
            quantity: الكمية
            stop_loss: وقف الخسارة
            sl_reason: سبب وقف الخسارة
            order_type: نوع الأمر (اختياري)
            exchange: المنصة (اختياري)
            position_value: قيمة المركز (اختياري)
        """
        header = self._get_layer_header(LogCategory.EXECUTION)
        
        # حساب نسبة المخاطرة
        risk_percent = abs((stop_loss - price) / price * 100)
        
        # بناء الرسالة
        parts = []
        parts.append(f"{ConsoleColors.CYAN}Price={ConsoleColors.YELLOW}{price:.4f}{ConsoleColors.RESET}")
        parts.append(f"Qty={ConsoleColors.GREEN}{quantity:.2f}{ConsoleColors.RESET}")
        parts.append(f"SL={ConsoleColors.RED}{stop_loss:.4f} ({sl_reason}){ConsoleColors.RESET}")
        parts.append(f"Risk={risk_percent:.2f}%")
        parts.append(f"Type={order_type}")
        parts.append(f"Exchange={exchange}")
        
        # قيمة المركز إذا كانت موجودة
        if position_value:
            parts.append(f"Value=${position_value:.2f}")
        
        message = f"{ConsoleColors.LIGHT_WHITE} | ".join(parts) + ConsoleColors.RESET
        
        # الطباعة
        if self.show_timestamp:
            print(f"{ConsoleColors.LIGHT_BLACK}[{self._format_timestamp()}]{ConsoleColors.RESET} {header} {message}")
        else:
            print(f"{header} {message}")
        
        # إضافة خط فاصل
        print(f"{ConsoleColors.LIGHT_BLACK}{'─' * 80}{ConsoleColors.RESET}")
        
        # تسجيل في الملف
        self.file_logger.info(f"EXECUTION | Price={price:.4f} | Qty={quantity:.2f} | SL={stop_loss:.4f}")
    
    def log_management(self,
                      phase: str,
                      action: str,
                      reason: str,
                      current_pnl: Optional[float] = None,
                      new_stop_loss: Optional[float] = None,
                      trimmed_qty: Optional[float] = None,
                      extra_details: Dict[str, Any] = None):
        """
        طباعة إدارة الصفقة
        
        Args:
            phase: المرحلة (PROTECT/BREAKEVEN/TRAIL/TRIM)
            action: الإجراء (MOVE_SL/TRIM/ADD/BREAKEVEN)
            reason: السبب
            current_pnl: الربح/الخسارة الحالي % (اختياري)
            new_stop_loss: وقف الخسارة الجديد (اختياري)
            trimmed_qty: الكمية المقتطعة (اختياري)
            extra_details: تفاصيل إضافية (اختياري)
        """
        header = self._get_layer_header(LogCategory.MANAGEMENT)
        
        # تحديد لون الربح/الخسارة
        if current_pnl is not None:
            if current_pnl >= 0:
                pnl_color = ConsoleColors.GREEN
                pnl_icon = self.STATUS_ICONS['ARROW_UP']
            else:
                pnl_color = ConsoleColors.RED
                pnl_icon = self.STATUS_ICONS['ARROW_DOWN']
            pnl_text = f"PnL={pnl_color}{pnl_icon}{abs(current_pnl):.2f}%{ConsoleColors.RESET}"
        else:
            pnl_text = ""
        
        # بناء الرسالة
        parts = []
        if pnl_text:
            parts.append(pnl_text)
        
        parts.append(f"Phase={phase}")
        parts.append(f"Action={ConsoleColors.CYAN}{action}{ConsoleColors.RESET}")
        
        # إضافة وقف الخسارة الجديد إذا كان موجوداً
        if new_stop_loss is not None:
            parts.append(f"NewSL={ConsoleColors.YELLOW}{new_stop_loss:.4f}{ConsoleColors.RESET}")
        
        # إضافة الكمية المقتطعة إذا كانت موجودة
        if trimmed_qty is not None:
            parts.append(f"Trimmed={ConsoleColors.LIGHT_MAGENTA}{trimmed_qty:.2f}{ConsoleColors.RESET}")
        
        # تفاصيل إضافية
        if extra_details:
            for key, value in extra_details.items():
                parts.append(f"{key}={value}")
        
        message = f"{ConsoleColors.LIGHT_WHITE} | ".join(parts) + ConsoleColors.RESET
        
        # الطباعة
        if self.show_timestamp:
            print(f"{ConsoleColors.LIGHT_BLACK}[{self._format_timestamp()}]{ConsoleColors.RESET} {header} {message}")
        else:
            print(f"{header} {message}")
        
        # سبب الإجراء
        if reason:
            why_text = self._format_reason(reason)
            indent = " " * (len(self._format_timestamp()) + 3 if self.show_timestamp else 0)
            print(f"{indent}{why_text}")
        
        # تسجيل في الملف
        self.file_logger.info(f"MANAGEMENT | Phase={phase} | Action={action} | Reason={reason}")
    
    def log_exit(self,
                reason: str,
                final_pnl: float,
                risk_reward: Optional[float] = None,
                exit_price: Optional[float] = None,
                trade_duration: Optional[str] = None,
                summary: Optional[Dict[str, Any]] = None):
        """
        طباعة الخروج من الصفقة
        
        Args:
            reason: سبب الخروج
            final_pnl: الربح/الخسارة النهائي (%)
            risk_reward: نسبة المخاطرة/العائد (اختياري)
            exit_price: سعر الخروج (اختياري)
            trade_duration: مدة الصفقة (اختياري)
            summary: ملخص الصفقة (اختياري)
        """
        header = self._get_layer_header(LogCategory.EXIT)
        
        # تحديد لون الربح/الخسارة والأيقونة
        if final_pnl > 0:
            pnl_color = ConsoleColors.GREEN
            pnl_icon = self.STATUS_ICONS['ROCKET'] if final_pnl > 2 else self.STATUS_ICONS['FIRE']
        else:
            pnl_color = ConsoleColors.RED
            pnl_icon = self.STATUS_ICONS['CROSS']
        
        # بناء الرسالة
        parts = []
        parts.append(f"Reason={reason}")
        parts.append(f"PnL={pnl_color}{pnl_icon}{final_pnl:+.2f}%{ConsoleColors.RESET}")
        
        # إضافة نسبة المخاطرة/العائد إذا كانت موجودة
        if risk_reward is not None:
            if risk_reward >= 2:
                rr_color = ConsoleColors.GREEN
                rr_icon = self.STATUS_ICONS['STAR']
            elif risk_reward >= 1:
                rr_color = ConsoleColors.YELLOW
                rr_icon = self.STATUS_ICONS['CHECK']
            else:
                rr_color = ConsoleColors.RED
                rr_icon = self.STATUS_ICONS['WARNING']
            parts.append(f"RR={rr_color}{rr_icon}1:{risk_reward:.1f}{ConsoleColors.RESET}")
        
        # إضافة سعر الخروج إذا كان موجوداً
        if exit_price is not None:
            parts.append(f"ExitPrice={exit_price:.4f}")
        
        # إضافة مدة الصفقة إذا كانت موجودة
        if trade_duration:
            parts.append(f"Duration={trade_duration}")
        
        message = f"{ConsoleColors.LIGHT_WHITE} | ".join(parts) + ConsoleColors.RESET
        
        # الطباعة
        if self.show_timestamp:
            print(f"{ConsoleColors.LIGHT_BLACK}[{self._format_timestamp()}]{ConsoleColors.RESET} {header} {message}")
        else:
            print(f"{header} {message}")
        
        # إضافة ملخص إذا كان موجوداً
        if summary:
            indent = " " * (len(self._format_timestamp()) + 3 if self.show_timestamp else 0)
            print(f"{indent}{ConsoleColors.LIGHT_BLACK}{'─' * 60}{ConsoleColors.RESET}")
            for key, value in summary.items():
                print(f"{indent}{ConsoleColors.LIGHT_BLACK}{key}: {ConsoleColors.LIGHT_WHITE}{value}{ConsoleColors.RESET}")
        
        # إضافة خط فاصل مزدوج
        print(f"{ConsoleColors.LIGHT_BLACK}{'═' * 80}{ConsoleColors.RESET}\n")
        
        # تسجيل في الملف
        self.file_logger.info(f"EXIT | Reason={reason} | PnL={final_pnl:.2f}%")
    
    def log_system(self, message: str, status: str = "INFO", details: Optional[Dict] = None):
        """
        طباعة رسائل النظام
        
        Args:
            message: الرسالة
            status: الحالة (INFO/SUCCESS/ERROR/WARNING)
            details: تفاصيل إضافية (اختياري)
        """
        header = self._get_layer_header(LogCategory.SYSTEM)
        
        # تحديد لون الحالة
        if status == "SUCCESS":
            status_color = ConsoleColors.GREEN
            status_icon = self.STATUS_ICONS['SUCCESS']
        elif status == "ERROR":
            status_color = ConsoleColors.RED
            status_icon = self.STATUS_ICONS['FAILURE']
        elif status == "WARNING":
            status_color = ConsoleColors.YELLOW
            status_icon = self.STATUS_ICONS['WARNING']
        else:
            status_color = ConsoleColors.CYAN
            status_icon = self.STATUS_ICONS['INFO']
        
        # إضافة التفاصيل إذا وجدت
        details_str = ""
        if details:
            details_list = [f"{k}: {v}" for k, v in details.items()]
            details_str = f" | {' | '.join(details_list)}"
        
        full_message = f"{status_color}{status_icon} {message}{details_str}{ConsoleColors.RESET}"
        
        if self.show_timestamp:
            print(f"{ConsoleColors.LIGHT_BLACK}[{self._format_timestamp()}]{ConsoleColors.RESET} {header} {full_message}")
        else:
            print(f"{header} {full_message}")
        
        # تسجيل في الملف
        self.file_logger.info(f"SYSTEM | {status} | {message}")
    
    def log_error(self, message: str, error: Optional[Exception] = None, context: Optional[str] = None):
        """
        طباعة الأخطاء
        
        Args:
            message: رسالة الخطأ
            error: كائن الاستثناء (اختياري)
            context: السياق (اختياري)
        """
        header = self._get_layer_header(LogCategory.ERROR)
        
        # بناء الرسالة
        error_msg = f"{self.STATUS_ICONS['ALERT']} {message}"
        
        if context:
            error_msg += f" | Context: {context}"
        
        if error:
            error_msg += f" | Error: {str(error)}"
        
        if self.show_timestamp:
            print(f"{ConsoleColors.LIGHT_BLACK}[{self._format_timestamp()}]{ConsoleColors.RESET} {header} {ConsoleColors.RED}{error_msg}{ConsoleColors.RESET}")
        else:
            print(f"{header} {ConsoleColors.RED}{error_msg}{ConsoleColors.RESET}")
        
        # تسجيل في الملف
        self.file_logger.error(f"ERROR | {message} | Error: {str(error) if error else 'N/A'} | Context: {context if context else 'N/A'}")
    
    def log_debug(self, message: str, data: Optional[Any] = None):
        """
        طباعة معلومات التصحيح
        
        Args:
            message: الرسالة
            data: البيانات (اختياري)
        """
        header = self._get_layer_header(LogCategory.DEBUG)
        
        if self.show_timestamp:
            print(f"{ConsoleColors.LIGHT_BLACK}[{self._format_timestamp()}]{ConsoleColors.RESET} {header} {ConsoleColors.LIGHT_WHITE}{message}{ConsoleColors.RESET}")
        else:
            print(f"{header} {ConsoleColors.LIGHT_WHITE}{message}{ConsoleColors.RESET}")
        
        if data is not None:
            indent = " " * (len(self._format_timestamp()) + 3 if self.show_timestamp else 0)
            print(f"{indent}{ConsoleColors.LIGHT_BLACK}↳ {ConsoleColors.LIGHT_WHITE}{data}{ConsoleColors.RESET}")
        
        # تسجيل في الملف
        self.file_logger.debug(f"DEBUG | {message}")
    
    def log_plan(self, plan_details: Dict, title: str = "TRADE PLAN"):
        """
        تسجيل خطة التداول
        
        Args:
            plan_details: تفاصيل الخطة
            title: عنوان اللوج
        """
        header = self._get_layer_header(LogCategory.PLAN)
        
        # طباعة عنوان الخطة
        border = "━" * 40
        if self.show_timestamp:
            print(f"{ConsoleColors.LIGHT_BLACK}[{self._format_timestamp()}]{ConsoleColors.RESET} {header} {ConsoleColors.LIGHT_MAGENTA}┏{border}┓{ConsoleColors.RESET}")
            print(f"{ConsoleColors.LIGHT_BLACK}[{self._format_timestamp()}]{ConsoleColors.RESET} {header} {ConsoleColors.LIGHT_MAGENTA}┃ {ConsoleColors.BOLD}{title}{ConsoleColors.RESET}{ConsoleColors.LIGHT_MAGENTA}{' ' * (38 - len(title))}┃{ConsoleColors.RESET}")
            print(f"{ConsoleColors.LIGHT_BLACK}[{self._format_timestamp()}]{ConsoleColors.RESET} {header} {ConsoleColors.LIGHT_MAGENTA}┣{border}┫{ConsoleColors.RESET}")
        else:
            print(f"{header} {ConsoleColors.LIGHT_MAGENTA}┏{border}┓{ConsoleColors.RESET}")
            print(f"{header} {ConsoleColors.LIGHT_MAGENTA}┃ {ConsoleColors.BOLD}{title}{ConsoleColors.RESET}{ConsoleColors.LIGHT_MAGENTA}{' ' * (38 - len(title))}┃{ConsoleColors.RESET}")
            print(f"{header} {ConsoleColors.LIGHT_MAGENTA}┣{border}┫{ConsoleColors.RESET}")
        
        # تفاصيل الخطة
        for key, value in plan_details.items():
            # تحديد اللون بناءً على نوع القيمة
            if key in ['side', 'signal']:
                if str(value).upper() == 'BUY':
                    value_color = ConsoleColors.GREEN
                    value_icon = '🟢'
                else:
                    value_color = ConsoleColors.RED
                    value_icon = '🔴'
                value_str = f"{value_icon} {value}"
            elif key in ['trend_class', 'trend']:
                if str(value).upper() == 'LARGE':
                    value_color = ConsoleColors.CYAN
                elif str(value).upper() == 'BULL':
                    value_color = ConsoleColors.GREEN
                elif str(value).upper() == 'BEAR':
                    value_color = ConsoleColors.RED
                else:
                    value_color = ConsoleColors.YELLOW
                value_str = str(value)
            elif 'liquidity' in key.lower():
                value_color = ConsoleColors.LIGHT_BLUE
                value_str = f"💧 {value}"
            elif 'structure' in key.lower():
                value_color = ConsoleColors.LIGHT_YELLOW
                value_str = f"🏗️ {value}"
            elif 'zone' in key.lower():
                value_color = ConsoleColors.LIGHT_MAGENTA
                value_str = f"📍 {value}"
            elif 'sl' in key.lower() or 'stop' in key.lower():
                value_color = ConsoleColors.RED
                value_str = f"🛑 {value}"
            elif 'tp' in key.lower() or 'target' in key.lower():
                value_color = ConsoleColors.GREEN
                value_str = f"🎯 {value}"
            elif 'confirmation' in key.lower():
                value_color = ConsoleColors.LIGHT_GREEN
                value_str = f"✅ {value}"
            elif 'invalid' in key.lower():
                value_color = ConsoleColors.LIGHT_RED
                value_str = f"❌ {value}"
            else:
                value_color = ConsoleColors.LIGHT_WHITE
                value_str = str(value)
            
            if self.show_timestamp:
                print(f"{ConsoleColors.LIGHT_BLACK}[{self._format_timestamp()}]{ConsoleColors.RESET} {header} {ConsoleColors.LIGHT_MAGENTA}┃ {ConsoleColors.LIGHT_BLACK}{key}: {value_color}{value_str}{ConsoleColors.LIGHT_MAGENTA}{' ' * (36 - len(key) - len(str(value_str)))}┃{ConsoleColors.RESET}")
            else:
                print(f"{header} {ConsoleColors.LIGHT_MAGENTA}┃ {ConsoleColors.LIGHT_BLACK}{key}: {value_color}{value_str}{ConsoleColors.LIGHT_MAGENTA}{' ' * (36 - len(key) - len(str(value_str)))}┃{ConsoleColors.RESET}")
        
        # نهاية الخطة
        if self.show_timestamp:
            print(f"{ConsoleColors.LIGHT_BLACK}[{self._format_timestamp()}]{ConsoleColors.RESET} {header} {ConsoleColors.LIGHT_MAGENTA}┗{border}┛{ConsoleColors.RESET}")
        else:
            print(f"{header} {ConsoleColors.LIGHT_MAGENTA}┗{border}┛{ConsoleColors.RESET}")
        
        # تسجيل في الملف
        self.file_logger.info(f"PLAN | {title} | Details: {json.dumps(plan_details)}")
    
    def log_blocked_entry(self, reason: str, details: Dict = None):
        """تسجيل منع دخول صفقة"""
        header = self._get_layer_header(LogCategory.ENTRY)
        
        message = f"{ConsoleColors.RED}🚫 ENTRY BLOCKED: {reason}{ConsoleColors.RESET}"
        
        if details:
            details_str = " | ".join([f"{k}={v}" for k, v in details.items()])
            message += f" | {ConsoleColors.LIGHT_BLACK}{details_str}{ConsoleColors.RESET}"
        
        if self.show_timestamp:
            print(f"{ConsoleColors.LIGHT_BLACK}[{self._format_timestamp()}]{ConsoleColors.RESET} {header} {message}")
        else:
            print(f"{header} {message}")
        
        # تسجيل في الملف
        self.file_logger.warning(f"ENTRY BLOCKED | Reason: {reason} | Details: {details if details else 'N/A'}")

# إنشاء كائن اللوجر العام
logger = ProConsoleLogger(show_timestamp=True, show_emoji=True)

# ============================================
#  TRADE PLAN STRUCTURE - قلب البوت الجديد
# ============================================

class TradePlan:
    """خطة التداول الذكية (قلب البوت)"""
    
    def __init__(self, side: str, trend_class: str):
        """
        تهيئة خطة التداول
        
        Args:
            side: اتجاه الصفقة (BUY/SELL)
            trend_class: نوع الاتجاه (MID/LARGE)
        """
        self.side = side.upper()                # BUY / SELL
        self.trend_class = trend_class.upper()  # MID / LARGE
        
        # أسباب الدخول
        self.entry_reason = {
            "liquidity": None,      # sweep_high / sweep_low
            "structure": None,      # BOS / CHoCH
            "zone": None,           # OB / FVG / DEMAND / SUPPLY
            "confirmation": None    # rejection / engulf / pin_bar
        }
        
        # مستويات الدخول والإبطال
        self.entry_price = None
        self.invalidation = None   # مستوى إبطال الصفقة
        
        # مستويات وقف الخسارة والأهداف
        self.stop_loss = None
        self.take_profit_1 = None
        self.take_profit_2 = None
        self.take_profit_3 = None
        
        # إدارة الصفقة
        self.trailing_mode = "STRUCTURE"  # STRUCTURE / ATR / NONE
        self.breakeven_rule = "AFTER_TP1" # AFTER_TP1 / NONE
        self.partial_exit_pct = 0.3       # نسبة الخروج الجزئي عند TP1
        
        # حالة الخطة
        self.created_at = time.time()
        self.valid = False
        self.reason = ""  # سبب إنشاء الخطة
        
        # تتبع الأداء
        self.tp1_hit = False
        self.tp2_hit = False
        self.tp3_hit = False
        self.stop_loss_hit = False
        
    def is_valid(self) -> bool:
        """التحقق من صلاحية الخطة"""
        required_fields = [
            self.stop_loss is not None,
            self.take_profit_1 is not None,
            self.invalidation is not None,
            self.valid,
            all(self.entry_reason.values())  # جميع أسباب الدخول موجودة
        ]
        
        return all(required_fields)
    
    def calculate_risk_reward(self, entry_price: float = None) -> float:
        """حساب نسبة المخاطرة/العائد"""
        if not entry_price:
            entry_price = self.entry_price
        
        if not all([entry_price, self.stop_loss, self.take_profit_1]):
            return 0.0
        
        # حساب المخاطرة
        if self.side == "BUY":
            risk = abs(entry_price - self.stop_loss)
            reward = abs(self.take_profit_1 - entry_price)
        else:  # SELL
            risk = abs(self.stop_loss - entry_price)
            reward = abs(entry_price - self.take_profit_1)
        
        if risk == 0:
            return 0.0
        
        return reward / risk
    
    def get_plan_summary(self) -> Dict[str, Any]:
        """الحصول على ملخص الخطة"""
        rr_ratio = self.calculate_risk_reward()
        
        return {
            'side': self.side,
            'trend_class': self.trend_class,
            'entry_reason': self.entry_reason,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit_1': self.take_profit_1,
            'take_profit_2': self.take_profit_2,
            'take_profit_3': self.take_profit_3,
            'invalidation': self.invalidation,
            'risk_reward': f"1:{rr_ratio:.2f}" if rr_ratio > 0 else "N/A",
            'valid': self.valid,
            'reason': self.reason,
            'created_at': datetime.fromtimestamp(self.created_at).isoformat()
        }

# ============================================
#  TRADE STATE MACHINE - نظام مراحل الصفقة (محدث)
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
    """محرك إدارة مراحل الصفقة (محدث مع TradePlan)"""
    
    def __init__(self, trade_plan: TradePlan, entry_price: float, logger: ProConsoleLogger):
        self.trade_plan = trade_plan
        self.entry_price = entry_price
        self.side = trade_plan.side
        self.entry_zone = trade_plan.entry_reason.get('zone', 'UNKNOWN')
        self.logger = logger
        self.current_state = TradeState.ENTRY
        self.state_changed_at = time.time()
        self.structure_levels = []  # مستويات الهيكل
        self.last_stop_loss = trade_plan.stop_loss
        self.trim_count = 0
        self.max_trims = 2
        self.state_log = []
        
        # إعدادات حسب نوع الصفقة
        self.protection_pct = 0.5  # حماية عند 0.5%
        self.be_pct = 0.3         # نقطة التعادل عند 0.3%
        self.trail_activation_pct = 0.8  # تفعيل التريل عند 0.8%
        self.trim_pct = 0.2       # تقليل 20% في كل ترايم
        
        # تتبع TP
        self.tp1_hit = False
        self.tp2_hit = False
        self.tp3_hit = False
        
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
        
        self.logger.log_management(
            phase=f"STATE_CHANGE",
            action=f"{old_state}→{new_state}",
            reason=reason,
            extra_details={'old_state': old_state, 'new_state': new_state}
        )
    
    def check_tp_hits(self, current_price: float) -> List[Dict]:
        """التحقق من ضرب مستويات TP"""
        tp_hits = []
        
        if not self.tp1_hit and self.trade_plan.take_profit_1:
            if (self.side == "BUY" and current_price >= self.trade_plan.take_profit_1) or \
               (self.side == "SELL" and current_price <= self.trade_plan.take_profit_1):
                self.tp1_hit = True
                tp_hits.append({'level': 'TP1', 'price': self.trade_plan.take_profit_1})
        
        if not self.tp2_hit and self.trade_plan.take_profit_2:
            if (self.side == "BUY" and current_price >= self.trade_plan.take_profit_2) or \
               (self.side == "SELL" and current_price <= self.trade_plan.take_profit_2):
                self.tp2_hit = True
                tp_hits.append({'level': 'TP2', 'price': self.trade_plan.take_profit_2})
        
        if not self.tp3_hit and self.trade_plan.take_profit_3:
            if (self.side == "BUY" and current_price >= self.trade_plan.take_profit_3) or \
               (self.side == "SELL" and current_price <= self.trade_plan.take_profit_3):
                self.tp3_hit = True
                tp_hits.append({'level': 'TP3', 'price': self.trade_plan.take_profit_3})
        
        return tp_hits

# ============================================
#  FAIL-FAST LOGIC - نظام قتل الصفقات الغلط
# ============================================

class FailFastSystem:
    """نظام Fail-Fast للخروج السريع من الصفقات الفاشلة"""
    
    def __init__(self, logger: ProConsoleLogger):
        self.logger = logger
        self.max_trade_time = 3600  # ساعة واحدة كحد أقصى للصفقة
        
    def check_fail_fast(self, trade_plan: TradePlan, current_price: float, 
                       candles: List[Dict], time_in_trade: float) -> Tuple[bool, str]:
        """
        التحقق من شروط Fail-Fast
        
        Returns:
            Tuple[bool, str]: (هل يجب الخروج فوراً, السبب)
        """
        reasons = []
        
        # 1. ضرب مستوى الإبطال
        if trade_plan.invalidation:
            if (trade_plan.side == "BUY" and current_price <= trade_plan.invalidation) or \
               (trade_plan.side == "SELL" and current_price >= trade_plan.invalidation):
                reasons.append("FAIL FAST — Invalidation hit")
        
        # 2. وقت الصفقة طويل بدون تقدم
        if time_in_trade > self.max_trade_time:
            profit_pct = self.calculate_profit_pct(trade_plan.side, trade_plan.entry_price, current_price)
            if abs(profit_pct) < 0.5:  # أقل من 0.5% ربح/خسارة
                reasons.append("FAIL FAST — Stuck trade (no progress)")
        
        # 3. Fake Breakout (اختراق كاذب)
        if self.detect_fake_breakout(candles, trade_plan.side):
            reasons.append("FAIL FAST — Fake breakout detected")
        
        # 4. إشارة عكسية قوية
        if self.detect_strong_reversal(candles, trade_plan.side):
            reasons.append("FAIL FAST — Strong reversal signal")
        
        # 5. اختراق هيكل ضد الصفقة
        if self.detect_structure_break(candles, trade_plan.side):
            reasons.append("FAIL FAST — Structure broken against trade")
        
        if reasons:
            return True, " | ".join(reasons)
        
        return False, ""
    
    def calculate_profit_pct(self, side: str, entry_price: float, current_price: float) -> float:
        """حساب نسبة الربح/الخسارة"""
        if side == "BUY":
            return ((current_price - entry_price) / entry_price) * 100
        else:
            return ((entry_price - current_price) / entry_price) * 100
    
    def detect_fake_breakout(self, candles: List[Dict], side: str) -> bool:
        """كشف الاختراق الكاذب"""
        if len(candles) < 3:
            return False
        
        current = candles[-1]
        prev = candles[-2]
        
        if side == "BUY":
            # اختراق كاذب صاعد: شمعة تخترق للأعلى ثم تغلق تحت
            if current['high'] > max([c['high'] for c in candles[-4:-1]]) and current['close'] < prev['close']:
                # حجم كبير على الشمعة الحالية
                if current['volume'] > sum([c['volume'] for c in candles[-4:-1]]) / 3 * 1.5:
                    return True
        else:  # SELL
            # اختراق كاذب هابط: شمعة تخترق للأسفل ثم تغلق فوق
            if current['low'] < min([c['low'] for c in candles[-4:-1]]) and current['close'] > prev['close']:
                # حجم كبير على الشمعة الحالية
                if current['volume'] > sum([c['volume'] for c in candles[-4:-1]]) / 3 * 1.5:
                    return True
        
        return False
    
    def detect_strong_reversal(self, candles: List[Dict], side: str) -> bool:
        """كشف إشارة انعكاس قوية"""
        if len(candles) < 3:
            return False
        
        current = candles[-1]
        prev = candles[-2]
        
        if side == "BUY":
            # شمعة هابطة قوية بعد صعود
            if current['close'] < current['open'] and \
               (current['open'] - current['close']) > (current['high'] - current['low']) * 0.7:
                # حجم كبير
                if current['volume'] > prev['volume'] * 1.3:
                    return True
        else:  # SELL
            # شمعة صاعدة قوية بعد هبوط
            if current['close'] > current['open'] and \
               (current['close'] - current['open']) > (current['high'] - current['low']) * 0.7:
                # حجم كبير
                if current['volume'] > prev['volume'] * 1.3:
                    return True
        
        return False
    
    def detect_structure_break(self, candles: List[Dict], side: str) -> bool:
        """كشف كسر الهيكل ضد الصفقة"""
        if len(candles) < 10:
            return False
        
        # البحث عن swing points
        highs = [c['high'] for c in candles[-10:]]
        lows = [c['low'] for c in candles[-10:]]
        
        if side == "BUY":
            # في الشراء: كسر آخر قاع مهم
            recent_lows = sorted(lows[-5:])
            if len(recent_lows) >= 2:
                last_swing_low = recent_lows[0]
                if candles[-1]['close'] < last_swing_low:
                    return True
        else:  # SELL
            # في البيع: كسر آخر قمة مهمة
            recent_highs = sorted(highs[-5:], reverse=True)
            if len(recent_highs) >= 2:
                last_swing_high = recent_highs[0]
                if candles[-1]['close'] > last_swing_high:
                    return True
        
        return False

# ============================================
#  TRADE PLAN BUILDER - بناء خطط التداول
# ============================================

class TradePlanBuilder:
    """بناء خطط التداول الذكية"""
    
    def __init__(self, logger: ProConsoleLogger):
        self.logger = logger
    
    def build_trade_plan(self, market_data: Dict) -> Optional[TradePlan]:
        """
        بناء خطة تداول بناءً على تحليل السوق
        
        Args:
            market_data: بيانات تحليل السوق
            
        Returns:
            TradePlan or None إذا فشل البناء
        """
        try:
            # استخراج البيانات الأساسية
            signal = market_data.get('signal', None)
            trend_class = market_data.get('trend_class', 'MID')
            
            if not signal:
                return None
            
            # إنشاء الخطة الأساسية
            plan = TradePlan(side=signal, trend_class=trend_class)
            
            # تعيين أسباب الدخول
            plan.entry_reason = {
                "liquidity": market_data.get('liquidity_event', 'UNKNOWN'),
                "structure": market_data.get('structure_event', 'UNKNOWN'),
                "zone": market_data.get('zone_type', 'UNKNOWN'),
                "confirmation": market_data.get('confirmation_type', 'UNKNOWN')
            }
            
            # التحقق من أسباب الدخول الأساسية
            if not plan.entry_reason['liquidity'] or not plan.entry_reason['zone']:
                self.logger.log_blocked_entry(
                    "Weak location / no liquidity",
                    plan.entry_reason
                )
                return None
            
            # تعيين مستويات الإبطال
            plan.invalidation = market_data.get('structure_invalid_level', None)
            if not plan.invalidation:
                self.logger.log_blocked_entry(
                    "No invalidation level defined",
                    plan.entry_reason
                )
                return None
            
            # تعيين وقف الخسارة
            plan.stop_loss = plan.invalidation
            
            # تعيين الأهداف بناءً على السيولة
            plan.take_profit_1 = market_data.get('internal_liquidity', None)
            plan.take_profit_2 = market_data.get('external_liquidity', None)
            
            if trend_class == "LARGE":
                plan.take_profit_3 = market_data.get('htf_liquidity', None)
            
            # التحقق من وجود أهداف
            if not plan.take_profit_1:
                self.logger.log_blocked_entry(
                    "No take profit levels defined",
                    plan.entry_reason
                )
                return None
            
            # تعيين إدارة الصفقة
            plan.trailing_mode = "STRUCTURE"
            plan.breakeven_rule = "AFTER_TP1"
            plan.partial_exit_pct = 0.3
            
            # حساب نسبة المخاطرة/العائد
            rr_ratio = plan.calculate_risk_reward(market_data.get('current_price', 0))
            
            # التحقق من نسبة المخاطرة/العائد
            if rr_ratio < 1.5:
                self.logger.log_blocked_entry(
                    f"Poor risk/reward ratio: 1:{rr_ratio:.2f}",
                    {
                        'RR_Ratio': f"1:{rr_ratio:.2f}",
                        'Min_Required': "1:1.5"
                    }
                )
                return None
            
            # تعليم الخطة كصالحة
            plan.valid = True
            plan.reason = f"Plan built based on {plan.entry_reason['structure']} at {plan.entry_reason['zone']} zone"
            
            # تسجيل الخطة
            plan_summary = plan.get_plan_summary()
            plan_summary['rr_ratio'] = rr_ratio
            self.logger.log_plan(plan_summary, "ENTRY PLAN APPROVED")
            
            return plan
            
        except Exception as e:
            self.logger.log_error(f"Failed to build trade plan: {str(e)}", e, "TradePlanBuilder")
            return None

# ============================================
#  SMART EXIT ENGINE - محرك الخروج الذكي
# ============================================

class SmartExitEngine:
    """محرك الخروج الذكي (Structure + Liquidity)"""
    
    def __init__(self, logger: ProConsoleLogger):
        self.logger = logger
        self.fail_fast_system = FailFastSystem(logger)
    
    def manage_trade_exits(self, trade_plan: TradePlan, trade_phase_engine: TradePhaseEngine,
                          current_price: float, candles: List[Dict], 
                          time_in_trade: float) -> List[Dict]:
        """
        إدارة عمليات الخروج من الصفقة
        
        Returns:
            List[Dict]: قائمة بأحداث الخروج التي تمت
        """
        exit_events = []
        
        # === FAIL-FAST CHECK ===
        should_fail_fast, fail_fast_reason = self.fail_fast_system.check_fail_fast(
            trade_plan, current_price, candles, time_in_trade
        )
        
        if should_fail_fast:
            exit_events.append({
                'type': 'FAIL_FAST',
                'reason': fail_fast_reason,
                'price': current_price,
                'partial': False
            })
            return exit_events
        
        # === CHECK TP HITS ===
        tp_hits = trade_phase_engine.check_tp_hits(current_price)
        
        for tp_hit in tp_hits:
            exit_events.append({
                'type': 'TAKE_PROFIT',
                'level': tp_hit['level'],
                'price': tp_hit['price'],
                'partial': True if tp_hit['level'] in ['TP1', 'TP2'] else False
            })
        
        # === STRUCTURE BREAK CHECK ===
        if self._check_structure_break(candles, trade_plan.side):
            exit_events.append({
                'type': 'STRUCTURE_BREAK',
                'reason': "Structure broken against trade direction",
                'price': current_price,
                'partial': False
            })
        
        # === LIQUIDITY TARGET REACHED ===
        if trade_plan.take_profit_3 and \
           ((trade_plan.side == "BUY" and current_price >= trade_plan.take_profit_3) or \
            (trade_plan.side == "SELL" and current_price <= trade_plan.take_profit_3)):
            exit_events.append({
                'type': 'FINAL_TARGET',
                'reason': "Final liquidity target reached",
                'price': current_price,
                'partial': False
            })
        
        return exit_events
    
    def _check_structure_break(self, candles: List[Dict], side: str) -> bool:
        """التحقق من كسر الهيكل"""
        if len(candles) < 5:
            return False
        
        # تحليل الهيكل الحالي
        highs = [c['high'] for c in candles[-5:]]
        lows = [c['low'] for c in candles[-5:]]
        closes = [c['close'] for c in candles[-5:]]
        
        if side == "BUY":
            # في الشراء: نبحث عن Lower Low
            if len(candles) >= 3:
                current_low = lows[-1]
                prev_low = lows[-2]
                if current_low < prev_low and closes[-1] < closes[-2]:
                    return True
        else:  # SELL
            # في البيع: نبحث عن Higher High
            if len(candles) >= 3:
                current_high = highs[-1]
                prev_high = highs[-2]
                if current_high > prev_high and closes[-1] > closes[-2]:
                    return True
        
        return False
    
    def calculate_partial_exit_size(self, trade_plan: TradePlan, exit_event: Dict) -> float:
        """حساب حجم الخروج الجزئي"""
        if exit_event['type'] == 'TAKE_PROFIT':
            if exit_event['level'] == 'TP1':
                return trade_plan.partial_exit_pct  # 30%
            elif exit_event['level'] == 'TP2':
                return 0.3  # 30% إضافية
            elif exit_event['level'] == 'TP3':
                return 1.0  # 100% (خروج كامل)
        
        elif exit_event['type'] == 'FINAL_TARGET':
            return 1.0  # خروج كامل
        
        elif exit_event['type'] in ['FAIL_FAST', 'STRUCTURE_BREAK']:
            return 1.0  # خروج كامل
        
        return 0.0  # لا خروج

# ============================================
#  EXECUTION GUARD - حماية التنفيذ
# ============================================

class ExecutionGuard:
    """حارس تنفيذ الأوامر مع Bybit"""
    
    def __init__(self, exchange, logger: ProConsoleLogger):
        self.exchange = exchange
        self.logger = logger
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
                self.logger.log_error(
                    f"Quantity {qty} < Minimum {min_qty} → ORDER CANCELLED",
                    context="Order Sanitization"
                )
                return None, f"Qty < Min: {qty} < {min_qty}"
            
            # التحقق من الحد الأقصى (إذا موجود)
            if 'max' in market['limits']['amount']:
                max_qty = market['limits']['amount']['max']
                if qty > max_qty:
                    qty = max_qty
                    self.logger.log_system(f"Quantity capped at maximum: {max_qty}", "INFO")
            
            self.logger.log_debug(f"Sanitized Qty: {qty} (Min: {min_qty}, Precision: {precision})")
            return qty, "VALID"
            
        except Exception as e:
            self.logger.log_error(f"Sanitization error: {str(e)}", e, "Order Sanitization")
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
        self.logger.log_debug("Order execution succeeded, resetting failure count")
    
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
        
        self.logger.log_error(
            f"Order failed ({self.failure_count}/{self.max_failures}): {error}",
            context="Order Execution"
        )

# ============================================
#  SMART TRADE MANAGER - المدير الرئيسي (محدث)
# ============================================

class SmartTradeManager:
    """المدير الذكي للصفقات (محدث مع TradePlan)"""
    
    def __init__(self, exchange, symbol: str, risk_percent: float = 0.6, logger: ProConsoleLogger = None):
        self.exchange = exchange
        self.symbol = symbol
        self.risk_percent = risk_percent
        self.logger = logger or ProConsoleLogger()
        
        # الأنظمة الفرعية
        self.execution_guard = ExecutionGuard(exchange, self.logger)
        self.trade_plan_builder = TradePlanBuilder(self.logger)
        self.smart_exit_engine = SmartExitEngine(self.logger)
        self.trade_phase_engine = None
        self.current_trade_plan = None
        
        # حالة التداول
        self.active_trade = False
        self.current_position = {
            'side': None,
            'entry_price': 0.0,
            'quantity': 0.0,
            'entry_time': None,
            'remaining_qty': 0.0
        }
        
        # إحصائيات
        self.trades_history = []
        self.total_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.fail_fast_exits = 0
    
    def open_trade_with_plan(self, trade_plan: TradePlan, balance: float, 
                            current_price: float) -> bool:
        """فتح صفقة بناءً على خطة محددة"""
        
        # التحقق من عدم وجود صفقة نشطة
        if self.active_trade:
            self.logger.log_error("Cannot open trade: Active trade exists", context="Trade Opening")
            return False
        
        # التحقق من صلاحية الخطة
        if not trade_plan or not trade_plan.is_valid():
            self.logger.log_error("Cannot open trade: Invalid trade plan", context="Trade Opening")
            return False
        
        # التحقق من صلاحية التنفيذ
        allow, allow_reason = self.execution_guard.should_allow_order()
        if not allow:
            self.logger.log_system(f"Order not allowed: {allow_reason}", "WARNING")
            return False
        
        # حساب حجم المركز
        qty = self.calculate_position_size(balance, current_price, trade_plan)
        if qty <= 0:
            return False
        
        # تعيين سعر الدخول في الخطة
        trade_plan.entry_price = current_price
        
        # تنفيذ الأمر (أو محاكاة)
        success = self.execute_order(trade_plan.side, qty, current_price, is_open=True)
        
        if success:
            # حفظ الخطة الحالية
            self.current_trade_plan = trade_plan
            
            # تحديث المركز الحالي
            self.current_position = {
                'side': trade_plan.side,
                'entry_price': current_price,
                'quantity': qty,
                'remaining_qty': qty,
                'entry_time': datetime.now(),
                'zone': trade_plan.entry_reason.get('zone', 'UNKNOWN'),
                'confidence': trade_plan.calculate_risk_reward(current_price) / 10  # ثقة بناءً على RR
            }
            
            # تهيئة نظام إدارة المراحل
            self.trade_phase_engine = TradePhaseEngine(trade_plan, current_price, self.logger)
            self.active_trade = True
            
            # تسجيل الصفقة
            trade_record = {
                'id': len(self.trades_history) + 1,
                'timestamp': datetime.now().isoformat(),
                'side': trade_plan.side,
                'entry_price': current_price,
                'qty': qty,
                'zone': trade_plan.entry_reason.get('zone', 'UNKNOWN'),
                'reason': trade_plan.reason,
                'plan': trade_plan.get_plan_summary(),
                'position_value': qty * current_price
            }
            self.trades_history.append(trade_record)
            
            # لوج الدخول
            self.logger.log_entry(
                side=trade_plan.side,
                zone_type=trade_plan.entry_reason.get('zone', 'UNKNOWN'),
                candle_pattern=trade_plan.entry_reason.get('confirmation', 'Signal'),
                confidence=min(trade_plan.calculate_risk_reward(current_price) / 10, 0.95),
                reason=trade_plan.reason,
                entry_price=current_price
            )
            
            # لوج تفاصيل التنفيذ
            self.logger.log_execution(
                price=current_price,
                quantity=qty,
                stop_loss=trade_plan.stop_loss,
                sl_reason=f"Invalidation level: {trade_plan.invalidation}",
                order_type="MARKET",
                exchange=self.exchange.name.upper(),
                position_value=qty * current_price
            )
            
            self.logger.log_system(
                f"Trade opened with plan | {trade_plan.side.upper()} @ {current_price:.4f} | Qty: {qty:.4f}",
                "SUCCESS",
                {
                    "Entry_Price": f"{current_price:.4f}",
                    "Quantity": f"{qty:.4f}",
                    "Zone": trade_plan.entry_reason.get('zone', 'UNKNOWN'),
                    "RR_Ratio": f"1:{trade_plan.calculate_risk_reward(current_price):.2f}"
                }
            )
            
            return True
        
        return False
    
    def calculate_position_size(self, balance: float, entry_price: float, 
                               trade_plan: TradePlan) -> float:
        """حساب حجم المركز الذكي بناءً على الخطة"""
        if not trade_plan or not trade_plan.stop_loss:
            return 0.0
        
        # حساب المخاطرة بالدولار
        risk_amount = balance * self.risk_percent
        
        # حساب المسافة بين الدخول ووقف الخسارة
        if trade_plan.side == "BUY":
            risk_distance = entry_price - trade_plan.stop_loss
        else:  # SELL
            risk_distance = trade_plan.stop_loss - entry_price
        
        if risk_distance <= 0:
            self.logger.log_error("Invalid risk distance (SL >= Entry for BUY or SL <= Entry for SELL)")
            return 0.0
        
        # حساب الكمية بناءً على المخاطرة
        raw_qty = risk_amount / risk_distance
        
        # تعديل حسب نوع الاتجاه
        if trade_plan.trend_class == "LARGE":
            raw_qty *= 1.2  # زيادة 20% للاتجاهات الكبيرة
        elif trade_plan.trend_class == "MID":
            raw_qty *= 1.0  # نفس الحجم للاتجاهات المتوسطة
        else:
            raw_qty *= 0.7  # تقليل 30% للاتجاهات الصغيرة
        
        # تنقية الكمية
        sanitized_qty, status = self.execution_guard.sanitize_order(self.symbol, raw_qty)
        
        if sanitized_qty is None:
            self.logger.log_error(f"Position size invalid: {status}")
            return 0.0
        
        self.logger.log_debug(
            f"Position Calculation | "
            f"Risk: ${risk_amount:.2f} | "
            f"Distance: {risk_distance:.6f} | "
            f"Raw: {raw_qty:.4f} | "
            f"Sanitized: {sanitized_qty:.4f} | "
            f"Trend Class: {trade_plan.trend_class}"
        )
        
        return sanitized_qty
    
    def manage_trade_with_plan(self, current_price: float, candles: List[Dict]):
        """إدارة الصفقة النشطة باستخدام الخطة"""
        if not self.active_trade or not self.trade_phase_engine or not self.current_trade_plan:
            return
        
        # حساب وقت الصفقة
        time_in_trade = time.time() - self.trade_phase_engine.state_changed_at
        
        # إدارة الخروجات باستخدام SmartExitEngine
        exit_events = self.smart_exit_engine.manage_trade_exits(
            self.current_trade_plan,
            self.trade_phase_engine,
            current_price,
            candles,
            time_in_trade
        )
        
        # معالجة أحداث الخروج
        for exit_event in exit_events:
            if exit_event['type'] == 'FAIL_FAST':
                self.close_trade(exit_event['reason'], current_price, is_fail_fast=True)
                self.fail_fast_exits += 1
                return
            
            elif exit_event['type'] == 'TAKE_PROFIT':
                exit_pct = self.smart_exit_engine.calculate_partial_exit_size(
                    self.current_trade_plan, exit_event
                )
                
                if exit_pct > 0:
                    if exit_pct >= 1.0:  # خروج كامل
                        self.close_trade(f"{exit_event['level']} reached", current_price)
                    else:  # خروج جزئي
                        self.execute_partial_exit(exit_pct, exit_event, current_price)
            
            elif exit_event['type'] in ['STRUCTURE_BREAK', 'FINAL_TARGET']:
                self.close_trade(exit_event['reason'], current_price)
                return
        
        # إدارة المراحل العادية (إذا لم يكن هناك خروج)
        self._update_trade_phase(current_price, candles)
        
        # حساب وقف الخسارة الحالي
        sl_price, sl_reason = self.trade_phase_engine.calculate_stop_loss(current_price, candles)
        
        # التحقق من وقف الخسارة
        if self._should_hit_stop_loss(current_price, sl_price):
            self.close_trade(f"Stop Loss: {sl_reason}", current_price)
            return
        
        # لوج حالة إدارة الصفقة
        profit_pct = self.calculate_profit_pct(current_price)
        state = self.trade_phase_engine.current_state
        
        current_time = time.time()
        if (current_time - self.trade_phase_engine.state_changed_at) < 30 or state != getattr(self, '_last_logged_state', None):
            self._last_logged_state = state
            
            self.logger.log_management(
                phase=state,
                action="HOLD",
                reason=f"Managing position with plan",
                current_pnl=profit_pct,
                new_stop_loss=sl_price,
                extra_details={
                    "State": state,
                    "Plan_Valid": self.current_trade_plan.valid,
                    "Remaining_Qty": f"{self.current_position['remaining_qty']:.4f}",
                    "Time_in_Trade": f"{int(time_in_trade)}s"
                }
            )
    
    def execute_partial_exit(self, exit_pct: float, exit_event: Dict, current_price: float):
        """تنفيذ خروج جزئي"""
        if not self.active_trade or self.current_position['remaining_qty'] <= 0:
            return
        
        # حساب الكمية للخروج
        exit_qty = self.current_position['remaining_qty'] * exit_pct
        
        # تحديث الكمية المتبقية
        self.current_position['remaining_qty'] -= exit_qty
        
        # تنفيذ الأمر (أو محاكاة)
        opposite_side = "sell" if self.current_trade_plan.side == "BUY" else "buy"
        success = self.execute_order(opposite_side, exit_qty, current_price, is_open=False)
        
        if success:
            # لوج الخروج الجزئي
            self.logger.log_management(
                phase="PARTIAL_EXIT",
                action="EXECUTING",
                reason=f"{exit_event['level']} reached",
                trimmed_qty=exit_qty,
                extra_details={
                    "Exit_Pct": f"{exit_pct*100:.0f}%",
                    "Remaining_Qty": f"{self.current_position['remaining_qty']:.4f}",
                    "Level": exit_event['level'],
                    "Price": current_price
                }
            )
            
            # إذا كان هذا هو TP1، نقل وقف الخسارة إلى نقطة التعادل
            if exit_event['level'] == 'TP1' and self.current_trade_plan.breakeven_rule == "AFTER_TP1":
                self.trade_phase_engine.update_state(TradeState.BREAKEVEN, 
                                                   "Moving to breakeven after TP1 hit")
    
    def close_trade(self, reason: str, exit_price: float, is_fail_fast: bool = False):
        """إغلاق الصفقة"""
        if not self.active_trade or self.trade_phase_engine is None:
            return
        
        # حساب الربح/الخسارة على الكمية المتبقية
        entry_price = self.trade_phase_engine.entry_price
        side = self.trade_phase_engine.side
        remaining_qty = self.current_position['remaining_qty']
        
        if side == "BUY":
            pnl_pct = ((exit_price - entry_price) / entry_price) * 100
            pnl_usd = (exit_price - entry_price) * remaining_qty
        else:
            pnl_pct = ((entry_price - exit_price) / entry_price) * 100
            pnl_usd = (entry_price - exit_price) * remaining_qty
        
        # حساب نسبة المخاطرة/العائد من الخطة الأصلية
        initial_risk_pct = abs((self.current_trade_plan.stop_loss - entry_price) / entry_price * 100) if self.current_trade_plan.stop_loss else 0
        risk_reward = abs(pnl_pct / initial_risk_pct) if initial_risk_pct > 0 else 0
        
        # مدة الصفقة
        if self.current_position['entry_time']:
            duration = datetime.now() - self.current_position['entry_time']
            duration_str = f"{duration.seconds // 3600}h {(duration.seconds % 3600) // 60}m {duration.seconds % 60}s"
        else:
            duration_str = "N/A"
        
        # تحديث الإحصائيات
        self.total_pnl += pnl_pct
        self.total_trades += 1
        
        if is_fail_fast:
            reason = f"FAIL-FAST: {reason}"
        elif pnl_pct > 0:
            self.winning_trades += 1
        
        # إغلاق المركز المتبقي (إذا كان هناك كمية متبقية)
        if remaining_qty > 0:
            opposite_side = "sell" if side == "BUY" else "buy"
            self.execute_order(opposite_side, remaining_qty, exit_price, is_open=False)
        
        # تحديث سجل الصفقة
        if self.trades_history:
            self.trades_history[-1].update({
                'exit_price': exit_price,
                'exit_reason': reason,
                'pnl_pct': pnl_pct,
                'pnl_usd': pnl_usd,
                'risk_reward': risk_reward,
                'exit_time': datetime.now().isoformat(),
                'duration': duration_str,
                'final_state': self.trade_phase_engine.current_state,
                'trim_count': self.trade_phase_engine.trim_count,
                'is_fail_fast': is_fail_fast,
                'tp1_hit': getattr(self.trade_phase_engine, 'tp1_hit', False),
                'tp2_hit': getattr(self.trade_phase_engine, 'tp2_hit', False),
                'tp3_hit': getattr(self.trade_phase_engine, 'tp3_hit', False)
            })
        
        # لوج الخروج
        self.logger.log_exit(
            reason=reason,
            final_pnl=pnl_pct,
            risk_reward=risk_reward,
            exit_price=exit_price,
            trade_duration=duration_str,
            summary={
                "Entry Price": f"{entry_price:.4f}",
                "Exit Price": f"{exit_price:.4f}",
                "Quantity": f"{remaining_qty:.4f}",
                "PnL USD": f"${pnl_usd:.2f}",
                "Trade Duration": duration_str,
                "Trade Phase": self.trade_phase_engine.current_state,
                "Trim Count": self.trade_phase_engine.trim_count,
                "Fail Fast": "Yes" if is_fail_fast else "No"
            }
        )
        
        # إعادة التعيين
        self.active_trade = False
        self.trade_phase_engine = None
        self.current_trade_plan = None
        self.current_position = {
            'side': None,
            'entry_price': 0.0,
            'quantity': 0.0,
            'remaining_qty': 0.0,
            'entry_time': None
        }
        
        self.logger.log_system(
            f"Trade closed | PnL: {pnl_pct:+.2f}% | Reason: {reason}",
            "INFO" if pnl_pct >= 0 else "WARNING",
            {
                "PnL": f"{pnl_pct:+.2f}%",
                "Reason": reason,
                "RR": f"1:{risk_reward:.1f}",
                "Fail_Fast": is_fail_fast
            }
        )
    
    def calculate_profit_pct(self, current_price: float) -> float:
        """حساب نسبة الربح/الخسارة الحالية"""
        if not self.active_trade or not self.current_position['entry_price']:
            return 0.0
        
        if self.current_position['side'] == "BUY":
            return ((current_price - self.current_position['entry_price']) / self.current_position['entry_price']) * 100
        else:
            return ((self.current_position['entry_price'] - current_price) / self.current_position['entry_price']) * 100
    
    def _should_hit_stop_loss(self, current_price: float, stop_loss: float) -> bool:
        """التحقق من ضرب وقف الخسارة"""
        if self.trade_phase_engine.side == "BUY":
            return current_price <= stop_loss
        else:
            return current_price >= stop_loss
    
    def _update_trade_phase(self, current_price: float, candles: List[Dict]):
        """تحديث مرحلة الصفقة (وظيفة مساعدة)"""
        # هذه وظيفة مساعدة أساسية
        pass
    
    def execute_order(self, side: str, qty: float, price: float, 
                      is_open: bool = True) -> bool:
        """تنفيذ الأمر (محاكاة أو حقيقي)"""
        # هذا مثال للتنفيذ المحاكى
        # في التنفيذ الحقيقي، استخدم exchange.create_order()
        
        global DRY_RUN, EXECUTE_ORDERS
        
        if DRY_RUN or not EXECUTE_ORDERS:
            order_type = "OPEN" if is_open else "CLOSE"
            self.logger.log_debug(
                f"DRY RUN: {order_type} {side.upper()} {qty:.4f} @ {price:.6f}",
                {"Side": side.upper(), "Qty": qty, "Price": price, "Type": order_type}
            )
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
            
            self.logger.log_system(
                f"ORDER FILLED: {'OPEN' if is_open else 'CLOSE'} {side.upper()} {qty:.4f} @ {price:.6f}",
                "SUCCESS",
                {"Side": side.upper(), "Qty": f"{qty:.4f}", "Price": f"{price:.6f}", "Order_ID": order.get('id', 'N/A')}
            )
            
            self.execution_guard.record_success()
            return True
            
        except Exception as e:
            error_msg = str(e)
            self.logger.log_error(f"Order execution failed: {error_msg}", e, "Order Execution")
            self.execution_guard.record_failure(error_msg)
            return False
    
    def get_trade_report(self) -> Dict:
        """تقرير عن أداء الصفقات"""
        total_trades = len(self.trades_history)
        winning_trades = len([t for t in self.trades_history if t.get('pnl_pct', 0) > 0])
        losing_trades = total_trades - winning_trades
        
        # حساب متوسط الربح/الخسارة
        if total_trades > 0:
            avg_pnl = sum(t.get('pnl_pct', 0) for t in self.trades_history) / total_trades
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            # أفضل وأسوأ صفقة
            winning_trades_list = [t for t in self.trades_history if t.get('pnl_pct', 0) > 0]
            losing_trades_list = [t for t in self.trades_history if t.get('pnl_pct', 0) < 0]
            
            best_trade = max(self.trades_history, key=lambda x: x.get('pnl_pct', 0)) if self.trades_history else None
            worst_trade = min(self.trades_history, key=lambda x: x.get('pnl_pct', 0)) if self.trades_history else None
            
            # إجمالي الربح/الخسارة
            total_pnl_usd = sum(t.get('pnl_usd', 0) for t in self.trades_history)
        else:
            avg_pnl = 0
            win_rate = 0
            best_trade = None
            worst_trade = None
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
            'best_trade': best_trade,
            'worst_trade': worst_trade,
            'recent_trades': self.trades_history[-5:] if self.trades_history else [],
            'current_position': self.current_position if self.active_trade else None,
            'fail_fast_exits': self.fail_fast_exits
        }

# ============================================
#  MARKET ANALYZER - محلل السوق (من الكود الأصلي)
# ============================================

class MarketAnalyzer:
    """محلل السوق المتقدم"""
    
    def __init__(self, logger: ProConsoleLogger):
        self.logger = logger
        self.market_states = deque(maxlen=100)  # حفظ آخر 100 حالة سوق
        
    def analyze_market(self, df: pd.DataFrame, timeframe: str = "15m") -> Dict[str, Any]:
        """
        تحليل شامل للسوق
        
        Args:
            df: بيانات OHLCV
            timeframe: الإطار الزمني
            
        Returns:
            dict: نتائج التحليل
        """
        if df.empty or len(df) < 20:
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
            self.logger.log_market(
                timeframe=timeframe,
                trend=trend['direction'],
                structure=structure['type'],
                liquidity=liquidity['level'],
                momentum=momentum['score'],
                volume_profile=volume_profile['profile'],
                reason=reason,
                extra_details={
                    'Trend_Strength': f"{trend['strength']:.1f}",
                    'Structure_Level': structure['key_level'],
                    'Liquidity_Zone': liquidity['zone'],
                    'Momentum_Direction': momentum['direction']
                }
            )
            
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
            self.logger.log_error(f"Market analysis error: {str(e)}", e, "Market Analysis")
            return {"error": str(e)}
    
    def _analyze_trend(self, df: pd.DataFrame) -> Dict[str, Any]:
        """تحليل الاتجاه"""
        closes = df['close'].astype(float).values
        
        # استخدام SMA قصيرة وطويلة الأجل
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
        
        # تأكيد الاتجاه باستخدام آخر 5 شمعات
        recent_closes = closes[-5:]
        if direction == "BULL":
            if all(recent_closes[i] <= recent_closes[i+1] for i in range(len(recent_closes)-1)):
                strength += 10  # زيادة القوة
        elif direction == "BEAR":
            if all(recent_closes[i] >= recent_closes[i+1] for i in range(len(recent_closes)-1)):
                strength += 10  # زيادة القوة
        
        return {
            'direction': direction,
            'strength': abs(strength),
            'sma_short': sma_short,
            'sma_long': sma_long,
            'confirmed': abs(strength) > 1.0
        }
    
    def _analyze_structure(self, df: pd.DataFrame) -> Dict[str, Any]:
        """تحليل الهيكل السعري"""
        highs = df['high'].astype(float).values
        lows = df['low'].astype(float).values
        
        # البحث عن قمم وقيعان محلية
        swing_highs = []
        swing_lows = []
        
        for i in range(2, len(highs) - 2):
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                swing_highs.append({'index': i, 'price': highs[i]})
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                swing_lows.append({'index': i, 'price': lows[i]})
        
        # تحديد نوع الهيكل
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            # Higher Highs و Higher Lows = BOS صاعد
            if swing_highs[-1]['price'] > swing_highs[-2]['price'] and swing_lows[-1]['price'] > swing_lows[-2]['price']:
                structure_type = "BOS_UP"
                key_level = swing_lows[-1]['price'] if swing_lows else None
            # Lower Highs و Lower Lows = BOS هابط
            elif swing_highs[-1]['price'] < swing_highs[-2]['price'] and swing_lows[-1]['price'] < swing_lows[-2]['price']:
                structure_type = "BOS_DOWN"
                key_level = swing_highs[-1]['price'] if swing_highs else None
            else:
                structure_type = "CONSOLIDATION"
                key_level = (max(highs[-10:]) + min(lows[-10:])) / 2
        else:
            structure_type = "NO_CLEAR_STRUCTURE"
            key_level = None
        
        return {
            'type': structure_type,
            'key_level': key_level,
            'swing_highs': swing_highs[-3:] if swing_highs else [],
            'swing_lows': swing_lows[-3:] if swing_lows else []
        }
    
    def _analyze_liquidity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """تحليل السيولة"""
        highs = df['high'].astype(float).values
        lows = df['low'].astype(float).values
        volumes = df['volume'].astype(float).values
        
        # تحليل السيولة بناءً على حجم التداول والمدى السعري
        avg_volume = np.mean(volumes[-10:])
        current_volume = volumes[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        # تحليل نطاق السعر
        price_range = max(highs[-10:]) - min(lows[-10:])
        current_range = highs[-1] - lows[-1]
        range_ratio = current_range / price_range if price_range > 0 else 1
        
        # تحديد مستوى السيولة
        if volume_ratio > 1.5 and range_ratio > 1.2:
            level = "HIGH"
            zone = "EXPANSION"
        elif volume_ratio > 1.2 and range_ratio > 1.0:
            level = "MEDIUM_HIGH"
            zone = "ACTIVE"
        elif volume_ratio > 0.8 and range_ratio > 0.8:
            level = "MEDIUM"
            zone = "NORMAL"
        elif volume_ratio > 0.5:
            level = "LOW"
            zone = "THIN"
        else:
            level = "VERY_LOW"
            zone = "ILLIQUID"
        
        return {
            'level': level,
            'zone': zone,
            'volume_ratio': volume_ratio,
            'range_ratio': range_ratio,
            'current_volume': current_volume,
            'avg_volume': avg_volume
        }
    
    def _analyze_momentum(self, df: pd.DataFrame) -> Dict[str, Any]:
        """تحليل الزخم"""
        closes = df['close'].astype(float).values
        
        if len(closes) < 14:
            return {'score': 0, 'direction': 'NEUTRAL'}
        
        # حساب RSI
        rsi = self._calculate_rsi(closes, 14)
        
        # حساب معدل التغير
        roc = ((closes[-1] - closes[-5]) / closes[-5]) * 100
        
        # تحديد اتجاه الزخم
        if rsi > 70:
            momentum_direction = "OVERBOUGHT"
            score = (rsi - 70) / 30  # تطبيع بين 0-1
        elif rsi < 30:
            momentum_direction = "OVERSOLD"
            score = (30 - rsi) / 30  # تطبيع بين 0-1
        elif roc > 0:
            momentum_direction = "BULLISH"
            score = min(abs(roc) / 5, 1)  # تطبيع
        elif roc < 0:
            momentum_direction = "BEARISH"
            score = min(abs(roc) / 5, 1)  # تطبيع
        else:
            momentum_direction = "NEUTRAL"
            score = 0
        
        return {
            'score': score,
            'direction': momentum_direction,
            'rsi': rsi,
            'roc': roc,
            'strength': "STRONG" if score > 0.7 else "MODERATE" if score > 0.3 else "WEAK"
        }
    
    def _analyze_volume(self, df: pd.DataFrame) -> Dict[str, Any]:
        """تحليل الحجم"""
        volumes = df['volume'].astype(float).values
        
        if len(volumes) < 10:
            return {'profile': 'UNKNOWN', 'trend': 'UNKNOWN'}
        
        # تحليل اتجاه الحجم
        recent_volumes = volumes[-5:]
        avg_volume = np.mean(volumes[-10:])
        
        volume_trend = "INCREASING" if recent_volumes[-1] > recent_volumes[0] else "DECREASING" if recent_volumes[-1] < recent_volumes[0] else "STABLE"
        
        # تحديد ملف الحجم
        if recent_volumes[-1] > avg_volume * 1.5:
            profile = "HIGH_ACCUMULATION" if volume_trend == "INCREASING" else "HIGH_DISTRIBUTION"
        elif recent_volumes[-1] > avg_volume * 1.2:
            profile = "MODERATE_ACCUMULATION" if volume_trend == "INCREASING" else "MODERATE_DISTRIBUTION"
        elif recent_volumes[-1] > avg_volume * 0.8:
            profile = "NORMAL"
        else:
            profile = "LOW_PARTICIPATION"
        
        return {
            'profile': profile,
            'trend': volume_trend,
            'current': recent_volumes[-1],
            'average': avg_volume,
            'ratio': recent_volumes[-1] / avg_volume if avg_volume > 0 else 1
        }
    
    def _generate_analysis_reason(self, trend: Dict, structure: Dict, liquidity: Dict) -> str:
        """توليد سبب التحليل"""
        reasons = []
        
        if trend['confirmed']:
            reasons.append(f"Trend: {trend['direction']} (Strength: {trend['strength']:.1f})")
        
        if structure['type'] != "NO_CLEAR_STRUCTURE":
            reasons.append(f"Structure: {structure['type']}")
        
        if liquidity['level'] in ["HIGH", "MEDIUM_HIGH"]:
            reasons.append(f"Liquidity: {liquidity['level']} ({liquidity['zone']})")
        
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
#  SIGNAL GENERATOR - مولد الإشارات (محدث)
# ============================================

class SignalGenerator:
    """مولد إشارات التداول (محدث مع تحليل متقدم)"""
    
    def __init__(self, logger: ProConsoleLogger):
        self.logger = logger
        self.last_signal_time = 0
        self.signal_cooldown = 60  # 60 ثانية بين الإشارات
    
    def generate_market_data(self, df: pd.DataFrame, market_analysis: Dict, 
                            current_price: float) -> Dict[str, Any]:
        """
        توليد بيانات السوق المتقدمة لبناء الخطط
        
        Returns:
            Dict: بيانات السوق الكاملة
        """
        if df.empty or len(df) < 20:
            return {}
        
        try:
            # استخراج البيانات الأساسية
            trend = market_analysis.get('trend', {})
            structure = market_analysis.get('structure', {})
            momentum = market_analysis.get('momentum', {})
            liquidity = market_analysis.get('liquidity', {})
            
            # تحليل متقدم للسيولة
            liquidity_event = self._analyze_liquidity_events(df, current_price)
            
            # تحليل الهيكل
            structure_event = self._analyze_structure_events(df, structure)
            
            # تحديد نوع المنطقة
            zone_type = self._determine_zone_type(df, current_price, structure)
            
            # تحديد نوع التأكيد
            confirmation_type = self._analyze_confirmation(df)
            
            # تحديد الإشارة
            signal = self._determine_signal(trend, structure, momentum, liquidity)
            
            # تحديد فئة الاتجاه
            trend_class = "LARGE" if trend.get('strength', 0) > 2.0 else "MID"
            
            # حساب مستويات السيولة
            internal_liquidity = self._calculate_internal_liquidity(df, signal)
            external_liquidity = self._calculate_external_liquidity(df, signal)
            htf_liquidity = self._calculate_htf_liquidity(df, signal) if trend_class == "LARGE" else None
            
            # مستوى إبطال الهيكل
            structure_invalid_level = self._calculate_structure_invalid(df, signal, structure)
            
            return {
                'signal': signal,
                'trend_class': trend_class,
                'liquidity_event': liquidity_event,
                'structure_event': structure_event,
                'zone_type': zone_type,
                'confirmation_type': confirmation_type,
                'internal_liquidity': internal_liquidity,
                'external_liquidity': external_liquidity,
                'htf_liquidity': htf_liquidity,
                'structure_invalid_level': structure_invalid_level,
                'current_price': current_price,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.log_error(f"Failed to generate market data: {str(e)}", e, "MarketData Generation")
            return {}
    
    def _analyze_liquidity_events(self, df: pd.DataFrame, current_price: float) -> str:
        """تحليل أحداث السيولة"""
        if len(df) < 5:
            return "NO_LIQUIDITY"
        
        highs = df['high'].values[-5:]
        lows = df['low'].values[-5:]
        
        # كشف Sweep
        current_high = highs[-1]
        current_low = lows[-1]
        
        if current_high > max(highs[:-1]):
            return "SWEEP_HIGH"
        elif current_low < min(lows[:-1]):
            return "SWEEP_LOW"
        
        # كشف Liquidity Tap
        if abs(current_high - max(highs[:-1])) < current_high * 0.001:
            return "TAP_HIGH"
        elif abs(current_low - min(lows[:-1])) < current_low * 0.001:
            return "TAP_LOW"
        
        return "NO_LIQUIDITY"
    
    def _analyze_structure_events(self, df: pd.DataFrame, structure: Dict) -> str:
        """تحليل أحداث الهيكل"""
        structure_type = structure.get('type', '')
        
        if 'BOS_UP' in structure_type:
            return "BOS_UP"
        elif 'BOS_DOWN' in structure_type:
            return "BOS_DOWN"
        elif 'CHoCH' in structure_type:
            return "CHOCH"
        
        return "NO_STRUCTURE"
    
    def _determine_zone_type(self, df: pd.DataFrame, current_price: float, structure: Dict) -> str:
        """تحديد نوع المنطقة"""
        key_level = structure.get('key_level')
        
        if not key_level:
            return "UNKNOWN"
        
        # حساب نسبة الاختلاف
        diff_pct = abs(current_price - key_level) / key_level * 100
        
        if diff_pct < 0.5:  # ضمن 0.5% من المستوى الرئيسي
            if current_price > key_level:
                return "SUPPLY_ZONE"
            else:
                return "DEMAND_ZONE"
        elif diff_pct < 1.0:  # ضمن 1% من المستوى الرئيسي
            return "RETEST_ZONE"
        
        return "NO_ZONE"
    
    def _analyze_confirmation(self, df: pd.DataFrame) -> str:
        """تحليل نوع التأكيد"""
        if len(df) < 3:
            return "NO_CONFIRMATION"
        
        current = df.iloc[-1]
        prev = df.iloc[-2]
        
        # تحليل أنماط الشمعات
        candle_range = current['high'] - current['low']
        
        # Pin Bar
        upper_wick = current['high'] - max(current['close'], current['open'])
        lower_wick = min(current['close'], current['open']) - current['low']
        body = abs(current['close'] - current['open'])
        
        if upper_wick > body * 2 and upper_wick > candle_range * 0.6:
            return "PIN_BAR_REJECTION"
        elif lower_wick > body * 2 and lower_wick > candle_range * 0.6:
            return "PIN_BAR_REJECTION"
        
        # Engulfing
        if current['close'] > current['open'] and prev['close'] < prev['open']:
            if current['open'] < prev['close'] and current['close'] > prev['open']:
                return "BULLISH_ENGULFING"
        elif current['close'] < current['open'] and prev['close'] > prev['open']:
            if current['open'] > prev['close'] and current['close'] < prev['open']:
                return "BEARISH_ENGULFING"
        
        return "PRICE_ACTION"
    
    def _determine_signal(self, trend: Dict, structure: Dict, momentum: Dict, liquidity: Dict) -> Optional[str]:
        """تحديد إشارة التداول"""
        # الشروط الأساسية
        if not trend.get('confirmed', False):
            return None
        
        if liquidity.get('level') in ["LOW", "VERY_LOW"]:
            return None
        
        # إشارة الشراء
        buy_conditions = [
            trend.get('direction') == "BULL",
            structure.get('type') in ["BOS_UP", "CHOCH_UP"],
            momentum.get('direction') in ["BULLISH", "NEUTRAL"],
            momentum.get('rsi', 50) < 70  # ليس مفرط في الشراء
        ]
        
        # إشارة البيع
        sell_conditions = [
            trend.get('direction') == "BEAR",
            structure.get('type') in ["BOS_DOWN", "CHOCH_DOWN"],
            momentum.get('direction') in ["BEARISH", "NEUTRAL"],
            momentum.get('rsi', 50) > 30  # ليس مفرط في البيع
        ]
        
        if all(buy_conditions):
            return "buy"
        elif all(sell_conditions):
            return "sell"
        
        return None
    
    def _calculate_internal_liquidity(self, df: pd.DataFrame, signal: str) -> Optional[float]:
        """حساب سيولة داخلية (TP1)"""
        if len(df) < 10:
            return None
        
        if signal == "buy":
            # للشراء: أعلى قمة حديثة
            return float(df['high'].rolling(5).max().iloc[-1] * 1.01)  # +1%
        else:
            # للبيع: أقل قاع حديث
            return float(df['low'].rolling(5).min().iloc[-1] * 0.99)  # -1%
    
    def _calculate_external_liquidity(self, df: pd.DataFrame, signal: str) -> Optional[float]:
        """حساب سيولة خارجية (TP2)"""
        if len(df) < 20:
            return None
        
        if signal == "buy":
            # للشراء: مقاومة رئيسية
            return float(df['high'].rolling(10).max().iloc[-1] * 1.02)  # +2%
        else:
            # للبيع: دعم رئيسي
            return float(df['low'].rolling(10).min().iloc[-1] * 0.98)  # -2%
    
    def _calculate_htf_liquidity(self, df: pd.DataFrame, signal: str) -> Optional[float]:
        """حساب سيولة إطار زمني أعلى (TP3)"""
        if len(df) < 50:
            return None
        
        if signal == "buy":
            # للشراء: قمة تاريخية قريبة
            return float(df['high'].max() * 1.03)  # +3%
        else:
            # للبيع: قاع تاريخي قريب
            return float(df['low'].min() * 0.97)  # -3%
    
    def _calculate_structure_invalid(self, df: pd.DataFrame, signal: str, structure: Dict) -> Optional[float]:
        """حساب مستوى إبطال الهيكل"""
        if signal == "buy":
            # للشراء: تحت آخر قاع هيكلي
            swing_lows = structure.get('swing_lows', [])
            if swing_lows:
                return min([s['price'] for s in swing_lows[-2:]]) * 0.995  # -0.5%
        else:
            # للبيع: فوق آخر قمة هيكلية
            swing_highs = structure.get('swing_highs', [])
            if swing_highs:
                return max([s['price'] for s in swing_highs[-2:]]) * 1.005  # +0.5%
        
        return None

# ============================================
#  MAIN BOT INTEGRATION - التكامل الرئيسي (محدث)
# ============================================

# إعدادات البوت (كما هي)
EXCHANGE_NAME = os.getenv("EXCHANGE", "bybit").lower()

if EXCHANGE_NAME == "bybit":
    API_KEY = os.getenv("BYBIT_API_KEY", "")
    API_SECRET = os.getenv("BYBIT_API_SECRET", "")
else:
    API_KEY = os.getenv("BINGX_API_KEY", "")
    API_SECRET = os.getenv("BINGX_API_SECRET", "")

MODE_LIVE = bool(API_KEY and API_SECRET)
SELF_URL = os.getenv("SELF_URL", "") or os.getenv("RENDER_EXTERNAL_URL", "")
PORT = int(os.getenv("PORT", 5000))

EXECUTE_ORDERS = True
SHADOW_MODE_DASHBOARD = False
DRY_RUN = os.getenv("DRY_RUN", "false").lower() == "true"

BOT_VERSION = "SUI ULTRA PRO AI v10.0 — SMART TRADE PLAN ENGINE"

# إعدادات التداول
SYMBOL = os.getenv("SYMBOL", "SUI/USDT:USDT")
INTERVAL = os.getenv("INTERVAL", "15m")
LEVERAGE = int(os.getenv("LEVERAGE", "10"))
RISK_ALLOC = float(os.getenv("RISK_ALLOC", "0.60"))
BASE_SLEEP = int(os.getenv("BASE_SLEEP", "5"))
NEAR_CLOSE_S = int(os.getenv("NEAR_CLOSE_S", "1"))

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

# الدوال المساعدة (كما هي)
def get_balance(exchange) -> float:
    """الحصول على الرصيد"""
    if not MODE_LIVE:
        return 100.0  # رصيد تجريبي
    try:
        b = exchange.fetch_balance(params={"type":"swap"})
        return b.get("total",{}).get("USDT") or b.get("free",{}).get("USDT", 0.0)
    except Exception as e:
        logger.log_error(f"Failed to fetch balance: {str(e)}", e, "Balance Check")
        return None

def get_current_price(exchange, symbol: str) -> float:
    """الحصول على السعر الحالي"""
    try:
        t = exchange.fetch_ticker(symbol)
        return t.get("last") or t.get("close")
    except Exception as e:
        logger.log_error(f"Failed to fetch price: {str(e)}", e, "Price Fetch")
        return None

def fetch_ohlcv_data(exchange, symbol: str, timeframe: str = "15m", limit: int = 100) -> pd.DataFrame:
    """جلب بيانات OHLCV"""
    try:
        rows = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit, params={"type":"swap"})
        return pd.DataFrame(rows, columns=["timestamp","open","high","low","close","volume"])
    except Exception as e:
        logger.log_error(f"Failed to fetch OHLCV: {str(e)}", e, "OHLCV Fetch")
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
#  MAIN BOT CLASS - الفئة الرئيسية للبوت (محدث)
# ============================================

class SUIUltraProBot:
    """الفئة الرئيسية للبوت (محدث مع TradePlan)"""
    
    def __init__(self):
        self.logger = logger
        self.exchange = None
        self.smart_trade_manager = None
        self.market_analyzer = None
        self.signal_generator = None
        self.running = False
        
    def initialize(self):
        """تهيئة البوت"""
        try:
            self.logger.log_system(f"🚀 Booting: {BOT_VERSION}", "SUCCESS")
            
            # تهيئة Exchange
            self.exchange = make_exchange()
            self.logger.log_system(f"Exchange: {EXCHANGE_NAME.upper()} | Symbol: {SYMBOL}", "INFO")
            self.logger.log_system(f"Mode: {'LIVE' if MODE_LIVE else 'PAPER'} | Dry Run: {DRY_RUN}", "INFO")
            
            # تهيئة الأنظمة
            self.smart_trade_manager = SmartTradeManager(
                exchange=self.exchange,
                symbol=SYMBOL,
                risk_percent=RISK_ALLOC,
                logger=self.logger
            )
            
            self.market_analyzer = MarketAnalyzer(logger=self.logger)
            self.signal_generator = SignalGenerator(logger=self.logger)
            
            self.logger.log_system("Smart Trade Plan System Initialized", "SUCCESS")
            self.logger.log_system(f"Symbol: {SYMBOL} | Risk: {RISK_ALLOC*100:.0f}% | Interval: {INTERVAL}", "INFO")
            
            return True
            
        except Exception as e:
            self.logger.log_error(f"Failed to initialize bot: {str(e)}", e, "Bot Initialization")
            return False
    
    def run_trade_loop(self):
        """تشغيل حلقة التداول الرئيسية"""
        self.logger.log_system("Starting Smart Trade Loop with Plan System", "INFO")
        self.running = True
        
        while self.running:
            try:
                # جمع بيانات السوق
                balance = get_balance(self.exchange)
                current_price = get_current_price(self.exchange, SYMBOL)
                df = fetch_ohlcv_data(self.exchange, SYMBOL, INTERVAL)
                
                if df.empty or current_price is None:
                    self.logger.log_debug("Waiting for market data...")
                    time.sleep(BASE_SLEEP)
                    continue
                
                # تحليل السوق
                market_analysis = self.market_analyzer.analyze_market(df, INTERVAL)
                
                # إذا كانت هناك صفقة نشطة
                if self.smart_trade_manager.active_trade:
                    # إدارة الصفقة الحالية باستخدام الخطة
                    candles = convert_candles_to_dicts(df)
                    self.smart_trade_manager.manage_trade_with_plan(current_price, candles[-20:])
                
                else:
                    # توليد بيانات السوق المتقدمة
                    market_data = self.signal_generator.generate_market_data(df, market_analysis, current_price)
                    
                    if market_data and market_data.get('signal') and balance and balance > 10:
                        # بناء خطة التداول
                        trade_plan = self.smart_trade_manager.trade_plan_builder.build_trade_plan(market_data)
                        
                        if trade_plan and trade_plan.is_valid():
                            # فتح صفقة بناءً على الخطة
                            success = self.smart_trade_manager.open_trade_with_plan(
                                trade_plan=trade_plan,
                                balance=balance,
                                current_price=current_price
                            )
                            
                            if success:
                                self.logger.log_system(
                                    f"Trade opened with plan | {trade_plan.side.upper()} @ {current_price:.4f}",
                                    "SUCCESS"
                                )
                        else:
                            if market_data.get('signal'):
                                self.logger.log_blocked_entry(
                                    "Plan validation failed",
                                    {
                                        'signal': market_data.get('signal'),
                                        'liquidity': market_data.get('liquidity_event'),
                                        'structure': market_data.get('structure_event')
                                    }
                                )
                
                # النوم حتى التكرار التالي
                time.sleep(BASE_SLEEP)
                
            except KeyboardInterrupt:
                self.logger.log_system("Trade loop stopped by user", "INFO")
                self.running = False
                break
                
            except Exception as e:
                self.logger.log_error(f"Trade loop error: {str(e)}", e, "Trade Loop")
                time.sleep(BASE_SLEEP * 2)  # زيادة وقت الانتظار عند الخطأ
    
    def stop(self):
        """إيقاف البوت"""
        self.running = False
        self.logger.log_system("Bot stopped", "INFO")
    
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
#  FLASK API SERVER - خادم API (محدث)
# ============================================

app = Flask(__name__)
bot_instance = None

def create_dashboard_html():
    """إنشاء واجهة HTML للرصد"""
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>SUI ULTRA PRO AI v10.0 - SMART TRADE PLAN ENGINE</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
                color: #ffffff;
                min-height: 100vh;
            }
            .container {
                max-width: 1400px;
                margin: 0 auto;
            }
            .header {
                text-align: center;
                margin-bottom: 30px;
                padding: 20px;
                background: rgba(255, 255, 255, 0.1);
                border-radius: 15px;
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.2);
            }
            .header h1 {
                margin: 0;
                color: #00ff88;
                font-size: 2.5em;
                text-shadow: 0 0 10px rgba(0, 255, 136, 0.5);
            }
            .header .subtitle {
                color: #88ffcc;
                font-size: 1.2em;
                margin-top: 10px;
            }
            .grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }
            .card {
                background: rgba(255, 255, 255, 0.1);
                border-radius: 15px;
                padding: 25px;
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.2);
                transition: transform 0.3s, box-shadow 0.3s;
            }
            .card:hover {
                transform: translateY(-5px);
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
            }
            .card h3 {
                color: #00ff88;
                margin-top: 0;
                border-bottom: 2px solid rgba(0, 255, 136, 0.3);
                padding-bottom: 10px;
            }
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 15px;
            }
            .stat-item {
                background: rgba(0, 0, 0, 0.3);
                padding: 15px;
                border-radius: 10px;
                text-align: center;
            }
            .stat-value {
                font-size: 1.8em;
                font-weight: bold;
                margin: 10px 0;
            }
            .positive {
                color: #00ff88;
            }
            .negative {
                color: #ff4444;
            }
            .neutral {
                color: #ffcc00;
            }
            .trade-list {
                max-height: 400px;
                overflow-y: auto;
            }
            .trade-item {
                background: rgba(0, 0, 0, 0.3);
                padding: 15px;
                border-radius: 10px;
                margin-bottom: 10px;
                border-left: 4px solid;
            }
            .trade-buy {
                border-left-color: #00ff88;
            }
            .trade-sell {
                border-left-color: #ff4444;
            }
            .refresh-btn {
                background: linear-gradient(45deg, #00ff88, #00ccff);
                color: white;
                border: none;
                padding: 12px 25px;
                border-radius: 25px;
                font-size: 1.1em;
                cursor: pointer;
                transition: all 0.3s;
                display: block;
                margin: 20px auto;
                font-weight: bold;
            }
            .refresh-btn:hover {
                transform: scale(1.05);
                box-shadow: 0 5px 20px rgba(0, 255, 136, 0.4);
            }
            .status-indicator {
                display: inline-block;
                width: 12px;
                height: 12px;
                border-radius: 50%;
                margin-right: 8px;
            }
            .status-active {
                background: #00ff88;
                box-shadow: 0 0 10px #00ff88;
            }
            .status-inactive {
                background: #ff4444;
            }
            .api-endpoints {
                margin-top: 30px;
                padding: 20px;
                background: rgba(255, 255, 255, 0.05);
                border-radius: 15px;
            }
            .api-endpoints h3 {
                color: #00ccff;
            }
            .endpoint-list {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 15px;
                margin-top: 15px;
            }
            .endpoint {
                background: rgba(0, 0, 0, 0.3);
                padding: 15px;
                border-radius: 10px;
                border-left: 4px solid #00ccff;
            }
            .endpoint code {
                background: rgba(0, 0, 0, 0.5);
                padding: 5px 10px;
                border-radius: 5px;
                display: block;
                margin: 10px 0;
                color: #88ffcc;
            }
            .timestamp {
                color: #88aaff;
                font-size: 0.9em;
                text-align: right;
                margin-top: 20px;
            }
            .plan-indicator {
                display: inline-block;
                padding: 2px 8px;
                border-radius: 10px;
                font-size: 0.8em;
                margin-left: 5px;
            }
            .plan-valid {
                background: rgba(0, 255, 136, 0.2);
                color: #00ff88;
            }
            .plan-invalid {
                background: rgba(255, 68, 68, 0.2);
                color: #ff4444;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🧠 SUI ULTRA PRO AI v10.0</h1>
                <div class="subtitle">SMART TRADE PLAN ENGINE WITH FAIL-FAST LOGIC</div>
            </div>
            
            <div class="grid">
                <!-- Bot Status Card -->
                <div class="card">
                    <h3>🤖 Bot Status</h3>
                    <div class="stats-grid">
                        <div class="stat-item">
                            <div>Status</div>
                            <div class="stat-value" id="bot-status">Loading...</div>
                        </div>
                        <div class="stat-item">
                            <div>Exchange</div>
                            <div class="stat-value" id="exchange">Loading...</div>
                        </div>
                        <div class="stat-item">
                            <div>Symbol</div>
                            <div class="stat-value" id="symbol">Loading...</div>
                        </div>
                        <div class="stat-item">
                            <div>Mode</div>
                            <div class="stat-value" id="mode">Loading...</div>
                        </div>
                    </div>
                </div>
                
                <!-- Trading Performance Card -->
                <div class="card">
                    <h3>📈 Trading Performance</h3>
                    <div class="stats-grid">
                        <div class="stat-item">
                            <div>Total Trades</div>
                            <div class="stat-value" id="total-trades">0</div>
                        </div>
                        <div class="stat-item">
                            <div>Win Rate</div>
                            <div class="stat-value positive" id="win-rate">0%</div>
                        </div>
                        <div class="stat-item">
                            <div>Total P&L</div>
                            <div class="stat-value" id="total-pnl">0%</div>
                        </div>
                        <div class="stat-item">
                            <div>Fail-Fast Exits</div>
                            <div class="stat-value" id="fail-fast">0</div>
                        </div>
                    </div>
                </div>
                
                <!-- Active Trade Card -->
                <div class="card">
                    <h3>⚡ Active Trade</h3>
                    <div id="active-trade-details">
                        <div style="text-align: center; padding: 20px; color: #88aaff;">
                            No active trade
                        </div>
                    </div>
                </div>
                
                <!-- Recent Trades Card -->
                <div class="card" style="grid-column: span 2;">
                    <h3>📊 Recent Trades</h3>
                    <div class="trade-list" id="recent-trades">
                        <div style="text-align: center; padding: 40px; color: #88aaff;">
                            Loading trades...
                        </div>
                    </div>
                </div>
            </div>
            
            <button class="refresh-btn" onclick="refreshData()">🔄 Refresh Dashboard</button>
            
            <!-- API Endpoints -->
            <div class="api-endpoints">
                <h3>🔗 API Endpoints</h3>
                <div class="endpoint-list">
                    <div class="endpoint">
                        <strong>Bot Status</strong>
                        <code>GET /api/status</code>
                        <small>Get current bot status and performance metrics</small>
                    </div>
                    <div class="endpoint">
                        <strong>Health Check</strong>
                        <code>GET /health</code>
                        <small>Check if bot is running</small>
                    </div>
                    <div class="endpoint">
                        <strong>Trades Report</strong>
                        <code>GET /api/trades</code>
                        <small>Get detailed trades history</small>
                    </div>
                    <div class="endpoint">
                        <strong>Trade Actions</strong>
                        <code>POST /api/trade/action</code>
                        <small>Manual trade control (emergency)</small>
                    </div>
                </div>
            </div>
            
            <div class="timestamp" id="last-update">Last updated: --</div>
        </div>
        
        <script>
            function refreshData() {
                fetch('/api/status')
                    .then(response => response.json())
                    .then(data => {
                        // Update bot status
                        document.getElementById('bot-status').innerHTML = 
                            `<span class="status-indicator ${data.running ? 'status-active' : 'status-inactive'}"></span>
                             ${data.running ? 'Running' : 'Stopped'}`;
                        document.getElementById('exchange').textContent = data.exchange;
                        document.getElementById('symbol').textContent = data.symbol;
                        document.getElementById('mode').textContent = `${data.mode} ${data.dry_run ? '(Dry Run)' : ''}`;
                        
                        // Update trading performance
                        const report = data.trade_report;
                        document.getElementById('total-trades').textContent = report.total_trades || 0;
                        document.getElementById('win-rate').textContent = report.win_rate ? report.win_rate.toFixed(1) + '%' : '0%';
                        document.getElementById('total-pnl').textContent = report.total_pnl_pct ? report.total_pnl_pct.toFixed(2) + '%' : '0%';
                        document.getElementById('total-pnl').className = 'stat-value ' + 
                            (report.total_pnl_pct > 0 ? 'positive' : report.total_pnl_pct < 0 ? 'negative' : 'neutral');
                        document.getElementById('fail-fast').textContent = report.fail_fast_exits || 0;
                        
                        // Update active trade
                        const activeTradeDiv = document.getElementById('active-trade-details');
                        if (report.active_trade && report.current_position) {
                            const pos = report.current_position;
                            const pnl = ((data.current_price - pos.entry_price) / pos.entry_price * 100 * (pos.side === 'buy' ? 1 : -1)).toFixed(2);
                            const pnlClass = pnl >= 0 ? 'positive' : 'negative';
                            
                            activeTradeDiv.innerHTML = `
                                <div style="margin-bottom: 10px;">
                                    <strong>${pos.side.toUpperCase()}</strong> 
                                    <span class="plan-indicator plan-valid">Plan Active</span>
                                </div>
                                <div style="font-size: 0.9em; margin-bottom: 5px;">
                                    Entry: ${pos.entry_price ? pos.entry_price.toFixed(4) : '--'}
                                </div>
                                <div style="font-size: 0.9em; margin-bottom: 5px;">
                                    Current: ${data.current_price ? data.current_price.toFixed(4) : '--'}
                                </div>
                                <div style="font-size: 0.9em; margin-bottom: 5px;">
                                    P&L: <span class="${pnlClass}">${pnl}%</span>
                                </div>
                                <div style="font-size: 0.8em; color: #88aaff;">
                                    ${pos.entry_time ? new Date(pos.entry_time).toLocaleTimeString() : ''}
                                </div>
                            `;
                        } else {
                            activeTradeDiv.innerHTML = '<div style="text-align: center; padding: 20px; color: #88aaff;">No active trade</div>';
                        }
                        
                        // Update recent trades
                        const tradesList = document.getElementById('recent-trades');
                        if (report.recent_trades && report.recent_trades.length > 0) {
                            let tradesHtml = '';
                            report.recent_trades.slice().reverse().forEach(trade => {
                                const pnlClass = trade.pnl_pct > 0 ? 'positive' : 'negative';
                                const sideClass = trade.side === 'buy' ? 'trade-buy' : 'trade-sell';
                                const failFastBadge = trade.is_fail_fast ? '<span style="background: #ff4444; color: white; padding: 2px 6px; border-radius: 3px; font-size: 0.7em; margin-left: 5px;">FAIL-FAST</span>' : '';
                                
                                tradesHtml += `
                                    <div class="trade-item ${sideClass}">
                                        <div style="display: flex; justify-content: space-between;">
                                            <div>
                                                <strong>${trade.side.toUpperCase()} #${trade.id}</strong>
                                                ${failFastBadge}
                                            </div>
                                            <span class="${pnlClass}">${trade.pnl_pct ? trade.pnl_pct.toFixed(2) + '%' : '--'}</span>
                                        </div>
                                        <div style="font-size: 0.9em; color: #88aaff;">
                                            Entry: ${trade.entry_price ? trade.entry_price.toFixed(4) : '--'} | 
                                            Exit: ${trade.exit_price ? trade.exit_price.toFixed(4) : '--'} | 
                                            ${trade.exit_reason || 'Active'}
                                        </div>
                                        <div style="font-size: 0.8em; color: #aaccff; margin-top: 5px;">
                                            ${trade.timestamp ? new Date(trade.timestamp).toLocaleString() : ''}
                                        </div>
                                    </div>
                                `;
                            });
                            tradesList.innerHTML = tradesHtml;
                        } else {
                            tradesList.innerHTML = '<div style="text-align: center; padding: 40px; color: #88aaff;">No trades yet</div>';
                        }
                        
                        // Update timestamp
                        document.getElementById('last-update').textContent = 
                            `Last updated: ${new Date().toLocaleTimeString()}`;
                    })
                    .catch(error => {
                        console.error('Error fetching data:', error);
                        alert('Error refreshing data. Check console for details.');
                    });
            }
            
            // Initial load
            refreshData();
            
            // Auto-refresh every 30 seconds
            setInterval(refreshData, 30000);
        </script>
    </body>
    </html>
    '''

@app.route('/')
def dashboard():
    """لوحة التحكم الرئيسية"""
    return render_template_string(create_dashboard_html())

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

@app.route('/api/trades')
def api_trades():
    """الحصول على تاريخ الصفقات"""
    if bot_instance and bot_instance.smart_trade_manager:
        return jsonify({
            'trades': bot_instance.smart_trade_manager.trades_history,
            'total_trades': len(bot_instance.smart_trade_manager.trades_history),
            'timestamp': datetime.now().isoformat()
        })
    return jsonify({'error': 'Trade manager not available'}), 500

@app.route('/api/trade/action', methods=['POST'])
def trade_action():
    """إجراء يدوي على الصفقة (لحالات الطوارئ)"""
    # هذه الدالة للاستخدام في حالات الطوارئ فقط
    if not bot_instance or not bot_instance.smart_trade_manager:
        return jsonify({'error': 'Bot not initialized'}), 500
    
    # في الواقع، يجب التحقق من البيانات المرسلة
    # لكن لأغراض التوضيح، سنقوم بإغلاق أي صفقة نشطة
    
    if bot_instance.smart_trade_manager.active_trade:
        current_price = get_current_price(bot_instance.exchange, SYMBOL)
        if current_price:
            bot_instance.smart_trade_manager.close_trade("Manual emergency close", current_price)
            return jsonify({'message': 'Trade closed manually', 'price': current_price}), 200
        else:
            return jsonify({'error': 'Cannot get current price'}), 500
    
    return jsonify({'message': 'No active trade to close'}), 200

# ============================================
#  MAIN EXECUTION - التنفيذ الرئيسي
# ============================================

def main():
    """الدالة الرئيسية"""
    global bot_instance
    
    try:
        # طباعة بانر البداية
        print(f"\n{ConsoleColors.LIGHT_CYAN}{'='*80}{ConsoleColors.RESET}")
        print(f"{ConsoleColors.LIGHT_GREEN}{BOT_VERSION}{ConsoleColors.RESET}")
        print(f"{ConsoleColors.LIGHT_CYAN}{'='*80}{ConsoleColors.RESET}\n")
        
        # إنشاء وتشغيل البوت
        bot_instance = SUIUltraProBot()
        
        if not bot_instance.initialize():
            logger.log_error("Failed to initialize bot", None, "Main Execution")
            return
        
        # بدء حلقة التداول في thread منفصل
        import threading
        trade_thread = threading.Thread(target=bot_instance.run_trade_loop, daemon=True)
        trade_thread.start()
        
        logger.log_system(f"Starting Flask server on port {PORT}", "INFO")
        
        # تشغيل خادم Flask
        app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
        
    except KeyboardInterrupt:
        logger.log_system("Bot stopped by user", "INFO")
    except Exception as e:
        logger.log_error(f"Fatal error in main: {str(e)}", e, "Main Execution")
    finally:
        if bot_instance:
            bot_instance.stop()

if __name__ == "__main__":
    main()
