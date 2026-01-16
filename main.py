# -*- coding: utf-8 -*-
"""
SUI ULTRA PRO AI BOT - الإصدار الذكي المتكامل مع نظام إدارة الصفقات المتقدم
• نظام إدارة المراحل الذكي (Entry → Protect → BE → Trail → Trim → Exit)
• لوج احترافي ذكي مع ألوان وأيقونات وأسباب القرارات
• Structure-Based Trailing (ليس ATR تقليدي)
• حماية تنفيذية من أخطاء Bybit/MinQty
• نظام Trim الذكي لتقليل المخاطرة
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
        'CONSOLIDATION': '↔️'
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

# إنشاء كائن اللوجر العام
logger = ProConsoleLogger(show_timestamp=True, show_emoji=True)

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
    
    def __init__(self, entry_price: float, side: str, entry_zone: str, logger: ProConsoleLogger):
        self.entry_price = entry_price
        self.side = side.upper()  # BUY/SELL
        self.entry_zone = entry_zone
        self.logger = logger
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
        
        self.logger.log_management(
            phase=f"STATE_CHANGE",
            action=f"{old_state}→{new_state}",
            reason=reason,
            extra_details={'old_state': old_state, 'new_state': new_state}
        )
        
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
            'entry_zone': self.entry_zone,
            'current_profit_pct': self.calculate_profit_pct(self.last_stop_loss if self.last_stop_loss else self.entry_price)
        }

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
#  SMART TRADE MANAGER - المدير الرئيسي
# ============================================

class SmartTradeManager:
    """المدير الذكي للصفقات"""
    
    def __init__(self, exchange, symbol: str, risk_percent: float = 0.6, logger: ProConsoleLogger = None):
        self.exchange = exchange
        self.symbol = symbol
        self.risk_percent = risk_percent
        self.logger = logger or ProConsoleLogger()
        
        # الأنظمة الفرعية
        self.execution_guard = ExecutionGuard(exchange, self.logger)
        self.trade_phase_engine = None
        self.active_trade = False
        self.current_position = {
            'side': None,
            'entry_price': 0.0,
            'quantity': 0.0,
            'entry_time': None
        }
        
        # إحصائيات
        self.trades_history = []
        self.total_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        
    def calculate_position_size(self, balance: float, entry_price: float, confidence: float = 0.7) -> float:
        """حساب حجم المركز الذكي"""
        # رأس المال المستخدم
        risk_capital = balance * self.risk_percent
        
        # تعديل حسب الثقة
        if confidence > 0.8:
            risk_multiplier = 1.2
            confidence_level = "HIGH"
        elif confidence > 0.6:
            risk_multiplier = 1.0
            confidence_level = "MEDIUM"
        elif confidence > 0.4:
            risk_multiplier = 0.7
            confidence_level = "LOW"
        else:
            risk_multiplier = 0.5
            confidence_level = "VERY_LOW"
        
        adjusted_capital = risk_capital * risk_multiplier
        
        # حساب الكمية
        raw_qty = adjusted_capital / entry_price
        
        # تنقية الكمية
        sanitized_qty, status = self.execution_guard.sanitize_order(self.symbol, raw_qty)
        
        if sanitized_qty is None:
            self.logger.log_error(f"Position size invalid: {status}")
            return 0.0
        
        self.logger.log_debug(
            f"Position Calculation | "
            f"Raw: {raw_qty:.4f} | "
            f"Sanitized: {sanitized_qty:.4f} | "
            f"Capital: ${adjusted_capital:.2f} | "
            f"Confidence: {confidence_level} ({confidence:.2f})"
        )
        
        return sanitized_qty
    
    def open_trade(self, side: str, entry_price: float, balance: float, 
                   entry_zone: str, confidence: float = 0.7, reason: str = "") -> bool:
        """فتح صفقة جديدة"""
        
        # التحقق من عدم وجود صفقة نشطة
        if self.active_trade:
            self.logger.log_error("Cannot open trade: Active trade exists", context="Trade Opening")
            return False
        
        # التحقق من صلاحية التنفيذ
        allow, allow_reason = self.execution_guard.should_allow_order()
        if not allow:
            self.logger.log_system(f"Order not allowed: {allow_reason}", "WARNING")
            return False
        
        # حساب حجم المركز
        qty = self.calculate_position_size(balance, entry_price, confidence)
        if qty <= 0:
            return False
        
        # حساب قيمة المركز
        position_value = qty * entry_price
        
        # تنفيذ الأمر (أو محاكاة)
        success = self.execute_order(side, qty, entry_price, is_open=True)
        
        if success:
            # تحديث المركز الحالي
            self.current_position = {
                'side': side,
                'entry_price': entry_price,
                'quantity': qty,
                'entry_time': datetime.now(),
                'zone': entry_zone,
                'confidence': confidence
            }
            
            # تهيئة نظام إدارة المراحل
            self.trade_phase_engine = TradePhaseEngine(entry_price, side, entry_zone, self.logger)
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
                'confidence': confidence,
                'position_value': position_value
            }
            self.trades_history.append(trade_record)
            
            # لوج الدخول
            self.logger.log_entry(
                side=side,
                zone_type=entry_zone,
                candle_pattern="Signal",  # يمكن تعديله بناءً على التحليل
                confidence=confidence,
                reason=reason,
                entry_price=entry_price
            )
            
            # حساب وقف الخسارة الأولي
            initial_sl, sl_reason = self.trade_phase_engine.calculate_stop_loss(entry_price, [])
            
            self.logger.log_execution(
                price=entry_price,
                quantity=qty,
                stop_loss=initial_sl,
                sl_reason=sl_reason,
                order_type="MARKET",
                exchange=self.exchange.name.upper(),
                position_value=position_value
            )
            
            self.logger.log_system(
                f"Trade opened successfully | {side.upper()} @ {entry_price:.4f} | Qty: {qty:.4f}",
                "SUCCESS",
                {"Entry_Price": f"{entry_price:.4f}", "Quantity": f"{qty:.4f}", "Zone": entry_zone}
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
        
        # حساب الربح الحالي
        profit_pct = self.trade_phase_engine.calculate_profit_pct(current_price)
        state = self.trade_phase_engine.current_state
        
        # لوج حالة إدارة الصفقة (كل 30 ثانية أو عند تغيير مهم)
        current_time = time.time()
        if (current_time - self.trade_phase_engine.state_changed_at) < 30 or state != getattr(self, '_last_logged_state', None):
            self._last_logged_state = state
            
            self.logger.log_management(
                phase=state,
                action="HOLD",
                reason=f"Managing position",
                current_pnl=profit_pct,
                new_stop_loss=sl_price,
                extra_details={
                    "State": state,
                    "SL_Reason": sl_reason,
                    "Position_Size": f"{self.current_position['quantity']:.4f}",
                    "Entry_Price": f"{self.current_position['entry_price']:.4f}"
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
                self.logger.log_management(
                    phase="PROTECT",
                    action="ACTIVATED",
                    reason=reason,
                    current_pnl=engine.calculate_profit_pct(current_price)
                )
        
        elif engine.current_state == TradeState.PROTECT:
            should_be, reason = engine.should_move_to_breakeven(current_price, candles)
            if should_be:
                engine.update_state(TradeState.BREAKEVEN, reason)
                self.logger.log_management(
                    phase="BREAKEVEN",
                    action="ACTIVATED",
                    reason=reason,
                    current_pnl=engine.calculate_profit_pct(current_price)
                )
        
        elif engine.current_state in [TradeState.BREAKEVEN, TradeState.TRAIL, TradeState.TRIM]:
            should_trail, reason = engine.should_move_to_trail(current_price, candles)
            if should_trail:
                engine.update_state(TradeState.TRAIL, reason)
                self.logger.log_management(
                    phase="TRAIL",
                    action="ACTIVATED",
                    reason=reason,
                    current_pnl=engine.calculate_profit_pct(current_price)
                )
            
            should_trim, trim_reason = engine.should_trim_position(current_price, candles)
            if should_trim:
                # تنفيذ ترام جزئي
                self._execute_trim(current_price, trim_reason)
                engine.trim_count += 1
                engine.update_state(TradeState.TRIM, f"Trim #{engine.trim_count}: {trim_reason}")
    
    def _execute_trim(self, current_price: float, reason: str):
        """تنفيذ تقليل المركز"""
        if self.trade_phase_engine and self.current_position['quantity'] > 0:
            # افتراضي: إغلاق 20% من المركز
            trim_percent = 0.2
            trim_qty = self.current_position['quantity'] * trim_percent
            
            # تحديث الكمية الحالية
            self.current_position['quantity'] -= trim_qty
            
            # لوج الترام
            self.logger.log_management(
                phase="TRIM",
                action="EXECUTING",
                reason=f"Closing {trim_percent*100:.0f}%: {reason}",
                trimmed_qty=trim_qty,
                extra_details={
                    "Trim_Pct": f"{trim_percent*100:.0f}%",
                    "Remaining_Qty": f"{self.current_position['quantity']:.4f}",
                    "Reason": reason
                }
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
        quantity = self.current_position['quantity']
        
        if side == "BUY":
            pnl_pct = ((exit_price - entry_price) / entry_price) * 100
            pnl_usd = (exit_price - entry_price) * quantity
        else:
            pnl_pct = ((entry_price - exit_price) / entry_price) * 100
            pnl_usd = (entry_price - exit_price) * quantity
        
        # حساب نسبة المخاطرة/العائد
        initial_risk_pct = abs((self.trade_phase_engine.last_stop_loss - entry_price) / entry_price * 100) if self.trade_phase_engine.last_stop_loss else 0
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
        if pnl_pct > 0:
            self.winning_trades += 1
        
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
                "Quantity": f"{quantity:.4f}",
                "PnL USD": f"${pnl_usd:.2f}",
                "Trade Duration": duration_str,
                "Trade Phase": self.trade_phase_engine.current_state,
                "Trim Count": self.trade_phase_engine.trim_count
            }
        )
        
        # إغلاق الصفقة (أو محاكاة)
        if quantity > 0:
            opposite_side = "sell" if side == "BUY" else "buy"
            self.execute_order(opposite_side, quantity, exit_price, is_open=False)
        
        # تسجيل النتيجة
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
                'trim_count': self.trade_phase_engine.trim_count
            })
        
        # إعادة التعيين
        self.active_trade = False
        self.trade_phase_engine = None
        self.current_position = {
            'side': None,
            'entry_price': 0.0,
            'quantity': 0.0,
            'entry_time': None
        }
        
        self.logger.log_system(
            f"Trade closed | PnL: {pnl_pct:+.2f}% | Reason: {reason}",
            "INFO" if pnl_pct >= 0 else "WARNING",
            {"PnL": f"{pnl_pct:+.2f}%", "Reason": reason, "RR": f"1:{risk_reward:.1f}"}
        )
    
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
            'current_position': self.current_position if self.active_trade else None
        }

# ============================================
#  MARKET ANALYZER - محلل السوق
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
#  MAIN BOT INTEGRATION - التكامل الرئيسي
# ============================================

# إعدادات البوت
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

BOT_VERSION = "SUI ULTRA PRO AI v9.0 — SMART TRADE MANAGEMENT ENGINE"

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

# الدوال المساعدة
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
#  SIGNAL GENERATOR - مولد الإشارات
# ============================================

class SignalGenerator:
    """مولد إشارات التداول"""
    
    def __init__(self, logger: ProConsoleLogger):
        self.logger = logger
        self.last_signal_time = 0
        self.signal_cooldown = 60  # 60 ثانية بين الإشارات
    
    def generate_signal(self, df: pd.DataFrame, market_analysis: Dict) -> Tuple[bool, str, float, str]:
        """
        توليد إشارة تداول
        
        Returns:
            Tuple[bool, str, float, str]: (هل توجد إشارة, الجانب, الثقة, السبب)
        """
        current_time = time.time()
        
        # فحص فترة التبريد
        if current_time - self.last_signal_time < self.signal_cooldown:
            return False, "", 0.0, f"Signal cooldown: {int(self.signal_cooldown - (current_time - self.last_signal_time))}s"
        
        if df.empty or len(df) < 20:
            return False, "", 0.0, "Insufficient data"
        
        # استخدام تحليل السوق لاتخاذ القرار
        trend = market_analysis.get('trend', {})
        structure = market_analysis.get('structure', {})
        momentum = market_analysis.get('momentum', {})
        liquidity = market_analysis.get('liquidity', {})
        
        # التحقق من شروط دخول أساسية
        if not self._check_basic_conditions(trend, structure, momentum, liquidity):
            return False, "", 0.0, "Basic conditions not met"
        
        # إشارة شراء
        buy_signal, buy_confidence, buy_reason = self._check_buy_signal(df, trend, structure, momentum)
        if buy_signal:
            self.last_signal_time = current_time
            return True, "buy", buy_confidence, buy_reason
        
        # إشارة بيع
        sell_signal, sell_confidence, sell_reason = self._check_sell_signal(df, trend, structure, momentum)
        if sell_signal:
            self.last_signal_time = current_time
            return True, "sell", sell_confidence, sell_reason
        
        return False, "", 0.0, "No clear signal"
    
    def _check_basic_conditions(self, trend: Dict, structure: Dict, momentum: Dict, liquidity: Dict) -> bool:
        """التحقق من الشروط الأساسية"""
        # يجب أن يكون هناك اتجاه واضح
        if not trend.get('confirmed', False):
            return False
        
        # يجب أن تكون السيولة كافية
        if liquidity.get('level') in ["LOW", "VERY_LOW"]:
            return False
        
        # يجب ألا يكون الزخم في أقصى درجاته (مفرط في الشراء/البيع)
        if momentum.get('direction') in ["OVERBOUGHT", "OVERSOLD"] and momentum.get('score', 0) > 0.8:
            return False
        
        return True
    
    def _check_buy_signal(self, df: pd.DataFrame, trend: Dict, structure: Dict, momentum: Dict) -> Tuple[bool, float, str]:
        """التحقق من إشارة الشراء"""
        reasons = []
        confidence = 0.0
        
        # 1. الاتجاه الصاعد
        if trend.get('direction') == "BULL" and trend.get('strength', 0) > 1.0:
            confidence += 0.3
            reasons.append(f"Bullish trend (strength: {trend['strength']:.1f})")
        
        # 2. هيكل BOS صاعد
        if structure.get('type') == "BOS_UP":
            confidence += 0.3
            reasons.append("Bullish BOS structure")
        
        # 3. زخم صاعد
        if momentum.get('direction') == "BULLISH" and momentum.get('score', 0) > 0.3:
            confidence += 0.2
            reasons.append(f"Bullish momentum (score: {momentum['score']:.2f})")
        
        # 4. اختبار دعم مع حجم
        recent_lows = [c['low'] for c in convert_candles_to_dicts(df)[-5:]]
        current_price = df['close'].iloc[-1]
        
        if structure.get('key_level') and current_price <= structure['key_level'] * 1.005:
            confidence += 0.2
            reasons.append(f"Testing support at {structure['key_level']:.4f}")
        
        if confidence >= 0.6 and reasons:
            return True, min(confidence, 0.95), " | ".join(reasons)
        
        return False, 0.0, ""
    
    def _check_sell_signal(self, df: pd.DataFrame, trend: Dict, structure: Dict, momentum: Dict) -> Tuple[bool, float, str]:
        """التحقق من إشارة البيع"""
        reasons = []
        confidence = 0.0
        
        # 1. الاتجاه الهابط
        if trend.get('direction') == "BEAR" and trend.get('strength', 0) > 1.0:
            confidence += 0.3
            reasons.append(f"Bearish trend (strength: {trend['strength']:.1f})")
        
        # 2. هيكل BOS هابط
        if structure.get('type') == "BOS_DOWN":
            confidence += 0.3
            reasons.append("Bearish BOS structure")
        
        # 3. زخم هابط
        if momentum.get('direction') == "BEARISH" and momentum.get('score', 0) > 0.3:
            confidence += 0.2
            reasons.append(f"Bearish momentum (score: {momentum['score']:.2f})")
        
        # 4. اختبار مقاومة مع حجم
        recent_highs = [c['high'] for c in convert_candles_to_dicts(df)[-5:]]
        current_price = df['close'].iloc[-1]
        
        if structure.get('key_level') and current_price >= structure['key_level'] * 0.995:
            confidence += 0.2
            reasons.append(f"Testing resistance at {structure['key_level']:.4f}")
        
        if confidence >= 0.6 and reasons:
            return True, min(confidence, 0.95), " | ".join(reasons)
        
        return False, 0.0, ""

# ============================================
#  MAIN BOT CLASS - الفئة الرئيسية للبوت
# ============================================

class SUIUltraProBot:
    """الفئة الرئيسية للبوت"""
    
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
            
            self.logger.log_system("Smart Trade System Initialized", "SUCCESS")
            self.logger.log_system(f"Symbol: {SYMBOL} | Risk: {RISK_ALLOC*100:.0f}% | Interval: {INTERVAL}", "INFO")
            
            return True
            
        except Exception as e:
            self.logger.log_error(f"Failed to initialize bot: {str(e)}", e, "Bot Initialization")
            return False
    
    def run_trade_loop(self):
        """تشغيل حلقة التداول الرئيسية"""
        self.logger.log_system("Starting Smart Trade Loop", "INFO")
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
                    # إدارة الصفقة الحالية
                    candles = convert_candles_to_dicts(df)
                    self.smart_trade_manager.manage_trade(current_price, candles[-20:])
                
                else:
                    # توليد إشارة تداول
                    signal, side, confidence, reason = self.signal_generator.generate_signal(df, market_analysis)
                    
                    if signal and balance and balance > 10:  # رصيد كافي
                        # تحديد منطقة الدخول بناءً على التحليل
                        entry_zone = self._determine_entry_zone(side, market_analysis)
                        
                        # محاولة فتح صفقة
                        success = self.smart_trade_manager.open_trade(
                            side=side,
                            entry_price=current_price,
                            balance=balance,
                            entry_zone=entry_zone,
                            confidence=confidence,
                            reason=reason
                        )
                        
                        if success:
                            self.logger.log_system(
                                f"Trade opened | {side.upper()} @ {current_price:.4f} | Confidence: {confidence:.2f}",
                                "SUCCESS"
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
    
    def _determine_entry_zone(self, side: str, market_analysis: Dict) -> str:
        """تحديد منطقة الدخول"""
        structure = market_analysis.get('structure', {})
        
        if side == "buy":
            if structure.get('type') == "BOS_UP":
                return "DEMAND_BOS"
            else:
                return "DEMAND_RETEST"
        else:  # sell
            if structure.get('type') == "BOS_DOWN":
                return "SUPPLY_BOS"
            else:
                return "SUPPLY_RETEST"
    
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
#  FLASK API SERVER - خادم API
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
        <title>SUI ULTRA PRO AI - Trading Dashboard</title>
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
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚀 SUI ULTRA PRO AI v9.0</h1>
                <div class="subtitle">SMART TRADE MANAGEMENT ENGINE</div>
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
                            <div>Active Trade</div>
                            <div class="stat-value" id="active-trade">No</div>
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
                        document.getElementById('active-trade').textContent = report.active_trade ? 'Yes' : 'No';
                        
                        // Update recent trades
                        const tradesList = document.getElementById('recent-trades');
                        if (report.recent_trades && report.recent_trades.length > 0) {
                            let tradesHtml = '';
                            report.recent_trades.slice().reverse().forEach(trade => {
                                const pnlClass = trade.pnl_pct > 0 ? 'positive' : 'negative';
                                const sideClass = trade.side === 'buy' ? 'trade-buy' : 'trade-sell';
                                tradesHtml += `
                                    <div class="trade-item ${sideClass}">
                                        <div style="display: flex; justify-content: space-between;">
                                            <strong>${trade.side.toUpperCase()} #${trade.id}</strong>
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
    # طباعة بانر البداية
    print(f"\n{ConsoleColors.LIGHT_CYAN}{'='*80}{ConsoleColors.RESET}")
    print(f"{ConsoleColors.LIGHT_GREEN}{BOT_VERSION}{ConsoleColors.RESET}")
    print(f"{ConsoleColors.LIGHT_CYAN}{'='*80}{ConsoleColors.RESET}\n")
    
    # تشغيل البوت
    main()
