# -*- coding: utf-8 -*-
"""
SUI ULTRA PRO AI BOT - الإصدار الذكي المتقدم المتكامل
• مجلس الإدارة الفائق الذكي مع 15 استراتيجية متقدمة  
• نظام ركوب الترند الذكي المحترف لتحقيق أقصى ربح متتالي
• السكالب الفائق الذكي بأهداف متعددة محسوبة
• إدارة صفقات ذكية متكيفة مع قوة الترند
• نظام Footprint + Diagonal Order-Flow المتقدم
• Multi-Exchange Support: BingX & Bybit
• HQ Trading Intelligence Patch - مناطق ذهبية + SMC + OB/FVG + BOX ENGINE + VOLUME ANALYSIS + VWAP INTEGRATION
• SMART PROFIT AI - نظام جني الأرباح الذكي المتقدم
• TP PROFILE SYSTEM - نظام جني الأرباح الذكي (1→2→3 مرات)
• COUNCIL STRONG ENTRY - دخول ذكي من مجلس الإدارة في المناطق القوية
• NEW INTELLIGENT PATCH - Advanced Market Analysis & Smart Monitoring
• FVG REAL vs FAKE + STOP HUNT - تمييز FVG الحقيقي من الفيك وكشف مصائد السيولة
• BOX REJECTION PRO - دخول محترف من رفض البوكس مع VWAP
• SMART MONEY HYBRID ENGINE - المحرك الهجين الذكي للسيولة والهيكل
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

# ============================================
#  SMART MONEY ENGINE IMPORT
# ============================================
try:
    # استيراد جميع المحركات الذكية
    from modules.smart_engine import SmartMoneyEngine, EntryDecision
    from modules.smc_zones_engine import SMCZonesEngine, OrderBlock, FVGZone
    from modules.smart_trailing_engine import SmartTrailingEngine
    from modules.tp_ladder_engine import TPLadderEngine
    from modules.trend_classifier_engine import TrendClassifier
    from modules.explosion_collapse_engine import ExplosionCollapseEngine
    from modules.liquidity_log_addon import LiquidityLogAddon
    from modules.fake_breakout_addon import FakeBreakoutAddon
    from modules.decision_matrix_engine import DecisionMatrix
    
    SMART_ENGINE_AVAILABLE = True
    log_i("✅ SMART MONEY ENGINE loaded successfully")
except ImportError as e:
    SMART_ENGINE_AVAILABLE = False
    log_w(f"⚠️ Smart Engine modules not available: {e}. Running in legacy mode.")

try:
    from termcolor import colored
except Exception:
    def colored(t,*a,**k): return t

# ============================================
#  SMART PATCH — HQ Trading Intelligence Engine
# ============================================

# ... [بقية الكود الحالي كما هو بدون تغيير] ...

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
BOT_VERSION = f"SUI ULTRA PRO AI v8.0 — {EXCHANGE_NAME.upper()} - SMART MONEY HYBRID ENGINE + PROFIT AI + TP PROFILE + COUNCIL ENTRY + BOX ENGINE + VOLUME ANALYSIS + VWAP"
print("🚀 Booting:", BOT_VERSION, flush=True)

# ... [بقية الكود الحالي كما هو بدون تغيير] ...

# =================== HELPERS ===================
_consec_err = 0
last_loop_ts = time.time()

# ... [بقية الكود الحالي كما هو بدون تغيير] ...

# =================== ADVANCED INDICATORS ===================
def sma(series, n: int):
    return series.rolling(n, min_periods=1).mean()

def ema(series, n: int):
    return series.ewm(span=n, adjust=False).mean()

# ... [بقية الكود الحالي كما هو بدون تغيير] ...

# =================== SUPER COUNCIL AI - ENHANCED VERSION ===================
def super_council_ai_enhanced(df):
    try:
        if len(df) < 50:
            return {"b": 0, "s": 0, "score_b": 0.0, "score_s": 0.0, "logs": [], "confidence": 0.0}
        
        ind = compute_indicators(df)
        
        # ... [الكود الحالي كما هو] ...
        
        # ===== SMART MONEY ENGINE INTEGRATION =====
        if SMART_ENGINE_AVAILABLE and len(df) >= 20:
            try:
                # تحويل DataFrame إلى قائمة candles للمحرك الذكي
                candles_list = []
                for i in range(len(df)):
                    candles_list.append({
                        'open': float(df['open'].iloc[i]),
                        'high': float(df['high'].iloc[i]),
                        'low': float(df['low'].iloc[i]),
                        'close': float(df['close'].iloc[i]),
                        'volume': float(df['volume'].iloc[i])
                    })
                
                # تشغيل المحرك الذكي
                smart_engine = SmartMoneyEngine(candles_list[-50:])
                smart_decision = smart_engine.evaluate()
                
                # دمج قرار المحرك الذكي مع مجلس الإدارة
                if smart_decision.allow_entry:
                    if smart_decision.side == "BUY":
                        score_b += smart_decision.confidence * 2.0
                        votes_b += int(smart_decision.confidence * 3)
                        logs.append(f"🧠 SMART ENGINE → BUY (conf: {smart_decision.confidence:.2f})")
                    elif smart_decision.side == "SELL":
                        score_s += smart_decision.confidence * 2.0
                        votes_s += int(smart_decision.confidence * 3)
                        logs.append(f"🧠 SMART ENGINE → SELL (conf: {smart_decision.confidence:.2f})")
                        
            except Exception as e:
                logs.append(f"🟨 Smart Engine error: {e}")
        
        # ... [بقية الكود الحالي كما هو] ...
        
        return {
            "b": votes_b, "s": votes_s,
            "score_b": round(score_b, 2), "score_s": round(score_s, 2),
            "logs": logs, "ind": ind, "gz": gz, "candles": candles,
            "confidence": round(confidence, 2),
            "momentum": momentum,
            "volume": volume_profile,
            "trend_strength": trend_strength,
            "early_trend": early_trend,
            "breakout": breakout,
            "fvg_ctx": fvg_ctx
        }
    except Exception as e:
        log_w(f"super_council_ai_enhanced error: {e}")
        import traceback
        log_w(f"Traceback: {traceback.format_exc()}")
        return {"b":0,"s":0,"score_b":0.0,"score_s":0.0,"logs":[],"ind":{},"confidence":0.0}

# ... [بقية الكود الحالي كما هو بدون تغيير] ...

# =================== ENHANCED TRADE EXECUTION ===================
def open_market_enhanced(side, qty, price):
    """نسخة محسنة من فتح الصفقة مع الذكاء الجديد"""
    if qty <= 0 or price is None:
        log_e("❌ كمية أو سعر غير صالح")
        return False

    # تحقق إضافي من الحجم
    balance = balance_usdt()
    expected_qty = compute_size(balance, price)
    
    if abs(qty - expected_qty) > (expected_qty * 0.1):
        log_w(f"⚠️ تصحيح الحجم: {qty:.4f} → {expected_qty:.4f}")
        qty = expected_qty

    df = fetch_ohlcv(limit=200)
    ind = compute_indicators(df)

    # --- تحديد المود (scalp / trend) حسب الدالة الحالية ---
    mode_info = classify_trade_mode(df, ind)
    mode = mode_info.get("mode", "scalp")
    why_mode = mode_info.get("why", "classify_trade_mode")

    # ===== SMART MONEY ENGINE VALIDATION =====
    if SMART_ENGINE_AVAILABLE:
        try:
            # تحويل DataFrame إلى candles للمحرك الذكي
            candles_list = []
            for i in range(len(df)):
                candles_list.append({
                    'open': float(df['open'].iloc[i]),
                    'high': float(df['high'].iloc[i]),
                    'low': float(df['low'].iloc[i]),
                    'close': float(df['close'].iloc[i]),
                    'volume': float(df['volume'].iloc[i])
                })
            
            # تشغيل المحرك الذكي للتحقق
            smart_engine = SmartMoneyEngine(candles_list[-50:])
            smart_decision = smart_engine.evaluate()
            
            # إذا كان المحرك الذكي يرفض الدخول
            if not smart_decision.allow_entry:
                log_i(f"🧠 SMART ENGINE BLOCKED: {smart_decision.reason}")
                return False
                
            # إذا كان المحرك الذكي يوافق على الاتجاه
            if smart_decision.side == side.upper():
                log_i(f"🧠 SMART ENGINE CONFIRMED: {side.upper()} | Confidence: {smart_decision.confidence:.2f}")
                
        except Exception as e:
            log_w(f"Smart Engine validation error: {e}")
    
    # ... [بقية الكود الحالي كما هو] ...
    
    # ✅ نحسب بيانات المجلس الحقيقية للصفقة
    council_data = super_council_ai_enhanced(df)

    # ✅ نحدد Profit Profile المناسب
    profit_profile = classify_profit_profile(df, ind, council_data, trend_info, mode)

    # إعدادات الإدارة المبنية على الـ profile الجديد
    management_config = {
        "tp1_pct": profit_profile["tp1_pct"],
        "tp2_pct": profit_profile["tp2_pct"],
        "tp3_pct": profit_profile["tp3_pct"],
        "be_activate_pct": profit_profile["tp1_pct"],
        "trail_activate_pct": profit_profile["trail_start_pct"],
        "atr_trail_mult": TREND_ATR_MULT if mode == "trend" else SCALP_ATR_TRAIL_MULT,
        "profile": profit_profile["label"],
        "profile_desc": profit_profile["desc"]
    }

    log_i(f"🎛 TRADE MODE DECISION: {mode.upper()} | profile={profit_profile['label']} | {why_mode}")

    # تنفيذ الأمر
    success = execute_trade_decision(side, price, qty, mode, council_data, golden_zone_check(df, ind))

    if success:
        trade_side = "long" if side.lower().startswith("b") else "short"
        
        STATE.update({
            "open": True,
            "side": trade_side,
            "entry": float(price),
            "qty": float(qty),
            "pnl": 0.0,
            "bars": 0,
            "mode": mode,
            "mode_why": why_mode,
            "management": management_config,
            "opened_at": time.time(),
            "tp1_done": False,
            "trail_active": False,
            "breakeven_armed": False,
            "highest_profit_pct": 0.0,
            "profit_targets_achieved": 0,
            "profit_profile": profit_profile["label"],
            "council_controlled": STATE.get("last_entry_source") == "COUNCIL_STRONG"
        })

        save_state({
            "in_position": True,
            "side": "LONG" if trade_side == "long" else "SHORT",
            "entry_price": price,
            "position_qty": qty,
            "leverage": LEVERAGE,
            "mode": mode,
            "mode_why": why_mode,
            "profit_profile": profit_profile["label"],
            "management": management_config,
            "opened_at": int(time.time())
        })

        # لوج ملوّن واضح
        profile_color = "🟢" if profit_profile["label"] == "TREND_STRONG" else "🟡" if profit_profile["label"] == "TREND_MEDIUM" else "🔵"
        log_g(
            f"{profile_color} COUNCIL TRADE OPENED | {side.upper()} {qty:.4f} @ {price:.6f} "
            f"| {mode.upper()} | {profit_profile['label']} | "
            f"TPs: {profit_profile['tp1_pct']}%"
            f"{f' → {profit_profile["tp2_pct"]}%' if profit_profile['tp2_pct'] else ''}"
            f"{f' → {profit_profile["tp3_pct"]}%' if profit_profile['tp3_pct'] else ''}"
        )
        
        print_position_snapshot(reason=f"OPEN - {mode.upper()}[{profit_profile['label']}]")
        return True

    return False

# ... [بقية الكود الحالي كما هو بدون تغيير] ...

# ============================================
#  ENHANCED TRADE LOOP WITH SMART MONEY ENGINE
# ============================================

def trade_loop_enhanced_with_smart_money():
    global wait_for_next_signal_side, compound_pnl
    loop_i = 0
    
    # إحصائيات الأداء
    performance_stats = {
        'total_trades': 0,
        'winning_trades': 0,
        'total_profit': 0.0,
        'consecutive_wins': 0,
        'consecutive_losses': 0
    }
    
    while True:
        try:
            current_time = time.time()
            bal = balance_usdt()
            px = price_now()
            df = fetch_ohlcv()
            
            if df.empty:
                time.sleep(BASE_SLEEP)
                continue
                
            # ✅ إضافة نظام جني الأرباح الذكي
            if STATE.get("open") and px:
                apply_smart_profit_strategy()
                
            # ============================================
            #  🧠 SMART MONEY ENGINE INTEGRATION
            # ============================================
            
            if SMART_ENGINE_AVAILABLE:
                try:
                    # تحويل DataFrame إلى candles
                    candles_list = []
                    for i in range(len(df)):
                        candles_list.append({
                            'open': float(df['open'].iloc[i]),
                            'high': float(df['high'].iloc[i]),
                            'low': float(df['low'].iloc[i]),
                            'close': float(df['close'].iloc[i]),
                            'volume': float(df['volume'].iloc[i])
                        })
                    
                    # تشغيل جميع المحركات الذكية
                    # 1. المحرك الأساسي
                    smart_engine = SmartMoneyEngine(candles_list[-50:])
                    smart_decision = smart_engine.evaluate()
                    
                    # 2. مناطق SMC
                    smc_engine = SMCZonesEngine(candles_list[-30:])
                    ob_signal = smc_engine.detect_order_block()
                    fvg_signal = smc_engine.detect_fvg()
                    
                    # 3. تصنيف الترند
                    ind = compute_indicators(df)
                    trend_classifier = TrendClassifier(
                        candles=candles_list[-50:],
                        adx=safe_get(ind, 'adx', 0),
                        di_plus=safe_get(ind, 'plus_di', 0),
                        di_minus=safe_get(ind, 'minus_di', 0),
                        volume=df['volume'].astype(float).values[-50:]
                    )
                    trend_state = trend_classifier.classify()
                    
                    # 4. كشف الانفجار/الانهيار
                    explosion_engine = ExplosionCollapseEngine(
                        candles=candles_list[-20:],
                        volume=df['volume'].astype(float).values[-20:],
                        atr=safe_get(ind, 'atr', 0)
                    )
                    explosion_signal = explosion_engine.detect()
                    collapse_signal = explosion_engine.detect_collapse()
                    
                    # 5. لوج السيولة
                    liquidity_log = LiquidityLogAddon(
                        candles=candles_list[-20:],
                        volume=df['volume'].astype(float).values[-20:]
                    ).snapshot()
                    
                    # 6. كشف الاختراقات الكاذبة
                    fake_detector = FakeBreakoutAddon(
                        candles=candles_list[-20:],
                        volume=df['volume'].astype(float).values[-20:],
                        atr=safe_get(ind, 'atr', 0)
                    )
                    fake_flags = fake_detector.verdict()
                    
                    # 7. مصفوفة القرار الذكية
                    decision_matrix = DecisionMatrix(
                        trend_state=trend_state,
                        explosion_signal=explosion_signal,
                        fake_breakout_flags=fake_flags,
                        liquidity_snapshot=liquidity_log,
                        smc_signal=ob_signal.type if ob_signal else None,
                        position_open=STATE["open"]
                    )
                    
                    final_decision = decision_matrix.decide()
                    
                    # لوج قرار المحرك الذكي
                    if LOG_ADDONS:
                        log_i(f"🧠 SMART ENGINE → Decision: {final_decision['action']} | Trend: {trend_state} | Liq: {liquidity_log}")
                    
                    # تطبيق قرار المحرك الذكي
                    if final_decision["action"] == "BLOCK" and not STATE["open"]:
                        log_i(f"🚫 SMART ENGINE BLOCKED: {final_decision['reason']}")
                        continue
                        
                    elif final_decision["action"] == "EXIT" and STATE["open"]:
                        log_i(f"🟡 SMART ENGINE EXIT: {final_decision['reason']}")
                        close_market_strict(f"Smart Engine: {final_decision['reason']}")
                        continue
                        
                    elif final_decision["action"] in ["BUY", "SELL"] and not STATE["open"]:
                        log_i(f"🟢 SMART ENGINE SIGNAL: {final_decision['action']} | Votes: {final_decision.get('votes', 0)}")
                        
                except Exception as e:
                    log_w(f"Smart Engine execution error: {e}")
            
            # ============================================
            #  END OF SMART MONEY ENGINE INTEGRATION
            # ============================================
                
            # تحديث جميع المحركات الذكية
            close_prices = df['close'].astype(float).tolist()
            volumes = df['volume'].astype(float).tolist()
            
            # تحديث السياق
            trend_ctx.update(close_prices[-1] if close_prices else 0)
            smc_detector.detect_swings(df)
            
            info = rf_signal_live(df)
            ind = compute_indicators(df)
            spread_bps = orderbook_spread_bps()
            
            # تحديث orderbook للـFlow Boost
            try:
                STATE["last_orderbook"] = ex.fetch_order_book(SYMBOL, limit=FLOW_STACK_DEPTH)
            except Exception as e:
                log_w(f"Orderbook update failed: {e}")
            
            snap = emit_snapshots(ex, SYMBOL, df,
                                balance_fn=lambda: float(bal) if bal else None,
                                pnl_fn=lambda: float(compound_pnl))
            
            if STATE["open"] and px:
                STATE["pnl"] = (px-STATE["entry"])*STATE["qty"] if STATE["side"]=="long" else (STATE["entry"]-px)*STATE["qty"]
            
            # ============================================
            #  SMART DECISION INTELLIGENCE BLOCK 
            # ============================================
            
            # ===== BOX ENGINE INTEGRATION =====
            boxes = build_sr_boxes(df)
            box_ctx = analyze_box_context(df, boxes)
            
            if box_ctx["ctx"] != "none":
                log_i(
                    f"📦 BOX CONTEXT: {box_ctx['ctx']} | tier={box_ctx['tier']} "
                    f"score={box_ctx['score']:.2f} rr={box_ctx['rr']:.2f} dir={box_ctx['dir']} "
                    f"| debug={box_ctx['debug']}"
                )
            
            # ===== VWAP CALCULATION =====
            vwap_ctx = compute_vwap(df)
            
            entry_reasons = []
            allow_buy = False
            allow_sell = False
            
            close_price = float(df['close'].iloc[-1]) if len(df) > 0 else px
            
            # ---- Volume Confirmation ----
            vol_ok = volume_is_strong(volumes)
            
            # ---- OB / FVG Detection ----
            ob_signal = detect_ob(df)
            fvg_signal = detect_fvg(df)
            
            # ---- Golden Zones ----
            golden_data = golden_zone_check(df, ind)
            gb = golden_data.get("ok", False) and golden_data.get("zone", {}).get("type") == "golden_bottom"
            gt = golden_data.get("ok", False) and golden_data.get("zone", {}).get("type") == "golden_top"
            
            # ---- SMC Liquidity Analysis ----
            liquidity_zones = smc_detector.detect_liquidity_zones(close_price)
            buy_liquidity = any(zone[0] == "buy_liquidity" for zone in liquidity_zones)
            sell_liquidity = any(zone[0] == "sell_liquidity" for zone in liquidity_zones)
            
            # ---- ADX Gate ----
            adx_ok = safe_get(ind, "adx", 0) >= ADX_GATE
            
            # ---- Zero Reversal Scalping Check ----
            scalper_ready, scalper_reason = zero_scalper.can_trade(current_time)
            
            # ===== BUY CONDITIONS =====
            buy_conditions = []
            
            # Golden Bottom
            if gb and trend_ctx.trend != "down" and adx_ok:
                allow_buy = True
                buy_conditions.append("Golden Bottom")
            
            # Bullish FVG
            if fvg_signal and fvg_signal[0] == "bullish":
                allow_buy = True
                buy_conditions.append("Bullish FVG")
            
            # Bullish OB
            if ob_signal and ob_signal[0] == "bullish":
                allow_buy = True
                buy_conditions.append("Bullish OB")
            
            # Buy Liquidity
            if buy_liquidity and vol_ok:
                allow_buy = True
                buy_conditions.append("Buy Liquidity Zone")
            
            # ===== SELL CONDITIONS =====
            sell_conditions = []
            
            # Golden Top
            if gt and trend_ctx.trend != "up" and adx_ok:
                allow_sell = True
                sell_conditions.append("Golden Top")
            
            # Bearish FVG
            if fvg_signal and fvg_signal[0] == "bearish":
                allow_sell = True
                sell_conditions.append("Bearish FVG")
            
            # Bearish OB
            if ob_signal and ob_signal[0] == "bearish":
                allow_sell = True
                sell_conditions.append("Bearish OB")
            
            # Sell Liquidity
            if sell_liquidity and vol_ok:
                allow_sell = True
                sell_conditions.append("Sell Liquidity Zone")
            
            # ===== SMART MONEY ENGINE BOOST =====
            if SMART_ENGINE_AVAILABLE and 'final_decision' in locals():
                if final_decision["action"] == "BUY":
                    allow_buy = True
                    buy_conditions.append("Smart Engine BUY")
                elif final_decision["action"] == "SELL":
                    allow_sell = True
                    sell_conditions.append("Smart Engine SELL")
                elif final_decision["action"] == "BLOCK":
                    allow_buy = False
                    allow_sell = False
                    entry_reasons.append(f"Smart Engine Block: {final_decision['reason']}")
            
            # ===== BOX + VWAP PRO ENTRY =====
            box_vol = box_ctx.get("box_vol", {}) if box_ctx else {}
            box_vol_label = box_vol.get("label", "normal")

            box_strong_enough = (
                box_ctx
                and box_ctx.get("ctx") in ("strong_reversal_short", "strong_reversal_long")
                and box_vol_label == "strong"
                and box_ctx.get("rr", 0) >= 1.6
            )

            box_rejection_side = None
            if box_ctx and box_ctx.get("ctx") in ("strong_reversal_short", "strong_reversal_long"):
                rej_cnt   = box_vol.get("rejects", 0)
                strong_ok = (box_vol_label == "strong") if BOX_REJECTION_REQUIRE_STRONG else True
                if rej_cnt >= BOX_REJECTION_MIN_REJECTS and strong_ok:
                    box_rejection_side = box_ctx.get("dir")
                    entry_reasons.append(
                        f"BOX_REJECTION_CONFIRMED({box_rejection_side},rej={rej_cnt},vol={box_vol_label})"
                    )

                    if box_rejection_side == "buy":
                        allow_buy = True
                        allow_sell = False
                    elif box_rejection_side == "sell":
                        allow_sell = True
                        allow_buy = False

            if box_strong_enough:
                v_pos   = vwap_ctx.get("position", "none")
                v_slope = vwap_ctx.get("slope_bps", 0.0)

                if box_ctx["dir"] == "sell":
                    if v_pos in ("above", "at") and v_slope <= 5.0:
                        allow_sell = True
                        entry_reasons.append(
                            f"BOX_STRONG_SELL(vol={box_vol.get('vol_ratio')},rej={box_vol.get('rejects')},vwap_pos={v_pos})"
                        )

                if box_ctx["dir"] == "buy":
                    if v_pos in ("below", "at") and v_slope >= -5.0:
                        allow_buy = True
                        entry_reasons.append(
                            f"BOX_STRONG_BUY(vol={box_vol.get('vol_ratio')},rej={box_vol.get('rejects')},vwap_pos={v_pos})"
                        )

            # ---- Volume Final Gate ----
            if not vol_ok:
                allow_buy = False
                allow_sell = False
                entry_reasons.append("Weak Volume - Blocked")
            else:
                entry_reasons.extend(buy_conditions)
                entry_reasons.extend(sell_conditions)
            
            # ---- Scalper Ready Check ----
            if not scalper_ready and SCALP_MODE:
                allow_buy = allow_buy and False
                allow_sell = allow_sell and False
                entry_reasons.append(f"Scalper Cooldown: {scalper_reason}")
            
            # ---- RF Signal Integration ----
            rf_buy = info.get("long", False)
            rf_sell = info.get("short", False)
            
            # ---- Missed Signals Logging ----
            if rf_buy and not allow_buy and not STATE["open"]:
                signal_logger.log_missed_signal("BUY", close_price, " | ".join(entry_reasons))
                
            if rf_sell and not allow_sell and not STATE["open"]:
                signal_logger.log_missed_signal("SELL", close_price, " | ".join(entry_reasons))
            
            # ================= BOX REJECTION SMART ENTRY =================
            box_reject_short = evaluate_box_rejection_for_entry(df, box_ctx, vwap_ctx.get("vwap"), side="short")
            box_reject_long  = evaluate_box_rejection_for_entry(df, box_ctx, vwap_ctx.get("vwap"), side="long")

            box_entry_signal = None
            box_entry_reason = None

            if box_reject_short["ok"]:
                box_entry_signal = "short"
                box_entry_reason = box_reject_short["reason"]
                log_y(f"📦 BOX REJECTION SELL: {box_entry_reason}")
            
            if box_reject_long["ok"]:
                box_entry_signal = "long"
                box_entry_reason = box_reject_long["reason"]
                log_y(f"📦 BOX REJECTION BUY: {box_entry_reason}")

            # ============================================
            #  FINAL ENTRY EXECUTION LAYER
            # ============================================

            council_data = super_council_ai_enhanced(df)
            final_signal   = None
            entry_source   = None

            cb   = int(council_data.get("b", 0))
            cs   = int(council_data.get("s", 0))
            sb   = float(council_data.get("score_b", 0.0))
            ss   = float(council_data.get("score_s", 0.0))
            conf = float(council_data.get("confidence", 0.0))
            total_score = sb + ss

            # ===== BOX ENGINE BOOST =====
            if box_ctx["ctx"] != "none":
                if box_ctx["dir"] == "buy":
                    cb += 3
                    sb += 1.5
                    log_i(f"📦 BOX BOOST: +3 votes BUY | score +1.5")
                elif box_ctx["dir"] == "sell":
                    cs += 3
                    ss += 1.5
                    log_i(f"📦 BOX BOOST: +3 votes SELL | score +1.5")
            
            council_side = None
            if COUNCIL_STRONG_ENTRY and conf >= COUNCIL_STRONG_CONF and total_score >= COUNCIL_STRONG_SCORE:
                if cb >= COUNCIL_STRONG_VOTES and sb > ss:
                    council_side = "buy"
                elif cs >= COUNCIL_STRONG_VOTES and ss > sb:
                    council_side = "sell"

                if council_side:
                    log_i(
                        f"🏛 COUNCIL STRONG SIDE → {council_side.upper()} | "
                        f"votes={cb}/{cs} score={sb:.1f}/{ss:.1f} conf={conf:.2f}"
                    )

            # ===== المسار الأساسي: RF + SMC / GOLDEN =====
            if rf_buy and allow_buy:
                final_signal = "buy"
                entry_source = "RF+SMC"
            elif rf_sell and allow_sell:
                final_signal = "sell"
                entry_source = "RF+SMC"

            # ===== المسار الذكي: دخول مجلس الإدارة القوي =====
            if final_signal is None and council_side is not None:
                safe_to_enter = True

                if COUNCIL_BLOCK_STRONG_TREND and trend_ctx.is_strong_trend():
                    if council_side == "buy" and trend_ctx.trend == "down" and not gb:
                        safe_to_enter = False
                    if council_side == "sell" and trend_ctx.trend == "up" and not gt:
                        safe_to_enter = False

                if safe_to_enter:
                    final_signal = council_side
                    entry_source = "COUNCIL_STRONG"
                    entry_reasons.append("COUNCIL_STRONG_ENTRY")
                    log_g(
                        f"🏛 COUNCIL STRONG ENTRY → {final_signal.upper()} | "
                        f"votes={cb}/{cs} score={sb:.1f}/{ss:.1f} conf={conf:.2f}"
                    )
                else:
                    log_i("🏛 COUNCIL STRONG ENTRY blocked by opposite strong trend")

            # ===== دمج SMART MONEY ENGINE =====
            if final_signal is None and SMART_ENGINE_AVAILABLE and 'final_decision' in locals():
                if final_decision["action"] == "BUY" and allow_buy:
                    final_signal = "buy"
                    entry_source = "SMART_ENGINE"
                elif final_decision["action"] == "SELL" and allow_sell:
                    final_signal = "sell"
                    entry_source = "SMART_ENGINE"

            # ===== دمج BOX REJECTION =====
            if final_signal is None and box_entry_signal:
                final_signal = box_entry_signal
                entry_source = "BOX_REJECTION"
                entry_reasons.append(box_entry_reason)

            # ===== فلتر BALANCED MODE =====
            combined_score = total_score + box_ctx.get("score", 0.0)

            if combined_score < BALANCED_MIN_SCORE or box_ctx.get("tier") == "weak":
                if council_side or allow_buy or allow_sell:
                    log_y(f"⚠️ BALANCED FILTER: skipped weak setup | combined_score={combined_score:.2f}")
                council_side = None
                allow_buy = False
                allow_sell = False
                final_signal = None

            # ===== تنفيذ الدخول إن وجد إشارة نهائية =====
            if final_signal and not STATE["open"]:
                allow_wait, wait_reason = wait_gate_allow(df, info)

                max_score = max(council_data.get("score_b", 0.0), council_data.get("score_s", 0.0))
                max_votes = max(council_data.get("b", 0), council_data.get("s", 0))
                conf = council_data.get("confidence", 0.0)

                strong_council = (
                    conf >= COUNCIL_STRONG_ENTRY_CONF and
                    max_score >= COUNCIL_STRONG_ENTRY_SCORE and
                    max_votes >= COUNCIL_STRONG_MIN_VOTES
                )

                rf_side = "buy" if info.get("long") else ("sell" if info.get("short") else None)
                wait_side = wait_for_next_signal_side

                override_wait = False
                if not allow_wait and strong_council and rf_side and wait_side and rf_side == wait_side:
                    override_wait = True
                    log_i(f"🏆 COUNCIL STRONG ENTRY override wait-for-next-RF({wait_side})")

                if not allow_wait and not override_wait:
                    log_i(f"⏳ Waiting: {wait_reason}")
                else:
                    qty = compute_size(bal, px or info["price"])
                    if qty > 0:
                        if box_strong_enough:
                            entry_source = "BOX+VWAP"
                        elif override_wait:
                            entry_source = "COUNCIL_STRONG"
                        else:
                            entry_source = "RF+SMC"
                            
                        STATE["last_entry_source"] = entry_source
                        STATE["last_entry_reasons"] = " | ".join(entry_reasons) if entry_reasons else ""
                        STATE["last_balance"] = float(bal or 0.0)

                        # تحديد قوة الإشارة وملف TP
                        signal_strength = "weak"
                        tp_profile = "SCALP_1"

                        if box_ctx["tier"] == "strong" and trend_ctx.trend == "trend":
                            signal_strength = "strong"
                            tp_profile = "TREND_3"
                        elif box_ctx["tier"] in ("mid", "strong"):
                            signal_strength = "mid"
                            tp_profile = "MID_2"

                        STATE["signal_strength"] = signal_strength
                        STATE["tp_profile"] = tp_profile

                        ok = open_market_enhanced(final_signal, qty, px or info["price"])
                        if ok:
                            wait_for_next_signal_side = None
                            log_i(f"🎯 SMART EXECUTION: {final_signal.upper()} | src={entry_source} | "
                                  f"Reasons: {' | '.join(entry_reasons)} | Strength: {signal_strength} | TP: {tp_profile}")
                            if SCALP_MODE:
                                zero_scalper.record_trade(current_time, True)
                    else:
                        log_w("❌ Quantity <= 0")

            # إدارة الصفقة المفتوحة
            if STATE["open"]:
                manage_after_entry_enhanced_with_smart_patch(df, ind, {
                    "price": px or info["price"], 
                    "bm": snap["bm"],
                    "flow": snap["flow"],
                    "trend_ctx": trend_ctx,
                    "vol_ok": vol_ok,
                    **info
                }, performance_stats)
            
            # Legacy Logging
            if LOG_LEGACY:
                pretty_snapshot(bal, {"price": px or info["price"], **info}, ind, spread_bps, " | ".join(entry_reasons), df)
            
            loop_i += 1
            sleep_s = NEAR_CLOSE_S if time_to_candle_close(df) <= 10 else BASE_SLEEP
            time.sleep(sleep_s)
            
        except Exception as e:
            log_e(f"Smart loop error: {e}\n{traceback.format_exc()}")
            time.sleep(BASE_SLEEP)

# استبدال الدورة الرئيسية
trade_loop = trade_loop_enhanced_with_smart_money

# =================== إدارة الصفقات مع SMART MONEY ENGINE ===================

def manage_trade_with_smart_engine(df, ind, info):
    """إدارة الصفقة مع المحرك الذكي"""
    if not STATE["open"] or STATE["qty"] <= 0:
        return

    px = info.get("price") or price_now()
    if not px:
        return
        
    entry = STATE["entry"]
    side = STATE["side"]
    qty = STATE["qty"]
    
    # حساب الربح/الخسارة
    if side == "long":
        pnl_pct = ((px - entry) / entry) * 100
    else:
        pnl_pct = ((entry - px) / entry) * 100
        
    STATE["pnl"] = pnl_pct
    
    if pnl_pct > STATE["highest_profit_pct"]:
        STATE["highest_profit_pct"] = pnl_pct
    
    # ===== SMART TRAILING ENGINE =====
    if SMART_ENGINE_AVAILABLE and STATE.get("trail_active"):
        try:
            # تحويل DataFrame إلى candles
            candles_list = []
            for i in range(len(df)):
                candles_list.append({
                    'open': float(df['open'].iloc[i]),
                    'high': float(df['high'].iloc[i]),
                    'low': float(df['low'].iloc[i]),
                    'close': float(df['close'].iloc[i]),
                    'volume': float(df['volume'].iloc[i])
                })
            
            # تطبيق التريل الذكي
            trailing_engine = SmartTrailingEngine(
                candles=candles_list[-20:],
                side="BUY" if side == "long" else "SELL"
            )
            
            current_sl = STATE.get("trail") or STATE["entry"]
            new_sl = trailing_engine.trailing_stop(entry, current_sl)
            
            if new_sl != current_sl:
                STATE["trail"] = new_sl
                log_i(f"🔄 SMART TRAILING: {current_sl:.6f} → {new_sl:.6f}")
                
        except Exception as e:
            log_w(f"Smart trailing error: {e}")
    
    # ===== TP LADDER ENGINE =====
    if SMART_ENGINE_AVAILABLE and pnl_pct > 0.5:
        try:
            # تحويل DataFrame إلى candles
            candles_list = []
            for i in range(len(df)):
                candles_list.append({
                    'open': float(df['open'].iloc[i]),
                    'high': float(df['high'].iloc[i]),
                    'low': float(df['low'].iloc[i]),
                    'close': float(df['close'].iloc[i]),
                    'volume': float(df['volume'].iloc[i])
                })
            
            # تحديد نوع الترند
            trend_classifier = TrendClassifier(
                candles=candles_list[-50:],
                adx=safe_get(ind, 'adx', 0),
                di_plus=safe_get(ind, 'plus_di', 0),
                di_minus=safe_get(ind, 'minus_di', 0),
                volume=df['volume'].astype(float).values[-50:]
            )
            trend_type = trend_classifier.classify()
            
            # تحويل التصنيف إلى MID/LARGE
            if trend_type == "LARGE":
                tp_trend = "LARGE"
            elif trend_type == "MID":
                tp_trend = "MID"
            else:
                tp_trend = "MID"
            
            # تطبيق TP Ladder
            tp_engine = TPLadderEngine(
                candles=candles_list[-60:],
                side="BUY" if side == "long" else "SELL",
                trend_type=tp_trend
            )
            
            tp_ladder = tp_engine.ladder(entry)
            
            # تطبيق مستويات TP
            for i, tp_level in enumerate(tp_ladder):
                tp_key = f"smart_tp_{i+1}_done"
                if not STATE.get(tp_key, False) and pnl_pct >= (tp_level["tp"] - entry) / entry * 100:
                    close_qty = safe_qty(qty * tp_level["close_pct"])
                    if close_qty > 0:
                        close_side = "sell" if side == "long" else "buy"
                        if MODE_LIVE and EXECUTE_ORDERS and not DRY_RUN:
                            try:
                                params = exchange_specific_params(close_side, is_close=True)
                                ex.create_order(SYMBOL, "market", close_side, close_qty, None, params)
                                log_g(f"🎯 SMART TP{i+1}: {pnl_pct:.2f}% | closed {tp_level['close_pct']*100:.0f}%")
                                STATE["qty"] = safe_qty(STATE["qty"] - close_qty)
                            except Exception as e:
                                log_e(f"❌ Smart TP{i+1} close failed: {e}")
                        STATE[tp_key] = True
                        
        except Exception as e:
            log_w(f"TP Ladder Engine error: {e}")
    
    # ===== COLLAPSE DETECTION =====
    if SMART_ENGINE_AVAILABLE:
        try:
            candles_list = []
            for i in range(len(df)):
                candles_list.append({
                    'open': float(df['open'].iloc[i]),
                    'high': float(df['high'].iloc[i]),
                    'low': float(df['low'].iloc[i]),
                    'close': float(df['close'].iloc[i]),
                    'volume': float(df['volume'].iloc[i])
                })
            
            explosion_engine = ExplosionCollapseEngine(
                candles=candles_list[-20:],
                volume=df['volume'].astype(float).values[-20:],
                atr=safe_get(ind, 'atr', 0)
            )
            
            collapse_signal = explosion_engine.detect_collapse()
            
            if collapse_signal and pnl_pct > 0:
                log_w(f"💥 COLLAPSE DETECTED during trade! PnL: {pnl_pct:.2f}%")
                close_market_strict("Smart Engine Collapse Detection")
                return
                
        except Exception as e:
            log_w(f"Collapse detection error: {e}")
    
    STATE["bars"] += 1

# =================== تثبيت الدوال المحسنة ===================

# استبدال دالة إدارة الصفقات
manage_after_entry_enhanced_with_smart_patch = manage_trade_with_smart_engine

# =================== إعداد الملفات والتهيئة ===================

def initialize_smart_engine():
    """تهيئة المحرك الذكي"""
    if SMART_ENGINE_AVAILABLE:
        log_g("🚀 SMART MONEY HYBRID ENGINE INITIALIZED")
        log_i("   • Smart Money Engine")
        log_i("   • SMC Zones Engine")
        log_i("   • Smart Trailing Engine")
        log_i("   • TP Ladder Engine")
        log_i("   • Trend Classifier")
        log_i("   • Explosion/Collapse Detector")
        log_i("   • Liquidity Log Addon")
        log_i("   • Fake Breakout Detector")
        log_i("   • Decision Matrix")
        log_i("")
        log_i("🧠 ENGINE MODE: HYBRID (Legacy + Smart Money)")
        log_i("📊 DECISION FLOW: RF+SMC → COUNCIL → SMART ENGINE → BOX")
        log_i("🛡️ PROTECTION: Fake Breakout + Collapse Detection")
        log_i("🎯 TP STRATEGY: Dynamic Ladder based on Trend Type")
        return True
    else:
        log_w("⚠️ SMART ENGINE NOT AVAILABLE - Running in Legacy Mode")
        return False

# =================== البداية ===================

if __name__ == "__main__":
    # تهيئة المحرك الذكي
    init_success = initialize_smart_engine()
    
    # بدء التداول
    log_g("🚀 STARTING ULTRA PRO AI BOT WITH SMART MONEY ENGINE")
    
    try:
        trade_loop()
    except KeyboardInterrupt:
        log_i("🛑 Bot stopped by user")
    except Exception as e:
        log_e(f"❌ Fatal error: {e}")
        traceback.print_exc()
