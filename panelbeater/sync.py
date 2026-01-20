"""Synchronise the account to the latest information."""

# pylint: disable=too-many-locals,broad-exception-caught,too-many-arguments,too-many-positional-arguments,superfluous-parens,line-too-long,too-many-branches
import os
import time

import pandas as pd
from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import (OrderClass, OrderSide, OrderType,
                                  QueryOrderStatus, TimeInForce)
from alpaca.trading.requests import (GetOrdersRequest, LimitOrderRequest,
                                     MarketOrderRequest, ReplaceOrderRequest,
                                     StopLossRequest, StopOrderRequest,
                                     TakeProfitRequest)

# Minimum change in position (in USD) required to trigger a trade
MIN_TRADE_USD = 50.0
# Safety factor to account for Alpaca's 2% price collar on market orders
SAFETY_FACTOR = 0.95


def sync_positions(df: pd.DataFrame):
    """Sync the portfolio, now with explicit Options and Crypto/Equity handling."""
    trading_client = TradingClient(
        os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"], paper=True
    )
    clock = trading_client.get_clock()
    # Check if we should skip options entirely right now
    is_market_open = clock.is_open  # type: ignore
    account = trading_client.get_account()
    available_funds = float(account.buying_power) * SAFETY_FACTOR  # type: ignore

    total_conviction = df["kelly_fraction"].sum()
    # Handle the case where total_conviction is 0 to avoid DivisionByZero
    if total_conviction > 0:
        df["target_usd"] = (df["kelly_fraction"] / total_conviction) * available_funds
    else:
        df["target_usd"] = 0.0

    # Get all positions (Equities, Crypto, and Options)
    raw_positions = trading_client.get_all_positions()
    positions = {p.symbol: p for p in raw_positions}  # type: ignore

    for _, row in df.iterrows():
        # --- THE FIX: IDENTIFY THE CORRECT SYMBOL ---
        # If option_symbol is present and not null, use it.
        is_option = pd.notna(row.get("option_symbol")) and row.get(
            "option_symbol"
        ) != row.get("ticker")
        if is_option and not is_market_open:  # pyright: ignore
            # Silent skip for options during off-hours
            continue
        symbol = row["option_symbol"] if is_option else row["ticker"]  # pyright: ignore

        # Standardize for Crypto detection
        is_crypto = "-" in symbol or "/" in symbol
        trade_symbol = symbol.replace("-", "/") if is_crypto else symbol  # pyright: ignore

        # 1. Determine Current State
        # Position symbols in Alpaca for options match the OCC format (e.g., SPY260115C00640000)
        pos = positions.get(symbol.replace("/", "").replace("-", ""))  # pyright: ignore

        price = float(pos.current_price) if pos else float(row["ask"])  # type: ignore
        current_qty = float(pos.qty) if pos else 0.0  # type: ignore

        # 2. Calculate Target Quantity
        # Note: For options, the price is per share. 1 contract = 100 shares.
        # target_qty here is the number of CONTRACTS.
        multiplier = 100.0 if is_option else 1.0  # pyright: ignore
        target_qty = row["target_usd"] / (price * multiplier)

        if row["type"] in ["spot_short", "put_short", "call_short"]:
            if is_crypto:
                target_qty = 0.0
            else:
                target_qty = -target_qty

        # 3. Decision Logic (USD Delta)
        current_usd_value = current_qty * price * multiplier
        if row["type"] == "spot_short" and is_crypto:
            target_usd = 0.0  # Force liquidation for crypto shorts
        else:
            target_usd = row["target_usd"]

        # Calculate the actual change needed
        diff_usd = target_usd - current_usd_value

        if abs(diff_usd) < MIN_TRADE_USD:
            update_exits(
                trade_symbol, row["tp_target"], row["sl_target"], trading_client
            )
            continue

        # 4. Execute
        clear_orders(trade_symbol, trading_client)
        side = OrderSide.BUY if diff_usd > 0 else OrderSide.SELL

        if is_crypto:
            execute_crypto_strategy(
                symbol=trade_symbol,  # e.g., 'BTC/USD'
                trade_notional=abs(diff_usd),  # The dollar amount to buy/sell now
                total_target_usd=row[
                    "target_usd"
                ],  # The total dollar value we want to guard
                side=side,  # OrderSide.BUY or OrderSide.SELL
                tp=row["tp_target"],  # Take profit price
                sl=row["sl_target"],  # Stop loss price
                trading_client=trading_client,  # The active TradingClient instance
            )
        elif is_option:  # pyright: ignore
            execute_option_strategy(
                trade_symbol,
                abs(round(target_qty, 0)),
                side,
                row["tp_target"],
                row["sl_target"],
                trading_client,
            )
        else:
            execute_equity_strategy(
                trade_symbol,
                abs(round(target_qty, 0)),
                side,
                row["tp_target"],
                row["sl_target"],
                trading_client,
            )


def clear_orders(symbol, trading_client):
    """Cancels all open orders for a symbol to avoid conflicts."""
    open_orders = trading_client.get_orders(
        GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
    )
    for order in open_orders:
        trading_client.cancel_order_by_id(order.id)


def execute_crypto_strategy(
    symbol, trade_notional, total_target_usd, side, tp, sl, trading_client
):
    """Handles crypto using Notional values to satisfy price collars."""
    try:
        # Use Notional for the entry to let Alpaca handle the collar/buffer
        trading_client.submit_order(
            MarketOrderRequest(
                symbol=symbol,
                notional=round(trade_notional, 2),
                side=side,
                time_in_force=TimeInForce.GTC,
            )
        )

        if total_target_usd <= 0:
            print(f"[{symbol}] Position liquidated. No exits set.")
            return

        time.sleep(2.0)  # Brief pause for order to fill and position to update

        # Get new position to set accurate TP/SL quantities
        new_pos = trading_client.get_open_position(symbol)
        abs_qty = abs(float(new_pos.qty))
        exit_side = OrderSide.SELL if float(new_pos.qty) > 0 else OrderSide.BUY

        trading_client.submit_order(
            LimitOrderRequest(
                symbol=symbol,
                qty=abs_qty,
                side=exit_side,
                limit_price=round(tp, 2),
                time_in_force=TimeInForce.GTC,
            )
        )
        trading_client.submit_order(
            StopOrderRequest(
                symbol=symbol,
                qty=abs_qty,
                side=exit_side,
                stop_price=round(sl, 2),
                time_in_force=TimeInForce.GTC,
            )
        )
    except Exception as e:
        print(f"Crypto Strategy failed for {symbol}: {e}")


def execute_equity_strategy(symbol, qty, side, tp, sl, trading_client):
    """Uses Bracket Orders for Equities."""
    try:
        # Validation for Alpaca Bracket rules: Buy TP > SL, Sell TP < SL
        if (side == OrderSide.BUY and tp <= sl) or (
            side == OrderSide.SELL and tp >= sl
        ):
            print(f"[{symbol}] TP/SL validation failed. Skipping bracket.")
            return

        order_req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.GTC,
            order_class=OrderClass.BRACKET,
            take_profit=TakeProfitRequest(limit_price=round(tp, 2)),
            stop_loss=StopLossRequest(stop_price=round(sl, 2)),
        )
        trading_client.submit_order(order_req)
    except Exception as e:
        print(f"Equity Trade failed: {e}")


def update_exits(symbol, model_tp, model_sl, trading_client):
    """Replaces open exit orders with refined logic for options and threshold sensitivity."""
    # Determine sensitivity: 0.01 for options/low-price, 0.5 for others
    is_option = len(symbol) > 12
    threshold = 0.01 if is_option else 0.25

    open_orders = trading_client.get_orders(
        GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[symbol])
    )

    for order in open_orders:
        try:
            # 1. Update Take Profit (Limit Orders)
            if order.type == OrderType.LIMIT and model_tp > 0:
                if abs(float(order.limit_price) - model_tp) > threshold:
                    print(f"[{symbol}] Updating TP to {model_tp}")
                    trading_client.replace_order_by_id(
                        order.id, ReplaceOrderRequest(limit_price=round(model_tp, 2))
                    )

            # 2. Update Stop Loss (Stop or Stop-Limit Orders)
            elif order.type in [OrderType.STOP, OrderType.STOP_LIMIT] and model_sl > 0:
                # ReplaceOrderRequest uses 'stop_price' for both Stop and Stop-Limit types
                if abs(float(order.stop_price) - model_sl) > threshold:
                    print(f"[{symbol}] Updating SL to {model_sl}")
                    trading_client.replace_order_by_id(
                        order.id, ReplaceOrderRequest(stop_price=round(model_sl, 2))
                    )

            # 3. Handle 'Canceled' Signal
            elif model_tp == 0 or model_sl == 0:
                print(f"[{symbol}] Model target is 0. Canceling order {order.id}")
                trading_client.cancel_order_by_id(order.id)

        except Exception as e:
            # Common error: order is already 'pending_replace' or 'filled'
            print(f"Update failed for {symbol} ({order.type}): {e}")


def execute_option_strategy(symbol, qty, side, tp, sl, trading_client):
    """Executes sequential orders for Options with market-hours error suppression."""
    print(f"[{symbol}] Executing Option Sequential Orders (TIF: DAY)...")
    try:
        # Step 1: Market Order (Must be DAY)
        trading_client.submit_order(
            MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.DAY,
            )
        )

        # Step 2: Brief pause to allow for fill
        time.sleep(2.0)

        # Step 3: Set independent TP/SL based on the NEW total position
        new_pos = trading_client.get_open_position(symbol)
        abs_qty = abs(float(new_pos.qty))
        exit_side = OrderSide.SELL if float(new_pos.qty) > 0 else OrderSide.BUY

        # Take Profit (Limit Order)
        trading_client.submit_order(
            LimitOrderRequest(
                symbol=symbol,
                qty=abs_qty,
                side=exit_side,
                limit_price=round(tp, 2),
                time_in_force=TimeInForce.DAY,
            )
        )

        # Stop Loss (Stop Order)
        trading_client.submit_order(
            StopOrderRequest(
                symbol=symbol,
                qty=abs_qty,
                side=exit_side,
                stop_price=round(sl, 2),
                time_in_force=TimeInForce.DAY,
            )
        )

    except APIError as e:
        # Suppress the "market hours" error specifically
        if e.code == 42210000 and "market hours" in str(e).lower():
            # Silent return or a simple non-error print
            print(f"[{symbol}] Trade skipped: Options market is closed.")
            return
        # Print other API-related errors normally
        print(f"[{symbol}] Alpaca API Error: {e}")

    except Exception as e:
        # Catch unexpected non-API errors (e.g. connection issues)
        print(f"[{symbol}] Unexpected execution error: {e}")
