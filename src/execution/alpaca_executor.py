"""
Paper Trading Execution Layer using Alpaca API.

Connects signals to orders, implements order sizing, retry logic, 
and tracks live vs expected positions. Strict enforcement of paper trading.
"""

import time
import logging
from typing import Dict
from alpaca_trade_api.rest import REST, TimeFrame
from alpaca_trade_api.common import URL

logger = logging.getLogger(__name__)

class AlpacaExecutor:
    def __init__(self, key_id: str, secret_key: str, paper: bool = True):
        """
        Initialise the Alpaca Executor.
        
        Args:
            key_id: Alpaca API Key ID
            secret_key: Alpaca Secret Key
            paper: Must be True. Real capital is NOT allowed yet.
        """
        if not paper:
            raise ValueError(
                "CRITICAL WARNING: Real capital execution is currently disabled for safety. "
                "Please use paper trading only (paper=True)."
            )
            
        self.base_url = "https://paper-api.alpaca.markets"
        
        self.api = REST(
            key_id=key_id,
            secret_key=secret_key,
            base_url=URL(self.base_url),
            api_version='v2'
        )
        
        logger.info("AlpacaExecutor initialised securely in PAPER TRADING mode.")

    def get_account_equity(self) -> float:
        """Returns the current account equity in USD."""
        account = self.api.get_account()
        return float(account.equity)

    def get_live_positions(self) -> Dict[str, float]:
        """Returns a dict of {symbol: qty} for all currently held positions."""
        positions = self.api.list_positions()
        return {p.symbol: float(p.qty) for p in positions}

    def _get_latest_prices(self, symbols: list[str]) -> Dict[str, float]:
        """Fetch the latest trade price for a list of symbols."""
        if not symbols:
            return {}
        
        # We can use get_latest_trades for a list of symbols
        trades = self.api.get_latest_trades(symbols)
        return {sym: trade.price for sym, trade in trades.items()}

    def execute_signals(self, target_weights: Dict[str, float], retry_count: int = 3):
        """
        Takes target portfolio weights, calculates shares, and submits orders.
        Includes retry logic and order sizing.
        """
        equity = self.get_account_equity()
        live_positions = self.get_live_positions()
        
        # Gather all symbols 
        symbols_to_price = list(set(target_weights.keys()).union(set(live_positions.keys())))
        
        if not symbols_to_price:
            logger.info("No target weights and no open positions. Nothing to execute.")
            return

        latest_prices = self._get_latest_prices(symbols_to_price)
        
        target_shares = {}
        for symbol, weight in target_weights.items():
            if weight == 0:
                target_shares[symbol] = 0
                continue
                
            if symbol not in latest_prices:
                logger.warning(f"Could not fetch price for {symbol}. Skipping.")
                continue
                
            dollar_allocation = equity * weight
            price = latest_prices[symbol]
            shares = int(dollar_allocation // price)  # simple floor integer shares
            target_shares[symbol] = shares

        # Missing symbols from target should be 0 
        for symbol in live_positions.keys():
            if symbol not in target_shares:
                target_shares[symbol] = 0
                
        # Compare and issue orders
        for symbol, target_qty in target_shares.items():
            current_qty = live_positions.get(symbol, 0.0)
            diff_qty = target_qty - current_qty

            if abs(diff_qty) < 1.0:
                # No change needed if diff is < 1 share (or zero)
                continue
                
            side = 'buy' if diff_qty > 0 else 'sell'
            qty = abs(int(diff_qty))
            
            self._place_order_with_retry(
                symbol=symbol, 
                qty=qty, 
                side=side, 
                retry_count=retry_count
            )

    def _place_order_with_retry(self, symbol: str, qty: int, side: str, retry_count: int):
        """Places a market order with exponential backoff retry logic."""
        for attempt in range(retry_count):
            try:
                logger.info(f"Placing order: {side.upper()} {qty} shares of {symbol} (Attempt {attempt+1})")
                self.api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    type='market',
                    time_in_force='day'
                )
                logger.info(f"Order for {symbol} successful.")
                break
            except Exception as e:
                logger.warning(f"Failed to place {side} {symbol} order: {e}")
                if attempt == retry_count - 1:
                    logger.error(f"Max retries reached for {symbol}. Order failed permanently.")
                else:
                    backoff = 2 ** attempt
                    logger.info(f"Retrying in {backoff} seconds...")
                    time.sleep(backoff)
