# Resolved Findings: Missing API Error Handling in Alpaca Executor (Validation Tests)
"""Unit tests for AlpacaExecutor."""
import pytest
from unittest.mock import MagicMock, patch
from src.execution.alpaca_executor import AlpacaExecutor


class TestAlpacaExecutor:
    def setup_method(self):
        # We mock REST client during initialization
        with patch("src.execution.alpaca_executor.REST") as mock_rest:
            self.executor = AlpacaExecutor(key_id="test_key", secret_key="test_secret", paper=True)
            self.mock_api = mock_rest.return_value
            self.executor.api = self.mock_api

    def test_paper_trading_required(self):
        with pytest.raises(ValueError):
            AlpacaExecutor(key_id="test_key", secret_key="test_secret", paper=False)

    def test_get_account_equity_success(self):
        mock_account = MagicMock()
        mock_account.equity = "100000.00"
        self.mock_api.get_account.return_value = mock_account
        
        equity = self.executor.get_account_equity()
        assert equity == 100000.0
        self.mock_api.get_account.assert_called_once()

    def test_get_account_equity_failure(self):
        self.mock_api.get_account.side_effect = Exception("API Error")
        equity = self.executor.get_account_equity()
        assert equity == 0.0

    def test_get_live_positions_success(self):
        pos1 = MagicMock()
        pos1.symbol = "AAPL"
        pos1.qty = "10.0"
        pos2 = MagicMock()
        pos2.symbol = "MSFT"
        pos2.qty = "-5.0"
        self.mock_api.list_positions.return_value = [pos1, pos2]
        
        positions = self.executor.get_live_positions()
        assert positions == {"AAPL": 10.0, "MSFT": -5.0}

    def test_get_live_positions_failure(self):
        self.mock_api.list_positions.side_effect = Exception("Connection Timeout")
        positions = self.executor.get_live_positions()
        assert positions is None

    def test_get_latest_prices_success(self):
        trade1 = MagicMock()
        trade1.price = 150.0
        trade2 = MagicMock()
        trade2.price = 250.0
        self.mock_api.get_latest_trades.return_value = {"AAPL": trade1, "MSFT": trade2}
        
        prices = self.executor._get_latest_prices(["AAPL", "MSFT"])
        assert prices == {"AAPL": 150.0, "MSFT": 250.0}

    def test_get_latest_prices_failure(self):
        self.mock_api.get_latest_trades.side_effect = Exception("Rate Limit Exceeded")
        prices = self.executor._get_latest_prices(["AAPL"])
        assert prices is None

    def test_execute_signals_aborts_on_zero_equity(self):
        mock_account = MagicMock()
        mock_account.equity = "0.00"
        self.mock_api.get_account.return_value = mock_account
        
        self.executor.execute_signals({"AAPL": 0.5})
        self.mock_api.list_positions.assert_not_called()

    def test_execute_signals_aborts_on_positions_failure(self):
        mock_account = MagicMock()
        mock_account.equity = "100000.00"
        self.mock_api.get_account.return_value = mock_account
        self.mock_api.list_positions.side_effect = Exception("API offline")
        
        self.executor.execute_signals({"AAPL": 0.5})
        self.mock_api.get_latest_trades.assert_not_called()

    def test_execute_signals_aborts_on_prices_failure(self):
        mock_account = MagicMock()
        mock_account.equity = "100000.00"
        self.mock_api.get_account.return_value = mock_account
        self.mock_api.list_positions.return_value = []
        self.mock_api.get_latest_trades.side_effect = Exception("Connection lost")
        
        self.executor.execute_signals({"AAPL": 0.5})
        self.mock_api.submit_order.assert_not_called()

    def test_execute_signals_places_orders(self):
        mock_account = MagicMock()
        mock_account.equity = "10000.00"
        self.mock_api.get_account.return_value = mock_account
        
        pos1 = MagicMock()
        pos1.symbol = "AAPL"
        pos1.qty = "10.0"
        self.mock_api.list_positions.return_value = [pos1]
        
        trade1 = MagicMock()
        trade1.price = 100.0
        self.mock_api.get_latest_trades.return_value = {"AAPL": trade1}
        
        # Target weight 0.2 means allocation = 2000.0. At price 100.0, target is 20 shares.
        # Current AAPL is 10 shares. Diff = +10. Market order buy 10.
        self.executor.execute_signals({"AAPL": 0.2})
        self.mock_api.submit_order.assert_called_once_with(
            symbol="AAPL",
            qty=10,
            side="buy",
            type="market",
            time_in_force="day"
        )
