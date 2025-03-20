"""
Main entry point for the AI-powered trading bot.

This script initializes all components and runs the trading bot.
"""

import asyncio
import logging
import signal
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

from config.settings import (
    API_KEY, API_SECRET, TRADING_PAIRS, METRICS_INTERVAL,
    ENVIRONMENT, CONFIDENCE_THRESHOLD
)
from config.logging_config import setup_logging
from data.websocket_client import WebSocketClient
from data.rest_client import RESTClient
from data.data_processor import DataProcessor
from models.feature_engineering import FeatureEngineer
from models.ml_signals import MLSignalGenerator
from models.rl_optimizer import RLTradeOptimizer
from execution.order_manager import OrderManager
from execution.risk_manager import RiskManager
from execution.position_sizer import PositionSizer
from monitoring.metrics import MetricsTracker
from monitoring.alerts import AlertManager, AlertRule, AlertLevel, AlertType, LogAlertHandler
from utils.helpers import setup_signal_handlers

# Global variables for graceful shutdown
shutdown_requested = False
active_tasks = set()

async def process_market_data(
    symbol: str,
    data: Dict,
    data_processor: DataProcessor,
    feature_engineer: FeatureEngineer,
    signal_generator: MLSignalGenerator,
    trade_optimizer: RLTradeOptimizer,
    risk_manager: RiskManager,
    position_sizer: PositionSizer,
    order_manager: OrderManager,
    metrics_tracker: MetricsTracker,
    alert_manager: AlertManager
) -> None:
    """
    Process market data and execute trading logic.
    
    Args:
        symbol: Trading symbol
        data: Market data
        data_processor: Data processor
        feature_engineer: Feature engineer
        signal_generator: ML signal generator
        trade_optimizer: RL trade optimizer
        risk_manager: Risk manager
        position_sizer: Position sizer
        order_manager: Order manager
        metrics_tracker: Metrics tracker
        alert_manager: Alert manager
    """
    try:
        # Process data
        if "ticker" in data.get("channel", ""):
            # Process ticker data
            ticker_df = data_processor.process_ticker(data)
            if ticker_df is None:
                return
            
            # Get account information
            account = await order_manager.rest_client.get_account()
            balances = account.get("balances", {})
            
            # Get current balance and position
            base_currency = symbol.split('/')[0]
            quote_currency = symbol.split('/')[1]
            
            base_balance = float(balances.get(base_currency, {}).get("free", 0))
            quote_balance = float(balances.get(quote_currency, {}).get("free", 0))
            
            # Calculate current position as a fraction of total portfolio value
            current_price = ticker_df.iloc[0]["price"]
            portfolio_value = quote_balance + (base_balance * current_price)
            current_position = (base_balance * current_price) / portfolio_value if portfolio_value > 0 else 0
            
            # Update metrics
            metrics_tracker.add_portfolio_value(portfolio_value)
            
            # Calculate features
            historical_data = data_processor.get_klines(symbol, "1h", limit=100)
            if historical_data.empty:
                return
            
            technical_df = feature_engineer.calculate_technical_indicators(historical_data)
            
            # Get orderbook data
            orderbook = await order_manager.rest_client.get_orderbook(symbol)
            bids_df, asks_df = data_processor.process_orderbook({
                "symbol": symbol,
                "timestamp": int(time.time() * 1000),
                "bids": orderbook.get("bids", []),
                "asks": orderbook.get("asks", [])
            })
            
            # Get recent trades
            trades = await order_manager.rest_client.get_trades(symbol, limit=50)
            trades_df = data_processor.process_trades({
                "symbol": symbol,
                "trades": trades
            })
            
            # Calculate all features
            orderbook_features = feature_engineer.calculate_orderbook_features(bids_df, asks_df)
            trade_features = feature_engineer.calculate_trade_features(trades_df)
            combined_features = feature_engineer.combine_features(technical_df, orderbook_features, trade_features)
            
            # Generate trading signal
            signal = signal_generator.predict(symbol, combined_features)
            
            # Check confidence threshold
            if signal.get("confidence", 0) < CONFIDENCE_THRESHOLD:
                return
            
            # Update risk metrics
            risk_manager.update_volatility(symbol, historical_data)
            
            # Optimize trade
            decision = trade_optimizer.optimize_trade(
                symbol=symbol,
                features=combined_features,
                current_position=current_position,
                balance=portfolio_value
            )
            
            # Validate trade with risk management
            validation = risk_manager.validate_trade(
                symbol=symbol,
                signal=signal,
                current_position=current_position,
                target_position=decision.get("target_position", current_position)
            )
            
            # Execute trade if valid
            if validation["valid"]:
                # Calculate position size
                position_size = position_sizer.calculate_position_size(
                    symbol=symbol,
                    signal=signal,
                    balance=portfolio_value,
                    price=current_price
                )
                
                # Execute trade
                execution_result = await order_manager.execute_trade_decision(
                    symbol=symbol,
                    decision={
                        "action": decision.get("action", "HOLD"),
                        "target_position": validation["adjusted_position"],
                        "confidence": signal.get("confidence", 0)
                    },
                    current_balance=portfolio_value,
                    current_position=current_position
                )
                
                # Track trade
                if execution_result.get("status") == "SUCCESS":
                    metrics_tracker.add_trade(execution_result.get("order", {}))
                    
                    # Create alert for trade
                    alert_context = {
                        "symbol": symbol,
                        "action": decision.get("action", "HOLD"),
                        "price": current_price,
                        "position_value": position_size.get("position_value", 0),
                        "confidence": signal.get("confidence", 0)
                    }
                    
                    alert_manager.check_rules(alert_context)
            
            # Check for alerts
            alert_context = {
                "symbol": symbol,
                "current_price": current_price,
                "portfolio_value": portfolio_value,
                "current_position": current_position,
                "daily_volatility": risk_manager.volatility.get(symbol, 0) * 100,
                "current_drawdown": risk_manager.drawdowns.get(symbol, 0) * 100,
                "signal": signal.get("signal", "NEUTRAL"),
                "confidence": signal.get("confidence", 0)
            }
            
            alert_manager.check_rules(alert_context)
            
        elif "orderbook" in data.get("channel", ""):
            # Process orderbook data
            data_processor.process_orderbook(data)
            
        elif "trades" in data.get("channel", ""):
            # Process trades data
            data_processor.process_trades(data)
            
    except Exception as e:
        logging.error(f"Error processing market data for {symbol}: {e}")

async def update_metrics(
    metrics_tracker: MetricsTracker,
    alert_manager: AlertManager
) -> None:
    """
    Periodically update metrics and check for alerts.
    
    Args:
        metrics_tracker: Metrics tracker
        alert_manager: Alert manager
    """
    while not shutdown_requested:
        try:
            # Update metrics
            metrics = metrics_tracker.update_metrics()
            
            # Check for alerts
            alert_manager.check_rules(metrics)
            
            # Log summary
            logging.info(f"Portfolio value: {metrics.get('portfolio_value', 0):.2f}, "
                        f"Return: {metrics.get('total_return', 0):.2f}%, "
                        f"Drawdown: {metrics.get('current_drawdown', 0):.2f}%")
            
        except Exception as e:
            logging.error(f"Error updating metrics: {e}")
        
        # Wait for next update
        await asyncio.sleep(METRICS_INTERVAL)

async def main():
    """Main function to run the trading bot."""
    global shutdown_requested
    
    # Setup logging
    setup_logging()
    
    # Setup signal handlers
    setup_signal_handlers(lambda: setattr(sys.modules[__name__], 'shutdown_requested', True))
    
    logging.info(f"Starting trading bot in {ENVIRONMENT} environment")
    
    try:
        # Initialize components
        rest_client = RESTClient()
        ws_client = WebSocketClient()
        data_processor = DataProcessor()
        feature_engineer = FeatureEngineer()
        signal_generator = MLSignalGenerator()
        trade_optimizer = RLTradeOptimizer()
        order_manager = OrderManager(rest_client)
        risk_manager = RiskManager()
        position_sizer = PositionSizer()
        metrics_tracker = MetricsTracker()
        alert_manager = AlertManager()
        
        # Add alert handlers
        alert_manager.add_handler(LogAlertHandler())
        
        # Add alert rules
        add_alert_rules(alert_manager)
        
        # Connect to WebSocket
        await ws_client.connect()
        
        # Define callback for market data
        async def on_market_data(data):
            if shutdown_requested:
                return
            
            symbol = data.get("symbol")
            if not symbol:
                channel_parts = data.get("channel", "").split(":")
                if len(channel_parts) > 1:
                    symbol = channel_parts[1]
            
            if symbol:
                task = asyncio.create_task(
                    process_market_data(
                        symbol=symbol,
                        data=data,
                        data_processor=data_processor,
                        feature_engineer=feature_engineer,
                        signal_generator=signal_generator,
                        trade_optimizer=trade_optimizer,
                        risk_manager=risk_manager,
                        position_sizer=position_sizer,
                        order_manager=order_manager,
                        metrics_tracker=metrics_tracker,
                        alert_manager=alert_manager
                    )
                )
                active_tasks.add(task)
                task.add_done_callback(active_tasks.discard)
        
        # Subscribe to channels for each trading pair
        for symbol in TRADING_PAIRS:
            await ws_client.subscribe("ticker", [symbol], on_market_data)
            await ws_client.subscribe("orderbook", [symbol], on_market_data)
            await ws_client.subscribe("trades", [symbol], on_market_data)
            
            # Load initial data
            klines = await rest_client.get_klines(symbol, "1h", limit=100)
            data_processor.process_klines({
                "symbol": symbol,
                "interval": "1h",
                "klines": klines
            })
            
            # Load models
            signal_generator.load_model(symbol)
            trade_optimizer.load_model(symbol)
        
        # Start metrics update task
        metrics_task = asyncio.create_task(update_metrics(metrics_tracker, alert_manager))
        active_tasks.add(metrics_task)
        
        # Keep the bot running
        while not shutdown_requested:
            await asyncio.sleep(1)
        
    except Exception as e:
        logging.error(f"Error in main loop: {e}")
    
    finally:
        # Cleanup
        logging.info("Shutting down trading bot")
        
        # Cancel all orders
        try:
            await order_manager.cancel_all_orders()
        except Exception as e:
            logging.error(f"Error canceling orders: {e}")
        
        # Close connections
        try:
            await ws_client.disconnect()
            await rest_client.close()
            data_processor.close()
        except Exception as e:
            logging.error(f"Error closing connections: {e}")
        
        # Wait for active tasks to complete
        if active_tasks:
            logging.info(f"Waiting for {len(active_tasks)} tasks to complete")
            await asyncio.gather(*active_tasks, return_exceptions=True)
        
        logging.info("Trading bot shutdown complete")

def add_alert_rules(alert_manager: AlertManager) -> None:
    """
    Add alert rules to the alert manager.
    
    Args:
        alert_manager: Alert manager
    """
    # High drawdown alert
    def check_high_drawdown(context):
        return context.get("current_drawdown", 0) > 5.0
    
    alert_manager.add_rule(
        AlertRule(
            name="high_drawdown",
            condition=check_high_drawdown,
            message="High drawdown detected: {current_drawdown:.2f}%",
            level=AlertLevel.WARNING,
            alert_type=AlertType.PERFORMANCE,
            cooldown_seconds=600
        )
    )
    
    # High volatility alert
    def check_high_volatility(context):
        return context.get("daily_volatility", 0) > 20.0
    
    alert_manager.add_rule(
        AlertRule(
            name="high_volatility",
            condition=check_high_volatility,
            message="High volatility detected: {daily_volatility:.2f}%",
            level=AlertLevel.WARNING,
            alert_type=AlertType.MARKET,
            cooldown_seconds=3600
        )
    )
    
    # Large trade alert
    def check_large_trade(context):
        return context.get("position_value", 0) > 1000.0
    
    alert_manager.add_rule(
        AlertRule(
            name="large_trade",
            condition=check_large_trade,
            message="Large trade executed: {symbol} {action} {position_value:.2f}",
            level=AlertLevel.INFO,
            alert_type=AlertType.TRADE,
            cooldown_seconds=60
        )
    )
    
    # High confidence signal alert
    def check_high_confidence(context):
        return context.get("confidence", 0) > 0.9
    
    alert_manager.add_rule(
        AlertRule(
            name="high_confidence",
            condition=check_high_confidence,
            message="High confidence signal: {symbol} {signal} {confidence:.2f}",
            level=AlertLevel.INFO,
            alert_type=AlertType.TRADE,
            cooldown_seconds=300
        )
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Bot stopped by user")
    except Exception as e:
        logging.critical(f"Unhandled exception: {e}", exc_info=True)
        sys.exit(1)

