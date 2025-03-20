import asyncio
import hashlib
import hmac
import json
import logging
import math
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import requests
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
LSTM = keras.layers.LSTM
Dense = keras.layers.Dense
Dropout = keras.layers.Dropout
Sequential = keras.models.Sequential
load_model = keras.models.load_model
Adam = keras.optimizers.Adam


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("RoostooTradingBot")

# Constants
API_URL = "https://mock-api.roostoo.com"
API_KEY = "USEAPIKEYASMYID" 
SECRET_KEY = "S1XP1e3UZj6A7H5fATj0jNhqPxxdSJYdInClVN65XAbvqqMKjVHjA7PZj4W12oep"  

# Configuration
class Config:
    # Trading parameters
    TRADING_PAIRS = ["BTC/USD", "ETH/USD", "BNB/USD"]  # Trading pairs to monitor
    POSITION_SIZE_PCT = 0.02  # Position size as percentage of portfolio (2%)
    MAX_POSITION_SIZE_PCT = 0.20  # Maximum allocation per asset (20%)
    STOP_LOSS_PCT = 0.02  # Stop loss percentage (2%)
    CONFIDENCE_THRESHOLD = 0.70  # Minimum confidence for trade execution (70%)
    VOLATILITY_THRESHOLD = 0.15  # Maximum volatility for trade execution (15%)
    
    # Technical indicators parameters
    RSI_PERIOD = 14
    BOLLINGER_PERIOD = 20
    BOLLINGER_STD = 2
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    
    # ML model parameters
    SEQUENCE_LENGTH = 60  # Number of time steps to look back
    PREDICTION_HORIZON = 3  # Predict price movement for next 3 time steps
    FEATURE_COLUMNS = ['close', 'volume', 'rsi', 'upper_band', 'lower_band', 'macd', 'macd_signal']
    MODEL_PATH = "lstm_model.h5"
    SCALER_PATH = "scaler.pkl"
    
    # Backtesting parameters
    BACKTEST_DAYS = 30
    
    # API request parameters
    REQUEST_TIMEOUT = 10  # seconds
    MAX_RETRIES = 3
    RETRY_DELAY = 1  # seconds

# Main Trading Bot Class
class RoostooTradingBot:
    def __init__(self, api_key: str, secret_key: str, config: Config = None):
        self.api_key = api_key
        self.secret_key = secret_key
        self.config = config or Config()
        self.data_cache = {}  # Cache for market data
        self.models = {}  # LSTM models for each trading pair
        self.scalers = {}  # Scalers for data normalization
        self.portfolio = {}  # Current portfolio state
        self.orders = {}  # Active orders
        self.trade_history = []  # Trade history for performance tracking
        self.server_time_offset = 0  # Time difference between local and server
        
        # Initialize portfolio
        self._initialize()
    
    def _initialize(self):
        """Initialize the trading bot by syncing time and loading models"""
        logger.info("Initializing trading bot...")
        
        # Sync time with server
        self._sync_time()
        
        # Load or train ML models
        for pair in self.config.TRADING_PAIRS:
            self._load_or_train_model(pair)
        
        # Get initial portfolio state
        self._update_portfolio()
        
        logger.info("Trading bot initialized successfully")
    
    def _sync_time(self):
        """Synchronize local time with server time"""
        try:
            response = self._make_request("GET", "/v3/serverTime")
            server_time = response.get("ServerTime", 0)
            local_time = int(time.time() * 1000)
            self.server_time_offset = server_time - local_time
            logger.info(f"Time synchronized. Offset: {self.server_time_offset}ms")
        except Exception as e:
            logger.error(f"Failed to sync time: {e}")
    
    def _get_server_time(self) -> int:
        """Get current server time in milliseconds"""
        return int(time.time() * 1000) + self.server_time_offset
    
    def _sign_request(self, params: Dict) -> str:
        """Generate HMAC SHA256 signature for API request"""
        # Sort parameters and create query string
        query_string = '&'.join([f"{k}={v}" for k, v in sorted(params.items())])
        
        # Create signature using HMAC SHA256
        signature = hmac.new(
            self.secret_key.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    def _make_request(self, method: str, endpoint: str, params: Dict = None, 
                      auth_required: bool = False, retries: int = 0) -> Dict:
        """Make HTTP request to Roostoo API with retry logic"""
        url = f"{API_URL}{endpoint}"
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        params = params or {}
        
        if auth_required:
            # Add timestamp for authenticated requests
            params['timestamp'] = str(self._get_server_time())
            
            # Add authentication headers
            signature = self._sign_request(params)
            headers['RST-API-KEY'] = self.api_key
            headers['MSG-SIGNATURE'] = signature
        
        try:
            if method == "GET":
                response = requests.get(
                    url, 
                    params=params, 
                    headers=headers, 
                    timeout=self.config.REQUEST_TIMEOUT
                )
            else:  # POST
                response = requests.post(
                    url, 
                    data=params, 
                    headers=headers, 
                    timeout=self.config.REQUEST_TIMEOUT
                )
            
            # Check if request was successful
            response.raise_for_status()
            return response.json()
            
        except (requests.RequestException, json.JSONDecodeError) as e:
            if retries < self.config.MAX_RETRIES:
                logger.warning(f"Request failed: {e}. Retrying ({retries+1}/{self.config.MAX_RETRIES})...")
                time.sleep(self.config.RETRY_DELAY)
                return self._make_request(method, endpoint, params, auth_required, retries + 1)
            else:
                logger.error(f"Request failed after {self.config.MAX_RETRIES} retries: {e}")
                raise
    
    def _update_portfolio(self):
        """Update current portfolio state"""
        try:
            response = self._make_request("GET", "/v3/balance", auth_required=True)
            if response.get("Success"):
                self.portfolio = response.get("Wallet", {})
                logger.info(f"Portfolio updated: {self.portfolio}")
            else:
                logger.error(f"Failed to update portfolio: {response.get('ErrMsg')}")
        except Exception as e:
            logger.error(f"Error updating portfolio: {e}")
            # Initialize with default portfolio for testing if API fails
            self.portfolio = {
                "USD": {"Free": 50000, "Lock": 0},
                "BTC": {"Free": 0, "Lock": 0},
                "ETH": {"Free": 0, "Lock": 0},
                "BNB": {"Free": 0, "Lock": 0}
            }
            logger.info("Using default portfolio for testing")
    
    def _fetch_market_data(self, pair: str, lookback_periods: int = None) -> pd.DataFrame:
        """Fetch market data for a trading pair"""
        try:
            # For a real implementation, you would fetch historical data
            # Since Roostoo API doesn't provide historical data directly,
            # we'll use the current ticker data and simulate historical data
            # In a real scenario, you would store this data over time
            
            params = {
                'pair': pair,
                'timestamp': str(self._get_server_time())
            }
            
            response = self._make_request("GET", "/v3/ticker", params)
            
            if not response.get("Success"):
                logger.error(f"Failed to fetch market data: {response.get('ErrMsg')}")
                # Generate synthetic data for testing
                current_price = 10000 if pair.startswith("BTC") else (2000 if pair.startswith("ETH") else 300)
                logger.info(f"Using synthetic data for {pair} with price {current_price}")
            else:
                ticker_data = response.get("Data", {}).get(pair, {})
                current_price = ticker_data.get("LastPrice", 0)
            
            # Generate synthetic data for demonstration
            # In a real implementation, you would use actual historical data
            lookback = lookback_periods or (self.config.SEQUENCE_LENGTH + 50)
            
            # Create synthetic price data with some randomness
            np.random.seed(42)  # For reproducibility
            price_changes = np.random.normal(0, 0.01, lookback)
            prices = [current_price]
            
            for change in price_changes:
                prices.append(prices[-1] * (1 + change))
            
            prices = prices[1:]  # Remove the initial price
            
            # Create synthetic volume data
            volumes = np.random.normal(1000, 500, lookback)
            volumes = np.abs(volumes)  # Ensure volumes are positive
            
            # Create DataFrame
            dates = pd.date_range(end=pd.Timestamp.now(), periods=lookback, freq='15min')
            df = pd.DataFrame({
                'timestamp': dates,
                'open': prices,
                'high': [p * (1 + np.random.uniform(0, 0.005)) for p in prices],
                'low': [p * (1 - np.random.uniform(0, 0.005)) for p in prices],
                'close': prices,
                'volume': volumes
            })
            
            # Add technical indicators
            df = self._add_technical_indicators(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return None
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the DataFrame"""
        # Calculate RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=self.config.RSI_PERIOD).mean()
        avg_loss = loss.rolling(window=self.config.RSI_PERIOD).mean()
        
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Calculate Bollinger Bands
        df['sma'] = df['close'].rolling(window=self.config.BOLLINGER_PERIOD).mean()
        df['std'] = df['close'].rolling(window=self.config.BOLLINGER_PERIOD).std()
        df['upper_band'] = df['sma'] + (df['std'] * self.config.BOLLINGER_STD)
        df['lower_band'] = df['sma'] - (df['std'] * self.config.BOLLINGER_STD)
        
        # Calculate MACD
        ema_fast = df['close'].ewm(span=self.config.MACD_FAST).mean()
        ema_slow = df['close'].ewm(span=self.config.MACD_SLOW).mean()
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=self.config.MACD_SIGNAL).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Calculate volatility (20-day standard deviation of returns)
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(window=20).std() * math.sqrt(365)
        
        # Drop NaN values
        df.dropna(inplace=True)
        
        return df
    
    def _prepare_model_data(self, df: pd.DataFrame, pair: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for LSTM model"""
        # Select features
        data = df[self.config.FEATURE_COLUMNS].copy()
        
        # Scale data
        if pair not in self.scalers:
            self.scalers[pair] = MinMaxScaler(feature_range=(0, 1))
            scaled_data = self.scalers[pair].fit_transform(data)
        else:
            scaled_data = self.scalers[pair].transform(data)
        
        # Create sequences for LSTM
        X, y = [], []
        for i in range(len(scaled_data) - self.config.SEQUENCE_LENGTH - self.config.PREDICTION_HORIZON):
            X.append(scaled_data[i:i + self.config.SEQUENCE_LENGTH])
            # Target is the price direction (1 for up, 0 for down)
            future_price = df['close'].iloc[i + self.config.SEQUENCE_LENGTH + self.config.PREDICTION_HORIZON]
            current_price = df['close'].iloc[i + self.config.SEQUENCE_LENGTH]
            y.append(1 if future_price > current_price else 0)
        
        return np.array(X), np.array(y)
    
    def _create_lstm_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """Create LSTM model for price prediction"""
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def _load_or_train_model(self, pair: str):
        """Load existing model or train a new one"""
        model_path = f"{pair.replace('/', '_')}_{self.config.MODEL_PATH}"
        
        try:
            # Try to load existing model
            if os.path.exists(model_path):
                logger.info(f"Loading existing model for {pair}")
                self.models[pair] = load_model(model_path)
                
                # Initialize scaler for this pair
                # This is the fix for the KeyError issue
                if pair not in self.scalers:
                    logger.info(f"Initializing scaler for {pair}")
                    # Fetch data to initialize scaler
                    df = self._fetch_market_data(pair, lookback_periods=500)
                    if df is not None and len(df) > 0:
                        # Initialize scaler with the data
                        data = df[self.config.FEATURE_COLUMNS].copy()
                        self.scalers[pair] = MinMaxScaler(feature_range=(0, 1))
                        self.scalers[pair].fit(data)
                        logger.info(f"Scaler initialized for {pair}")
                    else:
                        logger.error(f"Could not initialize scaler for {pair}: No data available")
                
                return
        except Exception as e:
            logger.warning(f"Could not load model for {pair}: {e}")
        
        # Train new model
        logger.info(f"Training new model for {pair}")
        
        # Fetch historical data
        df = self._fetch_market_data(pair, lookback_periods=500)
        if df is None or len(df) < self.config.SEQUENCE_LENGTH + self.config.PREDICTION_HORIZON:
            logger.error(f"Insufficient data to train model for {pair}")
            return
        
        # Prepare data
        X, y = self._prepare_model_data(df, pair)
        
        # Split data
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Create and train model
        model = self._create_lstm_model((X_train.shape[1], X_train.shape[2]))
        
        model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        # Save model
        model.save(model_path)
        self.models[pair] = model
        
        # Evaluate model
        _, accuracy = model.evaluate(X_test, y_test)
        logger.info(f"Model for {pair} trained with accuracy: {accuracy:.4f}")
    
    def _generate_trading_signals(self) -> Dict[str, Dict]:
        """Generate trading signals for all pairs"""
        signals = {}
        
        for pair in self.config.TRADING_PAIRS:
            try:
                # Fetch latest market data
                df = self._fetch_market_data(pair)
                if df is None or len(df) < self.config.SEQUENCE_LENGTH:
                    logger.warning(f"Insufficient data for {pair}")
                    continue
                
                # Prepare data for prediction
                data = df[self.config.FEATURE_COLUMNS].copy().tail(self.config.SEQUENCE_LENGTH)
                
                # Ensure scaler exists for this pair
                if pair not in self.scalers:
                    logger.warning(f"No scaler found for {pair}, initializing new one")
                    self.scalers[pair] = MinMaxScaler(feature_range=(0, 1))
                    self.scalers[pair].fit(data)
                
                scaled_data = self.scalers[pair].transform(data)
                X = np.array([scaled_data])
                
                # Make prediction
                if pair in self.models:
                    prediction = self.models[pair].predict(X)[0][0]
                    
                    # Get current price and volatility
                    current_price = df['close'].iloc[-1]
                    current_volatility = df['volatility'].iloc[-1]
                    
                    # Generate signal
                    signal = {
                        'pair': pair,
                        'price': current_price,
                        'timestamp': df['timestamp'].iloc[-1],
                        'prediction': prediction,
                        'volatility': current_volatility,
                        'rsi': df['rsi'].iloc[-1],
                        'upper_band': df['upper_band'].iloc[-1],
                        'lower_band': df['lower_band'].iloc[-1],
                        'macd': df['macd'].iloc[-1],
                        'macd_signal': df['macd_signal'].iloc[-1],
                        'action': 'HOLD'  # Default action
                    }
                    
                    # Determine action based on prediction and confidence
                    if (prediction > self.config.CONFIDENCE_THRESHOLD and 
                        current_volatility < self.config.VOLATILITY_THRESHOLD):
                        signal['action'] = 'BUY'
                    elif (prediction < (1 - self.config.CONFIDENCE_THRESHOLD) and 
                          current_volatility < self.config.VOLATILITY_THRESHOLD):
                        signal['action'] = 'SELL'
                    
                    signals[pair] = signal
                    logger.info(f"Signal for {pair}: {signal['action']} (confidence: {prediction:.4f}, volatility: {current_volatility:.4f})")
                else:
                    logger.warning(f"No model available for {pair}")
            
            except Exception as e:
                logger.error(f"Error generating signal for {pair}: {e}")
        
        return signals
    
    def _calculate_position_size(self, pair: str, action: str) -> float:
        """Calculate position size based on Kelly Criterion and portfolio constraints"""
        try:
            # Get available balance
            base_currency = pair.split('/')[1]  # e.g., USD from BTC/USD
            available_balance = float(self.portfolio.get(base_currency, {}).get('Free', 0))
            
            # Basic position size based on config
            position_size = available_balance * self.config.POSITION_SIZE_PCT
            
            # Apply maximum position constraint
            max_position = available_balance * self.config.MAX_POSITION_SIZE_PCT
            position_size = min(position_size, max_position)
            
            # For selling, check available asset balance
            if action == 'SELL':
                asset_currency = pair.split('/')[0]  # e.g., BTC from BTC/USD
                asset_balance = float(self.portfolio.get(asset_currency, {}).get('Free', 0))
                
                # Get current price
                ticker_params = {
                    'pair': pair,
                    'timestamp': str(self._get_server_time())
                }
                ticker_response = self._make_request("GET", "/v3/ticker", ticker_params)
                current_price = ticker_response.get("Data", {}).get(pair, {}).get("LastPrice", 0)
                
                # Calculate asset value in base currency
                asset_value = asset_balance * current_price
                position_size = min(position_size, asset_value)
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0
    
    def _place_order(self, pair: str, side: str, order_type: str, quantity: float, price: float = None) -> Dict:
        """Place an order on the exchange"""
        try:
            # Prepare order parameters
            params = {
                'pair': pair,
                'side': side,
                'type': order_type,
                'quantity': str(quantity)
            }
            
            # Add price for limit orders
            if order_type == 'LIMIT' and price is not None:
                params['price'] = str(price)
            
            # Place order
            response = self._make_request("POST", "/v3/place_order", params, auth_required=True)
            
            if response.get("Success"):
                order_detail = response.get("OrderDetail", {})
                logger.info(f"Order placed: {order_detail}")
                
                # Track order
                self.orders[order_detail.get("OrderID")] = order_detail
                
                # Update portfolio
                self._update_portfolio()
                
                return order_detail
            else:
                logger.error(f"Failed to place order: {response.get('ErrMsg')}")
                return {}
                
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return {}
    
    def _cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order"""
        try:
            params = {
                'order_id': order_id
            }
            
            response = self._make_request("POST", "/v3/cancel_order", params, auth_required=True)
            
            if response.get("Success"):
                logger.info(f"Order {order_id} canceled")
                
                # Remove from tracked orders
                if order_id in self.orders:
                    del self.orders[order_id]
                
                # Update portfolio
                self._update_portfolio()
                
                return True
            else:
                logger.error(f"Failed to cancel order: {response.get('ErrMsg')}")
                return False
                
        except Exception as e:
            logger.error(f"Error canceling order: {e}")
            return False
    
    def _check_stop_loss(self):
        """Check and execute stop loss orders"""
        try:
            # Get current prices
            ticker_params = {
                'timestamp': str(self._get_server_time())
            }
            ticker_response = self._make_request("GET", "/v3/ticker", ticker_params)
            
            if not ticker_response.get("Success"):
                logger.error(f"Failed to fetch prices for stop loss check: {ticker_response.get('ErrMsg')}")
                return
            
            ticker_data = ticker_response.get("Data", {})
            
            # Check positions for stop loss
            for pair in self.config.TRADING_PAIRS:
                asset_currency = pair.split('/')[0]  # e.g., BTC from BTC/USD
                asset_balance = float(self.portfolio.get(asset_currency, {}).get('Free', 0))
                
                # Skip if no position
                if asset_balance <= 0:
                    continue
                
                # Get current price
                current_price = ticker_data.get(pair, {}).get("LastPrice", 0)
                
                # Check if we have a recorded entry price for this asset
                entry_price = None
                for trade in reversed(self.trade_history):
                    if trade['pair'] == pair and trade['side'] == 'BUY' and trade['status'] == 'FILLED':
                        entry_price = trade['price']
                        break
                
                # Skip if no entry price found
                if entry_price is None:
                    continue
                
                # Check if stop loss is triggered
                if current_price < entry_price * (1 - self.config.STOP_LOSS_PCT):
                    logger.info(f"Stop loss triggered for {pair} at {current_price}")
                    
                    # Calculate quantity to sell
                    quantity = asset_balance
                    
                    # Place market sell order
                    self._place_order(pair, 'SELL', 'MARKET', quantity)
            
        except Exception as e:
            logger.error(f"Error checking stop loss: {e}")
    
    def _execute_trades(self, signals: Dict[str, Dict]):
        """Execute trades based on signals"""
        for pair, signal in signals.items():
            try:
                action = signal['action']
                
                # Skip if no action
                if action == 'HOLD':
                    continue
                
                # Determine order side
                side = 'BUY' if action == 'BUY' else 'SELL'
                
                # Calculate position size
                position_size = self._calculate_position_size(pair, action)
                
                # Skip if position size is too small
                if position_size <= 0:
                    logger.info(f"Skipping {action} for {pair}: insufficient funds")
                    continue
                
                # Get current price
                current_price = signal['price']
                
                # Calculate quantity
                quantity = position_size / current_price
                
                # Get pair precision from exchange info
                exchange_info = self._make_request("GET", "/v3/exchangeInfo")
                pair_info = exchange_info.get("TradePairs", {}).get(pair, {})
                amount_precision = pair_info.get("AmountPrecision", 6)
                
                # Round quantity to appropriate precision
                quantity = round(quantity, amount_precision)
                
                # Skip if quantity is too small
                if quantity <= 0:
                    logger.info(f"Skipping {action} for {pair}: quantity too small")
                    continue
                
                # Place market order
                order = self._place_order(pair, side, 'MARKET', quantity)
                
                # Record trade
                if order:
                    trade = {
                        'pair': pair,
                        'side': side,
                        'type': 'MARKET',
                        'quantity': quantity,
                        'price': current_price,
                        'timestamp': datetime.now().isoformat(),
                        'status': order.get('Status'),
                        'order_id': order.get('OrderID')
                    }
                    self.trade_history.append(trade)
            
            except Exception as e:
                logger.error(f"Error executing trade for {pair}: {e}")
    
    def _calculate_sharpe_ratio(self, returns: List[float], risk_free_rate: float = 0.001) -> float:
        """Calculate Sharpe Ratio"""
        if not returns or len(returns) < 2:
            return 0
        
        # Convert to numpy array
        returns_array = np.array(returns)
        
        # Calculate mean return and standard deviation
        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array)
        
        # Avoid division by zero
        if std_return == 0:
            return 0
        
        # Calculate Sharpe Ratio
        sharpe_ratio = (mean_return - risk_free_rate) / std_return
        
        return sharpe_ratio
    
    def _backtest_strategy(self):
        """Backtest the trading strategy"""
        logger.info("Starting backtest...")
        
        # Initialize backtest portfolio
        backtest_portfolio = {
            'USD': 50000  # Initial USD balance
        }
        
        # Initialize trade history
        backtest_trades = []
        
        # Initialize returns for Sharpe Ratio calculation
        hourly_returns = []
        
        # Run backtest for each pair
        for pair in self.config.TRADING_PAIRS:
            # Fetch historical data
            df = self._fetch_market_data(pair, lookback_periods=self.config.BACKTEST_DAYS * 24 * 4)  # 15-min intervals
            
            if df is None or len(df) < self.config.SEQUENCE_LENGTH:
                logger.warning(f"Insufficient data for backtesting {pair}")
                continue
            
            # Prepare model if not already loaded
            if pair not in self.models:
                self._load_or_train_model(pair)
            
            # Ensure scaler exists for this pair
            if pair not in self.scalers:
                logger.warning(f"No scaler found for {pair}, initializing new one")
                data = df[self.config.FEATURE_COLUMNS].copy()
                self.scalers[pair] = MinMaxScaler(feature_range=(0, 1))
                self.scalers[pair].fit(data)
            
            # Iterate through data
            for i in range(self.config.SEQUENCE_LENGTH, len(df) - self.config.PREDICTION_HORIZON):
                # Get current data window
                window = df.iloc[i-self.config.SEQUENCE_LENGTH:i]
                
                # Prepare data for prediction
                data = window[self.config.FEATURE_COLUMNS].copy()
                scaled_data = self.scalers[pair].transform(data)
                X = np.array([scaled_data])
                
                # Make prediction
                prediction = self.models[pair].predict(X)[0][0]
                
                # Get current price and volatility
                current_price = df['close'].iloc[i]
                current_volatility = df['volatility'].iloc[i]
                
                # Determine action
                action = 'HOLD'
                if (prediction > self.config.CONFIDENCE_THRESHOLD and 
                    current_volatility < self.config.VOLATILITY_THRESHOLD):
                    action = 'BUY'
                elif (prediction < (1 - self.config.CONFIDENCE_THRESHOLD) and 
                      current_volatility < self.config.VOLATILITY_THRESHOLD):
                    action = 'SELL'
                
                # Execute simulated trade
                if action != 'HOLD':
                    # Determine order side
                    side = 'BUY' if action == 'BUY' else 'SELL'
                    
                    # Calculate position size (simplified for backtest)
                    base_currency = pair.split('/')[1]  # e.g., USD from BTC/USD
                    asset_currency = pair.split('/')[0]  # e.g., BTC from BTC/USD
                    
                    available_balance = backtest_portfolio.get(base_currency, 0)
                    asset_balance = backtest_portfolio.get(asset_currency, 0)
                    
                    # Basic position size based on config
                    position_size = available_balance * self.config.POSITION_SIZE_PCT
                    
                    # Apply maximum position constraint
                    max_position = available_balance * self.config.MAX_POSITION_SIZE_PCT
                    position_size = min(position_size, max_position)
                    
                    # For selling, check available asset balance
                    if side == 'SELL':
                        asset_value = asset_balance * current_price
                        position_size = min(position_size, asset_value)
                    
                    # Skip if position size is too small
                    if position_size <= 0:
                        continue
                    
                    # Calculate quantity
                    quantity = position_size / current_price
                    
                    # Simulate trade execution
                    if side == 'BUY':
                        # Update portfolio
                        backtest_portfolio[base_currency] = available_balance - position_size
                        backtest_portfolio[asset_currency] = asset_balance + quantity
                    else:  # SELL
                        # Update portfolio
                        backtest_portfolio[base_currency] = available_balance + position_size
                        backtest_portfolio[asset_currency] = asset_balance - quantity
                    
                    # Record trade
                    trade = {
                        'pair': pair,
                        'side': side,
                        'quantity': quantity,
                        'price': current_price,
                        'timestamp': df['timestamp'].iloc[i],
                        'portfolio_value': self._calculate_portfolio_value(backtest_portfolio, df.iloc[i])
                    }
                    backtest_trades.append(trade)
                
                # Record hourly returns for Sharpe Ratio calculation
                if i % 4 == 0:  # Every hour (4 x 15-min intervals)
                    portfolio_value = self._calculate_portfolio_value(backtest_portfolio, df.iloc[i])
                    if len(hourly_returns) > 0:
                        previous_value = hourly_returns[-1]['value']
                        hourly_return = (portfolio_value - previous_value) / previous_value
                        hourly_returns.append({
                            'timestamp': df['timestamp'].iloc[i],
                            'value': portfolio_value,
                            'return': hourly_return
                        })
                    else:
                        hourly_returns.append({
                            'timestamp': df['timestamp'].iloc[i],
                            'value': portfolio_value,
                            'return': 0
                        })
        
        # Calculate Sharpe Ratio
        returns_list = [r['return'] for r in hourly_returns[1:]]  # Skip first entry (no return)
        sharpe_ratio = self._calculate_sharpe_ratio(returns_list)
        
        # Calculate final portfolio value
        final_portfolio_value = sum([
            backtest_portfolio.get(currency, 0) * 
            (1 if currency == 'USD' else df['close'].iloc[-1])
            for currency in backtest_portfolio
        ])
        
        # Calculate total return
        initial_value = 50000  # Initial USD balance
        total_return = (final_portfolio_value - initial_value) / initial_value
        
        logger.info(f"Backtest results:")
        logger.info(f"Initial portfolio value: ${initial_value:.2f}")
        logger.info(f"Final portfolio value: ${final_portfolio_value:.2f}")
        logger.info(f"Total return: {total_return:.2%}")
        logger.info(f"Sharpe Ratio: {sharpe_ratio:.4f}")
        logger.info(f"Number of trades: {len(backtest_trades)}")
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'total_return': total_return,
            'trades': backtest_trades,
            'hourly_returns': hourly_returns
        }
    
def _calculate_portfolio_value(self, portfolio: Dict[str, float], market_data: pd.Series) -> float:
    """Calculate total portfolio value in USD"""
    total_value = 0

    for currency, amount in portfolio.items():
        if currency == 'USD':
            # USD is the base currency, so its value is 1
            price = 1
        else:
            price = None  # initialize price

            # If market_data is for this currency, use its close price
            if hasattr(market_data, "name") and currency == market_data.name.split('/')[0]:
                price = market_data.get('close', 0)
            else:
                # If not, try to set a default price from known trading pairs
                for pair in self.config.TRADING_PAIRS:
                    if pair.startswith(currency + '/'):
                        if pair == "BTC/USD":
                            price = 10000  # default price for BTC/USD
                        elif pair == "ETH/USD":
                            price = 2000   # default price for ETH/USD
                        elif pair == "BNB/USD":
                            price = 300    # default price for BNB/USD
                        # Break if we have assigned a price
                        if price is not None:
                            break

            # If still no price, set it to 0 (or handle as needed)
            if price is None:
                price = 0

        total_value += amount * price

    return total_value

    async def run(self):
        """Main trading loop"""
        logger.info("Starting trading bot...")
        
        # Run backtest first
        backtest_results = self._backtest_strategy()
        logger.info(f"Backtest Sharpe Ratio: {backtest_results['sharpe_ratio']:.4f}")
        
        # Main trading loop
        while True:
            try:
                # Update portfolio
                self._update_portfolio()
                
                # Check stop loss
                self._check_stop_loss()
                
                # Generate trading signals
                signals = self._generate_trading_signals()
                
                # Execute trades
                self._execute_trades(signals)
                
                # Wait for next cycle (15 minutes)
                logger.info("Waiting for next trading cycle...")
                await asyncio.sleep(15 * 60)  # 15 minutes
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying

# Entry point
async def main():
    # Initialize trading bot
    bot = RoostooTradingBot(API_KEY, SECRET_KEY)
    
    # Run trading bot
    await bot.run()

if __name__ == "__main__":
    # Run the main function
    asyncio.run(main())

