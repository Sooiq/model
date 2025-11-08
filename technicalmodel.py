import yfinance as yf
import datetime
import pandas as pd
from ta.trend import SMAIndicator, EMAIndicator, MACD, PSARIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, MFIIndicator

stocks = ['NVDA', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'AVGO','TSLA', 'META', 'BRK-B', 'NFLX','META', 
            'PLTR','COST','ASML','AMD','CSCO','MU','AZN','TMUS','APP','LRXC','ISRG','SHOP','LIN','PEP',
            'PDD','AMAT','QCOM','INTC','INTU','AMGN','ADBE','TXN','BKNG','ZM','SNOW','NOW','ADSK','LRCX',
            'MRVL','WDAY','XLNX','CTSH','EBAY','SNDK','UAL','EXE','FISV','KDP','ANSS','CDNS','MCHP','VRSK',
            'DLTR','ALGN','CPRT','FAST','IDXX','XEL','SWKS',
        ]
def fetch_stock_data(stocks):
    today = datetime.date.today()
    all_stock_data = []
    for stock in stocks: #Download and calculate features for each stock
        data = yf.download(stock, start="2010-01-01", end=today)
        data.columns = data.columns.get_level_values(0)
        data['Ticker'] = stock
        data['Date'] = data.index
        data['Volume'] = data.Volume.shift(1).values/100_000_000 # Shift volume by 1 day and scale down
        data['MA_20'] = SMAIndicator(close=data['Close'], window=20).sma_indicator()
        data['MA_50'] = SMAIndicator(close=data['Close'], window=50).sma_indicator()
        data['EMA_20'] = EMAIndicator(close=data['Close'], window=20).ema_indicator()
        data['RSI_14'] = RSIIndicator(close=data['Close'], window=14).rsi()
        data['MACD'] = MACD(close=data['Close']).macd()
        data['BB_Width'] = BollingerBands(close=data['Close'], window=20).bollinger_wband()
        data['PSAR'] = PSARIndicator(high=data['High'], low=data['Low'], close=data['Close']).psar()
        data['ATR'] = AverageTrueRange(high=data['High'], low=data['Low'], close=data['Close'], window=14).average_true_range()
        data['OBV'] = OnBalanceVolumeIndicator(close=data['Close'], volume=data['Volume']).on_balance_volume()
        data['MFI'] = MFIIndicator(high=data['High'], low=data['Low'], close=data['Close'], volume=data['Volume'], window=14).money_flow_index()
        data['Close_Open_Ratio'] = data['Close'] / data['Open']
        data['Candle_Body_Size'] = abs(data['Close'] - data['Open']) / data['Open']
        data['Upper_Shadow'] = data['High'] - data[['Close', 'Open']].max(axis=1)
        data['Lower_Shadow'] = data[['Close', 'Open']].min(axis=1) - data['Low']

        all_stock_data.append(data)

    combined_data = pd.concat(all_stock_data)


