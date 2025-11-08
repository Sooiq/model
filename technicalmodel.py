import yfinance as yf
import datetime
import pandas as pd
from ta.trend import SMAIndicator, EMAIndicator, MACD, PSARIndicator
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, MFIIndicator

stocks = ['CRM','UBER','INTU','NOW','ADBE','ADP','SNOW','CDNS','MSTR','DDOG','ADSK','WDAY','ROP','GRAB','FIG',
          'JPM','BAC','WFC','C','RY','TD','BK','NTB',
          'MCD','SBUX','YUM','CMG','DRI','YUMC','DPZ','TXRH','BROS','WING','SHAK', 
          'ZTS','UTHR','NBIX','VTRS','ELAN','HIMS','RGC','ALKS','INDV','AMRX','LNTH'
          'TT','JCI','CARR','LII','CSL','MAS','BLDR','WMS','SPXC','AAON','OC','AWI','FBIN']
print(len(stocks))
def is_delisted(stock, try_download=True):
    if try_download:
        data = yf.download(stock, start="2018-01-01", end='2023-12-31')
        if data is None or data.empty:
            return True
    info = yf.Ticker(stock).info
    if 'delisted' in info and info['delisted']:
        return True
    return False

for stock in stocks:
    if is_delisted(stock):
        stocks.remove(stock)

def fetch_stock_data(stocks):
    all_stock_data = []
    for stock in stocks: #Download and calculate features for each stock
        data = yf.download(stock, start="2018-01-01", end='2023-12-31', interval='1wk')

        data.columns = data.columns.get_level_values(0)
        data['Ticker'] = stock
        data['Sector'] = yf.Sector(yf.Ticker(stock).info.get('sectorKey'))
        data['Industry'] = yf.Ticker(yf.Ticker(stock).info.get('industryKey'))
        data['Market_Weight_Sector'] = yf.Ticker(stock).info.get('marketWeightInSector', None)
        data['Market_Weight_Industry'] = yf.Ticker(stock).info.get('marketWeightInIndustry', None)
        data['Date'] = data.index
        data['Volume'] = data.Volume.shift(1).values/100_000_000 # Shift volume by 1 day and scale down
        if data is None or data.empty:
            continue
         # Calculate technical indicators
        data['MA_20'] = SMAIndicator(close=data['Close'], window=20).sma_indicator()
        data['MA_50'] = SMAIndicator(close=data['Close'], window=50).sma_indicator()
        data['EMA_20'] = EMAIndicator(close=data['Close'], window=20).ema_indicator()
        data['RSI_14'] = RSIIndicator(close=data['Close'], window=14).rsi()
        data['MACD'] = MACD(close=data['Close']).macd()
        data['BB_Width'] = BollingerBands(close=data['Close'], window=20).bollinger_wband()
        data['PSAR'] = PSARIndicator(high=data['High'], low=data['Low'], close=data['Close']).psar()
        if data is None or data.empty:
            continue
        else: data['ATR'] = AverageTrueRange(high=data['High'], low=data['Low'], close=data['Close']).average_true_range()
        data['OBV'] = OnBalanceVolumeIndicator(close=data['Close'], volume=data['Volume']).on_balance_volume()
        data['MFI'] = MFIIndicator(high=data['High'], low=data['Low'], close=data['Close'], volume=data['Volume'], window=14).money_flow_index()
        data['Close_Open_Ratio'] = data['Close'] / data['Open']
        data['Candle_Body_Size'] = abs(data['Close'] - data['Open']) / data['Open']
        data['Upper_Shadow'] = data['High'] - data[['Close', 'Open']].max(axis=1)
        data['Lower_Shadow'] = data[['Close', 'Open']].min(axis=1) - data['Low']

        all_stock_data.append(data)

    combined_data = pd.concat(all_stock_data)
    combined_data['Target'] = combined_data.groupby('Ticker')['Close'].transform(lambda x: ((x-x.shift(7))/x.shift(7)))
    combined_data.to_excel('technical_indicators_data.xlsx', index=True)
    return combined_data

if __name__ == "__main__":
    fetch_stock_data(stocks)