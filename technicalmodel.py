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
          'ZTS','UTHR','NBIX','VTRS','ELAN','HIMS','RGC','ALKS','INDV','AMRX','LNTH',
          'TT','JCI','CARR','LII','CSL','MAS','BLDR','WMS','SPXC','AAON','OC','AWI','FBIN']
print(f"Total stocks to process: {len(stocks)}")
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
    total = len(stocks)
    print(f"\n{'='*80}")
    print(f"FETCHING STOCK DATA AND CALCULATING TECHNICAL INDICATORS")
    print(f"{'='*80}\n")
    
    for idx, stock in enumerate(stocks, 1):
        print(f"Processing [{idx}/{total}] {stock}...", end=' ')
        try:
            data = yf.download(stock, start="2018-01-01", end='2023-12-31', interval='1wk', progress=False)

            if data is None or data.empty:
                print("❌ No data")
                continue
            
            data.columns = data.columns.get_level_values(0)
            data['Ticker'] = stock
            
            ticker_obj = yf.Ticker(stock)
            info = ticker_obj.info
            data['Sector'] = info.get('sectorKey', 'Unknown')
            data['Industry'] = info.get('industryKey', 'Unknown')
            data['Market_Weight_Sector'] = info.get('marketWeightInSector', None)
            data['Market_Weight_Industry'] = info.get('marketWeightInIndustry', None)
            data['Date'] = data.index
            data['Volume'] = data.Volume.shift(1).values/100_000_000 # Shift volume by 1 day and scale down
            # Calculate technical indicators
            data['MA_20'] = SMAIndicator(close=data['Close'], window=20).sma_indicator()
            data['MA_50'] = SMAIndicator(close=data['Close'], window=50).sma_indicator()
            data['EMA_20'] = EMAIndicator(close=data['Close'], window=20).ema_indicator()
            data['RSI_14'] = RSIIndicator(close=data['Close'], window=14).rsi()
            data['MACD'] = MACD(close=data['Close']).macd()
            data['BB_Width'] = BollingerBands(close=data['Close'], window=20).bollinger_wband()
            data['PSAR'] = PSARIndicator(high=data['High'], low=data['Low'], close=data['Close']).psar()
            data['ATR'] = AverageTrueRange(high=data['High'], low=data['Low'], close=data['Close']).average_true_range()
            data['OBV'] = OnBalanceVolumeIndicator(close=data['Close'], volume=data['Volume']).on_balance_volume()
            data['MFI'] = MFIIndicator(high=data['High'], low=data['Low'], close=data['Close'], volume=data['Volume'], window=14).money_flow_index()
            data['Close_Open_Ratio'] = data['Close'] / data['Open']
            data['Candle_Body_Size'] = abs(data['Close'] - data['Open']) / data['Open']
            data['Upper_Shadow'] = data['High'] - data[['Close', 'Open']].max(axis=1)
            data['Lower_Shadow'] = data[['Close', 'Open']].min(axis=1) - data['Low']

            all_stock_data.append(data)
            print(f"✅ {len(data)} weeks")
            
        except Exception as e:
            print(f"❌ Error: {str(e)[:50]}")
            continue

    print(f"\n{'='*80}")
    print(f"COMBINING DATA AND SAVING TO CSV")
    print(f"{'='*80}\n")
    
    combined_data = pd.concat(all_stock_data)
    # Calculate backward-looking weekly return: (current week high - last week high) / last week high
    combined_data['Target'] = combined_data.groupby('Ticker')['High'].transform(lambda x: ((x - x.shift(1)) / x.shift(1)))
    
    # Save to CSV instead of Excel
    csv_filename = 'technical_indicators_data.csv'
    combined_data.to_csv(csv_filename, index=True)
    
    print(f"✅ Saved to: {csv_filename}")
    print(f"   Total rows: {len(combined_data):,}")
    print(f"   Total columns: {len(combined_data.columns)}")
    print(f"   Date range: {combined_data.index.min()} to {combined_data.index.max()}")
    print(f"   Stocks processed: {combined_data['Ticker'].nunique()}")
    print(f"\n{'='*80}\n")
    
    return combined_data

if __name__ == "__main__":
    fetch_stock_data(stocks)