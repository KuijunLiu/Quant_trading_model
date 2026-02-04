import yfinance as yf
import pandas as pd
import os

def download_sp500_data():
    output_dir = "data/raw"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_file = os.path.join(output_dir, "sp500_prices.csv")
    
    print("step 1/3: downloading data from Wikipedia...")
    try:
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        payload = pd.read_html(url)
        sp500_table = payload[0]
        tickers = sp500_table['Symbol'].tolist()
        
        # corerct format：replace BRK.B by BRK-B (Yahoo Finance format)
        tickers = [ticker.replace('.', '-') for ticker in tickers]
        print(f"obtain {len(tickers)} stock index")
    except Exception as e:
        print(f"fail to obtain: {e}")
        return

    # time range setup
    start_date = "2018-01-01"  # fixed starting date
    end_date = pd.Timestamp.today().strftime('%Y-%m-%d')
    
    print(f"start downloading ({start_date} ~ {end_date})...")
    # print("...")
    
    # 3. download data
    group_by='column'
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True, group_by='ticker')
    
    # 这里的 data 是一个巨大的多层索引 DataFrame
    # 我们为了简化，暂时只提取所有股票的 'Close' (收盘价)
    # 注意：因为 yfinance 现在的返回格式比较复杂，我们提取 Close 的方式如下：
    
    # 重新下载一次仅包含 Close 的简洁版（防止结构混乱），或者直接清洗
    # 为了保险起见，我们用最稳妥的方式：只取收盘价
    df_close = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)['Close']
    
    # 4. simple data cleaning
    df_close = df_close.dropna(axis=1, how='all') # delete empty column
    
    print(f"step 3/3: save data to {output_file}...")
    df_close.to_csv(output_file)
    
    print(f"data saved! dimension: {df_close.shape}")

if __name__ == "__main__":
    download_sp500_data()