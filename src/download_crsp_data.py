"""
download_crsp_data.py

Description:
    This script connects to WRDS to fetch monthly stock data from CRSP.
    It filters for S&P 500 universe, handles delisting returns, and 
    cleans the data for momentum strategy backtesting.

Usage:
    python download_crsp_data.py --start_date 2010-01-01

Author: Kuijun Liu
Date: 2026-02-05
"""

import wrds
import pandas as pd
import numpy as np
import os  # create folder if not exists

def fetch_crsp_data():
    # 1. check if data/raw folder exists, if not create it
    if not os.path.exists("data/raw"):
        os.makedirs("data/raw")

    print("📡 Connecting to WRDS...")
    # if you have run this script before, WRDS will save your credentials and you won't need to input them again
    # if not, it will prompt you to enter your WRDS username and password
    db = wrds.Connection() 

    print("🚀 Querying CRSP Monthly Data (Filtered for Common Stocks)...")
    
    # we add a LEFT JOIN to get the company name, share code, and exchange code from the msenames table
    # shrcd IN (10, 11): "Ordinary Common Shares"，exclude ETF/REITs
    # exchcd IN (1, 2, 3): NYSE, AMEX, NASDAQ 
    
    sql_query = """
    SELECT 
        a.date, 
        a.permno, 
        a.ret, 
        a.prc, 
        a.shrout, 
        b.comnam, 
        b.shrcd, 
        b.exchcd
    FROM 
        crsp.msf AS a
    LEFT JOIN 
        crsp.msenames AS b
    ON 
        a.permno = b.permno 
        AND b.namedt <= a.date 
        AND a.date <= b.nameendt
    WHERE 
        a.date >= '2018-01-01' 
        AND b.shrcd IN (10, 11) 
        AND b.exchcd IN (1, 2, 3)
    """
    
    df = db.raw_sql(sql_query)
    
    print(f"✅ Downloaded {len(df)} rows.")
    
    # --- data cleaning ---
    print("🧹 Cleaning data...")
    df['date'] = pd.to_datetime(df['date'])
    df['prc'] = df['prc'].abs() # deal with negative prices (delisting returns are negative, but we want the absolute price for market cap calculation)
    df['mkt_cap'] = df['prc'] * df['shrout'] # 计算市值
    df['ret'] = pd.to_numeric(df['ret'], errors='coerce') # deal with non-numeric returns (e.g. delisting returns can be 'C' for "delisted with no price")
    
    # filter Penny Stocks
    original_count = len(df)
    df = df[df['prc'] > 5]
    print(f"📉 Filtered Penny Stocks: {original_count} -> {len(df)} rows")
    
    # save data to CSV
    output_path = "data/raw/crsp_monthly.csv"
    df.to_csv(output_path, index=False)
    print(f"💾 Saved clean data to {output_path}")

    # close WRDS connection
    db.close()
    
    return df

if __name__ == "__main__":
    fetch_crsp_data()
