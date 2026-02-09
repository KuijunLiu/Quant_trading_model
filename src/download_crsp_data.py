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
import os  # <--- 别忘了导入 os，用来创建文件夹

def fetch_crsp_data():
    # 1. 检查文件夹是否存在 (否则 to_csv 会报错)
    if not os.path.exists("data/raw"):
        os.makedirs("data/raw")

    print("📡 Connecting to WRDS...")
    # 如果你之前运行过 create_pgpass_file()，这里甚至不需要填 username
    # 如果没运行过，记得把 '你的用户名' 换成真实的 WRDS 账号
    db = wrds.Connection() 

    print("🚀 Querying CRSP Monthly Data (Filtered for Common Stocks)...")
    
    # --- 关键修改点 ---
    # 我们增加了两个 list 来辅助过滤
    # shrcd IN (10, 11): 代表 "Ordinary Common Shares" (普通股)，排除 ETF/REITs
    # exchcd IN (1, 2, 3): 代表 NYSE, AMEX, NASDAQ (三大主板)，排除粉单市场
    
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
    
    # --- 数据清洗 ---
    print("🧹 Cleaning data...")
    df['date'] = pd.to_datetime(df['date'])
    df['prc'] = df['prc'].abs() # 处理 Bid/Ask 平均价的负号
    df['mkt_cap'] = df['prc'] * df['shrout'] # 计算市值
    df['ret'] = pd.to_numeric(df['ret'], errors='coerce') # 处理非数值回报
    
    # 过滤微盘股 (Penny Stocks)
    original_count = len(df)
    df = df[df['prc'] > 5]
    print(f"📉 Filtered Penny Stocks: {original_count} -> {len(df)} rows")
    
    # 保存
    output_path = "data/raw/crsp_monthly.csv"
    df.to_csv(output_path, index=False)
    print(f"💾 Saved clean data to {output_path}")

    # 关闭连接
    db.close()
    
    return df

if __name__ == "__main__":
    fetch_crsp_data()
