import requests
import pandas as pd
import json
from dotenv import load_dotenv
import os
import datetime as dt
import json
import pyarrow.parquet as pq
import pyarrow as pa



class Stock:


    def __init__(self, interval, range, ticker, key):
        self.ticker = ticker
        self.range = range
        self.interval = interval
        self.key = key
        self.data = None
        self.stock_hist = None

      
    def get_data(self):
        
        payload = {
                     'range': self.range,
                     'interval': self.interval,
                     'dividends': 'true',
                     'fundamental': 'true',
                     'modules': 'summaryProfile,defaultKeyStatistics,balanceSheetHistoryQuarterly,incomeStatementHistory,incomeStatementHistoryQuarterly,financialData'
                    }


        header = {
                    'Authorization': 'Bearer ' + self.key
                    }        

       
        url = f'https://brapi.dev/api/quote/{self.ticker}'
        r = requests.get(url, params= payload, headers=header)
        self.data = r.json()
    
    def save_json(self, path):             
        with open(path + f'{self.ticker}.json', 'w') as f:
            json.dump(self.data, f)
                    

    def get_stock_prices(self, path):
        self.stock_hist = pd.DataFrame(data= self.data['results'][0]['historicalDataPrice'])
        self.stock_hist['date'] = pd.to_datetime(self.stock_hist['date'], unit='s')
        self.stock_hist['return'] = self.stock_hist['close'].pct_change()
        self.stock_hist['ticker'] = self.data['results'][0]['symbol']

        if path is None:
            pass
        else:
             self.stock_hist.to_parquet(path= path + f'{self.ticker}_ph.parquet', engine='fastparquet')
    

        
 

    def start(self, path):
        try:         
            self.get_data(self)
            self.save_json(self, path=path)
            self.get_stock_prices(self)
            print("Data obtained with Sucess!")
        except:
            print("An error has occurred")


def merge_data(input_dir, output_file):
    # List all Parquet files in the directory
    parquet_files = [f for f in os.listdir(input_dir) if f.endswith('.parquet')]
  
    # Read and concatenate all Parquet files
    dataframes = [pd.read_parquet(os.path.join(input_dir, file)) for file in parquet_files]
    merged_df = pd.concat(dataframes, ignore_index=True)
    merged_df = merged_df[['date', 'ticker', 'adjustedClose']]
    merged_df['date'] = merged_df['date'].dt.date
    merged_df['date'] = pd.to_datetime(merged_df['date'], format='%Y%m%d')
    merged_pivot = merged_df.pivot_table(index='date', columns='ticker', values='adjustedClose')
  
    # Save the merged DataFrame to a Parquet file
    merged_pivot.to_parquet(output_file)
    print(f"Merged DataFrame saved to {output_file}")

class Selic:   
    
    def __init__(self, key:str, start_date: str, end_date:str , country:str):
        self.start_date = None
        self.end_date = None
        self.country = country
        self.key = key
        self.data = None
        

    def get_data(self):
        

        params = {
                    'start': self.start_date,  # Start date in YYYY-MM-DD format
                    'end': self.end_date,  # Current date in YYYY-MM-DD format
                    'country': 'brazil', 
                    'token': self.key
        }
       
        url = "https://brapi.dev/api/v2/prime-rate"
        r = requests.get(url, params= params)
              
        if r.status_code == 200:
            self.data = r.json()
            return print("Data obtained with Sucess!")
        else:
            return print("An error has occurred")
    
    def save_json(self, path):             
        with open(path + 'selic.json', 'w') as f:
            json.dump(self.data, f)

