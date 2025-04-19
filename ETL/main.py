from modules.get_data import Stock, merge_data
import json
import os

with open('data\lista_tickers.json', 'r') as f:
    stocks = json.load(f)
    stocks = stocks['indexes']

print(stocks)

KEY_BRAPI = os.getenv('BRAPI_KEY')

for stock in stocks:
    stock_data = Stock(interval='1d', range='10y', ticker=stock, key=KEY_BRAPI)
    stock_data.get_data()
    stock_data.save_json(path="C:/tcc_dsa/data/raw/indexes_10y/")
    stock_data.get_stock_prices(path="C:/tcc_dsa/data/processed/prices_hist/indexes_10y/")

merge_data(input_dir="C:/tcc_dsa/data/processed/prices_hist/indexes_10y/", output_file="C:/tcc_dsa/data/processed/merged/merged_indexes_10y.parquet")


