from modules.get_data import Stock, merge_data
import json
import os
import pandas as pd

#ibov = pd.read_csv("C:/tcc_dsa/data/raw/ibov_carteira/IBOV_index.csv", sep=";", encoding="latin1")
#stocks = ibov["Código"]


with open('data\lista_tickers.json', 'r') as f:
    stocks = json.load(f)
    stocks = stocks['indexes']


print(stocks)

KEY_BRAPI = os.getenv('BRAPI_KEY')

for stock in stocks:
    stock_data = Stock(interval='1d', range='5y', ticker=stock, key=KEY_BRAPI)
    stock_data.get_data()
    stock_data.save_json(path="C:/tcc_dsa/data/raw/indices_5y/")
    stock_data.get_stock_prices(path="C:/tcc_dsa/data/processed/prices_hist/indices_5y/")

merge_data(input_dir="C:/tcc_dsa/data/processed/prices_hist/indices_5y/", output_file="C:/tcc_dsa/data/processed/merged/indices_5y.parquet")


