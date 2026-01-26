import yfinance as yf

# 下载 BTC
btc = yf.download("BTC-USD", start="2016-01-01")
btc.to_csv("BTC-USD.csv")

# 下载 S&P 500
spx = yf.download("^GSPC", start="2000-01-01")
spx.to_csv("GSPC.csv")

print("下载完成！")
