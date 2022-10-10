from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
key = 'cd46213aaamsh2d6e639f70782e2p15b68ajsnfad2d2a74008'

ts = TimeSeries(key, output_format="pandas")
ti = TechIndicators(key, output_format="pandas")

aapl_data, aapl_meta_data = ts.get_intraday(symbol='AAPL', interval='5min')
aapl_bbn, aapl_meta_bbn = ti.get_bbands(symbol='MSFT', interval='60min', time_period=60)


"figure(num=None, figsize=(15,6), dpi=80, facecolor='w',edgecolor='k')"
aapl_bbn.plot()
plt.tight_layout()
plt.grid()
plt.show()