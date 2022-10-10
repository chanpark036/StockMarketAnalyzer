from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.function_base import cov
import pandas as pd
from sklearn.linear_model import LinearRegression

key = 'cd46213aaamsh2d6e639f70782e2p15b68ajsnfad2d2a74008'

ts = TimeSeries(key, output_format="pandas")
ti = TechIndicators(key, output_format="pandas")

all_data, aapl_meta_data = ts.get_daily(symbol='MSFT')


print(all_data)


"all_data['4. close'].plot()"

"""print(type(all_data))
print(pd.DataFrame(all_data).index)"""

"separated the data into different arrays"
opendata = np.array(all_data['1. open'])
closeddata = np.array(all_data['4. close'])
volumedata = np.array(all_data['5. volume'])

initialtuple = (opendata, closeddata, volumedata)
initialdata = np.vstack(initialtuple)

"now need to either use standardization or normalization"
stanopendata = (opendata - np.mean(opendata))/np.std(opendata)
stancloseddata = (closeddata - np.mean(closeddata))/np.std(closeddata)
stanvolumedata = (volumedata - np.mean(volumedata))/np.std(volumedata)

"find covariance matrix"
tupleofvariables = (stanopendata, stancloseddata, stanvolumedata)
matrixofvariables = np.vstack(tupleofvariables)
covmatrix = np.cov(matrixofvariables)

"""after here you do not need to change anything no matter how many variables you are using"""

"find eigenvalues and eigenvectors of covariance matrix"
w,v = np.linalg.eig(covmatrix)
"""print(w)
print(v)"""

"order eigenvalues from descending order and eigenvectors according to the order of the eigenvalues"
wlist = []
vlist = []
templist = []
for i in range(len(w)):
    tempmax = 0
    for j in range(len(w)):
        if(w[j] > tempmax) and (w[j] not in templist):
            tempmax = w[j]
    templist.append(tempmax)
    wlist.append(tempmax)
    for k in range(len(w)):
        if(w[k] == tempmax):
            vlist.append(v[k])

wsorted = np.array(wlist)
vsorted = np.array(vlist)

"""print(wsorted)
print(vsorted)"""

"find importances of principal components"
eigsum = sum(w)
pcarray = w / sum(w)
print(pcarray)

"featurevector with 2 principal components"
featvecs = vsorted[0:1]
"print(featvecs)"

"calculate final data set"
finaldataset = np.dot(featvecs,initialdata)
print(finaldataset)


pddata = pd.DataFrame(finaldataset.T, index = pd.DataFrame(all_data).index, columns = ["PC1"])
print(pddata)

pddata.plot()
plt.tight_layout()
plt.grid()
plt.show()