import random
import pandas as pd
import numpy as np
from scipy.stats import norm


btc = pd.DataFrame(pd.read_csv("btc_bars_1d.csv"))
btcReturns = [np.log(btc.close[i]/btc.close[i+1]) for i in range(1380)]
print(btcReturns)
btcPrices = list(btc.close[:1460])[::-1]
print("This is BTCPRICES BELOW ")
print(btcPrices)
btcStd = np.std(btcReturns, ddof=1)
btcAvg = np.average(btcReturns)
btcVar = btcStd**2
drift = btcAvg - (btcVar/2)
over20k, over30k, over40k,over50k = [], [], [], []
numbOfSims = 1000
endPrice = 0
print("\n* * * * * * * * * * * * * * * * * *\nNumber of Simulations: ", numbOfSims)
for simulation in range(numbOfSims):
    btcPred = [btcPrices[-1]]
    for day in range(1381):
        btcPred.append(btcPred[-1]*np.exp(drift+btcStd *
                                          norm.ppf(random.SystemRandom.random(0))))
    over20k.append((0 if btcPred[-1] <= 20000 else 1))
    over30k.append((0 if btcPred[-1] <= 30000 else 1))
    over40k.append((0 if btcPred[-1] <= 40000 else 1))
    over50k.append((0 if btcPred[-1] <= 50000 else 1))
    endPrice += btcPred[-1]

prob20k, prob30k, prob40k, prob50k = (over20k.count(1)/numbOfSims),\
                                            (over30k.count(1)/numbOfSims),\
                                            (over40k.count(1)/numbOfSims),\
                                            (over50k.count(1)/numbOfSims)

print("Average predicted price of BTC: ${:.2f}".format(endPrice/numbOfSims))
print("Probability that BTC is over $20K by: {}\nProbability that BTC is over $30K: {}\nProbability that BTC is over $40K: {}\nProbability that BTC is over $50K: {}\n"
      .format(prob20k, prob30k, prob40k, prob50k))
print("* * * * * * * * * * * * * * * * * *")