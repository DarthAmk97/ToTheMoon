from fbprophet import Prophet
from fbprophet.plot import plot_plotly, add_changepoints_to_plot
import numpy as np


Model=Prophet(seasonality_mode="multiplicative",mcmc_samples=500,changepoint_range=0.8,n_changepoints=30,changepoint_prior_scale=0.07).add_seasonality(name='daily',period=1,fourier_order=18)#mcmc_samples=500

dfactual=pd.read_csv("btc_bars_2h_test.csv",usecols=["newvalue"])
dftest=pd.read_csv('btc_bars_2h_test.csv', usecols=["date"])
dftest['date'] = pd.to_datetime(dftest['date'], errors='coerce', format='%Y-%m-%d').dt.strftime("%Y%m%d")
confirmed = dfactual['newvalue'].values.tolist()
Dataframe101= pd.DataFrame(columns = ['ds','y'])
Dataframe101['ds']=list(dftest['Date'])
Dataframe101['y']=confirmed
Dataframe101= Dataframe101[Dataframe101['y'] != 0]

Model.fit(Dataframe101,control={'max_treedepth': 15})
FuturePrediction=Model.make_future_dataframe(periods=24,freq='H',include_history=False) #D means date. M means Monthly Stamp.


modelforecast=Model.predict(FuturePrediction)
modelforecast['yhat']=modelforecast['yhat'].astype(int)
print(modelforecast.columns)
forecast = modelforecast[['ds','yhat']]
figure=Model.plot_components(modelforecast)

