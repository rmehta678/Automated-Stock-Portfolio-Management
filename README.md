# Automated Stock Portfolio Management
 
Author: Rohan Mehta

I created an automatic portfolio management tracker utilizing AWS infrastructure and different deep learning frameworks which forecast intraday directional movements of S&P 500 constituent stocks and select the top/bottom X picks. From January 1996 to December 2022, I gathered all daily candlestick (OHLCV) data using the Finnhub API. Finnhub provides a real-time RESTful API for stock tickers with free and premium tiers based on the volume and transfer rate of data. The trading strategy is derived from Krauss et al. (2017) and Fischer & Krauss (2018) for simplicity. On each trading day, we buy the 10 stocks with the highest probability and short the 10 stocks with the lowest probability to outperform the market in terms of intraday returns – all with equal monetary weight. The best model is chosen to deploy to AWS Cloud where a user can invest based on the model forecast and be notified of deviations in near real-time.

#### Requirements
```
pip install scikit-learn==1.3.2
pip install tensorflow==2.14.0
pip install joblib
pip install modin
pip install pandas
pip install numpy
pip install boto3
pip install finnhub
pip install flask
pip install gunicorn
```
Additionally a Google Colab premium subscription was used with access to the NVIDIA Tesla V100 for training all LSTM models
