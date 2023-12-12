import finnhub
import json
import datetime
import pandas as pd
import numpy as np
import random
import time
import pickle
from sklearn.ensemble import RandomForestClassifier
from Statistics import Statistics

import os
SEED = 9
os.environ['PYTHONHASHSEED']=str(SEED)
random.seed(SEED)
np.random.seed(SEED)

name = []

def extract_names_from_json(json_file_path):
    with open(json_file_path, "r") as f:
        json_data = json.load(f)

    names = []
    for entry in json_data:
        names.append(entry["Symbol"])

    return names

def trainer(train_data):
    random.seed(SEED)
    np.random.seed(SEED)

    train_x,train_y = train_data[:,2:-2],train_data[:,-1]
    train_y = train_y.astype('int')

    print('Started training')
    clf = RandomForestClassifier(n_estimators=1000,
                                 max_depth=10,
                                 random_state = SEED,
                                 n_jobs=-1)
    clf.fit(train_x,train_y)
    print('Completed ',clf.score(train_x,train_y))
    return clf

def tester(clf, test_data):
    dates = list(set(test_data[:,0]))
    predictions = {}
    labels = {}
    for day in dates:
        test_d = test_data[test_data[:,0]==day]
        labels[day] = test_d[:,1]
        test_d = test_d[:,2:-2]
        predictions[day] = clf.predict_proba(test_d)[:,1]
    return labels, predictions


def simulate(test_data,predictions):
    rets = pd.DataFrame([],columns=['Long','Short'])
    k = 10
    for day in sorted(predictions.keys()):
        preds = predictions[day]
        test_returns = test_data[test_data[:,0]==day][:,-2]
        top_preds = predictions[day].argsort()[-k:][::-1]
        trans_long = test_returns[top_preds]
        worst_preds = predictions[day].argsort()[:k][::-1]
        trans_short = -test_returns[worst_preds]
        rets.loc[day] = [np.mean(trans_long),np.mean(trans_short)]
    return rets

def create_label(df_open,df_close,perc=[0.5,0.5]):
    if not np.all(df_close.iloc[:,0]==df_open.iloc[:,0]):
        print('Date Index issue')
        return
    perc = [0.]+list(np.cumsum(perc))
    label = (df_close.iloc[:,1:]/df_open.iloc[:,1:]-1).apply(
            lambda x: pd.qcut(x.rank(method='first'),perc,labels=False), axis=1)
    return label

def create_stock_data(df_close,df_open,st,label,test_year):
    st_data = pd.DataFrame([])
    st_data['Date'] = list(df_close['Date'])
    st_data['Name'] = [st]*len(st_data)

    daily_change = df_close[st]/df_open[st]-1
    # print(daily_change)
    # print(len(df_close[st]))
    # print(len(df_open[st]))
    # print(sum(df_close['Date'].value_counts() == df_open['Date'].value_counts()))
    m = list(range(1,20))+list(range(20,241,20))
    for k in m:
        st_data['IntraR'+str(k)] = df_close[st].shift(1)/df_open[st].shift(k)-1
        # print(st_data['IntraR'+str(k)])
        st_data['Close' + str(k)] = df_close[st].shift(1)/df_close[st].shift(k+1)-1
        st_data['Open' + str(k)] = df_open[st]/df_open[st].shift(k)-1

    st_data['R-future'] = daily_change
    st_data['label'] = list(label[st])
    st_data['Month'] = list(df_close['Date'].str[:-3])
    # st_data.to_excel('stock_data.xlsx')
    st_data = st_data.dropna()
    # print(st_data.head())
    trade_year = st_data['Month'].str[:4]
    st_data = st_data.drop(columns=['Month'])
    st_train_data = st_data[trade_year<str(test_year)]
    # print(st_train_data)
    st_test_data = st_data[trade_year==str(test_year)]
    return np.array(st_train_data),np.array(st_test_data)

def calculate_accuracy(predictions, test_data):
    accuracies = {}

    for date, daily_predictions in predictions.items():
        filtered_data = test_data[test_data[:, 0] == date]

        # Convert predictions to binary labels
        binary_predictions = [1 if prediction >= 0.5 else 0 for prediction in daily_predictions]

        # Calculate accuracy for the day
        correct_predictions = 0
        for prediction, label in zip(binary_predictions, filtered_data[:, -1]):
            if prediction == label:
                correct_predictions += 1

        accuracy = correct_predictions / len(binary_predictions)
        accuracies[date] = accuracy

    accuracy = np.mean(list(accuracies.values()))

    return accuracy

def main():
    json_file_path = "s&p500.json"
    names = extract_names_from_json(json_file_path)

    test_year=2022
    print(len(names))

    finnhub_client = finnhub.Client(api_key="cl056c1r01qhjei33va0cl056c1r01qhjei33vag")

    end_utc = datetime.datetime.utcnow()

    print(datetime.datetime.fromtimestamp(int(end_utc.timestamp())))

    start_utc = end_utc - datetime.timedelta(days=1)

    print(datetime.datetime.fromtimestamp(int(start_utc.timestamp())))

    # Get the stock candles for AAPL for the past minute
    candles = finnhub_client.stock_candles('AAPL', 'D', int(start_utc.timestamp()), int(end_utc.timestamp()))

    # Print the candles
    print(candles)


    result_folder = 'results-Intraday-240-1-RF'
    for directory in [result_folder]:
        if not os.path.exists(directory):
            os.makedirs(directory)


    filename = '/Users/rohan_mehta/Documents/GitHub/MSOL-Capstone/Open-1996.xlsx'
    df_open = pd.read_excel(filename)
    filename = '/Users/rohan_mehta/Documents/GitHub/MSOL-Capstone/Close-1996.xlsx'
    df_close = pd.read_excel(filename)

    constituents = {}

    for index, row in df_open.iterrows():
        date = row['Date']
        stocks = list(row.dropna().index.drop('Date'))
        constituents[date] = stocks
    
    importances = {}

    for test_year in range(2000, 2023):
        start = time.time()
        print('-'*40)
        print(test_year)
        print('-'*40)

        df_close_cur = df_close[(df_close['Date'].str[:4].astype(int) >= test_year -3) & (df_close['Date'].str[:4].astype(int) <= test_year)].reset_index(drop=True)
        df_open_cur = df_open[(df_open['Date'].str[:4].astype(int) >= test_year - 3) & (df_open['Date'].str[:4].astype(int) <= test_year)].reset_index(drop=True)

        # print(df_open_cur.head())
        label = create_label(df_open_cur,df_close_cur)

        earliest_date = ''

        for key in constituents:
            if key.startswith(str(test_year-3)):
                if earliest_date == '':
                    earliest_date = key
                elif key < earliest_date:
                    earliest_date = key

        stock_names = sorted(list(constituents[earliest_date]))
        train_data,test_data = [],[]
        for st in stock_names:
            st_train_data,st_test_data = create_stock_data(df_close_cur,df_open_cur,st,label,test_year)
            train_data.append(st_train_data)
            test_data.append(st_test_data)

        train_data = np.concatenate([x for x in train_data])
        test_data = np.concatenate([x for x in test_data])

        with open('train_data.pkl', 'wb') as f:
            pickle.dump(train_data, f)

        print('Created :',train_data.shape,test_data.shape,time.time()-start)

        clf = trainer(train_data)
        importances[test_year] = clf.feature_importances_
        labels, predictions = tester(clf, test_data)

        accuracy = calculate_accuracy(predictions, test_data)
        # print(labels.keys()[0])
        returns = simulate(test_data,predictions)
        # print(returns)
        result = Statistics(returns.sum(axis=1))
        result.shortreport()
        print('\nAverage returns prior to transaction charges')
        print(result.mean())

        returns.to_csv(result_folder+'/avg_daily_rets-'+str(test_year)+'.csv')
        with open(result_folder+"/report-" + str(test_year) + ".txt", "a") as f:
            res = '-'*30 + '\n'
            res += str(test_year) + '\n'
            res += 'Mean = ' + str(result.mean()) + '\n'
            res += 'Standard dev = '+str(result.std()) + '\n'
            res += 'Sharpe ratio = ' + str(result.sharpe()) + '\n'
            res += 'Standard Error = '+str(result.stderr()) + '\n'
            res += 'Share>0 = ' + str(result.pos_perc()) + '\n'
            res += 'Skewness = '+str(result.skewness()) + '\n'
            res += 'Kurtosis = ' + str(result.kurtosis()) + '\n'
            res += 'VaR_1 = '+str(result.VaR(1)) + '\n'
            res += 'VaR_2 = ' + str(result.VaR(2)) + '\n'
            res += 'VaR_5 = '+str(result.VaR(5)) + '\n'
            res += 'CVaR_1 = ' + str(result.CVaR(1)) + '\n'
            res += 'CVaR_2 = '+str(result.CVaR(2)) + '\n'
            res += 'CVaR_5 = '+str(result.CVaR(5)) + '\n'
            res += 'MDD = '+str(result.MDD()) + '\n'
            res += 'Percentiles = '+str(result.percentiles()) + '\n'
            res += '-'*30 + '\n'
            f.write(res)
        # with open(result_folder+"/avg_returns" + str(test_year) + '.txt', "a") as myfile:
        #     res = '-'*30 + '\n' 
        #     res += str(test_year) + '\n'
        #     res += 'Mean = ' + str(result.mean()) + '\n'
        #     res += 'Sharpe = '+str(result.sharpe()) + '\n'
        #     res += 'Accuracy = ' + str(accuracy) + '\n'
        #     res += '-'*30 + '\n'
        #     myfile.write(res)

        # Save the trained model
        # with open('random_forest_model.pkl', 'wb') as f:
        #     pickle.dump(clf, f)
    
        # with open(result_folder+'/labels-'+str(test_year)+'.pkl', "wb") as myfile:
        #     pickle.dump(labels, myfile)
            
if __name__ == '__main__':
    main()