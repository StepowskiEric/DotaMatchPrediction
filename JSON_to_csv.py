import json

import numpy
import numpy as np
import pandas as pd
import urllib3
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV


## https://api.opendota.com/api/Matches/5992559409?api_key=52dbfbf6-b204-43cf-9411-6715cd68f106
## 1 query parameter & thats the match ID

##Gets 100 match ID's from a random sample
def getPublicMatchesResponse():
    match_id_list = []
    match_id_list2 = []
    http = urllib3.PoolManager()
    first = http.request('GET',
                         'https://api.opendota.com/api/publicMatches?api_key=52dbfbf6-b204-43cf-9411-6715cd68f106')

    dataFirst = json.loads(first.data.decode('utf-8'))

    print(type(dataFirst))

    match_id = dataFirst[1]['match_id']  ##if you want to test just 1

    for i in dataFirst:
        match_id_list.append(i['match_id'])

    columns_needed2 = ['match_id', 'player_slot', 'assists', 'deaths', 'denies', 'duration',
                       'tower_damage',
                       'gold_per_min', 'hero_damage', 'hero_id', 'last_hits', 'level', 'kills',
                       'xp_per_min', 'kda', 'hero_healing',
                       'abandons',
                       'win']
    newMatchesData = []

    newMatch = http.request('GET',
                            f'https://api.opendota.com/api/matches/{match_id}?api_key=52dbfbf6-b204-43cf-9411-6715cd68f106')
    newData = json.loads(newMatch.data.decode('utf-8'))
    first_match = newData['players']
    print(first_match)
    df = pd.json_normalize(first_match)
    df.head()
    print(df)
    df.to_csv('PlayerData.csv', columns=columns_needed2)

    n = 0
    for j in match_id_list:
        newMatchData = http.request('GET',
                                    f'https://api.opendota.com/api/matches/{j}?api_key=52dbfbf6-b204-43cf-9411-6715cd68f106')
        newMatchesData.append(newMatchData)
        dataTest = json.loads(newMatchesData[n].data.decode('utf-8'))
        n += 1
        player_data = dataTest['players']
        print(player_data)
        extra_data = dataTest
        df = pd.json_normalize(player_data)
        df.to_csv('PlayerData.csv', mode='a', header=False, columns=columns_needed2)
        print("Done!")


# This function changes the player_slot to radiant and dire.
def change_player_slot():
    read_df = pd.read_csv('PlayerData.csv')
    read_df['player_slot'] = numpy.where(read_df['player_slot'] <= 127, 'Radiant', read_df.player_slot)
    read_df['player_slot'] = numpy.where(read_df['player_slot'] >= 128, 'Dire', read_df.player_slot)
    print(read_df)
    read_df.to_csv('PlayerData2.csv')


def train_model():
    df = pd.read_csv('PlayerData.csv')
    x = df.drop(['match_id', 'player_slot', 'hero_id', 'win', 'Unnamed: 0'], axis=1)
    y = df.win
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=4)
    logistic_regression = LogisticRegression(C=1e5, max_iter=1000, verbose=2)
    logistic_regression.fit(x_train, y_train)

    y_pred = logistic_regression.predict(x_test)
    accuracy = metrics.accuracy_score(y_test, y_pred)
    accuracy_percentage = 100 * accuracy
    print(accuracy_percentage)

    # Predict if a player will win or lose
    # Array values are: assists, deaths, denies, duration, tower dmg, gold_per_min, hero_dmg, last_hits, level, kills, xp_per_min, kda, hero_healing and abandons)
    first_test = logistic_regression.predict(
        (np.array([20, 8, 8, 3537, 19756, 587, 28076, 417, 28, 17, 660, 7, 626, 0]).reshape(1, -1)))
    print(first_test)
    second_test = logistic_regression.predict(
        (np.array([6, 7, 1, 1939, 0, 271, 12404, 67, 16, 2, 410, 1, 0, 0]).reshape(1, -1)))
    print(second_test)
    third_test = logistic_regression.predict(
        (np.array([10, 1, 10, 1164, 2450, 441, 4930, 45, 12, 3, 389, 6, 1029, 0]).reshape(1, -1)))
    print(third_test)
    live_test = logistic_regression.predict(
        (np.array([39,16,2,45270,4400,476,70800,239,25,11,492,0,0,0]).reshape(1, -1)))
    print(live_test)
