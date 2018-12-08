
# coding: utf-8

# In[62]:


import pandas as pd
import math
import csv
import random
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression


# In[63]:


base_elo = 1600
team_elos = {} 
team_stats = {}
X = []
y = []
# folder = '/Users/chaoyiyang/Documents/Course/CSE546-Machine Learning-18Au/Project/Data/' #directory to save the result
folder = '/Users/chaoyiyang/Documents/Course/CSE546-Machine Learning-18Au/Project/Data/UW_CSE546_Project/ELO Rating/training_data'


# In[64]:


# based on Miscellaneous Opponent，Team stats to initialize
def initialize_data(Mstat, Ostat, Tstat):
    new_Mstat = Mstat.drop(['Rk', 'Arena'], axis=1)
    new_Ostat = Ostat.drop(['Rk', 'G', 'MP'], axis=1)
    new_Tstat = Tstat.drop(['Rk', 'G', 'MP'], axis=1)

    team_stats1 = pd.merge(new_Mstat, new_Ostat, how='left', on='Team')
    team_stats1 = pd.merge(team_stats1, new_Tstat, how='left', on='Team')

    print(team_stats1.info())
    return team_stats1.set_index('Team', inplace=False, drop=True)

# get the elo rating of each team
def get_elo(team):
    try:
        return team_elos[team]
    except:
        # initially set as base_elo
        team_elos[team] = base_elo
        return team_elos[team]
    
# calculate the ELO rating of each team
def calc_elo(win_team, lose_team):
    winner_rank = get_elo(win_team)
    loser_rank = get_elo(lose_team)

    rank_diff = winner_rank - loser_rank
    exp = (rank_diff  * -1) / 400
    odds = 1 / (1 + math.pow(10, exp))
    # update the ELO rating
    if winner_rank < 2100:
        k = 32
    elif winner_rank >= 2100 and winner_rank < 2400:
        k = 24
    else:
        k = 16
    new_winner_rank = round(winner_rank + (k * (1 - odds)))
    new_rank_diff = new_winner_rank - winner_rank
    new_loser_rank = loser_rank - new_rank_diff

    return new_winner_rank, new_loser_rank

def  build_dataSet(all_data):
    print("Building data set..")
    for index, row in all_data.iterrows():

        Wteam = row['WTeam']
        Lteam = row['LTeam']

        # get the initial elo rating of each team
        team1_elo = get_elo(Wteam)
        team2_elo = get_elo(Lteam)

        # add 100 rating to the home team
        if row['WLoc'] == 'H':
            team1_elo += 100
        else:
            team2_elo += 100

        # set elo rating as the first feature
        team1_features = [team1_elo]
        team2_features = [team2_elo]

        # add the stats of each team
        for key, value in team_stats.loc[Wteam].iteritems():
            team1_features.append(value)
        for key, value in team_stats.loc[Lteam].iteritems():
            team2_features.append(value)

        # randomly set the order of features of each game for two team
        # give the corrsponding label
        # 0 as the latter team lose
        # 1 as the latter team win
        if random.random() > 0.5:
            X.append(team1_features + team2_features)
            y.append(0)
        else:
            X.append(team2_features + team1_features)
            y.append(1)

        # update the elo rating
        new_winner_rank, new_loser_rank = calc_elo(Wteam, Lteam)
        team_elos[Wteam] = new_winner_rank
        team_elos[Lteam] = new_loser_rank

    return np.nan_to_num(X), np.array(y)

def predict_winner(team_1, team_2, model):
    features = []

    # team 1，visitor team
    features.append(get_elo(team_1))
    for key, value in team_stats.loc[team_1].iteritems():
        features.append(value)

    # team 2，home team
    features.append(get_elo(team_2) + 100)
    for key, value in team_stats.loc[team_2].iteritems():
        features.append(value)

    features = np.nan_to_num(features)
    return model.predict_proba([features])


# In[65]:


def combine_data(folder, season1, season2):
    M_name = 'MiscellaneousStats'
    O_name = 'OpponentPerGameStats'
    T_name = 'TeamPerGameStats'
    file_format = 'csv'
    M1 = pd.read_csv("{}/{}_{}.{}".format(folder, M_name, season1, file_format))
    M2 = pd.read_csv("{}/{}_{}.{}".format(folder, M_name, season2, file_format))
    O1 = pd.read_csv("{}/{}_{}.{}".format(folder, O_name, season1, file_format))
    O2 = pd.read_csv("{}/{}_{}.{}".format(folder, O_name, season2, file_format))
    T1 = pd.read_csv("{}/{}_{}.{}".format(folder, T_name, season1, file_format))
    T2 = pd.read_csv("{}/{}_{}.{}".format(folder, T_name, season2, file_format))
    Mstat = M1.append(M2, ignore_index=True)
    Ostat = O1.append(O2, ignore_index=True)
    Tstat = T1.append(T2, ignore_index=True)
    return Mstat, Ostat, Tstat

def combine_results(folder, season1, season2):
    r_name = 'result'
    file_format = 'csv'
    r1 = pd.read_csv("{}/{}_{}.{}".format(folder, r_name, season1, file_format))
    r2 = pd.read_csv("{}/{}_{}.{}".format(folder, r_name, season2, file_format))
    # print(r2)
    result_data = r1.append(r2, ignore_index=True)
    return result_data


# In[67]:


if __name__ == '__main__':
    '''
    Mstat = pd.read_csv(folder + '/MiscellaneousStats.csv')
    Ostat = pd.read_csv(folder + '/OpponentPerGameStats.csv')
    Tstat = pd.read_csv(folder + '/TeamPerGameStats.csv')
    '''
    folder = '/Users/chaoyiyang/Documents/Course/CSE546-Machine Learning-18Au/Project/Data/UW_CSE546_Project/ELO Rating/training_data'
    season1 = '1718'
    season2 = '1819'
    Mstat, Ostat, Tstat = combine_data(folder, season1, season2)

    team_stats = initialize_data(Mstat, Ostat, Tstat)
    
    # result_data = pd.read_csv(folder + '/2017-18_result.csv')
    result_data = combine_results(folder, season1, season2)
    X_data, y_data = build_dataSet(result_data)

    # tarin the model
    print("Fitting on %d game samples.." % len(X))

    model = LogisticRegression()
    model.fit(X_data, y_data)

    # cross validation
    print("Doing cross-validation..")
    print(cross_val_score(model, X_data, y_data, cv = 10, scoring='accuracy', n_jobs=-1).mean())


    # predict on the new season
    print('Predicting on new schedule..')
    test_file = 'test_1819'
    test_format = 'csv'
    schedule1819 = pd.read_csv("{}/{}.{}".format(folder, test_file, test_format))
    result = []
    for index, row in schedule1819.iterrows():
        team1 = row['VTeam']
        team2 = row['HTeam']
        pred = predict_winner(team1, team2, model)
        prob = pred[0][0]
        if prob > 0.5:
            winner = team1
            loser = team2
            result.append([winner, loser, prob])
        else:
            winner = team2
            loser = team1
            result.append([winner, loser, 1 - prob])
    
    result_name = '1819_test_Result.csv'
    with open("{}/{}".format(folder, result_name), 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['win', 'lose', 'probability'])
        writer.writerows(result)

