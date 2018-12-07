
# coding: utf-8

# In[70]:


import pandas as pd
import math
import csv
import random
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression


# In[71]:


base_elo = 1600
team_elos = {} 
team_stats = {}
X = []
y = []
folder = '/Users/chaoyiyang/Documents/Course/CSE546-Machine Learning-18Au/Project/Data/' #directory to save the result


# In[72]:


# based on Miscellaneous Opponent，Team stats to initialize
def initialize_data(Mstat, Ostat, Tstat):
    new_Mstat = Mstat.drop(['Rk', 'Arena'], axis=1)
    new_Ostat = Ostat.drop(['Rk', 'G', 'MP'], axis=1)
    new_Tstat = Tstat.drop(['Rk', 'G', 'MP'], axis=1)

    team_stats1 = pd.merge(new_Mstat, new_Ostat, how='left', on='Team')
    team_stats1 = pd.merge(team_stats1, new_Tstat, how='left', on='Team')

    print(team_stats1.info())
    return team_stats1.set_index('Team', inplace=False, drop=True)


# In[73]:


# get the elo rating of each team
def get_elo(team):
    try:
        return team_elos[team]
    except:
        # initially set as base_elo
        team_elos[team] = base_elo
        return team_elos[team]


# In[74]:


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


# In[75]:


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


# In[76]:


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


# In[77]:


if __name__ == '__main__':

    Mstat = pd.read_csv(folder + '/MiscellaneousStats.csv')
    Ostat = pd.read_csv(folder + '/OpponentPerGameStats.csv')
    Tstat = pd.read_csv(folder + '/TeamPerGameStats.csv')

    team_stats = initialize_data(Mstat, Ostat, Tstat)
    
    result_data = pd.read_csv(folder + '/2017-18_result.csv')
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
    schedule1617 = pd.read_csv(folder + '/2018-19Schedule.csv')
    result = []
    for index, row in schedule1617.iterrows():
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

    with open('18-19Result.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['win', 'lose', 'probability'])
        writer.writerows(result)

