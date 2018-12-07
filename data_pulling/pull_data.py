import json 
import pandas as pd 
from pandas.io.json import json_normalize
from ohmysportsfeedspy import MySportsFeeds

# get all the NBA team abbreviations
# teams1 = ['ATL', 'BOS', 'BRO', 'CHA', 'CHI', 'CLE', 'DAL', 'DEN', 'DET', 'GSW', 'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA', 'MIL', 'MIN', 'NOP', 'NYK', 'OKL', 'ORL', 'PHI', 'PHX', 'POR', 'SAC', 'SAS', 'TOR', 'UTA', 'WAS']
teams_all = '''ATL
BOS
BRO
CHA
CHI
CLE
DAL
DEN
DET
GSW
HOU
IND
LAC
LAL
MEM
MIA
MIL
MIN
NOP
NYK
OKL
ORL
PHI
PHX
POR
SAC
SAS
TOR
UTA
WAS
'''
teams = teams_all.splitlines()

# set up data query for mysports feed
Data_query = MySportsFeeds(version='2.0',verbose=True)
Data_query.authenticate('3532c47b-606d-4b5a-b0a9-a736c2', 'MYSPORTSFEEDS')
for team_name in teams:
    print(team_name)
    Output = Data_query.msf_get_data(league='nba',season='2017-2018-regular',feed='seasonal_team_gamelogs',team=team_name,format='json')

    # process the json file
    dir = 'results'
    file_name = 'seasonal_team_gamelogs-nba-2017-2018-regular.json'
    with open("{}/{}".format(dir, file_name)) as f:
        d = json.load(f)
    df = pd.DataFrame.from_dict(json_normalize(d['gamelogs']), orient='columns')
    output_dir = 'team_game_logs'
    output_name = 'team_game_log'
    output_type = '.xlsx'
    df.to_excel("{}/{}_{}{}".format(output_dir, output_name, team_name, output_type), sheet_name='Sheet1')