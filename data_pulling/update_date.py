import pandas as pd

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
for team_name in teams:
    input_dir = 'team_game_logs'
    input_name = 'team_game_log'
    input_type = '.xlsx'
    df = pd.read_excel("{}/{}_{}{}".format(input_dir, input_name, team_name, input_type))
    df['date'] = pd.to_datetime(df['game.startTime'].str.split('T').str[0])
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    output_dir = 'date_time'
    df.to_excel("{}/{}_{}{}".format(output_dir, input_name, team_name, input_type), sheet_name='Sheet1')