from ohmysportsfeedspy import MySportsFeeds

'''
Data_query = MySportsFeeds('1.2',verbose=True)
Data_query.authenticate('YOUR_API_KEY', 'YOUR_ACCOUNT_PASSWORD')
Output = Data_query.msf_get_data(league='nba',season='2016-2017-regular',feed='player_gamelogs',format='json',player='stephen-curry',force='true')

print(Output)
'''
Data_query = MySportsFeeds(version='2.0',verbose=True)
Data_query.authenticate('3532c47b-606d-4b5a-b0a9-a736c2', 'MYSPORTSFEEDS')
'''
Output = Data_query.msf_get_data(league='nba',season='2016-2017-regular',feed='seasonal_player_gamelogs',format='json',player='stephen-curry')
'''
# Output = Data_query.msf_get_data(league='nba',season='2017-2018-regular',feed='seasonal_games',format='json')
# Output = Data_query.msf_get_data(league='nba',season='2018-2019-regular',feed='game_boxscore',date='20181115',game='20181115-ATL-DEN',format='json')
# Output = Data_query.msf_get_data(league='nba',season='2017-2018-regular',feed='seasonal_team_gamelogs',team='OKL',format='json')
Output = Data_query.msf_get_data(league='nba',season='2017-2018-regular',feed='seasonal_team_stats',format='json')
print(Output)

