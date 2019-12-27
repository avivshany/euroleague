import requests
import lxml.html as lh
import numpy as np
import pandas as pd
import math

_teams_codes_map_general = {
    'Anadolu Efes Istanbul': 'IST',
    'CSKA Moscow': 'CSK',
    'FC Bayern Munich': 'MUN',
    'Fenerbahce Beko Istanbul': 'ULK',
    'Khimki Moscow Region': 'KHI',
    'KIROLBET Baskonia Vitoria-Gasteiz': 'BAS',
    'Maccabi FOX Tel Aviv': 'TEL',
    'Olympiacos Piraeus': 'OLY',
    'Real Madrid': 'MAD',
    'Panathinaikos OPAP Athens': 'PAN',
    'Zalgiris Kaunas': 'ZAL'
}
_teams_codes_map_2018 = {
    'AX Armani Exchange Olimpia Milan': 'MIL',
    'FC Barcelona Lassa': 'BAR',
    'Buducnost VOLI Podgorica': 'BUD',
    'Darussafaka Tekfen Istanbul': 'DAR',
    'Herbalife Gran Canaria': 'CAN'
}
_teams_codes_map_2019 = {
    'AX Armani Exchange Milan': 'MIL',
    'FC Barcelona': 'BAR',
    'Crvena Zvezda mts Belgrade': 'RED',
    'Valencia Basket': 'PAM',
    'ALBA Berlin': 'BER',
    'LDLC ASVEL Villeurbanne': 'ASV',
    'Zenit St Petersburg': 'DYR'
}
teams_codes_map_all = dict(_teams_codes_map_general, **_teams_codes_map_2018,
                           **_teams_codes_map_2019)
teams_names_codes_map = {
    2018: dict(_teams_codes_map_general, **_teams_codes_map_2018),
    2019: dict(_teams_codes_map_general, **_teams_codes_map_2019)
}
_n_teams_per_season = {
    season: len(codes_map)
    for season, codes_map in teams_names_codes_map.items()
}
_recognizable_team_codes = {
    'MAD': 'RMD', 'TEL': 'MTA', 'PAM': 'VAL',
    'ULK': 'FNR', 'DYR': 'ZEN', 'IST': 'EFS'
}


def _scrape_data(url):
    """Scrape table data from a web page"""
    page = requests.get(url)           # Create a handle for page
    doc = lh.fromstring(page.content)  # Store page content
    tr_elements = doc.xpath('//tr')    # Parse data between <tr>..</tr> in HTML

    return tr_elements


def _scrape_team_data(team_code, season):
    """
    Scrape a team's games data for url defined by team code and season

    :param team_code: 3 letter team character code
    :param season: year of euroleague season start for which to scrape data
    :return: list of lxml.html.HtmlElement objects
    """
    teams_main_url = 'https://www.euroleague.net/competition/teams/showteam?'
    team_season_url = 'clubcode={}&seasoncode=E{}'.format(team_code, season)
    url = teams_main_url + team_season_url
    team_games_data = _scrape_data(url)

    return team_games_data


def get_teams_data(season, teams_codes=None,
                   names_codes_map=teams_names_codes_map):
    """
    Loop over all teams in a specific euroleague season and scrape their stats

    :param season: integer indicating season start year
    :param teams_codes: list, optional. codes of specific teams to scrape
    :param names_codes_map: dict, used to get all team codes if teams_codes=None
        nested dict where top level keys are year integers indicating the
        seasons start year, and  values are dictionaries where keys are full
        names of all teams in the relevant season, and values are their codes.

    :return: dict. keys are team codes and values are lists of
        lxml.html.HtmlElement objects with their stats data for that season
    """
    teams_data = dict()
    use_all_teams = False if teams_codes else True

    if use_all_teams:
        teams_codes = names_codes_map[season].values()

    for team in teams_codes:
        teams_data[team] = _scrape_team_data(team_code=team, season=season)

    return teams_data


def _clean_row(page_tables_row, remove_empty_elements=True):
    """
    Remove special formatting characters from web page tables data

    :param page_tables_row: lxml.html.HtmlElement object
    :param remove_empty_elements: bool. if False keeps number of elements as is
    :return: list of all elements in page_tables_row as strings
    """
    text = page_tables_row.text_content()
    stripped = [word.strip() for word in text.split("\r\n", )]

    if remove_empty_elements:
        stripped = [word for word in stripped if word != '']

    cleaned = [word.replace(u'\xa0', u' ') for word in stripped]

    return cleaned


def _raw_team_name_to_code(team_name_table_row,
                           names_codes_map=teams_codes_map_all):
    """
    Clean full team name from html table row and replace it with team code

    :param team_name_table_row: lxml.html.HtmlElement object
    :param names_codes_map: dict. keys are team names, values are team codes
    :return: 3 letter team character code
    """
    text = team_name_table_row.text_content()
    team_name = [char for char in text if char not in ['\r', '\n', '\t']]
    team_name = ''.join([char for char in team_name if not char.isdigit()])
    team_code = names_codes_map[team_name]

    return team_code


def _get_home_team(df_row):
    """Get home team for game. games df must include a 'location' column"""
    if df_row['location'] == 'home':
        home_team = df_row['team']
    elif df_row['location'] == 'away':
        home_team = df_row['opponent']
    else:
        home_team = np.nan

    return home_team


def _get_away_team(df_row):
    """Get away team for game. games df must include a 'location' column"""
    if df_row['location'] == 'home':
        away_team = df_row['opponent']
    elif df_row['location'] == 'away':
        away_team = df_row['team']
    else:
        away_team = np.nan

    return away_team


def _get_winner(df_row):
    """Get game's winning team. df must include {home, away}_score columns"""
    if df_row['home_score'] > df_row['away_score']:
        winner = df_row['home_team']
    elif df_row['home_score'] < df_row['away_score']:
        winner = df_row['away_team']
    else:
        winner = np.nan

    return winner


def _get_possessions(df_row):
    """Get number of possessions per team in game. to use on games_stats df"""
    poss = df_row['2PA'] + df_row['3PA'] + (0.44 * df_row['FTA']) -\
        df_row['OREB'] + df_row['TOV']

    return poss


def get_games_scores(season, teams_data):
    """
    Loop over all teams in a season, scrape scores from each game,
    parse tha data, and return in clean dataframe

    :param season: integer indicating season start year
    :param teams_data: dict. output of get_teams_data()
    :return: pandas.DataFrame
    """
    website_games_columns = ['round', 'outcome', 'location_opponent', 'score']
    games = pd.DataFrame()
    team_codes = teams_data.keys()
    n_rounds = 2 * (_n_teams_per_season[season] - 1)

    # loop over teams
    for team in team_codes:
        team_games = pd.DataFrame(
            columns=website_games_columns + ['team', 'season']
        )
        tr_elements = teams_data[team]

        # loop over games
        for game_num, game_score in enumerate(tr_elements[:n_rounds]):
            curr_game = _clean_row(game_score)

            # rows with 3 elements represent games that were not played yet
            if len(curr_game) > 3:
                team_games.loc[game_num, website_games_columns] = curr_game

        # join team games to all games
        team_games['team'] = team
        team_games['season'] = season
        games = pd.concat([games, team_games])

    # split columns to create a single column for each data point
    games[['location', 'opponent']] = games['location_opponent'].str\
        .split(' ', n=1, expand=True)
    games['location'].replace({'at': 'away', 'vs': 'home'}, inplace=True)
    games[['home_score', 'away_score']] = games['score'].str\
        .split(' - ', expand=True)

    # replace opponent teams names with their code
    games['opponent'] = games['opponent'].replace(teams_codes_map_all)

    # set correct dtype for integer columns
    int_cols = ['round', 'home_score', 'away_score']
    games[int_cols] = games[int_cols].astype(int)

    # assign home and away team columns instead of team/opponent columns
    games['home_team'] = games.apply(_get_home_team, axis=1)
    games['away_team'] = games.apply(_get_away_team, axis=1)
    games['home_team'] = games['home_team'].replace(_recognizable_team_codes)
    games['away_team'] = games['away_team'].replace(_recognizable_team_codes)
    games['winner'] = games.apply(_get_winner, axis=1)

    # create a game id column
    games['game_id'] = games['season'].map(str) + '_' + games['round'].map(str)\
        + '_' + games['home_team'] + '_' + games['away_team']

    # select relevant columns in the right order
    games = games[['game_id', 'season', 'round', 'home_team', 'away_team',
                   'home_score', 'away_score', 'winner']]

    # drop duplicated rows and sort
    games.drop_duplicates(inplace=True)
    games.sort_values(by='round', inplace=True)

    return games


def get_games_stats(season, completed_rounds):
    """
    Loop over all games in a season, scrape teams total stats, parse the data
    and return in a clean dataframe

    :param season: integer indicating season start year
    :param completed_rounds: number of rounds played in the season
    :return: pandas.DataFrame
    """
    games_stats = pd.DataFrame()
    games_per_round = _n_teams_per_season[season] / 2
    n_games = int(completed_rounds * games_per_round)

    # loop over games
    for game_code in range(1, n_games + 1):

        # get game stats
        main_games_url = 'https://www.euroleague.net/main/results/showgame'
        game_url = '?gamecode={}&seasoncode=E{}'.format(game_code, season)
        url = main_games_url + game_url
        tr_elements = _scrape_data(url)

        # get round number
        round_num = math.ceil(game_code / games_per_round)

        # team codes - 1st element for home team and 2nd for away team
        # work around because the website sometimes has unexpected behavior
        if 'By Quarter' in tr_elements[1].text_content():
            teams_element_nums = [2, 3]
            print('By Quarter in element 1')
            print('round: {}'.format(round_num))

            for element_num in range(5):
                print(tr_elements[element_num].text_content())

        elif 'By Quarter' in tr_elements[2].text_content():
            teams_element_nums = [3, 4]
            print('By Quarter in element 2')
            print('round: {}'.format(round_num))

            for element_num in range(5):
                print(tr_elements[element_num].text_content())

        else:
            teams_element_nums = [1, 2]

        curr_game_team_codes = [
            _raw_team_name_to_code(tr_elements[teams_element_nums[0]]),
            _raw_team_name_to_code(tr_elements[teams_element_nums[1]])
        ]

        # create empty dataframe for game stats
        game_stats_cols = [
            'Player', 'Min', 'Pts', '2FG', '3FG', 'FT', 'O', 'D',
            'T', 'As', 'St', 'To', 'Fv', 'Ag', 'Cm', 'Rv', 'PIR'
        ]
        added_cols = ['location', 'team', 'game_id', 'season', 'round']
        curr_game_stats = pd.DataFrame(
            columns=game_stats_cols + added_cols)
        location_counter = 0

        # loop over rows in page tables
        for table_stats_row in tr_elements:
            curr_row = _clean_row(table_stats_row)

            # find the 2 rows that have teams stats for the current game
            if curr_row[0] == 'Totals':

                # first team to appear is the home team
                location = ['home_team', 'away_team'][location_counter]
                team = curr_game_team_codes[location_counter]

                # set game id as: 'season_round_home-team_away-team'
                game_id = '{}_{}_{}_{}'.format(
                    season, round_num,
                    curr_game_team_codes[0], curr_game_team_codes[1]
                )
                # insert added columns and table columns into dataframe
                curr_row.extend([location, team, game_id, season, round_num])
                curr_game_stats.loc[location_counter, :] = curr_row

                location_counter += 1

        # concatenate teams stats from current game in all games stats
        games_stats = pd.concat([games_stats, curr_game_stats])

    # strip shooting columns
    games_stats[['2PM', '2PA']] = games_stats['2FG'].str.split('/', expand=True)
    games_stats[['3PM', '3PA']] = games_stats['3FG'].str.split('/', expand=True)
    games_stats[['FTM', 'FTA']] = games_stats['FT'].str.split('/', expand=True)

    # rename and select relevant columns
    games_stats = games_stats.rename(columns={
        'O': 'OREB', 'D': 'DREB', 'T': 'REB', 'As': 'AST', 'To': 'TOV',
        'Pts': 'PTS', 'St': 'STL', 'Fv': 'BLK', 'Ag': 'BLKA', 'Cm': 'PF',
        'Rv': 'PFD', 'Min': 'MTS'
    })
    games_stats = games_stats[[
        'game_id', 'season', 'round', 'team', 'location', 'MTS', 'PTS', '2PM',
        '2PA', '3PM', '3PA', 'FTM', 'FTA', 'OREB', 'DREB', 'REB', 'AST', 'STL',
        'TOV', 'BLK', 'BLKA', 'PF', 'PFD', 'PIR'
    ]]

    # set columns to the right dtype
    games_stats['MTS'] = games_stats['MTS'].str.split(':', expand=True).iloc[:, 0]
    int_cols = [
        'PTS', '2PM', '2PA', '3PM', '3PA', 'FTM', 'FTA', 'OREB',
        'DREB', 'REB', 'AST', 'STL', 'TOV', 'BLK', 'PF', 'PIR', 'MTS'
    ]

    for col in int_cols:
        games_stats[col] = games_stats[col].astype(int)

    # calculate possessions - baseline for advanced stats
    games_stats['POSS'] = games_stats.apply(_get_possessions, axis=1)

    games_stats.sort_values(by='round', inplace=True)

    return games_stats
