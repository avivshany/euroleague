import requests
import lxml.html as lh
import pandas as pd
import math

_teams_codes_map_general = {
    'Anadolu Efes Istanbul': 'IST',
    'CSKA Moscow': 'CSK',
    'Maccabi FOX Tel Aviv': 'TEL',
    'Olympiacos Piraeus': 'OLY',
    'Real Madrid': 'MAD',
    'Zalgiris Kaunas': 'ZAL'
}
_teams_codes_map_2016 = {
    'EA Emporio Armani Milan': 'MIL',
    'Panathinaikos Superfoods Athens': 'PAN',
    'Baskonia Vitoria Gasteiz': 'BAS',
    'Fenerbahce Istanbul': 'ULK',
    'Darussafaka Dogus Istanbul': 'DAR',
    'FC Barcelona Lassa': 'BAR',
    'Brose Bamberg': 'BAM',
    'Crvena Zvezda mts Belgrade': 'RED',
    'Galatasaray Odeabank Istanbul': 'GAL',
    'Unics Kazan': 'UNK'
}
_teams_codes_map_2017 = {
    'AX Armani Exchange Olimpia Milan': 'MIL',
    'Panathinaikos Superfoods Athens': 'PAN',
    'KIROLBET Baskonia Vitoria Gasteiz': 'BAS',
    'Fenerbahce Dogus Istanbul': 'ULK',
    'Khimki Moscow Region': 'KHI',
    'FC Barcelona Lassa': 'BAR',
    'Brose Bamberg': 'BAM',
    'Crvena Zvezda mts Belgrade': 'RED',
    'Unicaja Malaga': 'MAL',
    'Valencia Basket': 'PAM'
}
_teams_codes_map_2018 = {
    'AX Armani Exchange Olimpia Milan': 'MIL',
    'Panathinaikos OPAP Athens': 'PAN',
    'KIROLBET Baskonia Vitoria-Gasteiz': 'BAS',
    'Fenerbahce Beko Istanbul': 'ULK',
    'Khimki Moscow Region': 'KHI',
    'FC Barcelona Lassa': 'BAR',
    'Buducnost VOLI Podgorica': 'BUD',
    'Darussafaka Tekfen Istanbul': 'DAR',
    'FC Bayern Munich': 'MUN',
    'Herbalife Gran Canaria': 'CAN'
}
_teams_codes_map_2019 = {
    'AX Armani Exchange Milan': 'MIL',
    'Panathinaikos OPAP Athens': 'PAN',
    'KIROLBET Baskonia Vitoria-Gasteiz': 'BAS',
    'Fenerbahce Beko Istanbul': 'ULK',
    'Khimki Moscow Region': 'KHI',
    'FC Barcelona': 'BAR',
    'Crvena Zvezda mts Belgrade': 'RED',
    'Valencia Basket': 'PAM',
    'ALBA Berlin': 'BER',
    'LDLC ASVEL Villeurbanne': 'ASV',
    'FC Bayern Munich': 'MUN',
    'Zenit St Petersburg': 'DYR'
}
teams_codes_map_all = {
    **_teams_codes_map_general, **_teams_codes_map_2016,
    **_teams_codes_map_2017, **_teams_codes_map_2018, **_teams_codes_map_2019
}
teams_names_codes_map = {
    2016: dict(_teams_codes_map_general, **_teams_codes_map_2016),
    2017: dict(_teams_codes_map_general, **_teams_codes_map_2017),
    2018: dict(_teams_codes_map_general, **_teams_codes_map_2018),
    2019: dict(_teams_codes_map_general, **_teams_codes_map_2019)
}
_n_teams_per_season = {
    season: len(codes_map)
    for season, codes_map in teams_names_codes_map.items()
}
recognizable_team_codes = {
    'MAD': 'RMD', 'TEL': 'MTA', 'PAM': 'VAL', 'PAN': 'PAO',
    'ULK': 'FNR', 'DYR': 'ZEN', 'IST': 'EFS', 'UNK': 'KAZ'
}


def _scrape_data(url):
    """Scrape table data from a web page"""
    page = requests.get(url)           # Create a handle for page
    doc = lh.fromstring(page.content)  # Store page content
    tr_elements = doc.xpath('//tr')    # Parse data between <tr>..</tr> in HTML

    return tr_elements


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


def _get_possessions(df_row):
    """Get number of possessions per team in game. to use on games_stats df"""
    poss = df_row['2PA'] + df_row['3PA'] + (0.44 * df_row['FTA']) -\
        df_row['OREB'] + df_row['TOV']

    return poss


def get_games_stats(season, completed_rounds, rounds_already_scraped=0):
    """
    Loop over all games in a season, scrape teams total stats, parse the data
    and return in a tidy dataframe where each row is a team in a game

    :param season: integer indicating season start year
    :param completed_rounds: number of rounds played in the season
    :param rounds_already_scraped: int, optional. rounds previously scraped.
        if above 0 will start scraping from the next round
    :return: pandas.DataFrame
    """
    games_stats = pd.DataFrame()
    games_per_round = _n_teams_per_season[season] / 2
    n_games = int(completed_rounds * games_per_round)
    first_game_to_scrape = int(rounds_already_scraped * games_per_round) + 1

    # loop over games
    for game_code in range(first_game_to_scrape, n_games + 1):

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
        'season', 'round', 'PTS', '2PM', '2PA', '3PM', '3PA', 'FTM', 'FTA',
        'OREB', 'DREB', 'REB', 'AST', 'STL', 'TOV', 'BLK', 'PF', 'PIR', 'MTS'
    ]

    for col in int_cols:
        games_stats[col] = games_stats[col].astype(int)

    # calculate possessions - baseline for advanced stats
    games_stats['POSS'] = games_stats.apply(_get_possessions, axis=1)

    # replace team codes with more recognizable versions
    games_stats['team'].replace(recognizable_team_codes, inplace=True)
    games_stats['game_id'].replace(recognizable_team_codes,
                                   inplace=True, regex=True)
    games_stats.sort_values(by='round', inplace=True)

    return games_stats
