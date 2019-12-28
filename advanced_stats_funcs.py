import pandas as pd
import scrape_parse_funcs as sp


def get_overall_teams_opponents_stats(
        games_stats_df, season, team_codes=None, game_id_colname='game_id',
        team_colname='team', season_colname='season', opponent_vars_prefix='OP_'
):
    """
    Sum all box score stats for teams and their opponents per season

    :param games_stats_df: dataframe with teams scraped box-score stats
        from each game in a euroleague season
    :param season: integer indicating season start year
    :param team_codes: list, optional. specific team codes to calculate vars for
    :param game_id_colname: str. column name of game id in games_stats_df
    :param team_colname: str. team column name in games_stats_df
    :param season_colname: str. season column name in games_stats_df
    :param opponent_vars_prefix: str. prefix to add to opponents stats

    :return: pandas.DataFrame with a row for each team, containing the sum of
     all their, and their opponents, box scores stats for the season

    """
    teams_stats = pd.DataFrame()

    if not team_codes:
        team_codes = pd.Series(list(sp.teams_names_codes_map[season].values()))
        team_codes.replace(sp.recognizable_team_codes, inplace=True)

    for team_code in team_codes:

        # create df with team's stats sum
        team_mask_1 = (games_stats_df[game_id_colname].str.contains(team_code))
        team_mask_2 = (games_stats_df[team_colname] == team_code)
        season_mask = (games_stats_df[season_colname] == season)
        team_own_stats = games_stats_df.loc[
            team_mask_1 & team_mask_2 & season_mask
        ].select_dtypes(exclude='object')\
            .drop(['season', 'round'], axis=1)\
            .sum()\
            .to_frame().T

        # create df with team's opponents stats sum
        opp_mask_1 = (games_stats_df[game_id_colname].str.contains(team_code))
        opp_mask_2 = (games_stats_df[team_colname] != team_code)
        opponents_stats = games_stats_df.loc[
            opp_mask_1 & opp_mask_2 & season_mask
        ]\
            .select_dtypes(exclude='object')\
            .drop(['season', 'round'], axis=1)\
            .sum()\
            .to_frame().T
        opponents_stats.columns = [
            opponent_vars_prefix + colname
            for colname in opponents_stats.columns
        ]
        # concat team's and their opponents stats sums
        curr_team_stats = pd.concat([team_own_stats, opponents_stats], axis=1)
        curr_team_stats[team_colname] = team_code
        curr_team_stats[season_colname] = season

        # concat with teams_stats
        teams_stats = pd.concat([teams_stats, curr_team_stats])

    return teams_stats


def get_team_advanced_stats(df):
    """
    Calculate advanced stats and to teams_stats dataframe

    :param df: dataframe. output of get_overall_teams_opponents_stats()
    :return: pandas.DataFrame
    """
    # calculate field goal attempts and estimated chances
    df['FGA'] = df['2PA'] + df['3PA']
    df['CHANCES'] = df['FGA'] + (0.44 * df['FTA'])
    df['OP_FGA'] = df['OP_2PA'] + df['OP_3PA']
    df['OP_CHANCES'] = df['OP_FGA'] + (0.44 * df['OP_FTA'])

    # calculate percentage stats indicating shooting/rebounding efficiency
    df['3P%'] = 100 * df['3PM'] / df['3PA']
    df['OP_3P%'] = 100 * df['OP_3PM'] / df['OP_3PA']
    df['2P%'] = 100 * df['2PM'] / df['2PA']
    df['OP_2P%'] = 100 * df['OP_2PM'] / df['OP_2PA']
    df['FT%'] = 100 * df['FTM'] / df['FTA']
    df['eFG%'] = 100 * (df['2PM'] + (1.5 * df['3PM'])) / df['FGA']
    df['OP_eFG%'] = 100 * (df['OP_2PM'] + (1.5 * df['OP_3PM'])) / df['OP_FGA']
    df['TS%'] = 100 * df['PTS'] / (2 * df['CHANCES'])
    df['OP_TS%'] = 100 * df['OP_PTS'] / (2 * df['OP_CHANCES'])
    df['OREB%'] = 100 * df['OREB'] / (df['OREB'] + df['OP_DREB'])
    df['DREB%'] = 100 * df['DREB'] / (df['DREB'] + df['OP_OREB'])

    # calculate rate stats indicating style of play
    df['3PR'] = 100 * df['3PA'] / (df['FGA'])
    df['OP_3PR'] = 100 * df['OP_3PA'] / df['OP_FGA']
    df['FTR'] = 100 * df['FTA'] / (df['FGA'])
    df['OP_FTR'] = 100 * df['OP_FTA'] / df['OP_FGA']
    df['ASTR'] = 100 * df['AST'] / (df['2PM'] + df['3PM'])
    df['OP_ASTR'] = 100 * df['OP_AST'] / (df['OP_2PM'] + df['OP_3PM'])
    df['TOVR'] = 100 * df['TOV'] / df['POSS']
    df['OP_TOVR'] = 100 * df['OP_TOV'] / df['OP_POSS']
    df['AST-TOV_R'] = df['AST'] / df['TOV']
    df['OP_AST-TOV_R'] = df['OP_AST'] / df['OP_TOV']
    df['STLR'] = 100 * df['STL'] / df['OP_POSS']
    df['OP_STLR'] = 100 * df['OP_STL'] / df['POSS']
    df['BLKR'] = 100 * df['BLK'] / df['OP_2PA']
    df['OP_BLKR'] = 100 * df['OP_BLK'] / df['2PA']

    # calculate pace and rating stats indicating overall team efficiency
    df['PTS40'] = 40 * 5 * df['PTS'] / df['MTS']
    df['OP_PTS40'] = 40 * 5 * df['OP_PTS'] / df['OP_MTS']
    df['PACE'] = 40 * (df['POSS'] + df['OP_POSS']) / (2 * (df['MTS'] / 5))
    df['ORtg'] = 100 * df['PTS'] / df['POSS']
    df['DRtg'] = 100 * df['OP_PTS'] / df['OP_POSS']
    df['NETRtg'] = df['ORtg'] - df['DRtg']

    cols_to_use = [
        'team', 'PTS40', 'OP_PTS40', '3P%', 'OP_3P%', '2P%', 'OP_2P%', 'FT%',
        '3PR', 'OP_3PR', 'FTR', 'OP_FTR', 'OREB%', 'DREB%', 'ASTR', 'OP_ASTR',
        'TOVR', 'OP_TOVR', 'AST-TOV_R', 'OP_AST-TOV_R', 'STLR', 'OP_STLR',
        'BLKR', 'OP_BLKR', 'PACE', 'ORtg', 'DRtg', 'NETRtg', 'eFG%', 'OP_eFG%',
        'TS%', 'OP_TS%'
    ]

    return df[cols_to_use]
