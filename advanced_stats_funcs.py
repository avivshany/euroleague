import pandas as pd
from pandasql import sqldf
import scrape_parse_funcs as sp


def get_home_advantage_vars(games_stats):
    """
    Use SQL queries to calculate percentage of wins home and away for each team

    :param games_stats: dataframe with teams scraped box-score stats
        from each game in a euroleague season
    :return: pandas.DataFrame
    """
    # write query to create df containing teams, and wins by location per game
    game_location_data = sqldf("""
    SELECT h.game_id,
        h.team AS home_team,
        a.team AS away_team,
        h.PTS AS home_points,
        a.PTS AS away_points,
        (h.POSS + a.POSS) / 2 AS POSS,
        CASE WHEN h.PTS > a.PTS THEN 1 ELSE 0 END AS home_win,
        CASE WHEN h.PTS < a.PTS THEN 1 ELSE 0 END AS away_win
    FROM (SELECT * FROM games_stats WHERE location='home_team') AS h
    LEFT JOIN (SELECT * FROM games_stats WHERE location='away_team') AS a
        ON h.game_id = a.game_id
        AND h.team != a.team
    """)

    # create a df summarising wins per team by location
    wins_by_location = sqldf("""
    WITH home_wins AS(
        SELECT home_team,
            SUM(home_win) AS home_wins,
            SUM(away_win) AS home_losses,
            COUNT(home_win) AS home_games,
            100 * SUM(home_win) / COUNT(home_win) AS home_win_pct
        FROM game_location_data
        GROUP BY home_team
    ),
    away_wins AS (
        SELECT away_team,
            SUM(home_win) AS away_losses,
            SUM(away_win) AS away_wins,
            COUNT(away_win) AS away_games,
            100 * SUM(away_win) / COUNT(away_win) AS away_win_pct
        FROM game_location_data
        GROUP BY away_team
    )

    SELECT hw.home_team AS team,
        hw.home_win_pct,
        aw.away_win_pct,
        100 * (hw.home_wins + aw.away_wins) / (hw.home_games + aw.away_games) AS win_pct,
        hw.home_win_pct - aw.away_win_pct AS home_win_advantage
    FROM home_wins AS hw
    JOIN away_wins AS aw
        ON hw.home_team = aw.away_team
    """)

    # create a df summarising net rating per team by location
    rating_by_location = sqldf("""
    WITH home_team_ratings AS(
        SELECT home_team,
            100 * SUM(home_points) / SUM(POSS) AS home_ORtg,
            100 * SUM(away_points) / SUM(POSS) AS home_DRtg,
            100 * (SUM(home_points) / SUM(POSS)) - (SUM(away_points) / SUM(POSS)) AS home_NETRtg
        FROM game_location_data
        GROUP BY home_team
    ),
    away_team_ratings AS(
       SELECT away_team,
            100 * SUM(away_points) / SUM(POSS) AS away_ORtg,
            100 * SUM(home_points) / SUM(POSS) AS away_DRtg,
            100 * (SUM(away_points) / SUM(POSS)) - (SUM(home_points) / SUM(POSS)) AS away_NETRtg
        FROM game_location_data
        GROUP BY away_team
    )

    SELECT htr.home_team AS team,
        htr.home_ORtg,
        htr.home_DRtg,
        htr.home_NETRtg,
        atr.away_ORtg,
        atr.away_DRtg,
        atr.away_NETRtg,
        htr.home_NETRtg - atr.away_NETRtg AS home_NETRtg_advantage,
        htr.home_ORtg - atr.away_ORtg AS home_ORtg_advantage,
        atr.away_DRtg - htr.home_DRtg AS home_DRtg_advantage
    FROM home_team_ratings AS htr
    JOIN away_team_ratings AS atr
        ON htr.home_team = atr.away_team
    """)

    # join all location related variables into one dataframe
    home_advantage_df = wins_by_location.merge(rating_by_location, on='team')

    return home_advantage_df


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


def get_advanced_stats(df, opponents_stats=True):
    """
    Calculate advanced stats and to teams_stats dataframe

    :param df: dataframe. output of get_overall_teams_opponents_stats()
    :param opponents_stats: bool
        if True, assumes that teams opponents stats also included in df and
        calculates more stats and a more stable estimation of possessions
    :return: pandas.DataFrame
    """
    # calculate field goal attempts and estimated chances
    df['FGA'] = df['2PA'] + df['3PA']
    df['CHANCES'] = df['FGA'] + (0.44 * df['FTA'])

    # calculate percentage stats indicating shooting efficiency
    df['3P%'] = 100 * df['3PM'] / df['3PA']
    df['2P%'] = 100 * df['2PM'] / df['2PA']
    df['FT%'] = 100 * df['FTM'] / df['FTA']
    df['eFG%'] = 100 * (df['2PM'] + (1.5 * df['3PM'])) / df['FGA']
    df['TS%'] = 100 * df['PTS'] / (2 * df['CHANCES'])

    # calculate rate stats indicating style of play
    df['3PR'] = 100 * df['3PA'] / (df['FGA'])
    df['FTR'] = 100 * df['FTA'] / (df['FGA'])
    df['ASTR'] = 100 * df['AST'] / (df['2PM'] + df['3PM'])
    df['TOVR'] = 100 * df['TOV'] / df['POSS']
    df['AST-TOV_R'] = df['AST'] / df['TOV']

    # calculate pace and rating stats indicating overall team efficiency
    df['PTS40'] = 40 * 5 * df['PTS'] / df['MTS']
    df['PACE'] = (40 * df['POSS']) / (df['MTS'] / 5)
    df['ORtg'] = 100 * df['PTS'] / df['POSS']

    cols_to_use = [
        'team', 'PTS40', 'PTS', '3P%', '2P%', 'FT%', '3PR', 'FTR',
        'ASTR', 'TOVR', 'AST-TOV_R', 'PACE', 'ORtg', 'eFG%', 'TS%'
    ]

    # if opponents stats exists calculate more advanced stats
    if opponents_stats:

        # calculate a better estimation for pace, as mean pace of both teams
        df['PACE'] = 40 * (df['POSS'] + df['OP_POSS']) / (2 * (df['MTS'] / 5))

        # calculate field goal attempts and estimated chances
        df['OP_FGA'] = df['OP_2PA'] + df['OP_3PA']
        df['OP_CHANCES'] = df['OP_FGA'] + (0.44 * df['OP_FTA'])

        # calculate rebounding efficiency
        df['OREB%'] = 100 * df['OREB'] / (df['OREB'] + df['OP_DREB'])
        df['DREB%'] = 100 * df['DREB'] / (df['DREB'] + df['OP_OREB'])

        # calculate opponents % stats indicating shooting efficiency
        df['OP_3P%'] = 100 * df['OP_3PM'] / df['OP_3PA']
        df['OP_2P%'] = 100 * df['OP_2PM'] / df['OP_2PA']
        df['OP_eFG%'] = 100 * (df['OP_2PM'] + (1.5 * df['OP_3PM'])) / df['OP_FGA']
        df['OP_TS%'] = 100 * df['OP_PTS'] / (2 * df['OP_CHANCES'])

        # calculate ratings stats
        df['DRtg'] = 100 * df['OP_PTS'] / df['OP_POSS']
        df['NETRtg'] = df['ORtg'] - df['DRtg']

        # calculate defensive rate stats indicating style of play
        df['STLR'] = 100 * df['STL'] / df['OP_POSS']
        df['BLKR'] = 100 * df['BLK'] / df['OP_2PA']

        # calculate opponents rate stats indicating defensive style of play
        df['OP_3PR'] = 100 * df['OP_3PA'] / df['OP_FGA']
        df['OP_FTR'] = 100 * df['OP_FTA'] / df['OP_FGA']
        df['OP_ASTR'] = 100 * df['OP_AST'] / (df['OP_2PM'] + df['OP_3PM'])
        df['OP_TOVR'] = 100 * df['OP_TOV'] / df['OP_POSS']
        df['OP_AST-TOV_R'] = df['OP_AST'] / df['OP_TOV']
        df['OP_PTS40'] = 40 * 5 * df['OP_PTS'] / df['OP_MTS']
        df['OP_STLR'] = 100 * df['OP_STL'] / df['POSS']
        df['OP_BLKR'] = 100 * df['OP_BLK'] / df['2PA']

        cols_to_use = [
            'team', 'PTS40', 'OP_PTS40', 'PTS', 'OP_PTS', '3P%', 'OP_3P%',
            '2P%', 'OP_2P%', 'FT%', '3PR', 'OP_3PR', 'FTR', 'OP_FTR', 'OREB%',
            'DREB%', 'ASTR', 'OP_ASTR', 'TOVR', 'OP_TOVR', 'AST-TOV_R',
            'OP_AST-TOV_R', 'STLR', 'OP_STLR', 'BLKR', 'OP_BLKR', 'PACE',
            'ORtg', 'DRtg', 'NETRtg', 'eFG%', 'OP_eFG%', 'TS%', 'OP_TS%'
        ]

    return df[cols_to_use]
