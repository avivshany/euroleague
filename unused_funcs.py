###############################################################################
# Functions for getting teams data, and from that their games scores.
# Removed from scrape_parse_funcs.
# No longer required as game scores are obtained from get_teams_stats.
# get_teams_data (and its helper function _scrape_team_data) can be used
# in the future for getting players stats.
###############################################################################


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


###############################################################################
# Functions for getting games scores from teams data.
# Removed from scrape_parse_funcs.
# No longer required as game scores are obtained from get_teams_stats.
###############################################################################


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


def get_games_scores(season):
    """
    Loop over all teams in a season, scrape scores from each game,
    parse tha data, and return in clean dataframe

    :param season: integer indicating season start year
    :return: pandas.DataFrame
    """
    teams_data = get_teams_data(season)
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
    games['home_team'] = games['home_team'].replace(recognizable_team_codes)
    games['away_team'] = games['away_team'].replace(recognizable_team_codes)
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


###############################################################################
# Removed from advanced_stats_funcs and replaced by a function with the
# same name, which uses a more elegant solution in SQL
###############################################################################


def get_win_ratios(games_stats):
    """
    Calculate win% total and by location (home/away).
    Also calculates home advantage as home win% - away win%
    """
    # create df with a row per game containing teams, points, and location
    home = games_stats.loc[games_stats['location'] == 'home_team']
    away = games_stats.loc[games_stats['location'] == 'away_team']
    home.rename(columns={'team': 'home_team', 'PTS': 'home_points'}, inplace=True)
    away.rename(columns={'team': 'away_team', 'PTS': 'away_points'}, inplace=True)
    home.drop('location', axis=1, inplace=True)
    away.drop(['season', 'location'], axis=1, inplace=True)
    wins = home.merge(away, on='game_id')

    # get winner per game and winning team location
    wins['winner'] = wins.apply(
        lambda row: row['home_team']
        if row['home_points'] > row['away_points']
        else row['away_team'],
        axis=1
    )
    wins['home_win'] = wins['home_team'] == wins['winner']
    wins['away_win'] = wins['away_team'] == wins['winner']

    # create df summarizing home wins per team
    home_wins = wins.groupby(['season', 'home_team'])['home_win', 'away_win'].sum()
    home_wins.columns = ['home_wins', 'home_losses']
    home_wins['home_games'] = home_wins['home_wins'] + home_wins['home_losses']
    home_wins['home_win%'] = 100 * home_wins['home_wins'] / home_wins['home_games']
    home_wins.reset_index(inplace=True)
    home_wins.rename(columns={'home_team': 'team'}, inplace=True)

    # create df summarizing away wins per team
    away_wins = wins.groupby(['season', 'away_team'])['home_win', 'away_win'].sum()
    away_wins.columns = ['away_losses', 'away_wins']
    away_wins['away_games'] = home_wins['away_wins'] + home_wins['away_losses']
    away_wins['away_win%'] = 100 * away_wins['away_wins'] / away_wins['away_games']
    away_wins.reset_index(inplace=True)
    away_wins.rename(columns={'away_team': 'team'}, inplace=True)

    # join home_wins and away_wins to calculate win ratios per team
    home_advantage = home_wins.merge(away_wins, on=['season', 'team'])
    home_advantage['wins'] = home_advantage['home_wins'] + home_advantage['away_wins']
    home_advantage['losses'] = home_advantage['home_losses'] + home_advantage['away_losses']
    home_advantage['games'] = home_advantage['wins'] + home_advantage['losses']
    home_advantage['win%'] = 100 * home_advantage['wins'] / home_advantage['games']
    home_advantage['home_advantage'] = home_advantage['home_win%'] - home_advantage['away_win%']

    # set columns to return
    cols = ['season', 'team', 'home_win%', 'away_win%', 'win%', 'home_advantage']

    return home_advantage[cols].round(2)


###############################################################################
# Removed from euroleague module. was used as a more general solution for
# getting data than the get_teams_stats function. it is no longer needed as
# get_game_scores is no longer needed because tha same information is obtained
# from games_stats. can be used in the future if more get data funcs are used
###############################################################################


def get_data(df_name, seasons_rounds, scraped_until_round=0, overwrite=False):
    """"""
    os.chdir(data_dir)
    scrape_all = False
    df = pd.DataFrame()

    # loop over seasons
    for season in seasons_rounds.keys():
        n_rounds = seasons_rounds[season]
        df_path = '{}_{}_r{}.csv'.format(df_name, season, n_rounds)\
            .replace('teams_stats', 'games_stats')

        # set the right function to use for getting data
        if df_name == 'teams_stats':
            get_data_func = sp.get_games_stats
            kwargs = {'season': season, 'completed_rounds': n_rounds,
                      'rounds_already_scraped': scraped_until_round}
        elif df_name == 'games_scores':
            get_data_func = sp.get_games_scores
            kwargs = {'season': season}
        else:
            raise ValueError('df_name must be teams_stats or games_scores')

        # if updated file already exists, read it instead of scraping
        if (os.path.exists(df_path)) & (overwrite is False):
            curr_season_df = pd.read_csv(df_path)
        else:

            # if a scraped file until a specific round exists, read it,
            # scrape remaining rounds, and concatenate the two
            if (scraped_until_round > 0) & (overwrite is False):
                existing_df_path = '{}_{}_r{}.csv'.format(
                    df_name, season, scraped_until_round
                ).replace('teams_stats', 'games_stats')

                if os.path.exists(existing_df_path):
                    existing_season_df = pd.read_csv(existing_df_path)
                    txt = 'Scraping {} for season {} starting from round {}'
                    print(txt.format(df_name, season, scraped_until_round + 1))
                    remaining_season_df = get_data_func(**kwargs)
                    curr_season_df = pd.concat(
                        [existing_season_df, remaining_season_df], sort=False
                    )
                else:
                    print(existing_df_path + 'does not exist')
                    scrape_all = True
            # if no file exists or overwriting, scrape all games for the season
            else:
                scrape_all = True

        if scrape_all:
            print('Scraping all {} data for season {}'.format(df_name, season))
            curr_season_df = get_data_func(**kwargs)
            curr_season_df.to_csv(df_path, index=False)

        # if df is teams_stats calculate advanced statistics per team
        if df_name == 'teams_stats':
            curr_season_df = asf.get_overall_teams_opponents_stats(
                games_stats_df=curr_season_df, season=season
            )
            curr_season_df = asf.get_team_advanced_stats(curr_season_df)

        # add season column and concatenate current season's df with all seasons
        curr_season_df['season'] = season
        df = pd.concat([df, curr_season_df], sort=False)

    df.reset_index(inplace=True, drop=True)

    return df


################################################################################
# Removed from euroleague module. This function loops over all metrics and plots
# bar plots per team in season sorted by the metric value, in a subplots grid.
# replaced by a function with the same name that only takes one metric, as
# looping and subplots are intended to be done outside the functions.
# can be used in the future for plotting multiple metrics more easily
################################################################################


def sorted_barplot(
        df, metrics, nrows, ncols, figsize, marked_team='MTA',
        show_season=True, team_colname='team', ax_title_size=16, tick_rot=45,
        colors=sns.color_palette(), upper_offset=0.1, lower_offset=0.3
):
    """"""
    # if show_season=True show each team's seasons data separately
    # and set colors for bars to highlight bars of marked team
    if show_season:
        season_srs = df['season'].map(str).str.split('20', expand=True).iloc[:, 1]
        df['team_season'] = df[team_colname] + season_srs
        team_colname = 'team_season'
        teams_colors = {
            team_code: (colors[1] if marked_team in team_code else colors[0])
            for team_code in df[team_colname].unique()
        }
    else:
        teams_colors = {team: colors[0] for team in df[team_colname].unique()}
        teams_colors[marked_team] = colors[1]

    # create subplots
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    for facet_num, var in enumerate(metrics):

        # subset data
        data = df[[var, team_colname]].sort_values(by=var)
        data.reset_index(inplace=True, drop=True)

        # get handle for current axis
        if len(metrics) > 1:
            curr_ax = axes.ravel()[facet_num]
        else:
            curr_ax = axes

        sns.barplot(data=data, x=team_colname, y=var,
                    ax=curr_ax, palette=teams_colors)

        # set y axis limits
        if data[var].min() < 0:
            ylim_lower = data[var].min() + (data[var].min() * upper_offset)
        else:
            ylim_lower = max(data[var].min() - (data[var].min() * lower_offset), 0)

        ylim_upper = data[var].max() + (data[var].max() * upper_offset)
        curr_ax.set_ylim(ylim_lower, ylim_upper)

        # rotate x axis tick labels
        for tick in curr_ax.get_xticklabels():
            tick.set_fontsize(14)
            tick.set_rotation(tick_rot)

        # print values on top of bars
        for index, row in data.iterrows():
            cond = ('BLK' in var) | ('STL' in var) | ('TOV' in var) | ('NETRtg' in var)
            value_to_print = round(row[var], 1) if cond else round(row[var])
            va = 'bottom' if row[var] > 0 else 'top'
            curr_ax.text(x=index, y=row[var], s=str(value_to_print),
                         color='black', ha="center", va=va)

        # set titles and remove axis titles
        curr_ax.set_title(stats_descriptions[var], fontsize=ax_title_size)
        # curr_ax.set_xlabel('')

    fig.tight_layout()

    return fig, axes

