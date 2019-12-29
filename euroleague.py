import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scrape_parse_funcs as sp
import advanced_stats_funcs as asf

code_dir = os.getcwd()
data_dir = '..\\data'
plots_dir = '..\\teams_plots'

stats_descriptions = {
    '3PR': '3-point rate (% of shots that were 3-pointers)',
    'OP_3PR': 'Opponents 3-point rate (% of shots that were 3-pointers)',
    '3P%': '3 points %',
    'ORtg': 'Offensive rating (point per 100 possessions)',
    'DRtg': 'Defensive rating (points conceded per 100 possessions)',
    'NETRtg': 'Net rating (Points differential per 100 possessions)',
    'eFG%': 'Effective field goal %',
    'OP_eFG%': 'Oppoonents effective field goal %',
    'PTS40': 'Points per 40 minutes',
    'OP_PTS40': 'Points conceded per 40 minutes',
    'TS%': 'True shooting %',
    'OP_TS%': 'Opponents True shooting %',
    'OREB%': 'Offensive rebounds %\n(% of rebounds grabbed under opponents basket)',
    'DREB%': "Defensive rebounds %\n(% of rebounds grabbed under own team's basket)",
    'AST-TOV_R': 'Assist-Turnover ratio (assists per turnover)',
    'OP_AST-TOV_R': 'Opponents Assist-Turnover ratio (assists per turnover)',
    'FTR': 'Free throw rate (free throws per 100 field goal attempts)',
    'OP_FTR': 'Opponents free throw rate (free throws per 100 field goal attempts)',
    'ASTR': 'Assist rate (assists per 100 field goals made)',
    'OP_ASTR':'Opponent assist rate (assists per 100 field goals made)',
    'TOVR': 'Turnover rate (turnovers per 100 possessions)',
    'OP_TOVR': 'Opponent turnover rate (turnovers per 100 possessions)',
    'STLR': 'Steals rate (steals per 100 opponents posessions)',
    'OP_STLR': 'Opponents steals rate (opponent steals per 100 team posessions)',
    'BLKR': 'Blocks rate (blocks per 100 opponents 2-point attempts)',
    'OP_BLKR': 'Opponent Blocks rate (blocks against per 100 team 2-point attempts)',
    'PACE': 'PACE (possessions per 40 minutes)',
    'home_advantage': 'home win% - away win %',
    'win%': '% of games won'
}


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


def plot_bivariate(
        df, x, y, hue, show_season=True, annotate_only_hue_true=False,
        fit_reg=False, xyline=False, figsize=(10, 8), suptitle=None, ticks=None,
        ticklabels=None, xlim=None, ylim=None, points_color='tab:orange',
        line_color='black', annotations_colors=('tab:gray', 'tab:blue'),
        text_size='medium', axes_labels_size=14, suptitle_size=14
):
    """"""
    # create axis and plot
    fig, ax = plt.subplots(figsize=figsize)

    p1 = sns.regplot(
        data=df, x=x, y=y, ax=ax, fit_reg=fit_reg, ci=False, color=points_color,
        line_kws={'label': 'linear corr = {0:.2f}'.format(df[x].corr(df[y]))}
    )

    if fit_reg:
        ax.legend()

    # add annotations to points
    for row in range(df.shape[0]):
        color_loc = 1 if df[hue].iloc[row] else 0
        color = annotations_colors[color_loc]
        season = str(df['season'].iloc[row])[2:]

        if show_season:
            text = '{}{}'.format(df['team'].iloc[row], season)
        else:
            text = df['team'].iloc[row]

        if annotate_only_hue_true:

            if not df[hue].iloc[row]:
                text = ''

        p1.text(
            y=df[y].iloc[row], x=df[x].iloc[row], s=text,
            horizontalalignment='center', size=text_size, weight='semibold',
            color=color
        )

    # set labels
    ax.set_xlabel(stats_descriptions[x], fontsize=axes_labels_size)
    ax.set_ylabel(stats_descriptions[y], fontsize=axes_labels_size)

    if ticks:
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

    if ticklabels:
        ax.set_xticklabels(ticklabels)
        ax.set_yticklabels(ticklabels)

    if xyline:
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()]),
        ]
        # plot both limits against each other
        ax.plot(lims, lims, ls='--', label='{} = {}'.format(x, y),
                color=line_color, alpha=0.75)
        ax.set_aspect('equal')
        ax.legend()

    if suptitle:
        fig.suptitle(suptitle, fontsize=suptitle_size)

    if xlim:
        ax.set_xlim(xlim)

    if ylim:
        ax.set_ylim(ylim)

    return fig, ax


def sorted_barplot(
        df, metric, ax=None, marked_team='MTA', show_season=True,
        team_colname='team', title_size=16, tick_rot=45, figsize=(10, 6),
        colors=['tab:blue', 'tab:orange'], upper_offset=0.1, lower_offset=0.3
):
    """"""
    # if show_season=True show each team's seasons data separately
    # and set colors for bars to highlight bars of marked team
    if show_season:
        season_srs = df['season'].map(str).str.split('20', expand=True).iloc[:, 1]
        df['team_season'] = df[team_colname] + season_srs
        team_colname = 'team_season'
        team_colors = {
            team_code: (colors[1] if marked_team in team_code else colors[0])
            for team_code in df[team_colname].unique()
        }
    else:
        team_colors = {team: colors[0] for team in df[team_colname].unique()}
        team_colors[marked_team] = colors[1]

    # subset data
    data = df[[metric, team_colname]].sort_values(by=metric)
    data.reset_index(inplace=True, drop=True)

    # get axis to plot
    if not ax:
        fig, ax = plt.subplots(figsize)

    # plot bars
    sns.barplot(data=data, x=team_colname, y=metric, ax=ax, palette=team_colors)

    # set y axis limits
    if data[metric].min() < 0:
        ylim_lower = data[metric].min() + (data[metric].min() * upper_offset)
    else:
        ylim_lower = max(data[metric].min() - (data[metric].min() * lower_offset), 0)

    ylim_upper = data[metric].max() + (data[metric].max() * upper_offset)
    ax.set_ylim(ylim_lower, ylim_upper)

    # rotate x axis tick labels
    for tick in ax.get_xticklabels():
        tick.set_fontsize(14)
        tick.set_rotation(tick_rot)

    # print values on top of bars
    for index, row in data.iterrows():

            if metric in ['BLKR', 'OP_BLKR', 'STLR', 'OP_STLR', 'TOVR',
                          'OP_TOVR', 'AST-TOV_R', 'OP_AST-TOV_R', 'NETRtg']:
                value_to_print = round(row[metric], 1)
            else:
                value_to_print = round(row[metric])

            va = 'bottom' if row[metric] > 0 else 'top'
            ax.text(x=index, y=row[metric], s=str(value_to_print),
                    color='black', ha="center", va=va)

    # set titles and remove axis titles
    ax.set_title(stats_descriptions[metric], fontsize=title_size)

    return ax


# def sorted_barplot(
#         df, metrics, nrows, ncols, figsize, marked_team='MTA',
#         show_season=True, team_colname='team', ax_title_size=16, tick_rot=45,
#         colors=sns.color_palette(), upper_offset=0.1, lower_offset=0.3
# ):
#     """"""
#     # if show_season=True show each team's seasons data separately
#     # and set colors for bars to highlight bars of marked team
#     if show_season:
#         season_srs = df['season'].map(str).str.split('20', expand=True).iloc[:, 1]
#         df['team_season'] = df[team_colname] + season_srs
#         team_colname = 'team_season'
#         teams_colors = {
#             team_code: (colors[1] if marked_team in team_code else colors[0])
#             for team_code in df[team_colname].unique()
#         }
#     else:
#         teams_colors = {team: colors[0] for team in df[team_colname].unique()}
#         teams_colors[marked_team] = colors[1]
#
#     # create subplots
#     fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
#
#     for facet_num, var in enumerate(metrics):
#
#         # subset data
#         data = df[[var, team_colname]].sort_values(by=var)
#         data.reset_index(inplace=True, drop=True)
#
#         # get handle for current axis
#         if len(metrics) > 1:
#             curr_ax = axes.ravel()[facet_num]
#         else:
#             curr_ax = axes
#
#         sns.barplot(data=data, x=team_colname, y=var,
#                     ax=curr_ax, palette=teams_colors)
#
#         # set y axis limits
#         if data[var].min() < 0:
#             ylim_lower = data[var].min() + (data[var].min() * upper_offset)
#         else:
#             ylim_lower = max(data[var].min() - (data[var].min() * lower_offset), 0)
#
#         ylim_upper = data[var].max() + (data[var].max() * upper_offset)
#         curr_ax.set_ylim(ylim_lower, ylim_upper)
#
#         # rotate x axis tick labels
#         for tick in curr_ax.get_xticklabels():
#             tick.set_fontsize(14)
#             tick.set_rotation(tick_rot)
#
#         # print values on top of bars
#         for index, row in data.iterrows():
#             cond = ('BLK' in var) | ('STL' in var) | ('TOV' in var) | ('NETRtg' in var)
#             value_to_print = round(row[var], 1) if cond else round(row[var])
#             va = 'bottom' if row[var] > 0 else 'top'
#             curr_ax.text(x=index, y=row[var], s=str(value_to_print),
#                          color='black', ha="center", va=va)
#
#         # set titles and remove axis titles
#         curr_ax.set_title(stats_descriptions[var], fontsize=ax_title_size)
#         # curr_ax.set_xlabel('')
#
#     fig.tight_layout()
#
#     return fig, axes


def plot_parallel_pairs(
        df, metric, iv='team', time_var='season', ax=None, figsize=(6, 10),
        marked_iv_value='MTA', points_color='tab:blue',
        lines_color='tab:gray', marked_line_color='tab:orange',
        annot_size='small', x_offset=0.025, x_ticklabel_size=16
):
    """"""
    # restructure dataframe
    df = df.sort_values(by=time_var)
    time_points = list(df[time_var].dropna().unique())
    df = df[[iv, time_var, metric]].set_index([iv, time_var])
    df = df.unstack()
    df.columns = ['{}_{}'.format(metric, time) for time in time_points]
    df = df.reset_index()

    # get axis to plot
    if not ax:
        fig, ax = plt.subplots(figsize)

    n_obs = df.shape[0]

    # plot points for each time point
    for x_loc, time_point in enumerate(time_points):
        metric_name = '{}_{}'.format(metric, time_point)
        x_values = np.ones(n_obs) * x_loc
        ax.scatter(x=x_values, y=df[metric_name], c=points_color)

        # loop over rows in df
        for row_num in range(n_obs):
            iv_value = df[iv].iloc[row_num]
            curr_metric_value = df[metric_name].iloc[row_num]

            # add current annotation
            if np.isfinite(curr_metric_value):

                if x_loc == 0:
                    txt_x_loc = x_loc - x_offset
                    ha = 'right'
                else:
                    txt_x_loc = x_loc + x_offset
                    ha = 'left'

                ax.text(x=txt_x_loc, y=curr_metric_value, s=iv_value,
                        ha=ha, va='center', size=annot_size)

    # plot current line, work-around. assumes only 2 iv values.
    # fix later or use this loop to annotate as well instead of inner loop above
    for row_num in range(n_obs):
        iv_value = df[iv].iloc[row_num]
        metric_0 = df.iloc[row_num, 1]
        metric_1 = df.iloc[row_num, 2]
        color = marked_line_color if iv_value == marked_iv_value else lines_color
        ax.plot([0, 1], [metric_0, metric_1], c=color)

    ax.set_xlim((-0.25, 1.25))
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['2018/19', '2019/20'])

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(x_ticklabel_size)
