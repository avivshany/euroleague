import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scrape_parse_funcs as sp

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


def plot_bivariate(
        df, x, y, hue, fit_reg=False, xyline=False, figsize=(10, 8),
        suptitle=None, ticks=None, ticklabels=None, xlim=None,
        ylim=None, plot_color='orange', annotations_colors=('black', 'blue'),
        text_size='medium', axes_labels_size=14, suptitle_size=14
):
    # create axis and plot
    fig, ax = plt.subplots(figsize=figsize)

    p1 = sns.regplot(
        data=df, x=x, y=y, ax=ax, color=plot_color,
        fit_reg=fit_reg, ci=False,
        line_kws={'label': 'linear corr = {0:.2f}'.format(df[x].corr(df[y]))}
    )

    if fit_reg:
        ax.legend()

    # add annotations to points
    for row in range(df.shape[0]):
        color_loc = 1 if df[hue].iloc[row] is True else 0
        color = annotations_colors[color_loc]
        season = str(df['season'].iloc[row])[2:]
        text = '{}{}'.format(df['team'].iloc[row], season)
        p1.text(
            y=df[y].iloc[row], x=df[x].iloc[row], s=text,
            horizontalalignment='left', size=text_size, weight='semibold',
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
        ax.plot(lims, lims, ls='--', label='{} = {}'.format(x, y), alpha=0.75)
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
        df, metrics, nrows, ncols, figsize, highlighted_team='MTA',
        team_colname='team', ax_title_size=16, tick_rot=90,
        colors=sns.color_palette()
):
    """"""
    # set colors for teams
    teams_colors = {team: colors[0] for team in sp.teams_codes_map_all.values()}
    teams_colors[highlighted_team] = colors[1]

    # create subplots
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    for facet_num, var in enumerate(metrics):

        # subset data
        data = df[[var, team_colname]].sort_values(by=var)
        data.reset_index(inplace=True)

        # get handle for current axis
        if len(metrics) > 1:
            curr_ax = axes.ravel()[facet_num]
        else:
            curr_ax = axes

        sns.barplot(data=data, x=team_colname, y=var, ax=curr_ax, palette=teams_colors)

        # set y axis limits
        if data[var].min() < 0:
            ylim_lower = data[var].min() - (data[var].min() * 0.07)
        else:
            ylim_lower = max(data[var].min() - (data[var].min() * 0.3), 0)

        ylim_upper = data[var].max() + (data[var].max() * 0.07)
        curr_ax.set_ylim(ylim_lower, ylim_upper)

        # rotate x axis tick labels
        for tick in curr_ax.get_xticklabels():
            tick.set_fontsize(14)
            tick.set_rotation(tick_rot)

        # print values on top of bars
        for index, row in data.iterrows():
            cond = ('BLK' in var) | ('STL' in var) | ('TOV' in var)
            value_to_print = round(row[var], 1) if cond else round(row[var])
            va = 'bottom' if row[var] > 0 else 'top'
            curr_ax.text(x=index, y=row[var], s=str(value_to_print), color='black', ha="center", va=va)

        # set titles and remove axis titles
        curr_ax.set_title(stats_descriptions[var], fontsize=ax_title_size)
        curr_ax.set_xlabel('')

    fig.tight_layout()

    return fig, axes
