import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from analysis.core import core_functions

def get_colors():
    return ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'grey']

def get_default_sizes():
    default_col_size = 10
    default_row_size = 6
    return default_col_size, default_row_size

def faced_ratios_areas(dataframe, colums, class_column, n_rows=2, n_cols=3, plot_area=True, legend=True, colors=None, figure=None):
    if not colors:
        colors = get_colors()

    df_print = dataframe.copy()
    df_print['count'] = 1

    default_col_size, default_row_size = get_default_sizes()
    if not figure:
        fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols)
        fig.set_size_inches(default_col_size*n_cols,default_row_size*n_rows)    
    else:
        fig, ax = figure

    for i_tuple, col in enumerate(colums):
        curr_ax=ax[i_tuple//n_cols][i_tuple%n_cols]
        
        # range creation
        df_grouped = df_print[[col, class_column, 'count']].groupby([col, class_column]).count()
        col_levels = df_grouped.index.levels[0]
        target_values = list(set(df_grouped.index.labels[1].tolist()))
        range_col_value =  [i for i in col_levels.tolist() for _ in range(len(target_values))] 
        
        # fill gaps between indexes
        arrays = [range_col_value, target_values*(len(range_col_value)//len(target_values))]
        new_index = pd.MultiIndex.from_arrays(arrays, names=(col, class_column))
        df_grouped = df_grouped.reindex(new_index).fillna(0)
        
        # get completed data
        lines = []
        for c in range(len(target_values)):
            l = df_grouped[np.in1d(df_grouped.index.get_level_values(1), [c])]
            lines.append(l/l.sum())
        
        
        x = list(range(col_levels.shape[0]))
        for i, line in enumerate(lines):        
            y1 = line.values.squeeze()
            other_lines = []
            for i2, line2 in enumerate(lines):
                if i!=i2:
                    other_lines.append(line2)
            lines_matrix = tuple([l.values.squeeze() for l in other_lines])
            y2 = np.column_stack(lines_matrix).T.max(axis=0)

            a1, _ = core_functions.compute_areas_between_lines(y1, y2)
            curr_ax.bar(i, a1, color=colors[i])
        
        # other plot things
        if legend:
            legend = [f'{class_column}={t_v}' for t_v in target_values]
            curr_ax.legend(legend)
            
        curr_ax.set_ylim([0,1])

        curr_ax.set_xlabel(col)
    return fig, ax

def faced_ratios(dataframe, colums, class_column, n_rows=2, n_cols=3, plot_area=True, y_lim=None, legend=True, colors=None, figure=None):
    if not colors:
        colors = get_colors()

    df_print = dataframe.copy()
    df_print['count'] = 1

    default_col_size = 10
    default_row_size = 6
    if not figure:
        fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols)
        fig.set_size_inches(default_col_size*n_cols,default_row_size*n_rows)    
    else:
        fig, ax = figure

    for i_tuple, col in enumerate(colums):
        curr_ax=ax[i_tuple//n_cols][i_tuple%n_cols]
        
        # range creation
        df_grouped = df_print[[col, class_column, 'count']].groupby([col, class_column]).count()
        col_levels = df_grouped.index.levels[0]
        target_values = list(set(df_grouped.index.labels[1].tolist()))
        range_col_value =  [i for i in col_levels.tolist() for _ in range(len(target_values))] 
        
        # fill gaps between indexes
        arrays = [range_col_value, target_values*(len(range_col_value)//len(target_values))]
        new_index = pd.MultiIndex.from_arrays(arrays, names=(col, class_column))
        df_grouped = df_grouped.reindex(new_index).fillna(0)
        
        # get completed data
        lines = []
        for c in range(len(target_values)):
            l = df_grouped[np.in1d(df_grouped.index.get_level_values(1), [c])]
            lines.append(l/l.sum())
        
        # plotting
        for i, line in enumerate(lines):
            line.plot(ax=curr_ax, c=colors[i])            
        
        # print areas
        if plot_area:
            x = list(range(col_levels.shape[0]))
            for i, line in enumerate(lines):        
                y1 = line.values.squeeze()
                other_lines = []
                for i2, line2 in enumerate(lines):
                    if i!=i2:
                        other_lines.append(line2)
                lines_matrix = tuple([l.values.squeeze() for l in other_lines])
                y2 = np.column_stack(lines_matrix).T.max(axis=0)

                #y2 = (c2/c2.sum()).values.squeeze()
                curr_ax.fill_between(x, y1, y2, where=y1 >= y2, facecolor=colors[i], interpolate=True, alpha=0.3)
        
        
        # other plot things
        if legend:
            legend = [f'{class_column}={t_v}' for t_v in target_values]
            curr_ax.legend(legend)

        if y_lim:
            curr_ax.set_ylim(y_lim)

        curr_ax.set_xlabel(col)
        curr_ax.set_xticks(range(col_levels.shape[0]))
        curr_ax.set_xticklabels(col_levels)

        # if there are to much ticks
        x_ticks = curr_ax.xaxis.get_ticklabels()
        if len(x_ticks) > 20:
            every_nth = 5
            for n, label in enumerate(curr_ax.xaxis.get_ticklabels()):
                if n % every_nth != 0:
                    label.set_visible(False)
    return fig, ax

def prediction_probas(y_true, y_pred_probas):
    y_pred = np.argmax(y_pred_probas, axis=1)
    n_classes = np.unique(y_pred)
    fig, ax = plt.subplots(3,n_classes.shape[0])
    fig.set_size_inches((7*n_classes.shape[0],10))
    for class_n in sorted(n_classes):
        correct_preds = pd.Series(y_pred_probas[(y_true == y_pred) & (y_pred == class_n)][:,class_n], name="Correct")
        wrong_preds = pd.Series(y_pred_probas[(y_true != y_pred) & (y_pred == class_n)][:,class_n], name="Wrong")
        x = pd.Series(y_pred_probas[:,class_n], name="All")

        _ = sns.distplot(x, bins=20, kde=True, ax=ax[0][class_n], color='b', norm_hist=True).set(xlim=(0, 1))
        _ = sns.distplot(correct_preds, bins=20, kde=True, ax=ax[1][class_n], color='g', norm_hist=True).set(xlim=(0, 1))
        _ = sns.distplot(wrong_preds, bins=20, kde=True, ax=ax[2][class_n], color='r', norm_hist=True).set(xlim=(0, 1))


        ax[0][class_n].set_title(f'Model probability histogram, Class {class_n}')
        ax[0][class_n].set_ylabel('Number of y_pred')
    return fig, ax
