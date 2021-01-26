import numpy as np

def compute_areas_between_lines(y1,y2):
    diff = y1 - y2
    positive_values = np.maximum(diff, 0)
    negative_values = np.minimum(diff, 0)
    y1_area = np.trapz(positive_values)
    y2_area = np.trapz(negative_values)
    return y1_area, y2_area

def target_encode(df, by, on, m=300):
    '''
    Target encode a dataframe column using smooth mean
    https://maxhalford.github.io/blog/target-encoding-done-the-right-way/

    returns the column 'by' encoded as pd.Series and the dict with the mapping
    ////
    TODO: http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-munging/target-encoding.html
    docs: https://www.kaggle.com/c/porto-seguro-safe-driver-prediction/discussion/44659
    implement 
    holdout_type: whether or not a holdout should be used in constructing the target average
    blended_avg: whether to perform a blended average
    noise_level: whether to include random noise to the average
    '''
    target_encoding = smooth_mean_dict(df, by, on, m)
    return df[by].map(target_encoding), target_encoding

def smooth_mean(df, by, on, m):
    # Compute the global mean
    mean = df[on].mean()
    # Compute the number of values and the mean of each group
    agg = df.groupby(by)[on].agg(['count', 'mean'])
    counts = agg['count']
    means = agg['mean']
    # Compute the "smoothed" means
    smooth = (counts * means + m * mean) / (counts + m)
    # Replace each value by the according smoothed mean
    return smooth

def smooth_mean_dict(df, by, on, m):
    return smooth_mean(df, by, on, m).to_dict()

def correspondence_table(df, columns, table=True, freq=False):
    if len(columns)!=2:
        print('Columns must be exactly two.')
        return

    df_table = df.groupby(columns).size().reset_index(name='count').pivot(index=columns[0], columns=columns[1], values='count').fillna(0)
    if table:
        axis0 = df_table.sum(axis=0)
        df_table.loc['SUM'] = axis0
        axis1 = df_table.sum(axis=1)
        df_table['SUM'] = axis1
        max_axis = df_table.loc['SUM', ['SUM']].values
        if freq:
            df_table.loc['SUM'] = df_table.loc['SUM']/max_axis
            df_table['SUM'] = df_table['SUM']/max_axis
            df_table.loc['SUM', ['SUM']] = 1

    return df_table
