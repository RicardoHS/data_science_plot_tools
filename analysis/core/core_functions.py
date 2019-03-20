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

    returns the column 'by' encoded
    '''
    return df[by].map(smooth_mean(df, by, on, m))

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