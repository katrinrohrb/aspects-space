import matplotlib.pyplot as plt
from katspace.data import chunk_lengths, chunker
from katspace.core import space_types_pos, space_types_ext
import pandas as pd
import numpy as np

#this function exists two times, once in the notebooks (sb_data_vis_canon, compare_canon_non_compare)
def calculate_ratios(df, insert_values = False, set_index = True, suffix = "_rt"): 
    
    if insert_values == False:
        _df = df.copy()
    else: _df = df

    _df['all_space']=_df[space_types_pos].sum(axis=1)

    for space_type in space_types_ext:
        _df[space_type + suffix] = _df[space_type]/_df["total"]

    if insert_values == False:
        _df = _df[["year"] + [space_type + suffix for space_type in space_types_ext]]
    
    if set_index:
        _df.set_index("year", inplace = True)
    
    return _df

def calculate_ratios2(df, insert_values=False, set_index=True, suffix="_rt"): 
    if not insert_values:
        _df = df.copy()
    else:
        _df = df

    _df['all_space'] = _df[space_types_pos].sum(axis=1)

    for space_type in space_types_ext:
        _df[space_type + suffix] = _df[space_type] / _df["total"]


    if not insert_values:
        _df = _df[["year", "author_last", "title"] + [space_type + suffix for space_type in space_types_ext]]
    
    if set_index:
        _df.set_index("year", inplace=True)
    
    return _df


def smooth_df(df, half_window_size = 5, set_index = True):

    df = df.reset_index()
    min_year, max_year = df["year"].min(), df["year"].max()

    df_list = [df.copy() for i in range(-half_window_size+1, half_window_size)]

    for i, df in enumerate(df_list):
        df_list[i]["year"] =  df_list[i]["year"] + i

    smooth_df = pd.concat(df_list)
    smooth_df = smooth_df[smooth_df["year"].isin(range(min_year + 2 * half_window_size, max_year))]
    if set_index: 
        smooth_df.set_index("year", inplace = True)

    return smooth_df


def hist_heatmap(df, chunk_size_target = 5, vert_num_chunks = 20, space_types = ["action_space", "perceived_space", "visual_space", "descriptive_space", "all_space", "no_space"]):
    cols = list(space_types) + ["total", "year"]
    df = df[cols].set_index("year")
    df.loc[:,'books'] = 1 

    years = df.index.sort_values().unique()

    chunks = chunker(years, size = chunk_size_target)
    chunk_sizes = list(map(len, chunks))
    num_chunks = len(chunk_sizes)

    def grouper_f(idx):
        for c, chunk in enumerate(chunks):
            if idx in chunk: 
                break
        return c 

    df["chunk"] = df.index.map(grouper_f)
    num_books = df[["books", "chunk"]].groupby("chunk").sum()
    num_books = num_books["books"].values

    color = {"perceived_space": "Blues",
                    "action_space": "Greens",
                    "visual_space": "Reds",
                    "descriptive_space": "GnBu",
                    "no_space": "Greys",
                    "all_space": 'Purples'
                    }
    fig, axs = plt.subplots(ncols=1, nrows=len(space_types), figsize=(3, len(space_types) * 3),
                    layout="constrained") 

    for i, space_type in enumerate(space_types):
        title = space_type 
        y = []
        x = []

        for idx, book in df.iterrows():
            this_chunk = book["chunk"]
            nr_books_in_chunk = num_books[this_chunk]
            if (book.total == 0):
                continue

            norm_count = book[space_type] / book["total"]

            y += [norm_count]
            year = chunks[this_chunk][0]
            x += [year]
        
        axs[i].set_title(title)
        axs[i].hist2d(x, y, bins = [num_chunks, vert_num_chunks], cmap = color[space_type])

def plot_p_values_heatmap(res, genres, space_type):

    values = np.round(res[space_type].pvalue, 3)

    fig, ax = plt.subplots(figsize = (4,6))
    im = ax.imshow(values)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(genres)), labels=genres)
    ax.set_yticks(np.arange(len(genres)), labels=genres)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(genres)):
        for j in range(len(genres)):
            if i >= j: 
                text = ax.text(j, i, values[i, j],
                            ha="center", va="center", color="w")
            else: 
                value = res[space_type].statistic[j,i]
                value = np.round(100 * value, 1)
                ax.text(j, i, value, 
                            ha="center", va="center", color="w")

    ax.set_title(space_type)
    plt.show()