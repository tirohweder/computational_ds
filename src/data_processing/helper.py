import numpy as np
import pandas as pd


def merge_data(large_master_file_df, new_data_df, column = 'Semantic roberta twitter' ):


    new_data_df = new_data_df.loc[:,['Headline', column]]

    master_file_merged = pd.merge(df1, new_data_df, on='Headline', how='left')

    return master_file_merged