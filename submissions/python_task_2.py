
"""# task 2

# Q1
"""

import pandas as pd
import numpy as np
import networkx as nx


def calculate_distance_matrix(df)->pd.DataFrame():
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """



    G = nx.Graph()
    for _, row in df.iterrows():
      G.add_edge(row['id_start'], row['id_end'], weight=row['distance'])

    matrix = nx.floyd_warshall_numpy(G, weight='weight')

    np.fill_diagonal(matrix, 0)

    result_df = pd.DataFrame(matrix, index=G.nodes(), columns=G.nodes())



    return result_df

df = pd.read_csv('dataset-3.csv')
df1 = calculate_distance_matrix(df)
print(df1)

"""# Q2"""

def unroll_distance_matrix(df)->pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    nodes = df.index

    result_df = pd.DataFrame(columns=['id_start', 'id_end', 'distance'])

    for id_start in nodes:
        for id_end in nodes:
            if id_start != id_end:
                distance = df.loc[id_start, id_end]

                result_df = result_df.append({'id_start': id_start, 'id_end': id_end, 'distance': distance}, ignore_index=True)

    return result_df

df1 = unroll_distance_matrix(df1)
df1

"""# Q3"""

def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Write your logic here
    reference_rows = df[df['id_start'] == reference_id]

    reference_avg_distance = reference_rows['distance'].mean()

    lower_threshold = reference_avg_distance * 0.9
    upper_threshold = reference_avg_distance * 1.1

    result_df = df[(df['distance'] >= lower_threshold) & (df['distance'] <= upper_threshold)]



    return df

reference_value = 1001404
df2 = find_ids_within_ten_percentage_threshold(df1, reference_value)
df2

"""# Q4"""

def calculate_toll_rate(df)->pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Wrie your logic here
    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }

    for vehicle_type, rate_coefficient in rate_coefficients.items():
        df[vehicle_type] = df['distance'] * rate_coefficient

    df.drop('distance', axis=1, inplace=True)


    return df

df3 = calculate_toll_rate(df1)
df3

"""# Q5"""

from datetime import datetime, time, timedelta

def calculate_time_based_toll_rates(input_df)->pd.DataFrame():
    """
    Calculate time-based toll rates for different time intervals within a day.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Write your logic here
    time_ranges = [
        (time(0, 0, 0), time(10, 0, 0), 0.8),
        (time(10, 0, 0), time(18, 0, 0), 1.2),
        (time(18, 0, 0), time(23, 59, 59), 0.8)
    ]

    weekend_discount_factor = 0.7

    def calculate_rate(row):
        day_of_week = pd.to_datetime(row['id_start'], unit='s').day_name()

        if day_of_week in ['Saturday', 'Sunday']:
            discount_factor = weekend_discount_factor
        else:
            for start_range, end_range, factor in time_ranges:
                if start_range <= row['id_start'] % timedelta(days=1) <= end_range:
                    discount_factor = factor
                    break
            else:
                discount_factor = 1.0

        return row['distance'] * discount_factor

    input_df['time_based_rate'] = input_df.apply(calculate_rate, axis=1)



    return df

calculate_time_based_toll_rates(df2)

