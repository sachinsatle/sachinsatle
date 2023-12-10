

import pandas as pd

def generate_car_matrix(df_og)->pd.DataFrame:
    """
    Creates a DataFrame  for id combinations.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Matrix generated with 'car' values,
                          where 'id_1' and 'id_2' are used as indices and columns respectively.
    """
    # Write your logic here
    df = df_og.pivot(index='id_1', columns='id_2', values='car').fillna(0)
    return df

df_og = pd.read_csv('dataset-1.csv')
generate_car_matrix(df_og)



def get_type_count(df)->dict:
    """
    Categorizes 'car' values into types and returns a dictionary of counts.

    Args:
        df (pandas.DataFrame)

    Returns:
        dict: A dictionary with car types as keys and their counts as values.
    """
    # Write your logic here
    bins = [0, 15, 25, float('inf')]
    labels = ['Low', 'Medium', 'High']

    df['car_type'] = pd.cut(df['car'], bins=bins, labels=labels, right=False)
    car_counts = df['car_type'].value_counts().to_dict()
    sorted_car_count = dict(sorted(car_counts.items()))
    return sorted_car_count

df = pd.read_csv('dataset-1.csv')
get_type_count(df)



def get_bus_indexes(df)->list:
    """
    Returns the indexes where the 'bus' values are greater than twice the mean.

    Args:
        df (pandas.DataFrame)

    Returns:
        list: List of indexes where 'bus' values exceed twice the mean.
    """
    # Write your logic here
    mean = df['bus'].mean()
    index_of_bus = df[df['bus'] > 2*mean].index.tolist()
    sorted_bus = sorted(index_of_bus)
    return sorted_bus

df = pd.read_csv('dataset-1.csv')
get_bus_indexes(df)



def filter_routes(df)->list:
    """
    Filters and returns routes with average 'truck' values greater than 7.

    Args:
        df (pandas.DataFrame)

    Returns:
        list: List of route names with average 'truck' values greater than 7.
    """
    # Write your logic here
    route_mean = df.groupby('route')['truck'].mean()
    routes = route_mean[route_mean > 7].index.tolist()

    return sorted(routes)

df = pd.read_csv('dataset-1.csv')
filter_routes(df)



def multiply_matrix(matrix)->pd.DataFrame:
    """
    Multiplies matrix values with custom conditions.

    Args:
        matrix (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Modified matrix with values multiplied based on custom conditions.
    """
    # Write your logic here
    modify_function = lambda x: round(x * 0.75, 1) if x < 20 else round(x * 1.25, 1)
    df = matrix.applymap(modify_function)

    return df

matrix = generate_car_matrix(df_og)
multiply_matrix(matrix)



def time_check(df)->pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    # Write your logic here


    return pd.Series()

df = pd.read_csv('dataset-2.csv')

df.head()

