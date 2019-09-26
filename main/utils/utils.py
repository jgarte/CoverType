import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s', 
                    datefmt='%Y-%m-%d %H:%M:%S')

def compact_columns(row, dataframe, features):
    """
    Function for compacting the one hot encoded features into one single variable
    which better fits non parametric models.
    """
    for feat in features:
        if row[feat] == 1:
            return feat
        else:
            continue
        
def calculate_stoneyness(dataframe):
    """
    This function creates a new variabel based on the stone level of the soil.
    """
    logging.info('Creating stoneyness variable')
    dataframe['Stoneyness'] = sum(i * dataframe['Soil_Type{}'.format(i)] for i in range(1, 41))

    stoneyness = [4, 3, 1, 1, 1, 2, 0, 0, 3, 1, 
                1, 2, 1, 0, 0, 0, 0, 3, 0, 0, 
                0, 4, 0, 4, 4, 3, 4, 4, 4, 4, 
                4, 4, 4, 4, 1, 4, 4, 4, 4, 4]

    dataframe['Stoneyness'] = dataframe['Stoneyness'].replace(range(1, 41), stoneyness)
    
    return dataframe

def prepare_dataset(df):
    
    soil_types = ['Soil_Type1', 'Soil_Type2', 'Soil_Type3',
            'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type7', 'Soil_Type8',
            'Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12',
            'Soil_Type13', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16',
            'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20',
            'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24',
            'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28',
            'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32',
            'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36',
            'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40']

    wilderness_areas = ['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3',
                        'Wilderness_Area4']
    
    df = calculate_stoneyness(df)

    logging.info('Aggregating wilderness dimensions')
    df['Wilderness'] = df.apply(lambda x: compact_columns(x, df, wilderness_areas), axis=1)
    logging.info('Aggregating soiltype dimensions')
    df['SoilType'] = df.apply(lambda x: compact_columns(x, df, soil_types), axis=1)
    
    logging.info('Dropping one hot encoded dimensions')
    df = df.drop(soil_types, 1)
    df = df.drop(wilderness_areas, 1)
    
    return df

def reduce_mem_usage(df, verbose=True):
    
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                c_prec = df[col].apply(lambda x: np.finfo(x).precision).max()
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max and c_prec == np.finfo(np.float32).precision:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    
    if verbose: logging.info('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    
    return df