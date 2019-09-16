import pandas as pd
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s', 
                    datefmt='%Y-%m-%d %H:%M:%S')

data_input_path = Path('data/input/')
validation_path = Path('data/validation')
train_output_file = Path('train.csv')
test_output_file = Path('test.csv')
validation_file = Path('validation.csv')
file = Path('covtype.csv')
test_index = 15121

column_names = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
                'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
                'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
                'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area1',
                'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4',
                'Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4', 'Soil_Type5',
                'Soil_Type6', 'Soil_Type7', 'Soil_Type8', 'Soil_Type9', 'Soil_Type10',
                'Soil_Type11', 'Soil_Type12', 'Soil_Type13', 'Soil_Type14',
                'Soil_Type15', 'Soil_Type16', 'Soil_Type17', 'Soil_Type18',
                'Soil_Type19', 'Soil_Type20', 'Soil_Type21', 'Soil_Type22',
                'Soil_Type23', 'Soil_Type24', 'Soil_Type25', 'Soil_Type26',
                'Soil_Type27', 'Soil_Type28', 'Soil_Type29', 'Soil_Type30',
                'Soil_Type31', 'Soil_Type32', 'Soil_Type33', 'Soil_Type34',
                'Soil_Type35', 'Soil_Type36', 'Soil_Type37', 'Soil_Type38',
                'Soil_Type39', 'Soil_Type40', 'Cover_Type']


logging.info('Reading data')
data = pd.read_csv(data_input_path/file, names=column_names).reset_index()
data.rename({'index': 'Id'}, axis=1, inplace=True)

logging.info('Spliting train, test and validation sets')
train_set = data.iloc[:15121]
test_set = data.iloc[15121:]
validation_set = test_set[['Id', 'Cover_Type']]

logging.info('Saving training set')
train_set.to_csv(data_input_path/train_output_file)
logging.info('Saving testing set')
test_set.to_csv(data_input_path/test_output_file)
logging.info('Saving validation set')
validation_set.to_csv(validation_path/validation_file)