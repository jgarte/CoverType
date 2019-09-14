import pandas as pd
from pathlib import Path
from utils.utils import (compact_columns, calculate_stoneyness, 
                         prepare_dataset, reduce_mem_usage)
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s', 
                    datefmt='%Y-%m-%d %H:%M:%S')

def main():

    input_folder = Path('data/input/')
    train_file = Path('train.csv')
    test_file = Path('test.csv')

    logging.info('> Loading data')
    train = pd.read_csv(input_folder/train_file, index_col=0)
    test = pd.read_csv(input_folder/test_file, index_col=0)
    
    train = reduce_mem_usage(train)
    test = reduce_mem_usage(test)
    
    train = prepare_dataset(train)
    test = prepare_dataset(test)
    
    logging.info('> Exporting train dataset')
    train.to_csv(input_folder/train_file)
    logging.info('> Exporting test dataset')
    test.to_csv(input_folder/test_file)
    
if __name__ == "__main__":
    logging.info('> Starting application')
    main()

