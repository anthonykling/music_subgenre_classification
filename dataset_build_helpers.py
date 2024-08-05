import pandas as pd
import json
import os

def get_folder_size(folder_path):
    '''
    Args:
        folder_path (str): path to folder
    Returns:
        num_files (int): number of files
        total_size (int): total size in bytes
    '''
    total_size = 0
    num_files = 0
    for dirpath, dirnames, filenames in os.walk(folder_path):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            total_size += os.path.getsize(file_path)
            num_files += 1
    return num_files, total_size


def json_to_df_list(folder_path):
    '''
    Converts a collection of .json files in a folder to a list of DataFrames.
    Each DataFrame is a single row corresponding to a single .json file.
    Args:
        folder_path (str): location of folder
    Returns:
        list: list of DataFrames       
    '''
    num_files = get_folder_size(folder_path)[0]
    dfs = []
    saved = 0

    files = os.listdir(folder_path)
    for filename in files:
        file_path = os.path.join(folder_path, filename)
        with open(file_path, 'r') as file:
            data = json.load(file)
            df = pd.DataFrame(pd.json_normalize(data))
            dfs.append(df)
            saved += 1
            if saved % 100 == 0:
                print(round(100 * saved / num_files,2), '%', end=' ')
                if saved % 2000 == 0:
                    print()
    
    print('List of DataFrames has been made!\nConcatenate them next!')

    return dfs