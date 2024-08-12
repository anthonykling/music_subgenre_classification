import pandas as pd
import json
import os
import sys
from queue import LifoQueue
import numpy as np
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

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

def json_to_dictionary(filename) :
  '''
  Loads a json file into a dictionary.
  '''
  if (filename[-4:] != 'json') :
    return
  with open(filename, "r") as json_data :
    data = json.load(json_data)
    json_data.close()
  return data

def unpack_keys_single_level(raw_data, sep='_') -> pd.DataFrame:
  '''
     Unpacks a dictionary or list into a dataframe where each column represents a value or entry,
     or returns a dataframe with a single entry for any other type.  
  '''
  if type(raw_data) == dict :
    return pd.DataFrame(raw_data)
  elif type(raw_data) == list :
    length = len(raw_data)
    return pd.DataFrame(raw_data, columns=list(range(0,length)))
  else :
    return pd.DataFrame(raw_data, columns=[sep])
  
def dictionary_to_dataframe(raw_dict_data, include_meta=True, sep='_') :
  '''
  Performs a depth first search of the dictionary, to create a pandas dataframe where each column
  is a single bool, string, or float.  Column names are the nested key names concatenated and separated by
  the sep argument.
  '''
  Q = LifoQueue()
  names = LifoQueue()
  cols = []
  raw_data = []

  if 'mbdata' in raw_dict_data:
    mbdata_df = pd.json_normalize(raw_dict_data.pop('mbdata'))
    mbdata_df = mbdata_df.rename(columns=lambda x: 'mbdata' + sep + x)

  for k in raw_dict_data.keys():

    if (k != 'metadata' or include_meta) :
      Q.put(raw_dict_data[k])
      names.put(k)


  while(not Q.empty()) :
    r = Q.get()
    name = names.get()
    if (type(r) == dict) :
      for k1 in r.keys() :
        if (k1 == "beats_position"):
          continue
        Q.put(r[k1])
        names.put(name+sep+k1)
    elif (type(r) == list) :
      for i in range(0, len(r)) :
        Q.put(r[i])
        names.put(name+sep+str(i))
    else :
      cols.append(name)
      raw_data.append(r)

  df = pd.DataFrame([raw_data], columns=cols)

  df = pd.concat([df.reset_index(drop=True), mbdata_df.reset_index(drop=True)], axis=1)

  return df
  
def read_folder_to_dataframe(dir, sep) :
  '''
  Reads all json files in a directory dir into a single pandas dataframe.
  '''
  list_df = []
  list_dir = os.listdir(dir)
  num_dirs = len(list_dir)
  one_perc = num_dirs//100
  completed = 0

  for subdir in list_dir :
    if(subdir[-4:] != 'json') :
      continue
    d = json_to_dictionary(dir+subdir)
    df = dictionary_to_dataframe(d, sep=sep)
    list_df.append(df)
    completed += 1
    print("\r{}".format(completed).rjust(5) + " / " + str(num_dirs), end="")
    sys.stdout.flush()

  while len(list_df) > 1:
    new_list = []

    for i in range(0, len(list_df) - 1, 2):
        combined_df = pd.concat([list_df[i], list_df[i + 1]], ignore_index=True)
        new_list.append(combined_df)

    if len(list_df) % 2 == 1:
        new_list.append(list_df[-1])

    list_df = new_list

  return list_df[0]



    

  #while (len(list_df) > 1) :
  #  new_list = [pd.concat(list_df[i:i+2]) for i in range(0, len(list_df)//2)]
  #  if (len(list_df) % 2 == 1) :
  #    new_list.append(list_df[-1])
  #  list_df = new_list

  return pd.concat(list_df)