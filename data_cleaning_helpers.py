import pandas as pd
import json
import os

## This functions concern removing certain phrases in the mbdata of a recording.

# NOTE: This function will likely not need to be used in light of using the more robust
# remove_phrease function

def get_joinphrase(col):
    '''
    Given the mbdata.artist-credit column, this returns the value of the
    'joinphrase' key.
    Input:
        col (list): Based on how our DataFarme is formatted, we expect this column to be
        a list of dictionaries.  The key 'joinphrase' is always present if the length of this 
        list is greater than 1.
    Returns:
        (str) if we have a 'joinphrase' otherwise returns the string 'None'.
    '''
    if len(col) >1: return col[0]['joinphrase']
    else: return 'None'

def flatten_dict(d, parent_key='', sep='.'):
    '''
    Does a depth search on a dictionary to essentially 'un-nest' the contents.
    This makes retrieving 'all' values of a dictionary easier.
    Inputs:
        d (dict): dictionary 
        parent_key (str): empty by default
        sep (str): separating the names for each 'layer'.  Defaults to '.'
    Returns:
        (dict) with any nested dictionaries 'spread out'.
    '''
    items = []
    for k, v in d.items():
        new_key = f'{parent_key}{sep}{k}' if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def dict_contains(entry, phrase):
    '''
    Given a dictionary with possibly nested dictionaries, determines if any
    values contain the given phrase.  This uses flatten_dict on the input dictionary.
    Inputs:
        entry (dict): the given dictionary
        phrase (str): determines if the phrase is a substring of any of the values.
            Not case sensitive.
    Returns:
        bool: True if any of the values contain the phrase.
    '''
    # This function is specific to how electronic_df is formatted.
    # Might be best to change this to isinstance(entry, dict) == False to
    # take care of non-dictionary instances, but this should suffice for now.
    if isinstance(entry, float):
        return False
    
    phrase = phrase.lower()
    N = len(entry)
    for i in range(N):
        for v in flatten_dict(entry[i]).values():
            if phrase in str(v).lower():
                return True
            
    return False



def remove_phrase(df, phrase, get_index=False):
    '''
    This function searches through mbdata.title, mbdata.disambiguation,
    mbdata.tags, and mbdata.releases for the given phrase from the DataFrame df.  
    By default, it returns df rows containing those phrases removed.
    Args:
        df (DataFrame)
        phrase (str): we automatically make phrase lowercase, so case doesn't matter
        get_index (bool): If True, then this returns the indices where such phrases occur instead
            removing them
    Returns:
        DataFrame
    '''
    phrase = phrase.lower()

    title_index = df[df['mbdata.title'].apply(lambda x: phrase in x.lower())].index
    disambig_index = df[df['mbdata.disambiguation'].apply(lambda x: phrase in str(x).lower())].index
    tags_index = df[df['mbdata.tags'].apply(lambda x: dict_contains(x, phrase))].index
    releases_index = df[df['mbdata.releases'].apply(lambda x: dict_contains(x, phrase))].index

    index = title_index.union(disambig_index).union(tags_index).union(releases_index)

    if get_index: return index

    return df.drop(index)

def genre_extractor(tags):
    '''
    Input (list): The input should be a list of dictionaries as present in the
        mbdata.tags column
    Returns:
        set of strings of the genre names present in tags
    '''
    genres = set()
    N = len(tags)
    for i in range(N):
        genres.add(tags[i]['name'])
    
    return genres

def setstring_replace(set_of_strings, old, new):
    '''
    Given a set of strings, replaces any substring 'old' with the substring 'new'.
    Input:
        set_of_strings (set): set of strings
        old (str): old substring
        new (str): new substring
    Returns:
        new set of strings
    '''
    return {x.replace(old, new) for x in set_of_strings}


def genre_labeler(tags, genres = ['house', 'techno', 'trance', 'drum and bass']):
    '''
    Given a set of strings, determines if any of the strings contain a string in 
    genres.  Returns those genres.  
    Input:
        tags (set): set of strings
        genres (list): list of strings to replace
    Return:
        set of strings from genres
    '''
    genre = set()
    for label in genres:
        for tag in tags:
            if label in tag:
                genre.add(label)
    
    return genre