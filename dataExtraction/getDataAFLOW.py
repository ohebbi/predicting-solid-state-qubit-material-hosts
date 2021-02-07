#!/usr/bin/env python3

# Tables
import pandas as pd

# Visual representation of the query process
from tqdm import tqdm

# Query library
from aflow import *

import os

def get_data_AFLOW(compound_list, keys, batch_size, catalog="icsd", fileName = False):
    """
    A function used to make a query to AFLOW.
    ...
    Args
    ----------
    compound_list : list (dim:N)
        A list of strings containing full formula, eg. H2O1 or Si1C1
    keys : list (dim:M)
        A list containing the features of the compound, found in documentation of AFLUX.
        eg. Egap
    batch_size : int
        Number of data entries to return per HTTP request
    catalog : str
        "icsd" for ICSD
    fileName : str
        Path to file, e.g. "data/aflow_ml.csv"
        Writing to a file during iterations. Recommended for a large compound list.

    Returns
    -------
    dict
        A dictionary containing the resulting matching queries. This can result
        in several matching compounds for each compound.
    """
    def writeToFile(fileNamePath, row):
        row.to_csv(fileNamePath,
            sep=",",
            index=False,
            header=False,
            mode='a')
    index = 0
    aflow_dict = {k: [] for k in keys}
    for compound in tqdm(compound_list):
        print("Current query: {}".format(compound))

        results = search(catalog=catalog, batch_size=batch_size)\
            .filter(K.compound==compound)

        if len(results)>0:
            for result in tqdm(results):
                for key in keys:
                    try:
                        aflow_dict[key].append(getattr(result,key))
                    except:
                        aflow_dict[key].append("None")
                if (fileName) and (index % 50 == 0):
                    pd.DataFrame.from_dict(aflow_dict).to_csv(fileName, sep=",")

                index += 1
        else:
            print("No compound is matching the search")
            continue

    if fileName:
        pd.DataFrame.from_dict(aflow_dict).to_csv(fileName, sep=",")

    return aflow_dict

def get_dataframe_AFLOW(compound_list, keys, batch_size, catalog="icsd", fileName = False):
    """
    A function used to make a query to AFLOW.
    ...
    Args
    ----------
    See get_Data_AFLOW()

    Returns
    -------
    pd.DataFrame (dim:MxN)
        A DataFrame containing the resulting matching queries. This can result
        in several matching compounds for each compound.
    """
    return pd.DataFrame.from_dict(get_data_AFLOW(compound_list, keys, batch_size, catalog, fileName))


def dataMiningCandidates():

    if not os.path.exists('../dataMining/data/databases/AFLOW'):
        os.makedirs('../dataMining/data/databases/AFLOW')

    directory = "../dataMining/data"

    #reading entries from MP
    MP_entries = pd.read_csv(directory+"/stage_2/MP_data_stage_2.csv", sep=",")
    compound_list = list(MP_entries['full_formula'])

    #choosing keys used in AFLOW. We will here use all features in AFLOW.
    keys = list(pd.read_csv(directory+"/AFLOW/AFLOW_keywords.txt", sep=",").columns)
    aflow_dict = {k: [] for k in keys}

    AFLOW_entries = get_dataframe_AFLOW(compound_list=compound_list, keys=keys, batch_size=1000, catalog="icsd",fileName=directory+"/AFLOW/AFLOW_data.csv")

    #writing to csv
    AFLOW_entries.to_csv(directory+"/AFLOW/AFLOW_data.csv", sep=",", index = False)


def initialMP():
    directory = "../dataMining/data/databases/initialDataMP"
    if not os.path.exists(directory + "/AFLOW"):
        os.makedirs(directory + "/AFLOW")
        
    #reading entries from MP
    MP_entries = pd.read_csv(directory+"/MP/MP_FLAGBIGFILE.csv", sep=",")
    compound_list = list(MP_entries['full_formula'])

    #choosing keys used in AFLOW. We will here use all features in AFLOW.
    keys = list(pd.read_csv(directory+"/AFLOW/AFLOW_keywords.txt", sep=",").columns)
    aflow_dict = {k: [] for k in keys}

    AFLOW_entries = get_dataframe_AFLOW(compound_list=compound_list, keys=keys, batch_size=1000, catalog="icsd",fileName=directory+"/AFLOW/AFLOW_data.csv")

    #writing to csv
    AFLOW_entries.to_csv(directory+"/AFLOW/AFLOW_data.csv", sep=",", index = False)


if __name__ == '__main__':
    #dataMiningCandidates()
    initialMP()
