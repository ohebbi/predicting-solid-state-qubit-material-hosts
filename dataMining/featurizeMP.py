import os

from matminer.data_retrieval.retrieve_MP import MPDataRetrieval

from matminer.featurizers.bandstructure import BandFeaturizer
from matminer.featurizers.dos import DOSFeaturizer

from tqdm import tqdm
import pandas as pd

def apply_featurizers(criterion, properties, mpdr):
    df = mpdr.get_dataframe(criteria=criterion, properties=properties)
    df = BandFeaturizer().featurize_dataframe(df, col_id="bandstructure",ignore_errors=True)
    df =  DOSFeaturizer().featurize_dataframe(df, col_id="dos",ignore_errors=True)
    return df.drop(["bandstructure", "dos"], axis=1)

def featurize_by_material_id(material_ids, api_key, fileName = False, props=None):
    """
    LOW MEMORY DEMAND function (compared to matminer, pymatgen), but without
    returning "bandstructure"- and "dos"- objects as features.
    Args
    ----------
    material_ids : list
        List containing strings of materials project IDs
    api_key : string
        Individual API-key for Materials Project.

    fileName : str
        Path to file, e.g. "data/aflow_ml.csv"
        Writing to a file during iterations. Recommended for large entries.
    props : list (Cannot contain "band_gap")
        Containing the wanted properties from Materials Project

    Returns
    -------
    Pandas DataFrame
        A DataFrame containing "bandstructure"- and "dos"-featurized features,
        in addition to features given in props.
    """


    mpd = MPDataRetrieval(api_key)

    if props == None:
        properties = ["material_id","full_formula", "bandstructure", "dos"]

    elif props != None:
        properties=props

    firstIndex=True

    for i, mpid in tqdm(enumerate(material_ids)):
        criteria = {"task_id":{"$eq": mpid}}
        if firstIndex:
            FeaturizedEntries = apply_featurizers(criteria, properties, mpd)
            firstIndex = False
            continue
        currentIterationBandStructures = apply_featurizers(criteria, properties, mpd)
        FeaturizedEntries = pd.concat([FeaturizedEntries,currentIterationBandStructures])
        if (fileName) and (i % 50 == 0):
            FeaturizedEntries.to_csv(fileName, sep=",")
    if fileName:
        FeaturizedEntries.to_csv(fileName, sep=",")
    return FeaturizedEntries

def runFeaturizeMP():
    api_key = "b7RtVfJTsUg6TK8E"
    MP_entries = pd.read_csv("data/databases/initialDataMP/MP/MP.csv", sep=",")
    print(MP_entries.shape)
    fileName = "data/databases/initialDataMP/MP/MP_featurized.csv"
    featurize_by_material_id(MP_entries["material_id"].values, api_key, fileName)

def concatenateReQueueFeaturizers(directory):
    if os.path.exists(directory+"/MP_featurized_final.csv"):
        raise ValueError("The file already exist. You should rename it before overwriting it.")
    numbers=[]
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            if (str(filename[:14]) == "MP_featurized_"):
                numbers.append(int(filename[14:-4]))
            continue
        else:
            continue
    numbers.sort()

    MP_featurized = pd.DataFrame({})
    for num in numbers:
        fileDirectory = directory+"/MP_featurized_" + str(num) + ".csv"
        MP_featurized_portion = pd.read_csv(fileDirectory, sep=",")
        if os.path.exists(fileDirectory):
            os.remove(fileDirectory)
        MP_featurized = pd.concat([MP_featurized,MP_featurized_portion]).reset_index(drop=True)
    MP_featurized_portion = pd.read_csv(directory+"/MP_featurized.csv", sep=",")
    MP_featurized = pd.concat([MP_featurized,MP_featurized_portion]).reset_index(drop=True)
    MP_featurized.to_csv(directory+"/MP_featurized.csv",sep=",")

def reQueueMPFeaturizer():
    #reading entries from MP
    directory="data/databases/initialDataMP/MP"
    MP_entries =    pd.read_csv(directory+"/MP_FLAGBIGFILE.csv", sep=",")
    MP_featurized = pd.read_csv(directory+"/MP_featurized.csv", sep=",")

    howFar = MP_entries[MP_entries["material_id"] == MP_featurized["material_id"].iloc[-1]].index.values
    MP_featurized.to_csv(direcctory+"/MP_featurized_" + str(howFar[0]) + ".csv", sep=",", index=False)

    api_key = "b7RtVfJTsUg6TK8E"
    fileName = directory+"/MP_featurized.csv"
    print(MP_entries["material_id"].iloc[howFar[0]+1:].values.shape)

    #featurize again
    featurize_by_material_id(MP_entries["material_id"].iloc[howFar[0]+1:].values, api_key, fileName)

    #add the files together
    concatenateReQueueFeaturizers(directory)

if __name__ == '__main__':
    #runFeaturizeMP()
    #if the above fails during first run, run the following and keep rerunning it when it fails:
    #reQueueMPFeaturizer()
