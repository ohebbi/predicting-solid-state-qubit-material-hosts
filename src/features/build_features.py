import os
import pandas as pd
import numpy as np
import logging
import wget
from src.features.utils.utils import LOG

from src.features import preset
from src.features import featurizer
from src.features.utils.utils import LOG
from matminer.data_retrieval.retrieve_MP import MPDataRetrieval
from tqdm import tqdm
from pathlib import Path
from src.data.get_data_MP import data_MP
import dotenv

def featurize_by_material_id(material_ids: np.array, featurizerObject: featurizer.extendedMODFeaturizer, MAPI_KEY: str) -> pd.DataFrame:
    """ Run all of the preset featurizers on the input dataframe.
    Arguments:
        df: the input dataframe with a `"structure"` column
            containing `pymatgen.Structure` objects.
    Returns:
        The featurized DataFrame.
    """
    def apply_featurizers(criterion, properties, mpdr, featurizerObject):
        LOG.info("Downloading dos and bandstructure objects..")
        df_portion = mpdr.get_dataframe(criteria=criterion, properties=properties)
        LOG.info(df_portion)
        df_portion = featurizerObject.featurize(df_portion)
        return df_portion

    properties = ["material_id","full_formula", "bandstructure", "dos", "structure"]

    mpdr = MPDataRetrieval(MAPI_KEY)

    steps = 50
    leftover = len(material_ids)%steps
    df = pd.DataFrame({})
    for i in tqdm(range(0,len(material_ids),steps)):
        portionReturned = True
        if not (i+steps > len(material_ids)):
            LOG.info(list(material_ids[i:i+steps]))
            criteria = {"task_id":{"$in":list(material_ids[i:i+steps])}}
            while (portionReturned):
                try:
                    df_portion = apply_featurizers(criteria, properties, mpdr, featurizerObject)
                    portionReturned = False
                except:
                    LOG.info("Except - try again.")
            df = pd.concat([df,df_portion])
            LOG.info("CURRENT SHAPE:{}".format(df.shape))
            df.to_pickle(Path(__file__).resolve().parents[2] / "data" / "raw" / "featurizer" / "raw.pkl")
    if (leftover):
        criteria = {"task_id":{"$in":list(material_ids[i:i+leftover])}}
        df_portion = apply_featurizers(criteria, properties, mpdr, featurizerObject)
        df = pd.concat([df,df_portion])
        df.to_pickle(Path(__file__).resolve().parents[2] / "data" / "raw" / "featurizer" / "raw.pkl")

    return df


def run_featurizer():

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]
    data_dir = project_dir / "data"

    dotenv.load_dotenv(project_dir / ".env")

    MAPI_KEY = os.getenv("MAPI_KEY")
    MP = data_MP(API_KEY=MAPI_KEY)
    entries = MP.get_dataframe()
    entries = entries["material_id"].values

    featurizerObject = preset.PRESET_HEBNES_2021()
    df = featurize_by_material_id(entries, featurizerObject, MAPI_KEY)



def does_file_exist(filepath:Path)-> bool:
    """
    Checks if file path exists.

    """

    if os.path.exists(filepath):
        LOG.info("Data path detected:\n{}\.".format(filepath))
        return True
    else:
        LOG.info("Data path\n{}\nnot detected. Downloading now...".format(filepath))
        return False

def get_featurized_data():

    featurized_data_path = Path(__file__).resolve().parents[2] / \
                            "data" / "interim"  / "featurized" \
                            / "featurized-11-04-2021.pkl"

    if not does_file_exist(featurized_data_path):
        # Add unique url id for figshare endpoint
        url = "https://ndownloader.figshare.com/files/26777699"
        file = wget.download(url)

        # Read and load pkl
        with open(file, 'rb') as f:
            df = pickle.load(f)
            df.to_pickle(featurized_data_path)
            os.remove(file)
    else:
        LOG.info("Reading data..")
        df = pd.read_pickle(featurized_data_path)
    return df

def main():
    get_featurized_data()

    LOG.info("Done")
if __name__ == '__main__':
    #main()
    run_featurizer()
