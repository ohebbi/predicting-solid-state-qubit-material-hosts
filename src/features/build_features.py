import os
import pandas as pd
import numpy as np
import logging
import wget

from src.features import preset
from src.features import featurizer
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
        print("Downloading dos and bandstructure objects..")
        df_portion = mpdr.get_dataframe(criteria=criterion, properties=properties)
        print(df_portion)
        df_portion = featurizerObject.featurize(df_portion)#BandFeaturizer().featurize_dataframe(df, col_id="bandstructure",ignore_errors=True)
        return df_portion

    properties = ["material_id","full_formula", "bandstructure", "dos", "structure"]

    mpdr = MPDataRetrieval(MAPI_KEY)

    steps = 50
    leftover = len(material_ids)%steps
    df = pd.DataFrame({})
    for i in tqdm(range(0,len(material_ids),steps)):
        if not (i+steps > len(material_ids)):
            print(list(material_ids[i:i+steps]))
            criteria = {"task_id":{"$in":list(material_ids[i:i+steps])}}
            df_portion = apply_featurizers(criteria, properties, mpdr, featurizerObject)
            df = pd.concat([df,df_portion])
            df.to_pickle(Path(__file__).resolve().parents[2] / "data" / "raw" / "featurizer" / "raw.pkl")
    if (leftover):
        criteria = {"task_id":{"$in":list(material_ids[i:i+leftover])}}
        df_portion = apply_featurizers(criteria, properties, mpdr, featurizerObject)
        df = pd.concat([df,df_portion])
        df.to_pickle(Path(__file__).resolve().parents[2] / "data" / "raw" / "featurizer" / "raw.pkl")

    return df

def does_file_exist(filepath:Path)-> bool:
    """
    Checks if file path exists.

    """
    logger = logging.getLogger(__name__)

    if os.path.exists(filepath):
        logger.info("Data path detected:\n{}\.".format(filepath))
        return True
    else:
        logger.info("Data path\n{}\nnot detected. Downloading now...".format(filepath))
        return False

def get_featurized_data() -> pd.DataFrame:
    logger = logging.getLogger(__name__)

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
        logger.info("Reading data..")
        df = pd.read_pickle(featurized_data_path)
    return df

def main():
    # Initialise logger
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)

    get_featurized_data()

    logger.info("Done")
def run_featurizer():

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]
    data_dir = project_dir / "data"

    dotenv.load_dotenv(project_dir / ".env")

    MAPI_KEY = os.getenv("MAPI_KEY")
    MP = data_MP(API_KEY=MAPI_KEY)
    entries = MP.get_dataframe()

    featurizerObject = preset.PRESET_HEBNES_2021()
    df = featurize_by_material_id(entries["material_id"].values, featurizerObject, MAPI_KEY)

if __name__ == '__main__':
    #main()
    run_featurizer()
