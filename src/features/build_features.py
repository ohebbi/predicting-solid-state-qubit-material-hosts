import os
import pandas as pd
import numpy as np
import logging
import wget
import time
import pickle

from src.features import preset
from src.features import featurizer
from src.data.utils import LOG

from matminer.data_retrieval.retrieve_MP import MPDataRetrieval
from tqdm import tqdm
from pathlib import Path
from src.data.get_data_MP import data_MP
import dotenv


def featurize_by_material_id(material_ids: np.array,
                             featurizerObject: featurizer.extendedMODFeaturizer,
                             MAPI_KEY: str,
                             writeToFile: bool = True) -> pd.DataFrame:
    """ Run all of the preset featurizers on the input dataframe.
    Arguments:
        df: the input dataframe with a `"structure"` column
            containing `pymatgen.Structure` objects.
    Returns:
        The featurized DataFrame.
    """
    def apply_featurizers(criterion, properties, mpdr, featurizerObject):
        LOG.info("Downloading dos and bandstructure objects..")

        timeDownloadStart = time.time()
        df_portion = mpdr.get_dataframe(
            criteria=criterion, properties=properties)
        timeDownloadEnd = time.time()

        LOG.info(df_portion)
        df_time, df_portion = featurizerObject.featurize(df_portion)
        df_time["download_objects"] = [timeDownloadEnd-timeDownloadStart]

        return df_time, df_portion

    properties = ["material_id", "full_formula",
                  "bandstructure", "dos", "structure"]

    mpdr = MPDataRetrieval(MAPI_KEY)

    steps = 1
    leftover = len(material_ids) % steps

    df = pd.DataFrame({})
    df_timers = pd.DataFrame({})

    for i in tqdm(range(0, len(material_ids), steps)):
        portionReturned = True
        if not (i+steps > len(material_ids)):

            LOG.info(list(material_ids[i:i+steps]))
            criteria = {"task_id": {"$in": list(material_ids[i:i+steps])}}

            while (portionReturned):
                try:
                    df_time, df_portion = apply_featurizers(
                        criteria, properties, mpdr, featurizerObject)
                    portionReturned = False
                except:
                    LOG.info("Except - try again.")

            # Add ID to recognize afterwards
            df_portion["material_id"] = material_ids[i:i+steps]

            df = pd.concat([df, df_portion])
            df_timers = pd.concat([df_timers, df_time])

            LOG.info("CURRENT SHAPE:{}".format(df.shape))
            if writeToFile:
                df.to_pickle(Path(__file__).resolve(
                ).parents[2] / "data" / "raw" / "featurizer" / "featurized.pkl")
                df_timers.to_csv(Path(__file__).resolve(
                ).parents[2] / "data" / "raw" / "featurizer" / "timing.csv")

    if (leftover):
        LOG.info(list(material_ids[i:i+leftover]))
        criteria = {"task_id": {"$in": list(material_ids[i:i+leftover])}}
        df_time, df_portion = apply_featurizers(
            criteria, properties, mpdr, featurizerObject)
        df_portion["material_id"] = material_ids[i:i+leftover]

        df = pd.concat([df, df_portion])
        df_timers = pd.concat([df_timers, df_time])
        if writeToFile:
            df.to_pickle(Path(__file__).resolve(
            ).parents[2] / "data" / "raw" / "featurizer" / "featurized.pkl")
            df_timers.to_csv(Path(__file__).resolve(
            ).parents[2] / "data" / "raw" / "featurizer" / "timing.csv")

    return df


def run_featurizer() -> None:
    """ Function used to run, and rerun a featurization process of a large amount of entries.
        As default, we use the initial query from Materials Project. Initialised by
        "make features"

        If program stops, identify mistake (most likely an error in Materials Project
        (add to filterIDs)), remove raw data in Materials Project data folder, and
        rerun with "make features" command.

    """

    project_dir = Path(__file__).resolve().parents[2]
    data_dir = project_dir / "data"

    dotenv.load_dotenv(project_dir / ".env")

    MAPI_KEY = os.getenv("MAPI_KEY")
    MP = data_MP(API_KEY=MAPI_KEY)
    entries = MP.get_dataframe()
    material_ids = entries["material_id"]
    del entries, MP

    featurizerObject = preset.PRESET_HEBNES_2021()

    if Path(data_dir / "raw" / "featurizer" / "featurized.pkl").is_file():

        # If errors met, just rerun and this if-test will run.
        LOG.info("In-progress featurized data identified. Reading now...")

        entries_featurized = pd.read_pickle(
            data_dir / "raw" / "featurizer" / "featurized.pkl")
        time_featurized = pd.read_csv(
            data_dir / "raw" / "featurizer" / "timing.csv")

        LOG.info("Last featurized MPID: {}".format(
            entries_featurized.index[-1]))

        howFar = material_ids[material_ids ==
                              entries_featurized.index[-1]].index.values

        # Test if mpid index is the same, true if using the same dataset
        assert material_ids[howFar[0]
                            ] == entries_featurized.index[-1], "Are you sure this is the same dataset as earlier?"

        LOG.info("Index: {}".format(howFar))
        LOG.info("Preparing for new featurized data starting with MPID: {}".format(
            material_ids[howFar[0]]))

        entries_featurized.to_pickle(
            data_dir / "raw" / "featurizer" / Path("featurized-upto-" + str(howFar[0]) + ".pkl"))
        time_featurized.to_csv(
            data_dir / "raw" / "featurizer" / Path("timing-upto-" + str(howFar[0]) + ".csv"))

        del entries_featurized, time_featurized

        df = featurize_by_material_id(
            material_ids[howFar[0]+1:], featurizerObject, MAPI_KEY)

    else:
        # First time running featurizers.
        df = featurize_by_material_id(
            entries["material_id"], featurizerObject, MAPI_KEY)


def updateNumberFeaturizedEntries(entries: pd.DataFrame,
                                  featurizedEntries: pd.DataFrame,
                                  MAPI_KEY: str) -> pd.DataFrame:
    """ Function that checks if new entries that have not been featurized,
        and if true will featurize.
    """
    if entries.shape[0] > featurizedEntries.shape[0]:
        # Find new entries
        newEntries = entries.material_id[~entries.material_id.isin(
            featurizedEntries.material_id.values)]
        # Define featurizer preset
        featurizerObject = preset.PRESET_HEBNES_2021()
        # Update with new entries
        newEntries = featurize_by_material_id(
            newEntries, featurizerObject, MAPI_KEY, writeToFile=False)
        # Add new entries
        featurizedEntries = pd.concat([featurizedEntries, newEntries])
    elif entries.shape[0] < featurizedEntries.shape[0]:
        featurizedEntries = featurizedEntries[featurizedEntries.material_id.isin(
            entries.material_id.values)]
        featurizedEntries = featurizedEntries.reset_index(drop=True)

    if entries.shape[0] == featurizedEntries.shape[0]:
        print("Updated featurized entries, shape: {}".format(
            featurizedEntries.shape))
    assert (entries.shape[0] == featurizedEntries.shape[0]), "Not equal length, {}!={}"\
        .format(entries.shape[0], featurizedEntries.shape[0])

    return featurizedEntries


def testUpdateFeaturisedEntries(entries: pd.DataFrame,
                                featurizedEntries: pd.DataFrame,
                                MAPI_KEY: str):
    """ Test to see if updateNumberFeaturizedEntries work as intended

        Removes one entry and checks if the resulting featurized entry is equal to the
        removed one.
    """
    # Choosing an arbitrary featurizedEntry
    suddenlyLostEntry = featurizedEntries.iloc[0]
    # Woops! Where did it go?
    featurizedEntries = featurizedEntries[1:]
    # Puh, we can get it back!
    featurizedEntries = updateNumberFeaturizedEntries(
        entries, featurizedEntries, MAPI_KEY)
    # But is it back, though?
    assert featurizedEntries.iloc[0].equals(suddenlyLostEntry)
    # Yey, it's back!
    print("Test passed.")


def does_file_exist(filepath: Path) -> bool:
    """
    Checks if file path exists.

    """

    if os.path.exists(filepath):
        LOG.info("Data path detected:\n{}\.".format(filepath))
        return True
    else:
        LOG.info("Data path\n{}\nnot detected. Downloading now...".format(filepath))
        return False


def get_featurized_data() -> pd.DataFrame:
    """ A function that checks if featurized data is present in folder, if not,
        will download and store the data.

        Returns a dataframe
    """
    
    featurized_data_path = Path(__file__).resolve().parents[2] / \
        "data" / "raw" / "featurized" 

    featurized_file_path = featurized_data_path / "featurized-11-04-2021.pkl"
    if not does_file_exist(featurized_file_path):
        # Add unique url id for figshare endpoint
        url = "https://ndownloader.figshare.com/files/26777699"
        file = wget.download(url)

        # Read and load pkl
        with open(file, 'rb') as f:
            df = pickle.load(f)
            # Make directory if not present
            Path(featurized_data_path).mkdir(parents=True, exist_ok=True)
            df.to_pickle(featurized_file_path)
            os.remove(file)
    else:
        LOG.info("Reading data..")
        df = pd.read_pickle(featurized_file_path)
    return df


def main():
    get_featurized_data()

    LOG.info("Done")


if __name__ == '__main__':
    # main()
    run_featurizer()
