from typing import Optional, Iterable, Dict
import os
import pandas as pd
import numpy as np
import pickle
import wget
from pathlib import Path
from tqdm import tqdm
from src.data.utils import countSimilarEntriesWithMP, LOG, sortByMPID
# ML library and structural library
try:
    from src.data.aflowml.client import AFLOWmlAPI
except:
    raise NameError("AFLOWmlAPI not present. Have you remembered to download it?")

from pymatgen import Structure
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.io.cif import CifParser

from src.data.get_data_MP import data_MP
from src.data import get_data_base

class data_AFLOWML(get_data_base.data_base):
    def __init__(self, API_KEY: Optional[str] = None, MAPI_KEY: Optional[str] = None):

        self.API_KEY = API_KEY
        self.MAPI_KEY = MAPI_KEY
        self.data_dir = Path(__file__).resolve().parents[2] / "data"
        self.raw_data_path = self.data_dir / "raw" / "AFLOWML" / "AFLOWML.pkl"
        self.interim_data_path = self.data_dir / "interim" / "AFLOWML" / "AFLOWML.pkl"
        super().__init__()

    def get_data_AFLOWML(self, entries: pd.DataFrame)-> Dict:
        """
        A function used to initialise AFLOW-ML with appropiate inputs.
        ...
        Args
        ----------
        entries : Pandas DataFrame
        {
            "cif": {}
                - Materials Project parameter "cif", which is a dict
            "compound": []
                - list of strings
            "material id": []
                - list of strings
        }

        Returns
        -------
        dict
            A dictionary containing features as compound and material id,
            as well as the keys in the AFLOW-ML algorithm Property
            Labeled Material Fragments.
        """

        firstIteration = True
        for index, entry in tqdm(entries.iterrows()):

            struc = CifParser.from_string(entry["cif"]).get_structures()[0]

            poscar = Poscar(structure=struc)

            ml = AFLOWmlAPI()

            prediction = ml.get_prediction(poscar, 'plmf')

            if firstIteration:
                aflowml_dict = {k: [] for k in prediction.keys()}
                aflowml_dict["full_formula"] = []
                aflowml_dict["material_id"]  = []
                firstIteration = False

            for key in prediction.keys():
                aflowml_dict[key].append(prediction[key])

            aflowml_dict["full_formula"].append(entry["full_formula"])
            aflowml_dict["material_id"].append(entry["material_id"])
            if (index % 10 == 0):
                pd.DataFrame.from_dict(aflowml_dict).to_pickle(self.data_dir / "raw"  / "AFLOWML" / "new_AFLOW.pkl")

        return aflowml_dict

    def get_dataframe_AFLOWML(self, entries: pd.DataFrame)-> pd.DataFrame:
        """
        A function used to initialise AFLOW-ML with appropiate inputs.
        ...
        Args
        ----------
        See get_dataframe_AFLOW()

        Returns
        -------
        Pandas DataFrame
            A DataFrame containing features as compound and material id,
            as well as the keys in the AFLOW-ML algorithm Property
            Labeled Material Fragments.
        """
        return pd.DataFrame.from_dict(self.get_data_AFLOWML(entries))
    def _apply_query(self, sorted: Optional[bool])-> pd.DataFrame:

        # Add unique url id for figshare endpoint
        url = "https://ndownloader.figshare.com/files/26922764"
        file = wget.download(url)

        # Read and load pkl
        with open(file, 'rb') as f:
            df = pickle.load(f)
            os.remove(file)

        # Get data from Materials Project
        try:
            MP = data_MP(API_KEY = self.MAPI_KEY)
        except:
            raise ValueError("AFLOW-ML is dependent on MP data. Add MAPI_KEY argument\
            to class constructor.")
        entries = MP.get_dataframe()

        # Find if there are new entries in MP
        newEntries = entries[~entries["material_id"].isin(df["material_id"])]

        # Update if there are new entries
        if newEntries.shape[0]>0:
            LOG.info("New entries identified. Generating features...")

            AFLOWML_portion = self.get_dataframe_AFLOWML(entries=newEntries)

            df = pd.concat([df, AFLOWML_portion])
            df = sortByMPID(df)

        LOG.info("Writing to raw data...")
        df.to_pickle(self.data_dir / "raw"  / "AFLOWML" / "AFLOWML.pkl")

        return df;

    def _sort(self, df: pd.DataFrame, entries: pd.DataFrame)-> pd.DataFrame:


        sorted_df = df[df.material_id.isin(entries.material_id)]
        sorted_df = sorted_df.add_prefix("AFLOWML|")
        sorted_df = sorted_df.rename(columns={"AFLOWML|material_id": "material_id"})
        sorted_df = sorted_df.reset_index(drop=True)
        sorted_df.to_pickle(self.interim_data_path)
        return sorted_df

    def sort_with_MP(self, df: pd.DataFrame, entries: pd.DataFrame)-> pd.DataFrame:
        if os.path.exists(self.interim_data_path):
            sorted_df = pd.read_pickle(self.interim_data_path)
        else:
            sorted_df = self._sort(df, entries)
        countSimilarEntriesWithMP(sorted_df["AFLOWML|ml_egap"], "AFLOW-ML")
        return sorted_df
