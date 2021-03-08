from typing import Optional, Iterable, Dict
import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from src.data import utils

# ML library and structural library
try:
    from src.data.aflowml.client import AFLOWmlAPI
except:
    raise NameError("AFLOWmlAPI not present. Have you remembered to download it?")

from pymatgen import Structure
from pymatgen.io.vasp.inputs import Poscar
from pymatgen.io.cif import CifParser

from src.data.get_data_MP import data_MP
from . import get_data_base

class data_AFLOWML(get_data_base.data_base):
    def __init__(self, API_KEY: Optional[str] = None, MAPI_KEY: Optional[str] = None):

        self.API_KEY = API_KEY
        self.MAPI_KEY = MAPI_KEY
        self.data_dir = Path.cwd().parent / "data"
        self.raw_data_path = self.data_dir / "raw" / "AFLOWML" / "AFLOWML.pkl"
        self.interim_data_path = self.data_dir / "interim" / "AFLOWML" / "AFLOWML.pkl"
        self.df = None

    def _apply_query(self, sorted: Optional[bool])-> pd.DataFrame:

        # Get data from Materials Project
        try:
            MP = data_MP(API_KEY = self.MAPI_KEY)
        except:
            raise ValueError("AFLOW-ML is dependent on MP data. Add MAPI_KEY argument\
            to class constructor.")
        entries = MP.get_dataframe()

        self.df = get_dataframe_AFLOW(entries=entries)

        print("Writing to raw data...")
        self.df.to_pickle(self.data_dir / "raw"  / "AFLOWML" / "new_AFLOWML.pkl")

        return self.df;

    def get_data_AFLOWML(entries: pd.DataFrame)-> Dict:
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

    def get_dataframe_AFLOWML(entries: pd.DataFrame)-> pd.DataFrame:
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
        return pd.DataFrame.from_dict(get_data_AFLOWML(entries))

    def _sort(self, entries: pd.DataFrame)-> pd.DataFrame:

        bandgap = np.empty(len(entries))
        bandgap[:] = np.nan

        for i, mpid in tqdm(enumerate(entries["material_id"])):
            for j, mid in enumerate(self.df["material_id"]):
                if mpid==mid:
                    bandgap[i] = float(self.df["ml_egap"].iloc[j])

        sorted_df = pd.DataFrame({"aflowml_bg": bandgap})

        sorted_df.to_pickle(self.interim_data_path)
        return sorted_df

    def sort_with_MP(self, entries: pd.DataFrame)-> pd.DataFrame:

        if os.path.exists(self.interim_data_path):
            sorted_df = pd.read_pickle(self.interim_data_path)
        else:
            sorted_df = self._sort(entries)
        utils.countSimilarEntriesWithMP(sorted_df["aflowml_bg"], "AFLOW-ML")
        return sorted_df
