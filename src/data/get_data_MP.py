# -*- coding: utf-8 -*-
from pymatgen import MPRester
from typing import Optional, Iterable
import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from . import utils
from . import get_data_base

class data_MP(get_data_base.data_base):
    def __init__(self, API_KEY: str):

        self.API_KEY = API_KEY
        self.raw_data_path = Path.cwd().parent / "data" / "raw" / "MP" / "MP.pkl"
        self.df = None

    def _apply_query(self, sorted: Optional[bool] = True)-> pd.DataFrame:
        with MPRester(self.API_KEY) as mpr:

            # Initial criteria
            criteria = {"icsd_ids": {"$gt": 0}, #All compounds deemed similar to a structure in ICSD
                            "band_gap": {"$gt": 0.1}
                        }

            # Features
            props = ["material_id","full_formula","icsd_ids",
                    "spacegroup.number","band_gap","run_type",
                    "cif", "elements", "structure","pretty_formula"]

            # Query
            self.df = pd.DataFrame(mpr.query(criteria=criteria, properties=props))

        # Remove unsupported MPIDs
        self.df = utils.filterIDs(self.df)

        # Sort by ascending MPID order
        if (sorted):
            self.df = utils.sortByMPID(self.df)

        print("Writing to raw data...")
        self.df.to_pickle(self.raw_data_path)
        return self.df;

    def sort_with_MP(self, entries: pd.DataFrame)-> np.array:

        return self.df["full_formula"]
