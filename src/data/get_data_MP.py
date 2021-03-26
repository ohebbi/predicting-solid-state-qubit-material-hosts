# -*- coding: utf-8 -*-
from pymatgen import MPRester
from typing import Optional, Iterable
import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from src.data.utils import filterIDs, sortByMPID, LOG
from src.data import get_data_base

class data_MP(get_data_base.data_base):
    def __init__(self, API_KEY: str):

        self.API_KEY = API_KEY
        self.raw_data_path = Path(__file__).resolve().parents[2] / "data" / "raw" / "MP" / "MP.pkl"
        super().__init__()

    def _apply_query(self, sorted: Optional[bool] = True)-> pd.DataFrame:
        with MPRester(self.API_KEY) as mpr:

            # Initial criteria
            criteria = {"icsd_ids": {"$gt": 0}, #All compounds deemed similar to a structure in ICSD
                            "band_gap": {"$gt": 0.1}
                        }

            # Features
            props = ["material_id","full_formula","icsd_ids",
                    "spacegroup.number","spacegroup.point_group", "band_gap","run_type",
                    "cif", "structure","pretty_formula","total_magnetization",
                    "nelements", "efermi", "oxide_type"]

            # Query
            df = pd.DataFrame(mpr.query(criteria=criteria, properties=props))

        # Remove unsupported MPIDs
        df = filterIDs(df)
        LOG.info("Current shape of dataframe after filter applied: {}".format(df.shape))
        # Sort by ascending MPID order
        if (sorted):
            df = sortByMPID(df)

        LOG.info("Writing to raw data...")
        df.to_pickle(self.raw_data_path)
        return df;

    def sort_with_MP(self, entries: pd.DataFrame)-> np.array:

        return entries["full_formula"].values
