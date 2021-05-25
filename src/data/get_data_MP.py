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

        # TODO: Remove when all
        #if (isCurrentlyFeaturizing == False):

        with MPRester(self.API_KEY) as mpr:

            # Initial criteria
            criteria = {"icsd_ids": {"$gt": 0}, #All compounds deemed similar to a structure in ICSD
                        "band_gap": {"$gt": 0.1},
                        #"material_id":{"$in": featurizedData["material_id"].to_list()}
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

        bandgap_GGA = np.empty(len(entries["material_id"]))
        bandgap_GGA[:] = np.nan

        bandgaps_GGAU = np.copy(bandgap_GGA)

        bandgap_GGA[entries["run_type"]=="GGA"] = entries["band_gap"][entries["run_type"]=="GGA"]
        bandgaps_GGAU[entries["run_type"]=="GGA+U"] = entries["band_gap"][entries["run_type"]=="GGA+U"]

        sorted_df = pd.DataFrame({"mp_bg_gga":   bandgap_GGA,
                                  "mp_bg_gga_u": bandgaps_GGAU,
                                  "mp_bg":       entries["band_gap"],
                                  "material_id": entries["material_id"]})

        return sorted_df
