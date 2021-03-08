# -*- coding: utf-8 -*-
from typing import Optional
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
from . import utils
from . import get_data_base

from jarvis.db.figshare import data

class data_JARVIS(get_data_base.data_base):
    def __init__(self, API_KEY: Optional[str] = None):

        # Consistency - no need for API key for JARVIS
        self.API_KEY = API_KEY
        self.data_dir = Path.cwd().parent / "data"
        self.raw_data_path= self.data_dir / "raw" / "JARVIS" / "JARVIS.pkl"
        self.interim_data_path = self.data_dir / "interim" / "JARVIS" / "JARVIS.pkl"
        self.df = None

    def _apply_query(self, sorted: Optional[bool])-> pd.DataFrame:

        # Query
        self.df = pd.DataFrame(data('dft_3d'))\
                               .replace("na", np.nan)\
                               .replace("None", np.nan)\
                               .fillna(value=np.nan)\
                               .dropna(subset=['icsd'])

        icsd_list = []

        # ICSD-column is not consequent in notation, therefore we present a fix
        for icsd_jarvis in self.df["icsd"]:
            if isinstance(icsd_jarvis, str):

                if isinstance(eval(icsd_jarvis), int):
                    icsd_list.append([eval(icsd_jarvis)])

                elif isinstance(eval(icsd_jarvis), list):
                    icsd_list.append(eval(icsd_jarvis))

            elif isinstance(icsd_jarvis, float):
                icsd_list.append([icsd_jarvis])

        self.df["icsd"] = icsd_list
        self.df = self.df[self.df["optb88vdw_bandgap"]>0].reset_index(drop=True)

        print("Writing to raw data...")
        self.df.to_pickle(self.raw_data_path)

        return self.df;

    def _sort(self, entries: pd.DataFrame)-> pd.DataFrame:

        bandgaps_tbmbj = np.empty(len(entries))
        bandgaps_tbmbj[:] = np.nan

        bandgaps_opt = np.copy(bandgaps_tbmbj)
        spillage    = np.copy(bandgaps_tbmbj)

        print("total iterations: {}".format(len(entries)))
        for i, mp_icsd_list in tqdm(enumerate(entries["icsd_ids"])):
            #print("her", mp_icsd_list)
            for j, jarvis_icsd_list in enumerate(self.df["icsd"]):
                #print(mp_icsd_list)
                for icsd_mp in (mp_icsd_list):
                    #print(jarvis_icsd_list)
                    for icsd_jarvis in (jarvis_icsd_list):
                        if icsd_mp == int(icsd_jarvis):
                            bandgaps_tbmbj[i] = float(self.df["mbj_bandgap"].iloc[j])
                            bandgaps_opt[i]   = float(self.df["optb88vdw_bandgap"].iloc[j])
                            spillage[i]      = float(self.df["spillage"].iloc[j])

        sorted_df = pd.DataFrame({"jarvis_bg_tbmbj": bandgaps_tbmbj,
                                  "jarvis_bg_opt":   bandgaps_opt,
                                  "jarvis_spillage": spillage})

        sorted_df.to_pickle(self.interim_data_path)
        return sorted_df

    def sort_with_MP(self, entries: pd.DataFrame)-> pd.DataFrame:

        if os.path.exists(self.interim_data_path):
            sorted_df = pd.read_pickle(self.interim_data_path)
        else:
            sorted_df = self._sort(entries)
        utils.countSimilarEntriesWithMP(sorted_df["jarvis_bg_tbmbj"], "JARVIS tbmbj")
        utils.countSimilarEntriesWithMP(sorted_df["jarvis_bg_opt"],   "JARVIS opt")
        utils.countSimilarEntriesWithMP(sorted_df["jarvis_spillage"], "JARVIS spillage")

        return sorted_df
