# -*- coding: utf-8 -*-
from typing import Optional
from matminer.data_retrieval.retrieve_MDF import MDFDataRetrieval
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
from . import utils
from . import get_data_base

class data_OQMD(get_data_base.data_base):
    def __init__(self, API_KEY: Optional[str] = None):

        # Consistency - no need for API key for OQMD
        self.API_KEY = API_KEY
        self.data_dir = Path.cwd().parent / "data"
        self.raw_data_path= self.data_dir / "raw" / "OQMD" / "OQMD.pkl"
        self.interim_data_path = self.data_dir / "interim" / "OQMD" / "OQMD.pkl"
        self.df = None

    def _apply_query(self, sorted: Optional[bool])-> pd.DataFrame:

        # Query
        mdf = MDFDataRetrieval (anonymous = True)
        self.df = mdf.get_dataframe({
                    "source_names": ['oqmd'],
                    "match_fields": {"oqmd.converged": True}                    },
                    unwind_arrays=False)

        # Applying filters for unneccessary data
        self.df = self.df[["crystal_structure.space_group_number", "dft.exchange_correlation_functional", "material.composition", "crystal_structure.cross_reference.icsd", "oqmd.band_gap.value", "dc.relatedIdentifiers"]]
        self.df = self.df[self.df["oqmd.band_gap.value"]>0]
        self.df['crystal_structure.cross_reference.icsd'] = self.df['crystal_structure.cross_reference.icsd'].fillna(0)
        self.df["crystal_structure.space_group_number"]= self.df["crystal_structure.space_group_number"].astype(int)
        self.df["crystal_structure.cross_reference.icsd"] = self.df["crystal_structure.cross_reference.icsd"].astype(int)
        self.df = self.df.reset_index(drop=True)

        print("Writing to raw data...")
        self.df.to_pickle(self.raw_data_path)

        return self.df;

    def _sort(self, entries: pd.DataFrame)-> pd.DataFrame:

        bandgaps = np.empty(len(entries))
        bandgaps[:] = np.nan

        spacegroups = np.copy(bandgaps)
        ICSDs       = np.copy(bandgaps)

        print("total iterations: {}".format(len(entries)))
        for i, icsd_list in tqdm(enumerate(entries["icsd_ids"])):
            for j, oqmd_icsd in enumerate(self.df["crystal_structure.cross_reference.icsd"]):
                for icsd in eval(str(icsd_list)):
                    if icsd == oqmd_icsd:
                        spacegroups[i] = int(self.df["crystal_structure.space_group_number"].iloc[j])
                        bandgaps[i] = self.df["oqmd.band_gap.value"].iloc[j]
                        ICSDs[i] = int(oqmd_icsd)

        sorted_df = pd.DataFrame({"oqmd_bg":   bandgaps,
                                  "oqmd_sg":   spacegroups,
                                  "oqmd_icsd": ICSDs})

        sorted_df.to_pickle(self.interim_data_path)
        return sorted_df

    def sort_with_MP(self, entries: pd.DataFrame)-> pd.DataFrame:

        if os.path.exists(self.interim_data_path):
            sorted_df = pd.read_pickle(self.interim_data_path)
        else:
            sorted_df = self._sort(entries)
        utils.countSimilarEntriesWithMP(sorted_df["oqmd_bg"], "OQMD")
        return sorted_df
