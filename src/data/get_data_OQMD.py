# -*- coding: utf-8 -*-
from typing import Optional
from matminer.data_retrieval.retrieve_MDF import MDFDataRetrieval
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
from src.data.utils import countSimilarEntriesWithMP, LOG
from src.data import get_data_base

class data_OQMD(get_data_base.data_base):
    def __init__(self, API_KEY: Optional[str] = None):

        # Consistency - no need for API key for OQMD
        self.API_KEY = API_KEY
        self.data_dir = Path(__file__).resolve().parents[2] / "data"
        self.raw_data_path= self.data_dir / "raw" / "OQMD" / "OQMD.pkl"
        self.interim_data_path = self.data_dir / "interim" / "OQMD" / "OQMD.pkl"
        super().__init__()

    def _apply_query(self, sorted: Optional[bool])-> pd.DataFrame:

        # Query
        mdf = MDFDataRetrieval (anonymous = True)
        df = mdf.get_dataframe({
                    "source_names": ['oqmd'],
                    "match_fields": {"oqmd.converged": True}                    },
                    unwind_arrays=False)

        # Applying filters for unneccessary data
        df = df[["crystal_structure.space_group_number",
                "dft.exchange_correlation_functional",
                "material.composition",
                "crystal_structure.cross_reference.icsd",
                "oqmd.band_gap.value",
                "dc.relatedIdentifiers"]]
        df = df[df["oqmd.band_gap.value"]>0]
        df['crystal_structure.cross_reference.icsd'] = df['crystal_structure.cross_reference.icsd'].fillna(0)
        df["crystal_structure.space_group_number"]= df["crystal_structure.space_group_number"].astype(int)
        df["crystal_structure.cross_reference.icsd"] = df["crystal_structure.cross_reference.icsd"].astype(int)
        df = df.reset_index(drop=True)

        LOG.info("Writing to raw data...")
        df.to_pickle(self.raw_data_path)

        return df;

    def _sort(self, df: pd.DataFrame, entries: pd.DataFrame)-> pd.DataFrame:

        bandgaps = np.empty(len(entries))
        bandgaps[:] = np.nan

        spacegroups = np.copy(bandgaps)
        ICSDs       = np.copy(bandgaps)

        LOG.info("total iterations: {}".format(len(entries)))
        for i, icsd_list in tqdm(enumerate(entries["icsd_ids"])):
            for j, oqmd_icsd in enumerate(df["crystal_structure.cross_reference.icsd"]):
                for icsd in icsd_list:
                    if icsd == oqmd_icsd:
                        spacegroups[i] = int(df["crystal_structure.space_group_number"].iloc[j])
                        bandgaps[i] = df["oqmd.band_gap.value"].iloc[j]
                        ICSDs[i] = int(oqmd_icsd)

        sorted_df = pd.DataFrame({"oqmd_bg":   bandgaps,
                                  "oqmd_sg":   spacegroups,
                                  "oqmd_icsd": ICSDs,
                                  "material_id": entries["material_id"]})

        sorted_df.to_pickle(self.interim_data_path)
        return sorted_df

    def sort_with_MP(self, df: pd.DataFrame, entries: pd.DataFrame)-> pd.DataFrame:

        if os.path.exists(self.interim_data_path):
            sorted_df = pd.read_pickle(self.interim_data_path)
        else:
            sorted_df = self._sort(df, entries)
        countSimilarEntriesWithMP(sorted_df["oqmd_bg"], "OQMD")
        return sorted_df
