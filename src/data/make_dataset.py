# -*- coding: utf-8 -*-
import os
import click
import logging
import numpy as np
from pathlib import Path
import dotenv

import pandas as pd
from src.data import (
    get_data_AFLOW,
    get_data_AFLOWML,
    get_data_Citrine,
    get_data_JARVIS,
    get_data_MP,
    get_data_OQMD
    )


def get_all_data(data_dir):#MAPI_KEY:str, CAPI_KEY:str):
    MAPI_KEY = os.getenv("MAPI_KEY")
    CAPI_KEY = os.getenv("CAPI_KEY")

    #MP
    MP = get_data_MP.data_MP(API_KEY=MAPI_KEY)
    entries = MP.get_dataframe()

    # CI
    citrine = get_data_Citrine.data_Citrine(CAPI_KEY)
    experimental_entries = citrine.get_dataframe()
    sorted_citrine = citrine.sort_with_MP(entries)

    # OQMD
    OQMD = get_data_OQMD.data_OQMD()
    OQMDentries = OQMD.get_dataframe()
    sorted_oqmd = OQMD.sort_with_MP(entries)

    # AFLOW
    AFLOW = get_data_AFLOW.data_AFLOW()
    AFLOWentries = AFLOW.get_dataframe()
    sorted_aflow = AFLOW.sort_with_MP(entries)

    #AFLOW-ML
    AFLOWML = get_data_AFLOWML.data_AFLOWML()
    AFLOWML_entries = AFLOWML.get_dataframe()
    sorted_aflowml = AFLOWML.sort_with_MP(entries)

    #JARVIS
    JARVIS = get_data_JARVIS.data_JARVIS()
    JARVIS_entries = JARVIS.get_dataframe()
    sorted_jarvis = JARVIS.sort_with_MP(entries)


    bandGaps = pd.DataFrame({
    "material_id":     entries["material_id"],
    "MP_Eg":           entries["band_gap"],
    "OQMD_Eg":         sorted_oqmd["oqmd_bg"],
    "AFLOW_Eg":        sorted_aflow["aflow_bg"],
    "AFLOW-fitted_Eg": sorted_aflow["aflow_bg_fit"],
    "AFLOWML_Eg":      sorted_aflowml["aflowml_bg"],
    "JARVIS-TBMBJ_Eg": sorted_jarvis["jarvis_bg_tbmbj"],
    "JARVIS-OPT_Eg":   sorted_jarvis["jarvis_bg_opt"],
    "Exp_Eg":          sorted_citrine["citrine_bg"],
    "spillage":        sorted_jarvis["jarvis_spillage"] # Adding this here so we wont forget it
    })

    spaceGroups = pd.DataFrame({
        "material_id":    entries["material_id"],
        "MP_sg":          entries["spacegroup.number"],
        "OQMD_sg":        sorted_oqmd["oqmd_sg"],
        "AFLOW_sg_orig":  sorted_aflow["aflow_sg_orig"],
        "AFLOW_sg_relax": sorted_aflow["aflow_sg_relax"]
    })

    icsdIDs = pd.DataFrame({
        "material_id": entries["material_id"],
        "MP_icsd":     entries["icsd_ids"],
        "OQMD_icsd":   sorted_oqmd["oqmd_icsd"],
        "AFLOW_icsd":  sorted_aflow["aflow_icsd"]
    })

    def convertToInt(df, col):
        df[col] = df[col].fillna(-1)\
                         .astype(int)\
                         .astype(str)\
                         .replace('-1', np.nan)
        return df

    spaceGroups = convertToInt(spaceGroups, ["OQMD_sg",  "AFLOW_sg_orig","AFLOW_sg_relax"])
    icsdIDs     = convertToInt(icsdIDs,     ["OQMD_icsd","AFLOW_icsd"])


    bandGaps   .to_pickle(data_dir / "interim" / "bandgaps.pkl")
    spaceGroups.to_pickle(data_dir / "interim" / "spaceGroups.pkl")

    return bandGaps, spaceGroups, icsdIDs

#@click.command()
#@click.argument('data_dir', type=click.Path())
def main(data_dir):
    """ Runs data extracting scripts to save raw data (../raw) and organize
        the data ready to be preprocessed (saved in ../interim).
    """
    logger = logging.getLogger(__name__)
    logger.info('obtaining data')

    # Get all data
    bandGaps, spaceGroups, icsdIDs = get_all_data(data_dir)
    logger.info('Obtaining data is done')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]
    data_dir = project_dir / "data"

    dotenv.load_dotenv(project_dir / ".env")

    main(data_dir)
