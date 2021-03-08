# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from . import utils
from pymatgen import MPRester
import pandas as pd

from matminer.data_retrieval.retrieve_Citrine import CitrineDataRetrieval

#def get_all_data():
#    continue



def get_data_OQMD(filename):
    mdf = MDFDataRetrieval (anonymous = True)
    df = mdf.get_dataframe({
                "source_names": ['oqmd']#,
                },
                unwind_arrays=False)
    df.to_pickle(filename)
    return df


"""
def main(input_filepath, output_filepath):
    # Runs data processing scripts to turn raw data from (../raw) into
    #    cleaned data ready to be analyzed (saved in ../processed).

    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


if __name__ == '__main__':
    @click.command()
    @click.argument('input_filepath', type=click.Path(exists=True))
    @click.argument('output_filepath', type=click.Path())

    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]
    print(project_dir)
    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
"""
