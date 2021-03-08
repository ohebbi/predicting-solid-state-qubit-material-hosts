import pandas as pd
import numpy as np
from . import featurizeAll
from . import featurizer
from matminer.data_retrieval.retrieve_MP import MPDataRetrieval
from tqdm import tqdm
def featurize_by_material_id(material_ids: np.array, featurizerObject: featurizer.MPFeaturizer, MAPI_KEY: str) -> pd.DataFrame:
    """ Run all of the preset featurizers on the input dataframe.
    Arguments:
        df: the input dataframe with a `"structure"` column
            containing `pymatgen.Structure` objects.
    Returns:
        The featurized DataFrame.
    """
    def apply_featurizers(criterion, properties, mpdr, featurizerObject):
        print("Downloading dos and bandstructure objects..")
        df_portion = mpdr.get_dataframe(criteria=criterion, properties=properties)
        print(df_portion)
        df_portion = featurizerObject.featurize(df_portion)#BandFeaturizer().featurize_dataframe(df, col_id="bandstructure",ignore_errors=True)
        return df_portion

    properties = ["material_id","full_formula", "bandstructure", "dos", "structure"]

    mpdr = MPDataRetrieval(MAPI_KEY)

    steps = 50
    leftover = len(material_ids)%steps
    df = pd.DataFrame({})
    for i in tqdm(range(0,len(material_ids),steps)):
        if not (i+steps > len(material_ids)):
            print(list(material_ids[i:i+steps]))
            criteria = {"task_id":{"$in":list(material_ids[i:i+steps])}}
            df_portion = apply_featurizers(criteria, properties, mpdr, featurizerObject)
            df = pd.concat([df,df_portion])
    if (leftover):
        criteria = {"task_id":{"$in":list(material_ids[i:i+leftover])}}
        df_portion = apply_featurizers(criteria, properties, mpdr, featurizerObject)
        df = pd.concat([df,df_portion])
    return df
