import pandas as pd
import logging
import sys
from pymatgen.symmetry.groups import SYMM_DATA, sg_symbol_from_int_number


def sortByMPID(df: pd.DataFrame) -> pd.DataFrame:
    mpid_num = []
    for i in df["material_id"]:
        mpid_num.append(int(i[3:]))
    df["mpid_num"] = mpid_num
    df = df.sort_values(by="mpid_num").reset_index(drop=True)
    df = df.drop(columns=["mpid_num"])
    return df

def filterIDs(df: pd.DataFrame) -> pd.DataFrame:
    unsupportedMPIDs = ["mp-28709",  #C120S32
                        "mp-28905",  #Sr6C120
                        "mp-28979",  #Ba6C120
                        "mp-29281",  #Th24P132
                        "mp-555563", #PH6C2S2NCl2O4 #DOI: 10.17188/1268877
                        "mp-560718", #Te4H48Au4C16S12N4
                        "mp-568028", #C120
                        "mp-568259", #Ta4Si8P4H72C24N8Cl24
                        "mp-574148", #K16Zn8N96
                        "mp-583476", #Nb7S2I19      #DOI: 10.17188/1277059
                        "mp-600172", #Cu8H96C40S32N8
                        "mp-600205", #H10C5SeS2N3Cl          #DOI: -
                        "mp-600217", #H80C40Se8S16Br8N24     #DOI: -
                        "mp-603254", #P8H72Au8C24S24Cl8
                        "mp-645279", #C136O2F40
                        "mp-645316", #C140F60
                        "mp-645364", #Sr24P48N96
                        "mp-646059", #C156Cl36
                        "mp-646122", #C160Cl24
                        "mp-646669", #P112Pb20I8
                        "mp-647169", #C120F36
                        "mp-647192", #C112Cl20
                        "mp-647725", #Os20C68O64
                        "mp-648157", #Os24C76O80
                        "mp-680326", #P24C48S48N72
                        "mp-680329", #K48As112
                        "mp-698375", #Cu8H96C40S32N8
                        "mp-705194", #Mn16Sn8C80Br8O80
                        "mp-705526", #H64Au4C24S8N16Cl4O16
                        "mp-706304", #H72Ru4C24S12N12Cl4O12
                        "mp-707239", #H32C8Se4S8Br8N16
                        "mp-720895", #Re4H88C16S16N32Cl32O12 # not enough memory
                        "mp-722571", #Re20H20C80O80
                        "mp-744395", #Ni4H72C16S24N32O16
                        "mp-744919", #Mn6Mo4H68C44N32O10
                        "mp-782100", #As16H96C32S28N8
                        "mp-1195164", #Cu4B4P16H96C32S16F16
                        "mp-1195290", #Ga3Si5P10H36C12N4Cl11 #DOI: -
                        "mp-1195608", #C200Cl44
                        "mp-1195791", #Zr8H256C80I8N40
                        "mp-1196206", #C216Cl24
                        "mp-1196358", #P4H120Pt8C40I8N4Cl8   #DOI: -
                        "mp-1196439", #Sn8P4H128C44N12Cl8O4  #DOI: -
                        "mp-1198652", #Te4H72C36S24N12Cl4    #DOI: -
                        "mp-1198926", #Re8H96C24S24N48Cl48   #DOI: -
                        "mp-1199490", #Mn4H64C16S16N32Cl8    #DOI: -
                        "mp-1199686", #Mo4P16H152C52N16Cl16  #DOI: -
                        "mp-1203403", #C121S2Cl20            #DOI: -
                        "mp-1204279", #Si16Te8H176Pd8C64Cl16 #DOI: -
                        "mp-1204629"] #P16H216C80N32Cl8     #DOI: -
    print("A total of {} MPIDs are inconsistent with the rest."
          .format(len(unsupportedMPIDs)))

    dropped = 0
    for unsupportedMPID in unsupportedMPIDs:
        if unsupportedMPID in list(df["material_id"].values):
            df = df.drop(df[df["material_id"] == str(unsupportedMPID)].index)
            dropped += 1
    print("A total of {} MPIDs were dropped from the dataset provided."
          .format(len(unsupportedMPIDs)))

    df = df.reset_index(drop=True)
    return df

def countSimilarEntriesWithMP(listOfEntries, nameOfDatabase):
    similarEntries = 0
    for i in listOfEntries:
        if i>=0:
            similarEntries += 1

    LOG.info("The amount of similar entries between MP and {} is {},".format(nameOfDatabase, similarEntries))
    LOG.info("which is {} percent".format(similarEntries/len(listOfEntries)))

def polarGroupUsedInMP():
    """
    Materials Project has more space groups than normal convention. This function finds
    all the polar groups for materials project extended list of space groups.
    """
    # This is a list of the point groups as noted in pymatgen
    point_groups = []
    for i in range(1,231):
        symbol = sg_symbol_from_int_number(i)
        point_groups.append(SYMM_DATA['space_group_encoding'][symbol]['point_group'])

    # Note that there are 40 of them, rather than 32.
    print("Number of point groups denoted in pymatgen: {}".format(len(set(point_groups))))

    # This is because multiple conventions are used for the same point group.
    # This dictionary can be used to convert between them.
    point_group_conv = {'321' :'32', '312': '32', '3m1' :'3m', '31m': '3m',
                        '-3m1' : '-3m', '-31m': '-3m', '-4m2': '-42m', '-62m': '-6m2' }

    # Using this dictionary we can correct to the standard point group notation.
    corrected_point_groups = [point_group_conv.get(pg, pg) for pg in point_groups]
    # Which produces the correct number of point groups. 32.
    print("Number of point groups in conventional notation: {}".format(len(set(corrected_point_groups))))

    # There are 10 polar point groups
    polar_point_groups = ['1', '2', 'm', 'mm2', '4', '4mm', '3', '3m', '6', '6mm']

    # Polar spacegroups have polar point groups.
    polar_spacegroups = []

    # There are 230 spacegroups
    for i in range(1,231):
        symbol = sg_symbol_from_int_number(i)
        pg = SYMM_DATA['space_group_encoding'][symbol]['point_group']
        if point_group_conv.get(pg, pg) in polar_point_groups:
            polar_spacegroups.append(i)

    # 68 of the 230 spacegroups are polar.
    print("Number of polar spacegroups: {}".format(len(polar_spacegroups)))

    return polar_spacegroups

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
LOG.addHandler(handler)
