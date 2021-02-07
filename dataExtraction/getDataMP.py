from pymatgen import MPRester
import pandas as pd

def get_data_MP(api_key, filename, sorted=True):
    with MPRester(api_key) as mpr:

        criteria = {"icsd_ids": {"$gt": 0}, #All compounds deemed similar to a structure in ICSD
                        "band_gap": {"$gt": 0.1}
                    }

        props = ["material_id","full_formula","icsd_ids", "spacegroup.number","band_gap","run_type","cif","e_above_hull", "elements", "structure","pretty_formula"]#,'pretty_formula','e_above_hull',"band_gap"]

        entries = pd.DataFrame(mpr.query(criteria=criteria, properties=props))
    if (sorted):
        mpid_num = []
        for i in entries["material_id"]:
            mpid_num.append(int(i[3:]))
        entries["mpid_num"] = mpid_num
        entries = entries.sort_values(by="mpid_num").reset_index(drop=True)
        entries = entries.drop(columns=["mpid_num"])

    entries.to_csv(filename, sep=",", index = False)

    return entries;

def small_file_extract_all_MP_prop(api_key, filename):
    with MPRester(api_key) as mpr:

        criteria = {"icsd_ids": {"$gt": 0}, #All compounds deemed similar to a structure in ICSD
                        "band_gap": {"$gt": 0.1}
                    }

        props = ["material_id","full_formula","icsd_ids", "spacegroup.number","band_gap","run_type","cif","e_above_hull", "elements"]#, "structure"]#,'pretty_formula','e_above_hull',"band_gap"]

        entries = pd.DataFrame(mpr.query(criteria=criteria, properties=props))
    entries.to_csv(filename, sep=",", index = False)
    return;

if __name__ == '__main__':
    get_data_MP("b7RtVfJTsUg6TK8E","../dataMining/data/databases/initialDataMP/MP/MP_FLAGBIGFILE.csv",sorted=True)

    #small_file_extract_all_MP_prop("b7RtVfJTsUg6TK8E","MP_smallfile.csv")
