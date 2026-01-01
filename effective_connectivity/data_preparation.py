from statistical_model import Study
from helpers_functions import *
import os
import glob
from pathlib import Path
from tqdm import tqdm


def extract_conditions_from_path(file_path):
    """Extract measurement conditions from file path."""
    path = Path(file_path)
    
    # Extract city
    if "DTF_WAW" in str(path):
        city = "Warszawa"
    elif "DTF_KRK" in str(path):
        city = "Kraków"
    else:
        city = "Unknown"
    
    # Extract eye condition
    if "/EC/" in str(path) or "\\EC\\" in str(path):
        eyes = "eyes closed"
    elif "/EO/" in str(path) or "\\EO\\" in str(path):
        eyes = "eyes open"
    else:
        eyes = "Unknown"
    
    # Extract wave type
    frequency_band = {
        "delta": "delta",
        "theta": "theta",
        "alpha": "alpha",
        "beta" : "beta" ,
        "gamma": "gamma",
    }
    
    band = "Unknown"
    for key, value in frequency_band.items():
        if key in path.name.lower():
            band = value
            break
    
    return {
        "city": city,
        "eyes": eyes,
        "bands": band
    }

node_labels = {
    # Default Mode Network (DMN) nodes
    'L-mPFC': 'DMN',
    'R-mPFC': 'DMN',
    'L-IFG': 'DMN',
    'L-PREC': 'DMN',
    'R-PREC': 'DMN',
    'L-ANG': 'DMN',
    'R-ANG': 'DMN',
    'L-aSTG': 'DMN',
    'R-aSTG': 'DMN',
    
    # Salience Network (SN) nodes
    'L-ACC': 'SN',
    'R-ACC': 'SN',
    'L-INS': 'SN',
    'R-INS': 'SN',
    
    # Central Executive Network (CEN) nodes
    'R-IFG': 'CEN',
    'L-SFG': 'CEN',
    'R-SFG': 'CEN',
    'L-pSTG': 'CEN',
    'R-pSTG': 'CEN',
    'L-ITG': 'CEN',
    'R-ITG': 'CEN',
    'L-SUP': 'CEN',
    'R-SUP': 'CEN'
}


study = Study(nodes=node_labels, control_group_name = 9)
base_dir = "data"
excel_files = glob.glob(os.path.join(base_dir, "**/*.xlsx"), recursive=True)



for file_path in tqdm(excel_files):
    # Extract conditions from path
    conditions = extract_conditions_from_path(file_path)
    
    # Import the measurement with extracted conditions
    study.import_measurement_from_excel(file_path, 
                                        measurement_conditions=conditions,
                                        independent_samples=["city"])

study.summary()



study2 = study.merge_independent_condition(["city"])
study2.summary()
study2.permute(n_permutations=1000)
study2.save("study_merged_1000.cdb")

study.permute(n_permutations=1000)
study.save("study_unmerged_1000.cdb")

