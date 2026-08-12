from pathlib import Path
import sys

# define root path from which all relative paths are set
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# adding project root to list of locations to search for when
# importing libraries
sys.path.append(str(PROJECT_ROOT)) 
# paths for:
# data
DATA_DIR = PROJECT_ROOT / "data"
# outputs
FIG_DIR = PROJECT_ROOT/"figures"
RESULT_DIR = PROJECT_ROOT/"results"
# specific data directories
FELIX_DIR = DATA_DIR / "modes_surface"
CHAOS_DIR = DATA_DIR / "chaos_mats"
CHAOS_RESOL_DIR = DATA_DIR / "chaos_resolution_matrices"
CHAOS_COV_DIR = DATA_DIR/ "chaos_covariance"

# Defining optimal width for figures to fit on A4 paper
text_width = 7.25