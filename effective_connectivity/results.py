from statistical_model import Study
import importlib
import sys
# # Load unmerged and merged study datasets
study = Study.load("study_merged.cdb")
study2 = Study.load("study_unmerged.cdb")

print("Unmerged study summary:")
study.summary()