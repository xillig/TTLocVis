
# problem: i cant install qgis with conda

import sys
import pandas as pd
sys.path.append(r'C:\Users\gilli\PycharmProjects\TTLocVis') #add path to be able to import produced classes.
from TTLocVis_classes import LDAAnalyzer

pd.set_option('display.max_columns', None)  # show all columns
q = LDAAnalyzer.load_lda_analyzer_object(load_path=r'C:\Users\gilli\OneDrive\Desktop\test', obj_name='my_LDAAnalyzer_Object.pkl')


vlayer = QgsVectorLayer(uri, "layer", "delimitedtext")