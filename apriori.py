import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori


def dataset(path):
    dataset_obj = pd.read_csv(path)
    return dataset_obj

