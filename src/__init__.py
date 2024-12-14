# Mathmetcial packages
import numpy as np
import pandas as pd
import scipy
import scipy.stats as stats
import scipy.io as sio
import math
import random as rdm

# Deep learning packages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, MinMaxScaler
from tensorflow.python.client import device_lib
from tensorflow.keras.layers import Dense, LeakyReLU
# import h5py

# System packages
import os, sys, json, argparse, time, itertools
from time import time
from tqdm import tqdm
from functools import partial
tqdm = partial(tqdm, position=0, leave=True)

# Plotting packages
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()
from sklearn.metrics import confusion_matrix
from matplotlib.colors import LinearSegmentedColormap

sys.path.append('./src/')
import libs.pywarraychannels as pywarraychannels
# import libs.MOMP as MOMP
# import CommSys as CommSys
from libs.CommSysInfo import *





