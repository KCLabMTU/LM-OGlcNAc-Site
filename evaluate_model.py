""" 
      Author  : Suresh Pokharel
      Email   : sureshp@mtu.edu
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from Bio import SeqIO
from keras import backend as K
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef
from sklearn.svm import SVC
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (Conv1D, Dense, Dropout, Flatten, Input,
                                     LeakyReLU, MaxPooling1D, Reshape)
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import Adam


test_ids = np.array(['Q9NWS8', 'Q9ERB0', 'Q8CDG3', 'Q9Y467', 'Q5DU62', 'P08133', 'Q9DBE8', 'P00815', 'B1AQX6', 'O60487', 'O15230', 'Q68FF6', 'Q8NEV1', 'Q9LD55', 'D3Z2W0', 'A4FVR1', 'Q04758', 'Q96B23', 'Q08018', 'E9PYL2', 'Q9NRL3', 'Q640L6', 'Q6RHR9', 'P17252', 'O43318', 'Q9Y3R0', 'A3KMH1', 'P97355', 'Q6ZWR6', 'P26450', 'Q5HZA9', 'O95169', 'E9PHY5', 'O75400', 'P00387', 'P49432', 'Q5VW38', 'Q9DAU1', 'A0A080YU64', 'Q9BPZ2', 'Q9M7Q3', 'D3Z0V7', 'Q6EVK6', 'Q4FZZ4', 'Q9NW64', 'P38606', 'Q6ZMW3', 'Q9D0F1', 'P13010', 'Q9Y692', 'P35611', 'Q8R1S4', 'P40091', 'Q8TDJ6', 'P10860', 'O15357', 'Q8K0U8', 'Q9Z1Z9', 'P32432', 'Q8N3X1', 'Q12797', 'P23246', 'Q5C9Z4', 'O35927', 'P48556', 'Q68FF7', 'B0FRH4', 'Q9WVB2', 'Q9FM47', 'Q9UHY8', 'M7ZEC3', 'Q8IYB3', 'M7Z9B8', 'Q9Y6R9', 'P04844', 'Q8IWD5', 'Q04545', 'O43525', 'Q80V24', 'P39769', 'Q9D2G5', 'Q66HF1', 'Q15415', 'P47985', 'Q9H2P0', 'Q9BWF2', 'Q03496', 'O82263', 'H0YM25', 'P36123', 'Q8NE18', 'A2AIR2', 'Q6N022', 'Q9D2U0', 'P19224', 'O75506', 'Q9UKU6', 'Q9NUU6', 'G3XA57', 'Q9P265', 'Q6ZN03', 'E9PWM3', 'P84102', 'Q9P2K1', 'E9QLZ9', 'Q7Z434', 'Q9BSH4', 'Q12830', 'P35504', 'Q9NXD2', 'Q6P2Q9', 'P40227', 'Q99JG3', 'Q8N5M1', 'Q80VC9', 'Q9Y5V3', 'A6H619', 'O88492', 'Q04779', 'Q86U90', 'M7YV95', 'Q14207', 'O43852', 'Q9WVJ0', 'M7YJZ3', 'P35590', 'O14775', 'Q86UU0', 'Q9VXM5', 'Q06330', 'Q9DBF1', 'Q7Z417', 'Q02809', 'Q02819', 'E9Q7E2', 'Q52M93', 'Q13613', 'P40121', 'P55290', 'P14317', 'Q9VUH6', 'P40477', 'D4A7L4', 'A6NIV6', 'Q8NHP8', 'E9QMJ1', 'Q9Y490', 'Q9UGH3', 'Q9Y2F5', 'Q15393', 'Q7Z2K8', 'Q8IVF2', 'O00268', 'Q14671', 'Q8NBK3', 'P29419', 'Q16760', 'Q08050', 'Q6N069', 'O14818', 'P35612', 'Q9UMN6', 'Q9D809', 'Q9BSJ8', 'Q92794', 'O75821', 'Q63544', 'P35564', 'P25425', 'P98175', 'Q9LP92', 'P11951', 'Q7L590', 'Q6EMB2', 'Q922Y0', 'A2AHG0', 'P16949', 'P35222', 'A1Z877', 'P49748', 'Q8N7C4', 'Q9Z0U1', 'Q9UN86', 'Q6NZJ6', 'Q6NZI2', 'Q6H8Q1', 'A0A0G2K717', 'M7Z201', 'Q9Z0I7', 'P31270', 'Q9WTS4', 'A2ALS5', 'Q06647', 'O54824', 'Q9D5W4', 'Q96D71', 'Q9H992', 'P05453', 'P46937', 'Q86X10', 'Q96GQ7', 'Q8NBJ4', 'P18760', 'Q6UXD7', 'Q9SW80', 'Q6GR78', 'Q16623', 'Q96IZ7', 'E9Q0N0', 'P32909', 'P25588', 'P13726', 'P42227', 'Q07020', 'D6RH90', 'Q12912', 'P46527', 'Q8CBY1', 'P54259', 'P29972', 'Q8R4S0', 'Q8CFN5', 'Q15256', 'Q62523', 'O95628', 'P56385', 'Q96CP2', 'P57740', 'Q86XR8', 'Q96K83', 'P24814', 'Q9NP62', 'Q61818', 'Q91YE8-2', 'F4KBP5', 'Q9Y2M5', 'Q8CBY8', 'Q8R4X3', 'O55017-2', 'E9Q3T6', 'P27816', 'Q80ZX0', 'O14543', 'Q9WVI9', 'Q2KHT3', 'M7ZUJ1', 'Q8BJ05', 'Q9HAT2', 'Q9NWY4', 'Q8VYJ2', 'J3QNT7', 'O70507', 'Q99504', 'Q9C040', 'Q21251', 'Q18758', 'Q9UJU6', 'Q9BQI6', 'Q6ZQB7', 'Q12967', 'Q15054', 'Q6ZPY7', 'Q9QXN5', 'Q70JQ1', 'Q3UYH7', 'Q6ZSR9', 'Q921I1', 'P70670', 'Q6PIY7', 'Q6PDY2', 'P78347', 'Q9QUM9', 'Q5BLP8', 'Q9BQ75', 'Q9NR09', 'P04899', 'P48377', 'Q571K4', 'Q8WVZ7', 'Q3UKK2', 'Q4DS83', 'Q61026', 'Q14093', 'P06748', 'Q9ET43', 'Q99543', 'Q5SQX6', 'Q99729', 'P0C7U0', 'O60675', 'V5CWD8', 'Q9BVP2', 'Q8R1X6', 'P49257', 'P08048', 'Q96EV2', 'P05064', 'Q8K327', 'A2AGT5-2', 'P05784', 'Q0KL02', 'M7ZIY9', 'Q9NQC3', 'Q9NSV4', 'Q96PQ5', 'O15178', 'Q63HM2', 'P37108', 'M7ZSW9', 'P28331', 'Q32M84', 'P60879', 'A0A080YUM7', 'Q9HAP2', 'P51659', 'A0A061AE78', 'Q8N5N7', 'Q68CP9', 'Q8CGB6-3', 'Q13887', 'Q6NXI6', 'Q8IWB9', 'P20108', 'Q8N8X9', 'P42858', 'Q86YP4', 'Q5BJF6', 'B9EKL9', 'Q9UG63', 'P23471', 'P63011', 'Q96GM5', 'A0A0G2K3A0', 'P62500', 'Q16563', 'Q64536', 'Q08951', 'Q9NYF8', 'M7ZDS9', 'Q96S52', 'P07602', 'A2AJ19', 'P43121', 'Q9Z0M3', 'Q9P0T7', 'P43366', 'Q8BKX6', 'Q9NVQ4', 'P34246', 'Q86TZ1', 'P04799', 'P51858', 'P51608', 'Q3UH66-5', 'P53762', 'P04075', 'Q9Y3I1', 'Q07655', 'P46676', 'H2KYC5', 'Q04958', 'O75143', 'Q9BXT4', 'Q9HD90', 'Q17RS7', 'M7Z469', 'P08758', 'Q13480', 'M7Z4M0', 'P52179', 'D6RHQ6', 'Q76E23', 'Q12901', 'O95696', 'P13056', 'Q9CVI2', 'P0C7M7', 'P07758', 'Q68FD9', 'Q9ULD6', 'P05783', 'Q9Z2W9', 'Q6UXY1', 'P70662', 'Q9P202', 'Q8BI79', 'F5GUI4', 'Q810A1', 'Q96LW1', 'P10144', 'P39880', 'P70365', 'P20810', 'M7YT19', 'Q8RXE7', 'Q8N9W4', 'Q5SRN2', 'P10688', 'A6X8Z5', 'Q9ERC8', 'Q68DK7', 'O43390', 'P10605', 'O75175', 'Q71U36', 'A8JV18', 'P09605', 'P83876', 'P62995', 'P20676', 'Q6ZPK0', 'Q08874', 'Q03164', 'Q8CHH5', 'Q9V4C8', 'Q92625', 'P15705', 'Q0P678', 'Q6IMP4', 'Q8NI77', 'Q91YE2', 'Q9BRQ8', 'Q8IWA6', 'Q6ZUM4', 'B1AXH1', 'Q6NV74', 'Q9C0D5', 'P31948', 'P62910', 'Q8CHI8', 'Q9H2F5', 'G3UZ19', 'Q7TSC1', 'Q8K310', 'P61571', 'P49419', 'Q9GZY0', 'M7ZA56', 'Q3UU96', 'O62106', 'A6QL64', 'M7YFQ8', 'Q97520', 'Q13330', 'F4IS91', 'F7D291', 'P36041', 'Q7Z6G8', 'O60669', 'Q7TSG3', 'Q08379', 'Q9HCD5', 'Q15637', 'Q9UPP5', 'P17563', 'Q01518', 'Q63880', 'Q16620', 'P46100', 'O95922', 'P14209', 'Q92973', 'Q53YX8', 'Q86YN6', 'Q03968', 'P50502', 'Q6PCM2', 'P11279', 'Q7Z2Y8', 'P39053', 'Q9SD86', 'Q6BDI9', 'Q58A45', 'P11532', 'P32790', 'P55327', 'Q9W2U7', 'Q80TZ3', 'Q95Q32', 'Q9BXP5', 'P52564', 'Q92819', 'Q9Y450', 'P38716', 'Q96JY6', 'Q6F3F9', 'Q0KIW2', 'Q96A44', 'Q86VI3', 'Q96DY7', 'P14602', 'Q96Q83', 'F2Z3U3', 'P19429', 'Q5SV77', 'Q8WYQ9', 'C7IVR4', 'Q9NZN5', 'O95292', 'Q9WVL3', 'O61938', 'Q8BXR9', 'Q8C0I4', 'Q9H9B4', 'Q9NQI0', 'P10809', 'Q9BUJ2', 'P25444', 'O70263', 'Q6PKG0', 'Q15596', 'E9PV98', 'Q52M58', 'P08572', 'Q9DCS9', 'P82979', 'P16649', 'Q9D486', 'P04040', 'P16989', 'Q9Y2W1', 'Q9BZD4', 'P50991', 'Q8BG05', 'Q8CHC4', 'P23443', 'P56574', 'Q80YR6', 'Q86VF7'])


# read saved features
train_negative_unambiguous_df = pd.read_csv("data/features/ProtT5/train_negative_unambiguous_df.csv",header = 0)
train_positive_unambiguous_df = pd.read_csv("data/features/ProtT5/train_positive_unambiguous_df.csv",header = 0)

# select test data
test_positive_pt5 = train_positive_unambiguous_df[train_positive_unambiguous_df['accession'].isin(test_ids)].iloc[:,2:]
test_negative_pt5 = train_negative_unambiguous_df[train_negative_unambiguous_df['accession'].isin(test_ids)].iloc[:,2:]

# create labels
test_positive_labels = np.ones(test_positive_pt5.shape[0])
test_negative_labels = np.zeros(test_negative_pt5.shape[0])

# stack positive and negative data together
X_test_pt5 = np.vstack((test_positive_pt5,test_negative_pt5))
y_test = np.concatenate((test_positive_labels, test_negative_labels), axis = 0)

# load model
model = load_model('models/ANN_Final_Model.h5')

y_pred = model.predict(X_test_pt5).reshape(y_test.shape[0],)
y_pred = (y_pred > 0.59) # threshold calculated when training
y_pred = [int(i) for i in y_pred]
y_test = np.array(y_test)
y_pred = np.array(y_pred)

cm = confusion_matrix(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)

sn = cm[1][1]/(cm[1][1]+cm[1][0])
sp = cm[0][0]/(cm[0][0]+cm[0][1])

print("\n %s, %s, %s, %s, %s \n" %(str(acc), str(mcc), str(sn), str(sp), cm))
