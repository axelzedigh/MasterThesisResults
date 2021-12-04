"""Function for normalizing the collected test traces."""

import os
import random

import numpy as np
from tqdm import tqdm

import AES as ENCRYPT
import produce_round_key
from configs.variables import PROJECT_DIR

from utils.aes_utils import SBOX as AES_Sbox

# HW = [bin(n).count("1") for n in range(256)]
path = "datasets/test_traces/Zedigh_2021/office_corridor/15m/"
device = "device_8"
save_folder = os.path.join(PROJECT_DIR, path, device, "data")

pt_save = os.path.join(save_folder, 'pt.npy')
ct_save = os.path.join(save_folder, 'ct.npy')
key_save = os.path.join(save_folder, 'key.npy')
roundkey10_save = os.path.join(save_folder, '10th_roundkey.npy')
keylist_save = os.path.join(save_folder, 'keylist.npy')
trace_save = os.path.join(save_folder, 'traces.npy')
nor_trace_maxmin_save = os.path.join(save_folder, 'nor_traces_maxmin.npy')
nor_trace_meanstd_save = os.path.join(save_folder, 'nor_traces_meanstd.npy')
label_save = os.path.join(save_folder, 'label_lastround_Sout_0.npy')

# ===========================================================
# data_dir = os.getenv("MASTER_THESIS_RESULTS_RAW_DATA")
data_dir = os.getenv("HOME")
# load_folder = "datasets/test_traces/Zedigh_2021/office_corridor/2m/device_9"
load_folder = "Desktop/1120_office_hall/15m/device_8_pointed"
load_folder = os.path.join(data_dir, load_folder)
mis_index_path = os.path.join(load_folder, 'mis_index.npy')
plaintext_path = os.path.join(load_folder, 'pt_.txt')
key_path = os.path.join(load_folder, 'key_.txt')
trace_path = os.path.join(load_folder, 'all__')

total_number = 5000
select_number = 4900
averaging = 1

total_index = [i for i in range(total_number)]
missing_index = np.load(mis_index_path)
real_index = [x for x in total_index if x not in missing_index]

select = random.sample(range(len(real_index)), select_number)
#select = [x for x in range(select_number)]

# Read trace data
Trace = []
nor_trace_maxmin = []
nor_trace_meanstd = []

print(list(missing_index))

for i in tqdm(real_index[:], ncols=60):
    path = trace_path + str(i) + '.npy'
    all_trace = np.load(path)
    trace_select = random.sample(range(0, len(all_trace)), averaging)
    one_trace = all_trace[trace_select].mean(axis=0)
    Trace.append(one_trace)

    # Max-min normalization
    MAX = np.max(one_trace)
    MIN = np.min(one_trace)
    # MAX = np.max(one_trace[204:314])
    # MIN = np.min(one_trace[204:314])

    nor_one_trace_maxmin = (one_trace - MIN) / (MAX - MIN)
    nor_trace_maxmin.append(nor_one_trace_maxmin)

    '''
    nor_one_trace_maxmin = np.zeros(len(one_trace))
    for j in range(len(one_trace)):
        nor_one_trace_maxmin[j] = (one_trace[j] - MIN) / (MAX - MIN)

    nor_trace_maxmin.append(nor_one_trace_maxmin)

    # for mean-std normalization
    nor_one_trace_meanstd = (one_trace - np.mean(one_trace)) / np.std(one_trace)
    nor_trace_meanstd.append(nor_one_trace_meanstd)
    '''

Trace = np.array(Trace)[select]
nor_trace_maxmin = np.array(nor_trace_maxmin)[select]
# nor_trace_meanstd = np.array(nor_trace_meanstd)[select]

print('trace shape:', Trace.shape)

np.save(trace_save, Trace)  # for chipwhisperer and machine learning
np.save(nor_trace_maxmin_save, nor_trace_maxmin)  # for machine learning
# np.save(nor_trace_meanstd_save, nor_trace_meanstd)  # for machine learning

# read plaintext data-----------------------------------------------
plaintext = []
with open(plaintext_path, "r") as f:
    for line in f.readlines():
        line = line.strip('\n')  # delete /n
        plaintext.append(line)

for i in range(len(plaintext)):
    temp = []
    for j in range(16):
        byte = plaintext[i][2 * j: 2 * j + 2]
        dec = int(byte, 16)
        temp.append(dec)

    plaintext[i] = temp

plaintext = np.array(plaintext)

plaintext = plaintext[real_index]
plaintext = plaintext[select]

print('pt shape:', plaintext.shape)

np.save(pt_save, plaintext)  # for chipwhisperer

# read key data-----------------------------------------------
with open(key_path, 'r') as f:
    data = f.read()

key = []
for i in range(16):
    byte = data[2 * i: 2 * i + 2]
    dec = int(byte, 16)
    key.append(dec)

key = np.array(key)

hex_key = []
for x in key:
    hex_key.append(hex(x))

print('hex_key:', hex_key)

key_list = []
for i in range(len(select)):
    key_list.append(key)

key_list = np.array(key_list)

print('keylist length:', len(key_list))
np.save(key_save, key)  # for chipwhisperer
np.save(keylist_save, key_list)  # for chipwhisperer

# get ciphertext --------------------------------------------------------
ciphertexts = []
str_key = ''
for i in range(len(key)):
    str_key += format(key[i], '#04x').split('x')[-1]

print('str_key:', str_key)

for pt in tqdm(plaintext, ncols=60):
    str_pt = ''
    for i in range(len(pt)):
        str_pt += format(pt[i], '#04x').split('x')[-1]

    KEY = str_key
    DATA = str_pt
    aes = ENCRYPT.AES(mode='ecb', input_type='hex')
    ct = aes.encryption(DATA, KEY)
    ciphertexts.append(ct)

for i in range(len(ciphertexts)):
    temp = []
    for j in range(16):
        byte = ciphertexts[i][2 * j: 2 * j + 2]
        dec = int(byte, 16)
        temp.append(dec)

    ciphertexts[i] = temp

ciphertexts = np.array(ciphertexts)
np.save(ct_save, ciphertexts)

# get round key and get label of sbox out in the LAST round----------------------------------------------------
roundkey = produce_round_key.keyScheduleRounds(key, 0, 10)
roundkey = np.array(roundkey)
print('10th roundkey:', roundkey)
np.save(roundkey10_save, roundkey)

label_lastround_Sout_0 = []

for ct in ciphertexts:
    label_lastround_Sout_0.append(ct[0] ^ roundkey[0])

label_lastround_Sout_0 = np.array(label_lastround_Sout_0)
np.save(label_save, label_lastround_Sout_0)
