# 加载数据文件
# 130462条数据，每个有ID Atoms Distance U0四个字段
# 原子最多29个，distance矩阵对角线为0，其余取值范围是：0.95 -- 12.05, u0范围[-1101.48779008] [-19444.38734855]

import numpy as np
import pandas as pd
import csv

csv.register_dialect('tsv', delimiter = '\t', quoting = csv.QUOTE_ALL)


class NpzLoader:
    def __init__(self, file_path):
        self.file_path = file_path
    
    def load_file(self):
        self.npzout = np.load(self.file_path)
        return self.npzout


class PdLoader:
    def __init__(self, file_path):
        self.file_path = file_path
    
    def load_file(self):
        pdout = pd.read_csv(self.file_path, sep = '\t')
        return pdout

# example npzloader
# path = r'F:\ML\ML_mole\EnergyEst\Resources\QM9_nano.npz'
# loader = NpzLoader(path)
# result = loader.load_file()
# print(result['ID'][0][4:10])  # qm9: 000001
# print(result['Atoms'][0])
# print(result['Distance'][0])
# print(result['U0'][0])

# example pdloader
# path = r'F:\ML\ML_mole_c\EnergyEst\Resources\CM.tsv'
# loader = PdLoader(path)
# result = loader.load_file()
# print(result)

# with open(path,) as csvfile:
#     file_list = csv.reader(csvfile,'tsv')
#     for line in file_list:
#         print(line)
