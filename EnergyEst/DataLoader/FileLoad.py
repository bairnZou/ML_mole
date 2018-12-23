# 加载数据文件
# 130462条数据，每个有ID Atoms Distance U0四个字段

import numpy as np
import pandas as pd
class NpzLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_file(self):
        self.npzout = np.load(self.file_path)
        return self.npzout



# class PdLoader:
#     def __init__(self, file_path):
#         self.file_path = file_path
#
#     def load_file(self):
#         pdout = pd.read_csv(self.file_path, sep='\t')
#         return pdout

# example npzloader
# path = r'F:\ML\ML_mole\EnergyEst\Resources\QM9_nano.npz'
# loader = NpzLoader(path)
# result = loader.load_file()
# print(result['ID'][0][4:10])  # qm9: 000001
# print(result['Atoms'][0])
# print(result['Distance'][0])
# print(result['U0'][0])

# example pdloader
# path = r'F:\ML\ML_mole\EnergyEst\Resources\CM.tsv'
# loader = NpzLoader(path)
# result = loader.load_file()
# print(result)