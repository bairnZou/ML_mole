from DataLoader.FileLoad import *
import random


class QM9Data:
    def __init__(self, data_id, atoms, distance, u0, max_atoms_size):
        self.id = data_id
        self.atoms = np.zeros(max_atoms_size)
        self.inv_distance = np.zeros([max_atoms_size, max_atoms_size])
        self.u0 = u0
        self.atoms_num = len(atoms)
        for i in range(0, self.atoms_num):
            self.atoms[i] = atoms[i]
            for j in range(0, self.atoms_num):
                if distance[i][j] != 0:
                    self.inv_distance[i][j] = 1 / distance[i][j]
                else:
                    self.inv_distance[i][j] = 100
                    # 无穷大不存在，且不利于神经网络训练，数据集中distace_min = 0.95，100是inv_dis最大值得100倍。近似无穷大


class DataSet:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.data_list = []
        self.length = 0
    
    def add_data(self, qm9data):
        self.data_list.append(qm9data)
        self.length += 1
    
    def shuffle_data(self, shuffle_seed):
        random.seed(shuffle_seed)
        random.shuffle(self.data_list)


class PreProcesser:
    def __init__(self, training_proportion, shuffle_seed, max_atoms_size):
        self.training_set = DataSet(r'training data')
        self.test_set = DataSet(r'test data')
        self.all_data = DataSet(r'all data')
        self.training_proportion = training_proportion
        self.total_data = 0
        self.shuffle_seed = shuffle_seed
        self.max_atoms_size = max_atoms_size
    
    def load_data(self, file_path):
        print('load data:' + file_path)
        npz_loader = NpzLoader(file_path)
        npzout = npz_loader.load_file()
        id_list = npzout[r'ID']
        atoms_list = npzout[r'Atoms']
        distance_list = npzout[r'Distance']
        u0_list = npzout[r'U0']
        self.total_data = len(id_list)
        for i in range(0, self.total_data):
            qm9data = QM9Data(int(id_list[i][4:10]), atoms_list[i], distance_list[i], u0_list[i], self.max_atoms_size)
            self.all_data.add_data(qm9data)
        self.all_data.shuffle_data(self.shuffle_seed)
        for i in range(0, self.total_data):
            if i < int(self.total_data * self.training_proportion):
                self.training_set.add_data(self.all_data.data_list[i])
            else:
                self.test_set.add_data(self.all_data.data_list[i])
        print('load data complete')
