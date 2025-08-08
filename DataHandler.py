import pickle
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix
from Params import args
import scipy.sparse as sp
import torch as t
import torch.utils.data as data
import torch.utils.data as dataloader
import pandas as pd
from collections import Counter
import random
def read_file(file_path):
    data = pd.read_csv(file_path)
    selected_columns = data.iloc[:, [0, 3, 2]]
    selected_columns= selected_columns.dropna(subset=[selected_columns.columns[2]])
    selected_columns = selected_columns[selected_columns.iloc[:,2]!= 2.0]
    result_list = selected_columns.values.tolist()
    return result_list
def filter_data(data_list):
    # Counting user and item interactions
    user_counts = Counter([item[0] for item in data_list])
    item_counts = Counter([item[1] for item in data_list])

    # Filtering data
    filtered_data = [
        item for item in data_list
        if user_counts[item[0]] >= 10 and item_counts[item[1]] >= 10
    ]

    return filtered_data
def reindex_data(data):
    user_id_map = {}
    item_id_map = {}
    new_data = []

    # Create new IDs for user and item, starting from 0
    for record in data:
        user_id, item_id, user_age = record

        # Reindex user_id
        if user_id not in user_id_map:
            user_id_map[user_id] = len(user_id_map)

        # Reindex item_id
        if item_id not in item_id_map:
            item_id_map[item_id] = len(item_id_map)

        new_data.append([user_id_map[user_id], item_id_map[item_id], user_age])

    return new_data
class DataHandler:
	def __init__(self):
		if args.data == 'ml_100k':
			predir = './Datasets/ml_100k/'
		elif args.data == 'lastfm':
			predir = './Datasets/lastFM/'
		elif args.data == 'movielens':
			predir = './Datasets/movielens/'
		elif args.data == 'ijcai':
			predir = './Datasets/rent/'
		self.predir = predir
		self.trnfile = predir + 'trnMat.pkl'
		self.tstfile = predir + 'tstMat.pkl'
		if args.data == 'lastfm':  #  update
			self.senfile = predir + 'users_features.npy'
		elif args.data == 'ml_100k':
			self.senfile = predir + 'data/u.user'
		elif args.data == 'movielens':
			self.senfile = predir + 'users.dat'
		elif args.data == 'ijcai':
			self.senfile = predir + 'data_format2/new.csv'

	def loadOneFile(self, filename):
		with open(filename, 'rb') as fs:
			ret = (pickle.load(fs) != 0).astype(np.float32)
		if type(ret) != coo_matrix:
			ret = sp.coo_matrix(ret)
		return ret
	def loadSenFile(self, filename):
		if  args.data == 'movielens':
			sex_list = []
			with open(filename, 'r') as file:
				for line in file:
					_, sex, _, _, _ = line.strip().split('::')
					sex = 1 if sex == 'F' else 0
					sex_list.append(sex)     		#sex_list = [random.randint(0, 1) for _ in range(1892)]
		elif args.data == 'ml_100k':
			sex_list = []
			with open(filename, 'r') as file:
				for line in file:
					_, _, sex, _, _ = line.strip().split('|')
					sex = 1 if sex == 'F' else 0
					sex_list.append(sex)
		elif args.data == 'lastfm':
			train_data = np.load(filename, allow_pickle=True)
			sex_list = [int(i[0]) for i in train_data][:9900]
		elif args.data == 'ijcai':
			train_list = read_file(filename)
			filter_list = filter_data(train_list)
			reid_list = reindex_data(filter_list)
			user_genders = {}
			for record in reid_list:
				user_id, _, user_gender = record
				user_genders[user_id] = int(user_gender)
			sex_list = list(user_genders.values())
		return sex_list

	def normalizeAdj(self, mat):
		degree = np.array(mat.sum(axis=-1))
		dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
		dInvSqrt[np.isinf(dInvSqrt)] = 0.0
		dInvSqrtMat = sp.diags(dInvSqrt)
		return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()

	def makeTorchAdj(self, mat):
		# make ui adj
		a = sp.csr_matrix((args.user, args.user))
		b = sp.csr_matrix((args.item, args.item))
		mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
		mat = (mat != 0) * 1.0
		mat = (mat + sp.eye(mat.shape[0])) * 1.0
		mat = self.normalizeAdj(mat)

		# make cuda tensor
		idxs = t.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
		vals = t.from_numpy(mat.data.astype(np.float32))
		shape = t.Size(mat.shape)
		return t.sparse.FloatTensor(idxs, vals, shape).cuda()

	def LoadData(self):
		trnMat = self.loadOneFile(self.trnfile)
		args.user, args.item = trnMat.shape
		tstMat = self.loadOneFile(self.tstfile)
		senLabel = self.loadSenFile(self.senfile)
		self.trnMat = trnMat
		self.senLabel = senLabel
		self.torchBiAdj = self.makeTorchAdj(trnMat)
		trnData = TrnData(trnMat)
		self.trnLoader = dataloader.DataLoader(trnData, batch_size=args.batch, shuffle=True, num_workers=0)
		tstData = TstData(tstMat, trnMat)
		self.tstLoader = dataloader.DataLoader(tstData, batch_size=args.tstBat, shuffle=False, num_workers=0)

class TrnData(data.Dataset):
	def __init__(self, coomat):
		self.rows = coomat.row
		self.cols = coomat.col
		self.dokmat = coomat.todok()
		self.negs = np.zeros(len(self.rows)).astype(np.int32)

	def negSampling(self):
		for i in range(len(self.rows)):
			u = self.rows[i]
			while True:
				iNeg = np.random.randint(args.item)
				if (u, iNeg) not in self.dokmat:
					break
			self.negs[i] = iNeg

	def __len__(self):
		return len(self.rows)

	def __getitem__(self, idx):
		return self.rows[idx], self.cols[idx], self.negs[idx]

class ValData(data.Dataset):
	def __init__(self, coomat, trnMat):
		self.csrmat = (trnMat.tocsr() != 0) * 1.0

		valLocs = [None] * coomat.shape[0]
		valUsrs = set()
		for i in range(len(coomat.data)):
			row = coomat.row[i]
			col = coomat.col[i]
			if valLocs[row] is None:
				valLocs[row] = list()
			valLocs[row].append(col)
			valUsrs.add(row)
		tstUsrs = np.array(list(valUsrs))
		self.tstUsrs = tstUsrs
		self.tstLocs = valLocs

	def __len__(self):
		return len(self.valUsrs)

	def __getitem__(self, idx):
		return self.valUsrs[idx], np.reshape(self.csrmat[self.valUsrs[idx]].toarray(), [-1])

class TstData(data.Dataset):
	def __init__(self, coomat, trnMat):
		self.csrmat = (trnMat.tocsr() != 0) * 1.0

		tstLocs = [None] * coomat.shape[0]
		tstUsrs = set()
		for i in range(len(coomat.data)):
			row = coomat.row[i]
			col = coomat.col[i]
			if tstLocs[row] is None:
				tstLocs[row] = list()
			tstLocs[row].append(col)
			tstUsrs.add(row)
		tstUsrs = np.array(list(tstUsrs))
		self.tstUsrs = tstUsrs
		self.tstLocs = tstLocs

	def __len__(self):
		return len(self.tstUsrs)

	def __getitem__(self, idx):
		return self.tstUsrs[idx], np.reshape(self.csrmat[self.tstUsrs[idx]].toarray(), [-1])