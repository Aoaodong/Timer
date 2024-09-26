import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from utils.input_category import category_trans_by_logic
from utils.timefeatures import time_features

warnings.filterwarnings('ignore')


class DatasetCustom(Dataset):
    def __init__(self, root_path='dataset', flag='train', input_len=None, pred_len=None,
                 data_slicing_type='logic', scale=True, timeenc=1, freq='h', stride=1,
                 subset_rand_ratio=1.0, target_col='s_POW', time_col='date',
                 train_p=0.7, test_p=0.2):
        self.subset_rand_ratio = subset_rand_ratio
        self.data_slicing_type = data_slicing_type
        # size [seq_len, label_len, pred_len]
        # info
        self.input_len = input_len
        self.pred_len = pred_len
        self.seq_len = input_len + pred_len
        self.timeenc = timeenc
        self.scale = scale
        # input setup
        self.target_col = target_col
        self.time_col = time_col
        self.freq = freq
        self.train_p = train_p
        self.test_p = test_p
        assert self.train_p + self.test_p < 1.0, "Should not set train_p + test_p >= 1.0 !"
        self.data_x = {}
        self.data_y = {}
        self.n_timepoint = {}
        self.data_stamp = {}
        # init
        assert flag in ['train', 'test', 'val'], rf"{flag} not in ['train', 'test', 'val']"
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        if self.set_type == 0:
            self.internal = int(1 // self.subset_rand_ratio)
        else:
            self.internal = 1

        self.root_path = root_path
        self.dataset_name = self.root_path.split('/')[-1].split('.')[0]

        self.__read_data__()

    def __read_data__(self):
        # 读取数据
        if self.root_path.endswith('.csv'):
            df_raw = pd.read_csv(self.root_path)
        elif self.root_path.endswith('.txt'):
            df_raw = []
            with open(self.root_path, "r", encoding='utf-8') as f:
                for line in f.readlines():
                    line = line.strip('\n').split(',')
                    data_line = np.stack([float(i) for i in line])
                    df_raw.append(data_line)
            df_raw = np.stack(df_raw, 0)
            df_raw = pd.DataFrame(df_raw)

        elif self.root_path.endswith('.npz'):
            data = np.load(self.root_path, allow_pickle=True)
            data = data['data'][:, :, 0]
            df_raw = pd.DataFrame(data)
        else:
            raise ValueError('Unknown data format: {}'.format(self.root_path))

        # 数据分类
        self.scaler = StandardScaler()
        if self.data_slicing_type == 'logic':
            data = category_trans_by_logic(df_raw)
            category = set(data['input_category'])
            data_dict = {c: data[[self.time_col, self.target_col]][data['input_category'] == c] for c in category}
        elif self.data_slicing_type == 'order':
            data_dict = {0: df_raw[[self.time_col, self.target_col]].copy()}
        elif self.data_slicing_type == 'custom':
            category = set(df_raw['input_category'])
            data_dict = {c: df_raw[[self.time_col, self.target_col]][df_raw['input_category'] == c] for c in category}
        else:
            raise ValueError(f'Unknown data format: {self.data_slicing_type}')

        # 数据分割
        for key, df in data_dict.items():
            data_len = len(df)
            num_train = int(data_len * self.train_p)
            num_test = int(data_len * self.test_p)
            num_vali = data_len - num_train - num_test

            assert (num_train - self.input_len) > 0 and (
                data_len - num_test - self.input_len) > 0, f"Unable to find data that meets the criteria in category {key}"

            border1s = [0, num_train - self.input_len, data_len - num_test - self.input_len]
            border2s = [num_train, num_train + num_vali, data_len]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]

            data = df[[self.target_col]].values

            if self.scale:
                train_data = data[border1s[0]:border2s[0]]
                self.scaler.fit(train_data)
                data = self.scaler.transform(data)

            if self.timeenc == 0:
                df_stamp = df[[self.time_col]].copy()
                df_stamp['date'] = pd.to_datetime(df_stamp[self.time_col])
                df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
                df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
                df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
                df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
                df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
                df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
                data_stamp = df_stamp.drop(['date'], 1).values
            elif self.timeenc == 1:
                if self.time_col:
                    data_stamp = time_features(pd.to_datetime(pd.to_datetime(df[self.time_col]).values), freq=self.freq)
                    data_stamp = data_stamp.transpose(1, 0)
                else:
                    data_stamp = np.zeros((len(df_raw), 4))
            else:
                raise ValueError('Unknown timeenc: {}'.format(self.timeenc))

            self.data_x[key] = data[border1:border2]
            self.data_y[key] = data[border1:border2]
            self.data_stamp[key] = data_stamp[border1:border2]

            self.n_timepoint[key] = len(self.data_x[key]) - self.input_len - self.pred_len + 1
            print(data_len, num_train, num_vali, num_test)
        print(self.n_timepoint)

    def __getitem__(self, index):
        internal = 1
        if self.set_type == 0:
            index = index * self.internal
            internal = self.internal
        cumulative_n = 0
        for key, n in self.n_timepoint.items():
            cumulative_n += int(n * internal)
            if index >= cumulative_n:
                continue
            else:
                s_begin = (index - cumulative_n) % n  # select start time
                s_end = s_begin + self.input_len
                r_begin = s_end
                r_end = r_begin + self.pred_len
                seq_x = self.data_x[key][s_begin:s_end, 0]
                seq_y = self.data_y[key][r_begin:r_end, 0]
                seq_x_mark = self.data_stamp[key][s_begin:s_end]
                seq_y_mark = self.data_stamp[key][r_begin:r_end]

                return seq_x, seq_y, seq_x_mark, seq_y_mark

        raise KeyError('Could not find the index-th data!')

    def __len__(self):
        if self.set_type == 0:
            return max(sum([int(v * self.subset_rand_ratio) for v in self.n_timepoint.values()]), 1)
        else:
            return int(sum(self.n_timepoint.values()))

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class AutoRegressionDatasetCustom(DatasetCustom):
    def __init__(self, root_path='dataset', flag='train', input_len=None, label_len=None, pred_len=None,
                 data_slicing_type='custom', scale=True, timeenc=1, freq='h', stride=1, subset_rand_ratio=1.0,
                 target_col='s_POW', time_col='date', train_p=0.7, test_p=0.2):
        self.label_len = label_len
        super().__init__(root_path=root_path, flag=flag, input_len=input_len, pred_len=pred_len,
                         data_slicing_type=data_slicing_type, scale=scale, timeenc=timeenc, freq=freq, stride=stride,
                         subset_rand_ratio=subset_rand_ratio, target_col=target_col, time_col=time_col,
                         train_p=train_p, test_p=test_p)

    def __getitem__(self, index):
        internal = 1
        if self.set_type == 0:
            index = index * self.internal
            internal = self.internal
        cumulative_n = 0
        for key, n in self.n_timepoint.items():
            if index >= cumulative_n:
                cumulative_n += int(n * internal)
                continue
            else:
                s_begin = (index - cumulative_n) % n  # select start time
                s_end = s_begin + self.input_len
                r_begin = s_end - self.label_len
                r_end = r_begin + self.pred_len + self.label_len
                seq_x = self.data_x[key][s_begin:s_end, 0]
                seq_y = self.data_y[key][r_begin:r_end, 0]
                seq_x_mark = self.data_stamp[key][s_begin:s_end]
                seq_y_mark = self.data_stamp[key][r_begin:r_end]

                return seq_x, seq_y, seq_x_mark, seq_y_mark

        raise KeyError('Could not find the index-th data!')

