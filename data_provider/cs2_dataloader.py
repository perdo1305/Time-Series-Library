import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features


class Dataset_CS2(Dataset):
    def __init__(
        self,
        args,
        root_path,
        flag="train",
        size=None,
        features="S",
        data_path="combined_data.csv",
        target="Voltage(V)",
        scale=True,
        timeenc=0,
        freq="h",
        seasonal_patterns=None,
    ):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # Check if target exists in the data
        if self.target not in df_raw.columns:
            raise ValueError(
                f"Target column '{self.target}' not found in dataset. Available columns: {df_raw.columns.tolist()}"
            )

        # Convert Date_Time to pandas datetime
        if "Date_Time" in df_raw.columns:
            df_raw["date"] = pd.to_datetime(df_raw["Date_Time"])
        else:
            # Use Test_Time as a simpler index if Date_Time not available
            df_raw["date"] = pd.to_datetime(
                df_raw["Test_Time(s)"], unit="s", origin="2010-01-01"
            )

        # Ensure data is sorted by time
        df_raw = df_raw.sort_values("date")

        # Determine total size and split points
        num_samples = len(df_raw)
        num_train = int(num_samples * 0.7)
        num_test = int(num_samples * 0.2)
        num_vali = num_samples - num_train - num_test

        # Store row IDs for test results
        self.ids = np.arange(num_samples)

        # Initialize timeseries container for validation and testing
        self.timeseries = []

        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # Select features based on args
        if self.features == "M" or self.features == "MS":
            # Use relevant numeric columns as features
            numeric_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
            # Remove certain columns that might not be useful as features
            exclude_cols = ["Data_Point", "Cycle_Index", "Step_Index", "Is_FC_Data"]
            cols = [col for col in numeric_cols if col not in exclude_cols]
            df_data = df_raw[cols]
        elif self.features == "S":
            # Use only the target as feature
            df_data = df_raw[[self.target]]

        # Scale data if needed
        if self.scale:
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # Process timestamps
        df_stamp = df_raw[["date"]][border1:border2]
        if self.timeenc == 0:
            # Time encoding method 1
            df_stamp["month"] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp["weekday"] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(["date"], 1).values
        elif self.timeenc == 1:
            # Time encoding method 2
            data_stamp = time_features(
                pd.to_datetime(df_stamp["date"].values), freq=self.freq
            )
            data_stamp = data_stamp.transpose(1, 0)

        # Feature dimensions
        self.enc_in = data.shape[1]
        print(f"Feature dimensions: {self.enc_in}")

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

        # For validation/testing, prepare timeseries data
        if self.set_type == 1 or self.set_type == 2:  # val or test
            # Create target sequences for validation/testing
            # Find target column index - for single-variate, it's index 0
            target_idx = 0
            if self.features == "M" or self.features == "MS":
                # For multivariate, find the index of the target column
                if hasattr(df_data, "columns") and self.target in df_data.columns:
                    target_idx = list(df_data.columns).index(self.target)

            # Extract sequences
            for i in range(len(self.data_x) - self.seq_len - self.pred_len + 1):
                self.timeseries.append(
                    self.data_y[
                        i + self.seq_len : i + self.seq_len + self.pred_len, target_idx
                    ]
                )

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def last_insample_window(self):
        """
        The last window of insample size of all timeseries.
        This function is required for validation in the short-term forecasting experiment.

        :return: Last insample window of all timeseries and a mask (all ones for our case)
        """
        if not hasattr(self, "insample_window"):
            # Cache the result
            samples = min(1000, len(self.data_x) - self.seq_len)
            self.insample_window = np.zeros((samples, self.seq_len))
            insample_mask = np.ones((samples, self.seq_len))

            # For each validation sample, use its input sequence as insample window
            # For single-variate prediction, use the first feature column
            target_idx = 0
            if self.features == "M" or self.features == "MS":
                # For multivariate, we need to find the target column index
                # This is an approximation since we don't have direct column mapping
                # We'll use the first column as default
                pass

            for i in range(samples):
                self.insample_window[i, :] = self.data_x[
                    i : i + self.seq_len, target_idx
                ]

            return self.insample_window, insample_mask

        return self.insample_window, np.ones_like(self.insample_window)
