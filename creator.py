import pandas as pd
import os
import json
from sklearn.model_selection import train_test_split


class DataProcessor:
    def __init__(self, input_file="./data/Data.xlsx", test_size=0.15, seed=42):
        self.input_file = input_file
        self.test_size = test_size
        self.seed = seed
        self.df = None
        self.categories = None
        self.json_data = None
        self.label2id = None
        self.classes1 = None
        self.df_new = None
        self.df_test = None
        self.all_df = None
        self.mini_df = None
        self.all_classes = None
        self.setup()

    def get_classes1(self):
        return self.classes1

    def get_json(self):
        return self.json_data

    def get_df_new(self):
        return self.df_new

    def get_df_test(self):
        return self.df_test

    def get_all_df(self):
        return self.all_df

    def get_mini_df(self):
        return self.mini_df

    def get_all_classes(self):
        return self.all_classes

    def save_output(self, output_file=None):
        output_filename = os.path.splitext(os.path.basename(self.input_file))[0]
        data_dir = "data"

        if output_file is not None:
            data_dir = output_file

        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        # Kaydetme yolları
        train_path = os.path.join(data_dir, "df_train.xlsx")
        test_path = os.path.join(data_dir, "df_test.xlsx")
        json_name = os.path.join(data_dir, output_filename)
        json_name += ".json"

        print("***", json_name)

        # DataFrame'leri belirtilen yollara kaydet
        self.df_new.to_excel(train_path, index=False)
        self.df_test.to_excel(test_path, index=False)

        with open(json_name, "w") as json_file:
            json.dump(self.json_data, json_file, indent=4)
        return train_path, test_path, json_name

    def setup(self):
        self.df = pd.read_excel(self.input_file)
        self.df = self.df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        self.df['Abstract'] = self.df['Abstract'].str.lower()
        # self.df['Domain'] = self.df['Domain'].str.lower()
        # self.df['area'] = self.df['Domain'].str.lower()
        print("DF read success")
        self.categories = {}
        for index, row in self.df.iterrows():
            category = row["Domain"].strip()
            area = row["area"].strip()
            if category not in self.categories:
                self.categories[category] = set()
            self.categories[category].add(area)
        classes = sorted(self.categories.keys())
        categories = {k: sorted(v) for k, v in sorted(self.categories.items())}

        self.json_data = []
        for i, (key, values) in enumerate(categories.items()):
            category_json = {i: {j: {value: None} for j, value in enumerate(values)}}
            self.json_data.append(category_json)
        print("Json Data create success")

        self.label2id = dict(zip(classes, range(len(classes))))
        self.classes1 = classes

        self.df['encoded'] = self.df['Domain'].apply(lambda x: self.label2id[x])
        print("Class create success")

        if any(self.df.encoded.value_counts() <= 2):
            self.df_new, self.df_test = train_test_split(self.df, test_size=self.test_size, random_state=self.seed)
        else:
            self.df_new, self.df_test = train_test_split(self.df, test_size=self.test_size, random_state=self.seed,
                                                         stratify=self.df.encoded)
        self.all_df = []
        self.mini_df = []
        self.all_classes = []
        self.all_df = []
        for cls in self.classes1:
            new_df = self.df_new[self.df_new['Domain'] == cls]
            new_df.reset_index(drop=True, inplace=True)
            self.all_df.append(new_df)

        for i, df in enumerate(self.all_df):
            tmp_sinif = []
            for key, value in self.json_data[i][i].items():
                for key_in_value in value:
                    tmp_sinif.append(key_in_value)
            tmp_sinif.sort()
            self.all_classes.append(tmp_sinif)
            label2id = dict(zip(tmp_sinif, range(len(tmp_sinif))))
            tmp_df = {}
            tmp_df['Abstract'] = df['Abstract']
            tmp_df['area'] = df['area']
            tmp_df['encoded'] = df['area'].apply(lambda x: label2id[x])
            df_end = pd.DataFrame(tmp_df)
            self.mini_df.append(df_end)
        print("Mini DF, ALL CLASSES succes")
