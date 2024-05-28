import pandas as pd
import torch

import argparse
from torch.utils.data import DataLoader

from transformers import BertForSequenceClassification, AutoTokenizer

from tqdm import tqdm
from train import Create_Dataset
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from creator import DataProcessor


class Eval:
    def __init__(self, data_path, model_path, model_name, cls_id, batch, device):
        self.data_path = data_path
        self.model_path = model_path
        self.model_name = model_name
        self.cls_id = cls_id
        self.batch = batch
        self.device = device
        self.df = pd.DataFrame()

    def eval_root(self):
        dp = DataProcessor(self.data_path)
        classes = dp.get_classes1()
        print(classes)
        device = self.device
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if self.data_path is not None:
            self.df = pd.read_excel(self.data_path)

        n_labels = len(classes)
        test_dataset = Create_Dataset(self.df, tokenizer, classes)
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch, num_workers=4, shuffle=False)
        loaded_model = BertForSequenceClassification.from_pretrained(self.model_name, num_labels=n_labels)
        loaded_model.load_state_dict(torch.load(self.model_path, map_location=torch.device(device)))
        true_labels = []
        predicted_labels = []
        loaded_model.to(device)
        loaded_model.eval()
        with tqdm(total=len(test_dataloader), unit="batch") as pbar:
            for batch in test_dataloader:
                inputs, labels = batch
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)
                labels = labels.to(device)
                with torch.no_grad():
                    outputs = loaded_model(input_ids, attention_mask, labels=labels)
                    logits = outputs.logits
                    batch_predictions = torch.argmax(logits, axis=1)
                    true_labels.extend(labels.tolist())
                    predicted_labels.extend(batch_predictions.tolist())
                pbar.update(1)
        f1 = f1_score(true_labels, predicted_labels, average='macro')
        print("F1 Skoru:", f1)

        recall = recall_score(true_labels, predicted_labels, average='macro')
        print("Recall Skoru:", recall)

        precision = precision_score(true_labels, predicted_labels, average='macro')
        print("Precision Skoru:", precision)

        accuracy = accuracy_score(true_labels, predicted_labels)
        print("Accuracy:", accuracy)
        score = {"f1": f1, "recall": recall, "precision": precision, "accuracy": accuracy}
        return score

    def eval_subclass(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        device = self.device

        dp = DataProcessor(self.data_path)
        all_classes = dp.get_all_classes()
        mini_df = dp.get_mini_df()
        n_labels = len(all_classes[self.cls_id])

        loaded_model = BertForSequenceClassification.from_pretrained(self.model_name, num_labels=n_labels)
        loaded_model.load_state_dict(torch.load(self.model_path))

        test_dataset = Create_Dataset(mini_df[self.cls_id], tokenizer, all_classes[self.cls_id])
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch, num_workers=4, shuffle=False)

        true_labels = []
        predicted_labels = []
        loaded_model.to(device)
        loaded_model.eval()

        with tqdm(total=len(test_dataloader), unit="batch") as pbar:
            for batch in test_dataloader:
                inputs, labels = batch
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)
                labels = labels.to(device)
                with torch.no_grad():
                    outputs = loaded_model(input_ids, attention_mask, labels=labels)
                    logits = outputs.logits
                    batch_predictions = torch.argmax(logits, axis=1)
                    true_labels.extend(labels.tolist())
                    predicted_labels.extend(batch_predictions.tolist())
                pbar.update(1)

        f1 = f1_score(true_labels, predicted_labels, average='macro')
        print("F1 Skoru:", f1)

        recall = recall_score(true_labels, predicted_labels, average='macro')
        print("Recall Skoru:", recall)

        precision = precision_score(true_labels, predicted_labels, average='macro')
        print("Precision Skoru:", precision)

        accuracy = accuracy_score(true_labels, predicted_labels)
        print("Accuracy:", accuracy)
        score = {"f1": f1, "recall": recall, "precision": precision, "accuracy": accuracy}
        return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Argument")
    parser.add_argument("-data_path", type=str, required=True, help="Data Frame file's path. This can't be empty")
    parser.add_argument("-model_path", type=str, required=True, help="Model path. This can't be empty")
    parser.add_argument('-model_name', type=str, required=True, help="Model and tokenizer name.This can't be empty")
    parser.add_argument('-batch', type=int, default=32, help='Batch size. Default value = 16')
    parser.add_argument("-eval_type", choices=["root", "sub"], default="root",
                        help="Train type: root or subclass. Default value = root")
    parser.add_argument("-cls_id", type=int, default=0,
                        help="Enter subclass id as int, ['CS', 'Civil', 'ECE', 'MAE', 'Medical', 'Psychology', 'biochemistry']")
    parser.add_argument("-device", type=str, help="Select device cpu or cuda", default="cuda")

    args = parser.parse_args()
    evl = Eval(args.data_path, args.model_path, args.model_name, args.cls_id, args.batch, args.device)
    if args.eval_type == "root":
        evl.eval_root()
    elif args.eval_type == "sub":
        evl.eval_subclass()
