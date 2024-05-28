import os

import pandas as pd
import torch

import argparse
import torch.nn.functional as F

from transformers import BertForSequenceClassification, AutoTokenizer

from tqdm import tqdm

from creator import DataProcessor


class TextClassifierPipeline:
    def __init__(self, data_path, root_model_path, sub_models_paths, model_name, device):
        self.data_path = data_path
        self.root_model_path = root_model_path
        self.sub_models_paths = sub_models_paths
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.device = device
        dp = DataProcessor()
        self.classes1 = dp.get_classes1()
        self.classes2 = dp.get_all_classes()
        self.sub_models = []
        self.model = None

        n_labels = len(self.classes1)
        if os.path.exists(self.root_model_path):
            self.model = BertForSequenceClassification.from_pretrained(self.model_name, num_labels=n_labels)
            self.model.load_state_dict(torch.load(self.root_model_path))
            print("Success loading model")

        else:
            print("Error loading model")

        for i in range(len(sub_models_paths)):
            sb_mdl_path = sub_models_paths[i]
            n_labels = len(self.classes2[i])

            if os.path.exists(sb_mdl_path):
                loaded_model = BertForSequenceClassification.from_pretrained(self.model_name, num_labels=n_labels)
                loaded_model.load_state_dict(torch.load(sb_mdl_path))
                self.sub_models.append(loaded_model)
                print("Success loading model")
            else:
                print("Error loading model")

    def predict(self, text):
        tokens = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            return_tensors='pt',
            truncation=True,
            max_length=512,
            padding='max_length',
            return_attention_mask=True
        )
        device = self.device
        input_ids = tokens.input_ids.to(device)
        attention_mask = tokens.attention_mask.to(device)
        self.model.to(device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            softmax_logits = F.softmax(outputs.logits, dim=1)
            confidence1 = softmax_logits.max().item()
            pred = torch.argmax(softmax_logits, axis=1)
            pred = pred.item()

        sub_model = self.sub_models[pred]
        sub_model.eval()
        sub_model.to(device)
        with torch.no_grad():
            outputs = sub_model(input_ids, attention_mask)
            softmax_logits = F.softmax(outputs.logits, dim=1)
            confidence2 = softmax_logits.max().item()
            pred_out = torch.argmax(softmax_logits, axis=1)
            pred_out = pred_out.item()
        return {"text": text, "root_class": self.classes1[pred], "sub_class": self.classes2[pred][pred_out],
                "confidence": [confidence1, confidence2], "label": [self.classes1[pred], self.classes2[pred][pred_out]]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Argument")
    parser.add_argument("-test_data", type=str, default=None, help="Test data path")
    parser.add_argument("-data_path", type=str, default="./data/Data.xlsx", help="Data Frame file's path.")
    parser.add_argument("-root_model_path", type=str, default="./models/Root_Model",
                        help="Model path. This can't be empty")
    parser.add_argument('-model_name', type=str, default="bert-base-uncased",
                        help="Model and tokenizer name.Default= bert-base-uncased")
    parser.add_argument("-device", type=str, help="Select device cpu or cuda", default="cuda")
    parser.add_argument('-sub_models_paths', type=str, nargs='+',
                        default=['./models/SModel_CS.pth', './models/SModel_Civil.pth', './models/SModel_ECE.pth',
                                 './models/SModel_MAE.pth',
                                 './models/SModel_Medical.pth', './models/SModel_Psychology.pth',
                                 './models/SModel_biochemistry.pth'],
                        help='Enter the model path sequentially')
    args = parser.parse_args()
    pipeline = TextClassifierPipeline(args.data_path, args.root_model_path, args.sub_models_paths, args.model_name,
                                      args.device)
    if args.test_data is not None:
        df = pd.read_excel(args.test_data)
        domains = df['Domain']
        areas = df['area']
        all_text = df['Abstract']
        # Gerçek etiketleri (true_labels) oluştur
        y_true = list(zip(domains, areas))
        predicted_labels = []
        with tqdm(total=len(all_text), desc="Predicting") as pbar:
            for i, t in enumerate(all_text):
                x = pipeline.predict(t)['label']
                predicted_labels.append(x)
                pbar.update(1)
                pbar.set_postfix({"pred": x, "true": y_true[i]})
        t_count = 0
        f_count = 0
        for x, y in zip(y_true, predicted_labels):
            if all(a == b for a, b in zip(x, y)):
                t_count += 1
            else:
                f_count += 1
        score_end = (t_count / len(df)) * 100
        print(score_end)
    else:
        while True:
            input_text = input("Tahmin etmek istediğiniz metni girin (Çıkmak için 'q' yazın): ")
            if input_text.lower() == "q":
                break
            prediction = pipeline.predict(input_text)
            print("Tahmin:", prediction["label"])
