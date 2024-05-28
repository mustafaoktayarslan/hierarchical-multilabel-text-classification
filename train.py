import os
import pandas as pd
import torch
import random
import numpy as np
import torch.nn as nn
import math
import argparse
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import BertForSequenceClassification, AutoTokenizer
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
from creator import DataProcessor

scores = []


class Create_Dataset(Dataset):
    def __init__(self, data, tokenizer, attributes, max_token_len: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.attributes = attributes
        self.max_token_len = max_token_len
        self.text = ""

    def __len__(self):
        return len(self.data)

    def getText(self):
        return self.text

    def __getitem__(self, index):
        item = self.data['encoded'].values
        title = self.data["Abstract"].values
        comment = str(title[index])
        self.text = comment
        label = torch.tensor(item[index])
        encoding = self.tokenizer.encode_plus(comment,
                                              add_special_tokens=True,
                                              return_tensors='pt',
                                              truncation=True,
                                              max_length=self.max_token_len,
                                              padding='max_length',
                                              return_attention_mask=True
                                              )
        return {'input_ids': encoding.input_ids.flatten(), 'attention_mask': encoding.attention_mask.flatten()}, label


class Training:
    def __init__(self, model_name, data_path, epoch, batch, max_len, lr,
                 weight_decay, warmup, seed, device, save_directory=None, cls_id=0, ):
        self.model_name = model_name
        self.data_path = data_path
        self.epoch = epoch
        self.batch = batch
        self.max_len = max_len
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup = warmup
        self.seed = seed
        self.device = device
        self.save_directory = save_directory
        self.cls_id = cls_id
        self.json_data = []
        self.classes1 = []
        self.all_classes = []
        self.mini_df = []
        self.all_df = []
        self.df_new = pd.DataFrame()
        self.df_test = pd.DataFrame()
        self.set_seed()

    def set_seed(self):
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        random.seed(self.seed)
        np.random.seed(self.seed)

    def data_preparation(self):
        dp = DataProcessor(self.data_path)
        self.json_data = dp.get_json()
        self.classes1 = dp.get_classes1()
        self.all_classes = dp.get_all_classes()
        self.mini_df = dp.get_mini_df()
        self.all_df = dp.get_all_df()
        self.df_new = dp.get_df_new()
        self.df_test = dp.get_df_test()
        dp.save_output()

    def root_training(self):
        label2id = dict(zip(self.classes1, range(len(self.classes1))))
        id2label = dict(zip(label2id.values(), label2id.keys()))
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if any(self.df_new.encoded.value_counts() <= 2):
            traindf, valdf = train_test_split(self.df_new, test_size=0.15, random_state=42)

        else:
            traindf, valdf = train_test_split(self.df_new, test_size=0.15, random_state=42,
                                              stratify=self.df_new.encoded)

        train_dataset = Create_Dataset(traindf, tokenizer, self.classes1)
        val_dataset = Create_Dataset(valdf, tokenizer, self.classes1)

        train_dataloader = DataLoader(train_dataset, batch_size=self.batch, num_workers=4, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch, num_workers=4, shuffle=False)
        print(f"train:{traindf.shape} val:{valdf.shape} ")

        n_labels = len(self.classes1)
        model = BertForSequenceClassification.from_pretrained(self.model_name, num_labels=n_labels, id2label=id2label,
                                                              label2id=label2id)
        device = self.device
        num_epochs = self.epoch
        weight_decay = self.weight_decay
        warmup = self.warmup
        lr = self.lr
        best_val_loss = float('inf')
        patience = 2
        model.to(device)
        total_steps = len(train_dataloader) * num_epochs
        warmup_steps = math.floor(total_steps * warmup)
        warmup_steps = max(1, warmup_steps)

        print(device)
        # Model ve kayıp fonksiyonu
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=warmup_steps, T_mult=1)

        running_loss = 0.0
        model_name = "model"
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            num_batches = len(train_dataloader)
            with tqdm(total=num_batches, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as pbar:
                for batch_idx, batch in enumerate(train_dataloader):
                    inputs, labels = batch
                    input_ids = inputs["input_ids"].to(device)
                    attention_mask = inputs["attention_mask"].to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(input_ids, attention_mask, labels=labels)
                    loss = criterion(outputs.logits, labels)
                    loss.backward()
                    optimizer.step()
                    scheduler.step(epoch + batch_idx / num_batches)
                    running_loss += loss.item()
                    # Ayrıntılı bilgi gösterme
                    pbar.set_postfix({"loss": loss.item(), "running_loss": running_loss / (batch_idx + 1)})
                    pbar.update(1)  # İlerleme çubuğunu güncelleme
            # Doğrulama
            model.eval()
            val_loss = 0.0
            for batch_idx, batch in enumerate(val_dataloader):
                inputs, labels = batch
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)
                labels = labels.to(device)
                with torch.no_grad():
                    outputs = model(input_ids, attention_mask, labels=labels)
                    loss = criterion(outputs.logits, labels)
                val_loss += loss.item()
                # Early Stopping kontrolü
            val_loss /= len(val_dataloader)
            name = "Root_Model.pth"
            model_dir = "models"
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            model_path = os.path.join(model_dir, "best_model.pth")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if self.save_directory is None:
                    torch.save(model.state_dict(), model_path)
                else:
                    directory = os.path.join(self.save_directory, "best_model.pth")
                    torch.save(model.state_dict(), directory)
                model_name = f"{name}-{epoch:02d}-{best_val_loss:.2f}.pth"
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {running_loss / num_batches:.4f}, Validation Loss: {val_loss:.4f}")

        if self.save_directory is None:
            new_path = os.path.join(model_dir, name)
            os.rename(model_path, new_path)
        else:
            directory = os.path.join(self.save_directory, "best_model.pth")
            os.rename(directory, os.path.join(self.save_directory, name))

    def subclass_training(self):
        for i in range(len(self.all_df)):
            label2id = dict(zip(self.all_classes[i], range(len(self.all_classes[i]))))
            id2label = dict(zip(label2id.values(), label2id.keys()))

            tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            if any(self.mini_df[i]['encoded'].value_counts() <= 2):
                traindf, valdf = train_test_split(self.mini_df[i], test_size=0.15, random_state=42)
            else:
                traindf, valdf = train_test_split(self.mini_df[i], test_size=0.15, random_state=42,
                                                  stratify=self.mini_df[i].encoded)
            traindf = traindf.head(50)
            valdf = valdf.head(50)

            train_dataset = Create_Dataset(traindf, tokenizer, self.all_classes[i])
            val_dataset = Create_Dataset(valdf, tokenizer, self.all_classes[i])

            train_dataloader = DataLoader(train_dataset, batch_size=self.batch, num_workers=4, shuffle=True)
            val_dataloader = DataLoader(val_dataset, batch_size=self.batch, num_workers=4, shuffle=False)
            print(f"train:{traindf.shape}  val:{valdf.shape} ")

            n_labels = len(self.all_classes[i])
            model = BertForSequenceClassification.from_pretrained(self.model_name, num_labels=n_labels,
                                                                  id2label=id2label, label2id=label2id)
            device = self.device
            num_epochs = self.epoch
            weight_decay = self.weight_decay
            warmup = self.warmup
            lr = self.lr
            best_val_loss = float('inf')
            patience = 2
            model.to(device)
            total_steps = len(train_dataloader) * num_epochs
            warmup_steps = math.floor(total_steps * warmup)

            # Model ve kayıp fonksiyonu
            criterion = nn.CrossEntropyLoss().to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=warmup_steps, T_mult=1)

            running_loss = 0.0
            for epoch in range(num_epochs):
                model.train()
                running_loss = 0.0
                num_batches = len(train_dataloader)
                with tqdm(total=num_batches, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as pbar:
                    for batch_idx, batch in enumerate(train_dataloader):
                        inputs, labels = batch
                        input_ids = inputs["input_ids"].to(device)
                        attention_mask = inputs["attention_mask"].to(device)
                        labels = labels.to(device)
                        optimizer.zero_grad()
                        outputs = model(input_ids, attention_mask, labels=labels)
                        loss = criterion(outputs.logits, labels)
                        loss.backward()
                        optimizer.step()
                        scheduler.step(epoch + batch_idx / num_batches)
                        running_loss += loss.item()
                        # Ayrıntılı bilgi gösterme
                        pbar.set_postfix({"loss": loss.item(), "running_loss": running_loss / (batch_idx + 1)})
                        pbar.update(1)  # İlerleme çubuğunu güncelleme
                # Doğrulama
                model.eval()
                val_loss = 0.0
                for batch_idx, batch in enumerate(val_dataloader):
                    inputs, labels = batch
                    input_ids = inputs["input_ids"].to(device)
                    attention_mask = inputs["attention_mask"].to(device)
                    labels = labels.to(device)
                    with torch.no_grad():
                        outputs = model(input_ids, attention_mask, labels=labels)
                        loss = criterion(outputs.logits, labels)
                    val_loss += loss.item()
                    # Early Stopping kontrolü
                val_loss /= len(val_dataloader)
                name = "SModel_" + self.classes1[i] + ".pth"

                model_dir = "models"
                # Hedef dizini oluştur (eğer yoksa)
                if not os.path.exists(model_dir):
                    os.makedirs(model_dir)
                model_path = os.path.join(model_dir, "best_model.pth")

                # Modeli kaydet
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if self.save_directory is None:
                        torch.save(model.state_dict(), model_path)
                    else:
                        directory = os.path.join(self.save_directory, "best_model.pth")
                        torch.save(model.state_dict(), directory)

                    model_name = f"Model_{self.classes1[i]}-{epoch:02d}-{best_val_loss:.2f}.pth"
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {running_loss / num_batches:.4f}, Validation Loss: {val_loss:.4f}")
            if self.save_directory is None:
                new_path = os.path.join(model_dir, name)
                os.rename(model_path, new_path)
            else:
                directory = os.path.join(self.save_directory, "best_model.pth")

                os.rename(directory, os.path.join(self.save_directory, name))

    def subclass_only_training(self):
        label2id = dict(zip(self.all_classes[self.cls_id], range(len(self.all_classes[self.cls_id]))))
        id2label = dict(zip(label2id.values(), label2id.keys()))
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if any(self.mini_df[self.cls_id]['encoded'].value_counts() <= 2):
            traindf, valdf = train_test_split(self.mini_df[self.cls_id], test_size=0.15, random_state=42)
        else:
            traindf, valdf = train_test_split(self.mini_df[self.cls_id], test_size=0.15, random_state=42,
                                              stratify=self.mini_df[self.cls_id].encoded)
        traindf = traindf.head(50)
        valdf = valdf.head(50)

        train_dataset = Create_Dataset(traindf, tokenizer, self.all_classes[self.cls_id])
        val_dataset = Create_Dataset(valdf, tokenizer, self.all_classes[self.cls_id])

        train_dataloader = DataLoader(train_dataset, batch_size=self.batch, num_workers=4, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=self.batch, num_workers=4, shuffle=False)
        print(f"train:{traindf.shape}  val:{valdf.shape} ")

        n_labels = len(self.all_classes[self.cls_id])
        model = BertForSequenceClassification.from_pretrained(self.model_name, num_labels=n_labels,
                                                              id2label=id2label, label2id=label2id)
        device = self.device
        num_epochs = self.epoch
        weight_decay = self.weight_decay
        warmup = self.warmup
        lr = self.lr
        best_val_loss = float('inf')
        patience = 2
        model.to(device)
        total_steps = len(train_dataloader) * num_epochs
        warmup_steps = math.floor(total_steps * warmup)

        # Model ve kayıp fonksiyonu
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=warmup_steps, T_mult=1)

        running_loss = 0.0
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            num_batches = len(train_dataloader)
            with tqdm(total=num_batches, desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as pbar:
                for batch_idx, batch in enumerate(train_dataloader):
                    inputs, labels = batch
                    input_ids = inputs["input_ids"].to(device)
                    attention_mask = inputs["attention_mask"].to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(input_ids, attention_mask, labels=labels)
                    loss = criterion(outputs.logits, labels)
                    loss.backward()
                    optimizer.step()
                    scheduler.step(epoch + batch_idx / num_batches)
                    running_loss += loss.item()
                    # Ayrıntılı bilgi gösterme
                    pbar.set_postfix({"loss": loss.item(), "running_loss": running_loss / (batch_idx + 1)})
                    pbar.update(1)  # İlerleme çubuğunu güncelleme
            # Doğrulama
            model.eval()
            val_loss = 0.0
            for batch_idx, batch in enumerate(val_dataloader):
                inputs, labels = batch
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)
                labels = labels.to(device)
                with torch.no_grad():
                    outputs = model(input_ids, attention_mask, labels=labels)
                    loss = criterion(outputs.logits, labels)
                val_loss += loss.item()
                # Early Stopping kontrolü
            val_loss /= len(val_dataloader)
            name = "SModel_" + self.classes1[self.cls_id] + ".pth"

            model_dir = "models"
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            model_path = os.path.join(model_dir, "best_model.pth")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if self.save_directory is None:
                    torch.save(model.state_dict(), "./best_model.pth")
                else:
                    directory = os.path.join(self.save_directory, "best_model.pth")
                    torch.save(model.state_dict(), directory)
                # model_name = f"Model_{self.classes1[self.cls_id]}-{epoch:02d}-{best_val_loss:.2f}.pth"
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {running_loss / num_batches:.4f}, Validation Loss: {val_loss:.4f}")

        if self.save_directory is None:
            new_path = os.path.join(model_dir, name)
            os.rename(model_path, new_path)
        else:
            directory = os.path.join(self.save_directory, "best_model.pth")
            os.rename(directory, os.path.join(self.save_directory, name))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Argument")
    parser.add_argument('-model_name', type=str, required=True, help="Model and tokenizer name.This can't be empty")
    parser.add_argument("-data_path", type=str, required=True, help="Data Frame file's path. This can't be empty")
    parser.add_argument('-epoch', type=int, default=4, help='Epoch count. Default value = 4')
    parser.add_argument('-batch', type=int, default=16, help='Batch size. Default value = 16')
    parser.add_argument('-max_len', type=int, default=128, help='Maximum token length. Default value = 128')
    parser.add_argument('-lr', type=float, default=3e-5, help='Learning rate. Default value = 3e-5')
    parser.add_argument('-weight_decay', type=float, default=3e-4, help='Weight decay value. Default value = 3e-4')
    parser.add_argument('-warmup', type=float, default=0.2, help='Warmup steps.Default value = 0.2')
    parser.add_argument('-seed', default=42, type=int, help='Random seed.Default value = 42')
    parser.add_argument("-save_directory", default=None, help="Save model directory.")
    parser.add_argument("-training_type", choices=["root", "subclass", "only"], default="root",
                        help="Train type: root or subclass. Default value = root")
    parser.add_argument("-cls_id", type=int, default=0,
                        help="Enter subclass id as int, ['CS', 'Civil', 'ECE', 'MAE', 'Medical', 'Psychology', 'biochemistry']")
    parser.add_argument("-device", type=str, help="Select device cpu or cuda", default="cuda")
    args = parser.parse_args()

    training = Training(
        args.model_name,
        args.data_path,
        args.epoch,
        args.batch,
        args.max_len,
        args.lr,
        args.weight_decay,
        args.warmup,
        args.seed,
        args.device,
        args.save_directory,
        args.cls_id
    )

    if args.training_type == "root":
        training.data_preparation()
        training.root_training()
    elif args.training_type == "subclass":
        training.data_preparation()
        training.subclass_training()
    elif args.training_type == "only":
        training.data_preparation()
        training.subclass_only_training()
