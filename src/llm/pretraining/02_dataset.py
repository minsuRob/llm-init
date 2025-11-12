"""
데이터셋 및 데이터로더 정의
"""
import torch
from torch.utils.data import Dataset, DataLoader
import importlib.util
import os

# config_01 모듈 동적 로드
spec = importlib.util.spec_from_file_location("config_01", os.path.join(os.path.dirname(__file__), "01_config.py"))
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)
tokenizer = config_module.tokenizer


class MyDataset(Dataset):
    def __init__(self, txt, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)

        print("# of tokens in txt:", len(token_ids))

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader(file_path, max_length=32, stride=4, batch_size=128):
    """데이터로더 생성"""
    with open(file_path, 'r', encoding='utf-8-sig') as file:
        txt = file.read()
    
    dataset = MyDataset(txt, max_length=max_length, stride=stride)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    return train_loader
