"""
데이터셋 테스트 스크립트
"""
import importlib.util
import os

# 동적 모듈 로드
def load_module(module_name, file_name):
    spec = importlib.util.spec_from_file_location(
        module_name, os.path.join(os.path.dirname(__file__), file_name)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# dataset_02 모듈 로드
dataset_module = load_module("dataset_02", "02_dataset.py")
config_module = load_module("config_01", "01_config.py")

create_dataloader = dataset_module.create_dataloader
tokenizer = config_module.tokenizer

# 데이터 경로
data_path = "../assets/cleaned_norway_forest_en.txt"

# 데이터로더 생성
print("Creating dataloader...")
train_loader = create_dataloader(data_path, max_length=32, stride=4, batch_size=8)

print(f"Number of batches: {len(train_loader)}")

# 첫 번째 배치 확인
dataiter = iter(train_loader)
x, y = next(dataiter)

print(f"\nBatch shape: {x.shape}")
print(f"\nFirst input sequence:")
print(tokenizer.decode(x[0].tolist()))
print(f"\nFirst target sequence:")
print(tokenizer.decode(y[0].tolist()))
