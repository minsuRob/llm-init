"""
모델 훈련 스크립트
"""
import torch
import importlib.util
import os

# 동적 모듈 로드
def load_module(module_name, file_name):
    spec = importlib.util.spec_from_file_location(module_name, os.path.join(os.path.dirname(__file__), file_name))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

model_module = load_module("model_03", "03_model.py")
dataset_module = load_module("dataset_02", "02_dataset.py")

GPTModel = model_module.GPTModel
create_dataloader = dataset_module.create_dataloader


def train_model(data_path, num_epochs=100, lr=0.0004, weight_decay=0.1, save_dir="./"):
    """모델 훈련"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 데이터로더 생성
    train_loader = create_dataloader(data_path)
    
    # 모델 초기화
    torch.manual_seed(123)
    model = GPTModel()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    tokens_seen, global_step = 0, -1
    losses = []
    
    for epoch in range(num_epochs):
        model.train()
        
        epoch_loss = 0
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)
            
            logits = model(input_batch)
            loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1
            
            if global_step % 1000 == 0:
                print(f"Tokens seen: {tokens_seen}")
        
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        print(f"Epoch: {epoch + 1}, Loss: {avg_loss}")
        torch.save(model.state_dict(), f"{save_dir}/model_{str(epoch + 1).zfill(3)}.pth")
    
    return model, losses


if __name__ == "__main__":
    # 훈련 실행
    data_path = "../../assets/cleaned_norway_forest_en.txt"
    model, losses = train_model(data_path)
    
    # Loss 그래프 저장
    import matplotlib.pyplot as plt
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.savefig('training_loss.png')
    plt.show()
