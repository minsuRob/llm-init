"""
모델 추론 및 결과 확인 스크립트
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
generate_module = load_module("generate_05", "05_generate.py")
config_module = load_module("config_01", "01_config.py")

GPTModel = model_module.GPTModel
generate_text = generate_module.generate_text
tokenizer = config_module.tokenizer


def load_model(model_path, device):
    """저장된 모델 로드"""
    model = GPTModel()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    return model


def test_single_prediction(model, device, text="Dobby is"):
    """단일 단어 예측 테스트"""
    idx = tokenizer.encode(text)
    idx = torch.tensor(idx).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(idx)
    
    logits = logits[:, -1, :]
    
    # 가장 확률이 높은 단어 10개 출력
    print(f"\nTop 10 predictions for '{text}':")
    top_logits, top_indices = torch.topk(logits, 10)
    for p, i in zip(top_logits.squeeze(0).tolist(), top_indices.squeeze(0).tolist()):
        print(f"{p:.2f}\t {i}\t {tokenizer.decode([i])}")
    
    # 가장 확률이 높은 단어 출력
    idx_next = torch.argmax(logits, dim=-1, keepdim=True)
    flat = idx_next.squeeze(0)
    out = tokenizer.decode(flat.tolist())
    print(f"\nMost likely next token: {out}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 모델 로드
    model_path = "model_100.pth"
    model = load_model(model_path, device)
    
    # 단일 예측 테스트
    test_single_prediction(model, device, "Dobby is")
    
    # 텍스트 생성
    print("\n" + "="*50)
    print("Generating text samples:")
    print("="*50)
    start_context = input("Start context: ")
    generate_text(model, start_context, device)
