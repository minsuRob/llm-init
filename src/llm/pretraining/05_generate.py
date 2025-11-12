"""
텍스트 생성 함수
"""
import torch
import importlib.util
import os

# config_01 모듈 동적 로드
spec = importlib.util.spec_from_file_location("config_01", os.path.join(os.path.dirname(__file__), "01_config.py"))
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)
tokenizer = config_module.tokenizer


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    """텍스트 생성 함수"""
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        if idx_next == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def generate_text(model, start_context, device, max_new_tokens=50, top_k=50, temperature=0.5, num_samples=10):
    """주어진 시작 문맥으로부터 텍스트 생성"""
    idx = tokenizer.encode(start_context)
    idx = torch.tensor(idx).unsqueeze(0)
    
    context_size = model.pos_emb.weight.shape[0]
    
    results = []
    for i in range(num_samples):
        token_ids = generate(
            model=model,
            idx=idx.to(device),
            max_new_tokens=max_new_tokens,
            context_size=context_size,
            top_k=top_k,
            temperature=temperature
        )
        
        flat = token_ids.squeeze(0)
        out = tokenizer.decode(flat.tolist()).replace("\n", " ")
        results.append(out)
        print(f"{i}: {out}")
    
    return results
