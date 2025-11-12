"""
LLM 모델 설정 파일
"""
import tiktoken

# 토크나이저 설정
tokenizer = tiktoken.get_encoding("gpt2")

# 모델 하이퍼파라미터
VOCAB_SIZE = tokenizer.n_vocab  # 50257 Tiktoken
CONTEXT_LENGTH = 128  # Shortened context length (orig: 1024)
EMB_DIM = 768  # Embedding dimension
NUM_HEADS = 12  # Number of attention heads
NUM_LAYERS = 12  # Number of layers
DROP_RATE = 0.1  # Dropout rate
QKV_BIAS = False  # Query-key-value bias
