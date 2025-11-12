"""
토큰화 테스트 스크립트
"""
import tiktoken

# 토크나이저 초기화
tokenizer = tiktoken.get_encoding("gpt2")

# 테스트 텍스트
text = "Harry Potter was a wizard."

# 토큰화
tokens = tokenizer.encode(text)

# 결과 출력
print("글자수:", len(text), "토큰수", len(tokens))
print(tokens)
print(tokenizer.decode(tokens))

# 각 토큰별 디코딩
for t in tokens:
    print(f"{t}\t -> {tokenizer.decode([t])}")

print("\n" + "="*50)
print("글자별 토큰화 테스트")
print("="*50)

# 글자별 토큰화 테스트
for char in text:
    token_ids = tokenizer.encode(char)
    decoded = tokenizer.decode(token_ids)
    print(f"{char} -> {token_ids} -> {decoded}")
