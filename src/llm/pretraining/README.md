# LLM Pretraining

대형언어모델(LLM) 사전훈련 코드입니다.

## 파일 구조 (노트북 순서대로)

- `01_config.py`: 모델 하이퍼파라미터 및 토크나이저 설정
- `02_dataset.py`: 데이터셋 및 데이터로더 정의
- `03_model.py`: GPT 모델 아키텍처 정의
- `04_train.py`: 모델 훈련 스크립트
- `05_generate.py`: 텍스트 생성 함수
- `06_inference.py`: 모델 추론 및 결과 확인

## 사용 방법

### 1. 훈련

```bash
python 04_train.py
```

### 2. 추론

```bash
python 06_inference.py
```

## 요구사항

- Python 3.9+
- PyTorch 2.6+
- tiktoken
- matplotlib (훈련 loss 시각화용)

## 참고

- 원본 노트북: `../Pretraining Notebook.ipynb`
- 데이터 경로: `../../assets/cleaned_norway_forest_en.txt`
