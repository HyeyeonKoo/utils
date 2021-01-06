# Training RoBERTa from scratch & Tuning with S-BERT
transformers==3.5.0

## Pre-train RoBERTa
아래와 같은 파라미터 조합을 시도할 수 있음 (RoBERTa의 경우 base, large를 사용)

- BERT Architecture

| 구분 | BERT-Tiny | BERT-Mini | BERT-Small | BERT-Medium | BERT-Base | BERT-Large |
| --- | --- | --- | --- | --- | --- | --- |
| num_hidden_layers | 2 | 4 | 4 | 8 | 12 | 24 |
| hidden_size | 128 | 256 | 512 | 512 | 768 | 1024 |
| num_attention_heads | 2 | 4 | 8 | 8 | 12 | 16 |
| intermediate_size | 512 | 1024 | 2048 | 2048 | 3072 | 4096 |

- batch size & learning rate (BERT, RoBERTa pre-training, fine-tuning에 사용된 값)
  batch size : {8, 16, 32, 64, 128, ...} (GPU 메모리에 맞게 설정)
  learning rate : {3e-5, 5e-5, 1e-4, 3e-4, 4e-4, 6e-4, 1e-5, 1.5e-5}
  
- epochs
  BERT 논문 : 4
  ※ Huggingface에서는 gradient clipping을 사용하지 않는 한 일반적으로
    num_train_epochs = max_stpes / len(train_dataloader) 라고 함
    또한, 일반적으로 step = (n * epoch) / batch 가 성립함
  
