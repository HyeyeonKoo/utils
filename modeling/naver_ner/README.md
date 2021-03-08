# Naver Ner

### Directory

```
| dataset
	- naver_ner.txt
| KoBERT
	- ...
| data.py
| model.py
| train.py
| trainer.py
```

*naver ner dataset : https://github.com/naver/nlp-challenge/tree/master/missions/ner*

*KoBERT : https://github.com/SKTBrain/KoBERT*



### Requirements

*KoBERT에서 쿠다 11.0이 지원이 안되는 듯 함. 현재 CUDA 세팅을 변경할 수 없으므로 나중에 환경이 가능할 때 다시 테스트.. ㅜㅜ*



### Run

```sh
python train.py
```

