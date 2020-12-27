# utils
개발하면서 작성한 코드 중, 재사용이 가능한 코드들을 유틸리티로 저장하고자 함

### [ir_eval](https://github.com/HyeyeonKoo/utils/tree/main/ir_eval)
- ranking된 검색 결과에 따라, 검색 품질이 얼마나 좋은지 평가할 수 있는 클래스
- MRR, MAP, NDCG 세 개의 평가 클래스를 제공

### [preprocess](https://github.com/HyeyeonKoo/utils/tree/main/preprocess)
- kss를 이용한 문장 분리
- konlpy를 이용한 형태소 단위 tokenizing
- sentencepiece를 이용한 subword 분절

### [similarity](https://github.com/HyeyeonKoo/utils/tree/main/similarity)
- l1, l2, inf 거리
- 코사인, 자카드 유사도

### [word representation](https://github.com/HyeyeonKoo/utils/tree/main/word_representation)
- 간단한 인코딩
- TF-IDF, context 행렬 등 간단한 단어 벡터를 구현
