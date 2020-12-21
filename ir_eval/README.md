# Information Retrieval Evaluation Class
자연어로 검색했을 때, 모델의 predict로 반환되는 ranking된 결과를 비교하고 평가하기 위한 함수인 MRR, MAP, NDCG를 구현<br/>
정답 데이터와 결과 데이터 비교 시, exact match와 jaccard similarity 방식으로 비교 가능<br/>

### requirements
- konlpy

### Theory
- MRR : Mean Reciprocal Rank
  - 각 검색 문장별 상위 n개의 검색 결과와 정답을 비교해 가장 높은 위치를 역수로 계싼
  - 각 검색 문장별 계산 결과를 평균
  
- MAP : Mean Average Precision
  - 각 검색 문장별 상위 n개의 검색 결과에서 정답이 있는 모든 위치를 역수로 계산해 평균
  - 각 검색 문장별 계산 결과를 평균
  
- NDGC : Normalized Discounted Cumulative Gain
  - 정답의 위치 별 특정 점수를 매기고, 정답의 점수(Ideal DCG)를 구함
  - 각 검색 문장별로 위와 같은 점수 표를 기준으로 검색 결과의 점수를 구함
  - 각 검색 문장별로 "검색 결과 점수 / 정답 점수"를 구하고, 모든 계산 결과를 평균
  
*자세한 이론은 [여기](https://medium.com/swlh/rank-aware-recsys-evaluation-metrics-5191bba16832)를 참고하세요.*
