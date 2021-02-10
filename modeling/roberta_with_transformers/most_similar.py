import numpy as np
import pandas as pd
import json
import time
from sentence_transformers import SentenceTransformer, util, models
from datetime import datetime

# Data
sentences = set()

with open("data/sentences.txt", "r", encoding="utf-8") as f:
    for line in f:
        sentences.add(line[:-1].lower())
        
sentences = list(sentences)

test_sentence = [
    "검색 문장 1".lower(),
    "검색 문장 2".lower(),
    "검색 문장 3".lower(),
    "검색 문장 4".lower(),
    "검색 문장 5".lower()
]

# Cusom RoBERTa
start = datetime.now()

word_embedding_model = models.Transformer('pre_model/CustomRoBERTa', max_seq_length=512)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

end = datetime.now()
print("time : " + str(end-start))

# 문장임베딩 생성
start = datetime.now()

sent_encode = model.encode(sentences) 

end = datetime.now()
print("time : " + str(end-start))

# 검색 문장 임베딩 생성
start = datetime.now()

test_encode = model.encode(test_sentence)

end = datetime.now()
print("time : " + str(end-start))

# 유사도 추출
start = datetime.now()

cos_sim = util.pytorch_cos_sim(sent_encode, test_encode)

end = datetime.now()
print("time : " + str(end-start))

# 가장 유사한 5개 문장
start = datetime.now()

with open("data/output/result.txt", "w", encoding="utf-8") as f:
    
    for i in range(len(test_sentence)):
        simil = []

        for j in range(len(sentences)):
            simil.append(round(float(util.pytorch_cos_sim(test_encode[i], sent_encode[j])), 4))

        f.write("[" + test_sentence[i] + "]\n")

        top5 = np.argsort(simil)[::-1][1:6]
        for k, index in enumerate(top5): # 유사도 탑5 문장 제시
            f.write(str(k + 1) + " : " + sentences[index] + " (" + str(simil[index]) + ")\n")
        
    f.write("\n")

            
end = datetime.now()
print()
print("time : " + str(end-start))
