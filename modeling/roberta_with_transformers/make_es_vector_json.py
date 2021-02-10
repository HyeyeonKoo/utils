import numpy as np
import pandas as pd
import json
from sentence_transformers import SentenceTransformer,models
from datetime import datetime

# Data

data = []

index = 0
with open("data/data_for_es.txt", "r", encoding="utf-8") as f:
    for line in f:
        no, title, morphs = line[:-1].strip().split("\t")
        data.append({"title": title, "doc_id": no, "morphs": morphs})

        index += 1
       
# Model
start = datetime.now()

word_embedding_model = models.Transformer('pre_model/CusomRoBERTa', max_seq_length=512)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())

model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

end = datetime.now()
print("time : " + str(end-start))

# Make Data for ES
from tqdm import tqdm


 start = datetime.now()

for el in tqdm(data):
    el["title_vector"] = model.encode(el["morphs"]).tolist()
    del el["morphs"]
    
end = datetime.now()
print("time : " + str(end-start))

# Save
import json


per_data_len = [len(data), 10000, 20000, 50000, 100000]
file_prefix = [
    "data/output/es_vector/whole/",
    "data/output/es_vector/per_10000/",
    "data/output/es_vector/per_20000/",
    "data/output/es_vector/per_50000/", 
    "data/output/es_vector/per_100000/"
]

start_time = datetime.now()

for i in range(len(per_data_len)):
    start = 0
    end = per_data_len[i]
    whole_data_len = len(data)

    file_index = 0
    while start <= whole_data_len:
        if not start == end:
            result = {"vector_data": data[start:end]}
            with open(file_prefix[i] + str(file_index) + ".json", "w", encoding="utf-8") as f:
                json.dump(result, f)

        start += per_data_len[i]
        if end + per_data_len[i] >= whole_data_len:
            end = whole_data_len
        else:
            end += per_data_len[i]
        file_index += 1
    
end_time = datetime.now()
print("time : " + str(end_time-start_time))
