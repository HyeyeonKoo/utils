#-*-coding:utf-8-*-

# Data
import os
from datetime import datetime

start = datetime.now()

# 토크나징된 데이터로부터 한 문장씩 저장
i=0
with open("data/morphed_sentences.txt", "r", encoding="utf-8") as f1:
    for line in f1:
        try:
            with open(os.path.join("data/for_tokenizing", str(i)+'.txt'), "w", encoding="utf-8") as f2:
                f2.write(line[:-1])
        except Exception as e:
            print(line, e) 
        i+=1
        
# 사용자 사전 정보를 활용하기 위해 한 단어씩 저장
with open("data/duser_dic.txt", "r", encoding="utf-8") as f1:
    for line in f1:
        try:
            with open(os.path.join("data/for_tokenizing", str(i)+'.txt'), "w", encoding="utf-8") as f2:
                f2.write(line[:-1])
        except Exception as e:
            print(line, e) 
        i += 1
        
end = datetime.now()
print("prepring data : %s" % str(end - start))

# Huggingface의 tokenizers를 이용해 vocab 생성
sent_files = os.listdir("data/for_tokenizing")
sent_paths = [os.path.join("data/for_tokenizing", file) for file in sent_files]

from tokenizers import ByteLevelBPETokenizer

start = datetime.now()

tokenizer = ByteLevelBPETokenizer()

tokenizer.train(files=sent_paths, vocab_size=24000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

end = datetime.now()
print("train ByteLevelBPETokenizer : %s" % str(end-start))

tokenizer.save_model("vocab")

# 결과 확인
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

tokenizer = ByteLevelBPETokenizer(
    "vocab/vocab.json",
    "vocab/merges.txt",
)

tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")),
    ("<s>", tokenizer.token_to_id("<s>")),
)
tokenizer.enable_truncation(max_length=512)

tokenizer.encode("확인 문장(형태소 분석된 형태로 입력)").tokens
