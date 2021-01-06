#-*-coding:utf-8-*-

from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model="model",
    tokenizer="model"
)

fill_mask("텍스트 중 <mask>를 넣 어 요 .")

