#-*-coding:utf-8-*-

# cuda 동작 확인
import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))

# Config
from transformers import RobertaConfig

config = RobertaConfig(
    num_hidden_layers=4,    
    hidden_size=512,    
    hidden_dropout_prob=0.1,
    num_attention_heads=8,
    attention_probs_dropout_prob=0.1,    
    intermediate_size=2048,    
    vocab_size=34492,
    type_vocab_size=1,    
    initializer_range=0.02,
    max_position_embeddings=512,
    position_embedding_type="absolute"
)

# Tokenizer
from transformers import RobertaTokenizerFast

tokenizer = RobertaTokenizerFast.from_pretrained("vocab", max_len=512)

# Init model
from transformers import RobertaForMaskedLM

model = RobertaForMaskedLM(config=config)

# Build dataset
from datetime import datetime
from transformers import LineByLineTextDataset

start = datetime.now()

train_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="data/train.txt",
    block_size=tokenizer.max_len_single_sentence
)

end = datetime.now()
print("build dataset : %s" % str(end-start))

start = datetime.now()

eval_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="data/eval.txt",
    block_size=tokenizer.max_len_single_sentence
)

end = datetime.now()
print("build dataset : %s" % str(end-start))

from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

# Init Trainer
from transformers import Trainer, TrainingArguments

num_train_epochs = 1
max_steps = num_train_epochs * len(train_dataset)
warmup_steps = int(max_steps*0.05)

training_args = TrainingArguments(
    output_dir="log/ex1/checkpoints",
    overwrite_output_dir=True,
    
    do_train=True,
    max_steps=max_steps,
    warmup_steps=warmup_steps,
    num_train_epochs=num_train_epochs,

    per_device_train_batch_size=100,
    
    learning_rate=5e-5,
    
    weight_decay=0,
    max_grad_norm=1,
    
    adam_beta1=0.9,
    adam_beta2=0.98,
    adam_epsilon=1e-6,

    do_eval=True,
    per_device_eval_batch_size=100,
    evaluation_strategy="steps",
    eval_steps=1000,
    
#     disable_tqdm=True
    logging_dir="log/ex1/tensorboard",
    logging_first_step=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    prediction_loss_only=True,
)

# Training
start = datetime.now()

trainer.train()

end = datetime.now()
print("train time : %s" % str(end-start))

# Save
trainer.save_model("pre_model")
