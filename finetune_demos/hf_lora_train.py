import sys
import pandas as pd

from functools import partial
from datasets import Dataset
from peft import (
    TaskType,
    LoraConfig,
    get_peft_model
)
from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)

sys.path.append("../")
from utils.llm_utils import QWenUtils


if __name__ == "__main__":
    """
    messages = [{
        "instruction": insturction(System Prompt),
        "input": input(User Input),
        "output": output(Assistant Output),
    },...]
    """
    messages = QWenUtils.CMeEE_dataset_json_transfer(
        "../datasets/CMeEE-V2/CMeEE-V2_train.json",
    )
    
    model, tokenizer = QWenUtils.load_model_and_tokenizer("qwen2-7b-instruct")
    model.enable_input_require_grads()
    
    """
    从list of messages转化为Dataset
    """
    train_df = pd.DataFrame(data=messages)
    train_ds = Dataset.from_pandas(train_df)
    train_dataset = train_ds.map(
        partial(QWenUtils.process_func, tokenizer=tokenizer),
        remove_columns=train_ds.column_names,
    )
    
    """
    LoraConfig:
        target_modules: Lora影响的权重
        r: Lora-R
        alpha: Lora-Alpha
    """
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj", "k_proj", "v_proj", 
            "o_proj", "gate_proj", 
            "up_proj", "down_proj"],
        inference_mode=False,
        r=16, lora_alpha=32, lora_dropout=0.1
    )
    
    model = get_peft_model(model, config)
    
    """
    TrainingArguments:
        output_dir:微调后模型的输出位置
        per_device_train_batch_size: 每张卡上的batch size
        gradient_accumulation_steps: 梯度累计的步数（每多少步计算一次梯度）
        logging_steps: 每多少步打印一次log
        num_train_epochs: 训练轮次
        save_steps: 每多少步保存一次模型
        learning_rate: 学习率
    """
    args = TrainingArguments(
        output_dir="../finetune_outputs/qwen2",
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        logging_steps=10,
        num_train_epochs=2,
        save_steps=1000,
        learning_rate=1e-4,
        gradient_checkpointing=True,
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True)
    )
    
    trainer.train()
    
    