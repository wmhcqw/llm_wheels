import sys
import torch

from tqdm.auto import tqdm
from peft import PeftModel

sys.path.append("../")
from utils.llm_utils import QWenUtils, INSTRUCTIONS


if __name__ == "__main__":
    model, tokenizer = QWenUtils.load_model_and_tokenizer("qwen2-7b-instruct")
    model = PeftModel.from_pretrained(
        model,
        model_id="../finetune_outputs/qwen2/checkpoint-1876"
    )
    
    messages = QWenUtils.CMeEE_dataset_json_transfer(
        "../datasets/CMeEE-V2/CMeEE-V2_dev.json"
    )
    
    preds = []
    prompts = []
    gts = []
    for m in tqdm(messages):
        pred = QWenUtils.chat(
            m["input"], 
            model, 
            tokenizer, 
            INSTRUCTIONS["CMeEE"]
        )
        preds.append(pred)
        prompts.append(m["input"])
        gts.append(sorted(m["output"].split(";")))
    