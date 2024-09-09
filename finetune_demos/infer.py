import sys
import torch

from peft import PeftModel

sys.path.append("../")
from utils.llm_utils import QWenUtils, INSTRUCTIONS


if __name__ == "__main__":
    
    top_p = 0.95
    top_k = 3
    temperature = 0.6
    
    model, tokenizer = QWenUtils.load_model_and_tokenizer("qwen2-7b-instruct")
    model = PeftModel.from_pretrained(
        model,
        model_id="../finetune_outputs/qwen2/checkpoint-1876"
    )
    print(QWenUtils.chat(
        prompt="六、新生儿疾病筛查的发展趋势自1961年开展苯丙酮尿症筛查以来，随着医学技术的发展，符合进行新生儿疾病筛查标准的疾病也在不断增加，无论在新生儿疾病筛查的病种，还是在新生儿疾病筛查的技术方法上，都有了非常显著的进步。",
        model=model,
        tokenizer=tokenizer,
        instruction=INSTRUCTIONS["CMeEE"],
        top_p=top_p,
        top_k=top_k,
        temperature=temperature
    ))