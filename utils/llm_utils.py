import os
import json
import torch
import textwrap

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)
from huggingface_hub import snapshot_download

_cur_dir_path = os.path.dirname(__file__)


MAX_LENGTH = 384

HF_MODEL_PATHS = {
    # QWen
    "qwen2-0.5b-instruct": "Qwen/Qwen2-0.5B-Instruct",
    "qwen2-1.5b-instruct": "Qwen/Qwen2-1.5B-Instruct",
    "qwen2-7b-instruct":   "Qwen/Qwen2-7B-Instruct",
    "qwen2-72b-instruct":  "Qwen/Qwen2-72B-Instruct",
    
    # GLM
    "glm-4-9b-chat": "THUDM/glm-4-9b-chat",
    
    # Phi-3.5
    "phi-3.5-mini-instruct": "microsoft/Phi-3.5-mini-instruct",
    "phi-3.5-moe-instruct":  "microsoft/Phi-3.5-MoE-instruct"
}

MODEL_PATHS = {
    # QWen
    "qwen2-0.5b-instruct": os.path.join(_cur_dir_path, "../", "llm_models/Qwen2-0.5B-Instruct/"),
    "qwen2-1.5b-instruct": os.path.join(_cur_dir_path, "../", "llm_models/Qwen2-1.5B-Instruct/"),
    "qwen2-7b-instruct": os.path.join(_cur_dir_path, "../", "llm_models/Qwen2-7B-Instruct/"),
    "qwen2-72b-instruct": os.path.join(_cur_dir_path, "../", "llm_models/Qwen2-72B-Instruct/"),

    # GLM
    "glm-4-9b-chat": os.path.join(_cur_dir_path, "../", "llm_models/glm-4-9b-chat/"),
    
    # Phi-3.5
    "phi-3.5-mini-instruct": os.path.join(_cur_dir_path, "../", "llm_models/phi-3.5-mini-instruct/"),
    "phi-3.5-moe-instruct": os.path.join(_cur_dir_path, "../", "llm_models/phi-3.5-MoE-instruct/")
}

INSTRUCTIONS = {
    "CMeEE": "你是一个擅长实体提取的医学专家，你会收到一段文本，请从中提取出所有的实体，并标注出它们的类型，不要输出重复的实体。"
}


class LLMUtils:
    
    @staticmethod
    def load_model_and_tokenizer(model_name):
        """load model and tokenizer from local or huggingface hub

        Args:
            model_name (str): repo ID

        Returns:
            AutoModelForCasualLM: loaded model
            AutoTokenizer: model's tokenizer
        """
        print("Load Model From ", MODEL_PATHS[model_name])
        try:
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATHS[model_name],
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="cuda"
            )
            
        except Exception as e:
            print("Local Model Not Found, Downloading From HF.")
            snapshot_download(
                HF_MODEL_PATHS[model_name], 
                local_dir=MODEL_PATHS[model_name]
            )
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_PATHS[model_name],
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="cuda",
                cache_dir=MODEL_PATHS[model_name]
            )
            
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATHS[model_name],
            trust_remote_code=True,
        )
        
        return model, tokenizer
    
    @staticmethod
    def CMeEE_dataset_json_transfer(
            json_data_path, 
            instruction=INSTRUCTIONS["CMeEE"]
        ):
        """CMeEE dataset transfer from json to chat format

        Args:
            json_data_path (str, Path): Json data file path
            instruction (str, optional): Chat model's instruction. Defaults to INSTRUCTIONS["CMeEE"].

        Returns:
            list: list of CMeEE dicts
        """
        messages = []
        with open(json_data_path, "r") as f:
            datas = json.load(f)
        for data in datas:
            message = {
                "instruction": instruction,
                "input": data["text"],
                "output": ";".join([
                    d["entity"]+":"+d["type"] for d in data["entities"]
                ])
            }
            messages.append(message)
        return messages
    
    @staticmethod
    def chat(
        prompt: str,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        instruction: str,
        historys=[],
        top_k=3,
        top_p=0.95,
        temperature=0.6
    ):
        """LLM Model chat function

        Args:
            prompt (str): Chat input message
            model (AutoModelForCausalLM): Loaded LLM Model
            tokenizer (AutoTokenizer): Model's tokenizer
            instruction (str): Chat instruction
            historys (list): Chat History

        Returns:
            str: LLM Model chat output message
        """
        if historys:
            prompt = "\n".join(historys) + prompt
        
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": prompt}
        ]
        
        generate_kwargs = {
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
        }
        
        response = LLMUtils.predict(messages, model, tokenizer, **generate_kwargs)
        historys.append(prompt+response)
        return response
        
    @staticmethod
    def predict(
        messages, 
        model: AutoModelForCausalLM, 
        tokenizer: AutoTokenizer, 
        max_new_tokens=512, 
        **kwargs
    ):
        device = "cuda"
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(device)
        
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=max_new_tokens,
            **kwargs
        )
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return response
    
    @staticmethod
    def process_func():
        pass


class QWenUtils(LLMUtils):
    
    @staticmethod
    def process_func(example, tokenizer, max_length=MAX_LENGTH):
        input_ids, attention_mask, labels = [], [], []
        instruction = tokenizer(
            textwrap.dedent(
                f"""\
                <im_start>system
                {example['instruction']}<im_end>
                <im_start>user
                {example['input']}<im_end>
                <im_start>assistant
                """
            ), 
            add_special_tokens=False
        )
        response = tokenizer(f"{example['output']}", add_special_tokens=False)
        input_ids = instruction["input_ids"] + \
                    response["input_ids"] + \
                    [tokenizer.pad_token_id]
        attention_mask = (
            instruction["attention_mask"] + response["attention_mask"] + [1]
        )
        labels = [-100] * len(instruction["input_ids"]) + \
                 response["input_ids"] + \
                 [tokenizer.pad_token_id]
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]
            labels = labels[:max_length]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
      
  
class GLMUtils(LLMUtils):
    
    @staticmethod
    def process_func(example, tokenizer, max_length=MAX_LENGTH):
        input_ids, attention_mask, labels = [], [], []
        instruction = tokenizer(
            textwrap.dedent(
                f"""\
                <|system|>
                {example['instruction']}<|endoftext|>
                <|user|>
                {example['input']}<|endoftext|>
                <|assistant|>
                """
            ),
            add_special_tokens=False
        )
        response = tokenizer(f"{example['output']}", add_special_tokens=False)
        input_ids = instruction["input_ids"] + \
                    response["input_ids"] + \
                    [tokenizer.pad_token_id]
        attention_mask = (
            instruction["attention_mask"] + response["attention_mask"] + [1]
        )
        labels = [-100] * len(instruction["input_ids"]) + \
                 response["input_ids"] + \
                 [tokenizer.pad_token_id]
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
            attention_mask = attention_mask[:max_length]
            labels = labels[:max_length]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }