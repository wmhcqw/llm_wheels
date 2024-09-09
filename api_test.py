import requests

params = {
    "instruction": "You are a joke AI",
    "model_id": "qwen2-7b-instruct"
}


result = requests.get(
    "http://127.0.0.1:8000/chat/你好",
    params=params
)

# print(result.text)
result = eval(result.text)
print(result["result"]["response"])