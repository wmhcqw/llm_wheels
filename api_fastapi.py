from fastapi import FastAPI
from contextlib import asynccontextmanager

from utils.llm_utils import LLMUtils, MODEL_PATHS

@asynccontextmanager
async def lifespan(app: FastAPI):
    """这是根据FastAPI官方文档最新的推荐方式创建的Lifespan function
    
    参考链接: https://fastapi.tiangolo.com/advanced/events/#lifespan-function
    
    在yield前, 为before startup阶段
    在yield后, 为after finished阶段

    """
    
    # before startup
    app.state.model = None
    app.state.tokenizer = None
    app.state.model_name = None
    yield
    
    # after finished
    del app.state.model
    del app.state.tokenizer
    app.state.model_name = None

# pass lifespan to FastAPI
app = FastAPI(lifespan=lifespan)


def get_model(model_id, app):
    """根据输入的model_id获得模型, 更改app state中的model, tokenizer, model_id

    如果当前app state中存放的model_name和model_id相同,
    则不需要重新加载model, 否则需要根据model_id加载
    
    如果model_id不在可选的模型内, 则不做任何操作，返回空

    Args:
        model_id (str): api申请的模型model_id
        app (FastAPI): FastAPI app
    """
    
    if model_id is None:
        return
    if app.state.model_name != model_id:
        app.state.model_name = model_id
        app.state.model, app.state.tokenizer = \
            LLMUtils.load_model_and_tokenizer(app.state.model_name)
    else:
        if app.state.model is not None and app.state.tokenizer is not None:
            pass
        else:
            app.state.model, app.state.tokenizer = \
                LLMUtils.load_model_and_tokenizer(app.state.model_name)


@app.get("/chat/{prompt}")
def chat(prompt, instruction="", model_id=None, historys=[], top_k=3, top_p=0.95, temperature=0.6):
    """chat API

    Args:
        prompt (str): 用户输入
        instruction (str, optional): chatbot人设/system. Defaults to "".
        model_id (name, optional): 模型id. Defaults to None.

    Returns:
        dict: json/dict格式的返回结果
    """
    
    if model_id not in MODEL_PATHS.keys():
        return {
            "status": "failed",
            "model_id": model_id,
            "result": f"model id not available. Select from {','.join(list(MODEL_PATHS.keys()))}"
        }
    get_model(model_id, app)
    
    response = LLMUtils.chat(
        prompt=prompt,
        model=app.state.model,
        tokenizer=app.state.tokenizer,
        instruction=instruction,
        historys=historys,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature
    )
    
    return {
        "status": "success",
        "model_id": model_id,
        "result": {
            "prompt": prompt,
            "instruction": instruction,
            "response": response    
        }
    }

