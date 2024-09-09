import re
import gc
import json
import time
import torch
import random
import string

from typing import List, Literal, Optional, Union
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from asyncio.log import logger
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager
from transformers import AutoTokenizer, LogitsProcessor
from sse_starlette.sse import EventSourceResponse


MAX_MODEL_LENGTH = 8192


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        
        
app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    all_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def generate_id(prefix: str, k=29):
    suffix = ''.join(random.choices(string.ascii_letters + string.digits, k=k))
    return f"{prefix}{suffix}"


class ModelCard(BaseModel):
    id: str = ""
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "owner"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: Optional[list] = None
    
    
class ModelList(BaseModel):
    object:str = "list"
    data: List[ModelCard] = ["glm-4"]
    

class FunctionCall(BaseModel):
    name: Optional[str] = None
    arguments: Optional[str] = None