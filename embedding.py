import numpy as np
import torch
from transformers import BertTokenizer, BertModel

def getembedding(model,tokenizer,input_text):
    # 载入模型
    model = model
    # 读取模型对应的tokenizer
    tokenizer = tokenizer
    # 输入文本
    input_text = input_text
    # 通过tokenizer把文本变成 token_id
    input_ids = tokenizer.encode(input_text, add_special_tokens=True)
    # input_ids: [101, 2182, 2003, 2070, 3793, 2000, 4372, 16044, 102]
    input_ids = torch.tensor([input_ids])
    # 获得BERT模型最后一个隐层结果
    with torch.no_grad():
        last_hidden_states = model(input_ids)[1]  # Models outputs are now tuples

    out = last_hidden_states[0]
    out2 = out.cpu().detach().numpy()
    return out2







