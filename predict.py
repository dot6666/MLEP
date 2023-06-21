from embedding import getembedding
import time
from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from numpy.linalg import norm
import pandas as pd
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# -*- coding: UTF-8 -*-

app = FastAPI()
model_name = 'bert-base-chinese'
# 读取模型对应的tokenizer
tokenizer = BertTokenizer.from_pretrained(model_name)
# 载入模型
model = torch.load('data/ml-measure-model.pkl')
# 读取数据
df = pd.read_csv("data/all_event.csv", encoding = 'utf-8')
# 1列
df1 = df.iloc[:,0]
measure_list = list(df1)

def measure_predict(input):
    input1 = input.split(',')
    measure = input1[-1]
    best_similarity = -2
    rec_measure = input
    f = open("data/nursing_example.txt", encoding="utf-8")
    f = f.read()
    f = eval(f)
    if measure in f.keys():
        for i in measure_list:
            if i != measure:
                a = getembedding(model, tokenizer, measure)
                b = getembedding(model, tokenizer, i)
                # similarity = np.dot(a,b)/(norm(a)*norm(b))
                out1 = torch.tensor(a)
                out2 = torch.tensor(b)
                similarity = torch.cosine_similarity(out1, out2, dim=0)
                similarity = similarity.cpu().detach().numpy()
                if similarity >= best_similarity:
                    best_similarity = similarity
                    rec_measure = i
        rec_measure = f[measure]
    else:
        for i in measure_list:
            if i != measure:
                a = getembedding(model, tokenizer, measure)
                b = getembedding(model, tokenizer, i)
                # similarity = np.dot(a,b)/(norm(a)*norm(b))
                out1 = torch.tensor(a)
                out2 = torch.tensor(b)
                similarity = torch.cosine_similarity(out1, out2, dim=0)
                similarity = similarity.cpu().detach().numpy()
                if similarity >= best_similarity:
                    best_similarity = similarity
                    rec_measure = i
    return rec_measure

# for i in range(10):
#     print("请输入护理措施:")
#     measure = input()
#     rec_measure = measure_predict(measure)
#     print(rec_measure)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.post("/predict")
def generation(measure):
    result = measure_predict(measure)
    return result


if __name__ == '__main__':
    uvicorn.run(app='predict:app', host="0.0.0.0", port=9990, reload=True, debug=True)




