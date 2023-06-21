import pandas as pd
import torch
from torch import optim, nn
from torch.utils.data import Dataset, DataLoader
from PtransE_a import PtransE_a
from PtransE_b import PtransE_b
from PtransE_c import PtransE_c
from PtransE_d import PtransE_d
from prepare_data import TrainSet, TestSet
from tqdm import tqdm
import numpy as np

device = torch.device('cpu')
embed_dim = 50
num_epochs = 30
train_batch_size = 256
test_batch_size = 256
lr = 1e-2
gamma = 1
d_norm = 2
top_k = 20
is_train = True

def train(model_name='d'):
    train_dataset = TrainSet()
    test_dataset = TestSet()
    test_dataset.convert_word_to_index(train_dataset.entity_to_index, train_dataset.relation_to_index,
                                       test_dataset.raw_data)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)

    # 分类任务数据
    # test_pos, test_neg = test_dataset.generate_classify_data(train_dataset.entity_num, train_dataset.related_dic)

    # 模型
    if (model_name == 'a'):
        print('training model_a...')
        model_path = './model/ptranse_a.pth'
        model = PtransE_a(train_dataset.entity_num, train_dataset.relation_num, device, dim=embed_dim, d_norm=d_norm,
                          gamma=gamma, train_dataset=train_dataset, ).to(device)
    elif (model_name == 'b'):
        print('training model_b...')
        model_path = './model/ptranse_b.pth'
        model = PtransE_b(train_dataset.entity_num, train_dataset.relation_num, device, dim=embed_dim,d_norm=d_norm,
                          gamma=gamma, train_dataset=train_dataset, ).to(device)
    elif (model_name == 'c'):
        print('training model_c...')
        model_path = './model/ptranse_c.pth'
        model = PtransE_c(train_dataset.entity_num, train_dataset.relation_num, device, dim=embed_dim,
                          d_norm=d_norm, gamma=gamma, train_dataset=train_dataset, ).to(device)
    else:
        print('training model_d...')
        model_path = './model/ptranse_d.pth'
        model = PtransE_d(train_dataset.entity_num, train_dataset.relation_num, device, dim=embed_dim,
                          d_norm=d_norm, gamma=gamma, train_dataset=train_dataset, ).to(device)

    optimizer = optim.SGD(model.parameters(), lr=lr)
    for epoch in range(num_epochs):

        total_loss = 0
        for pos, neg in tqdm(train_loader):
            pos, neg = pos.to(device), neg.to(device)
            # pos: [batch_size, 3] => [3, batch_size]
            pos = torch.transpose(pos, 0, 1)
            # pos_head, pos_relation, pos_tail: [batch_size]
            pos_head, pos_relation, pos_tail = pos[0], pos[1], pos[2]

            neg = torch.transpose(neg, 0, 1)
            # neg_head, neg_relation, neg_tail: [batch_size]
            neg_head, neg_relation, neg_tail = neg[0], neg[1], neg[2]
            loss = model(pos_head, pos_relation, pos_tail, neg_head, neg_relation, neg_tail)
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print('total loss:', total_loss)
        print(f"epoch {epoch+1}, loss = {total_loss/train_dataset.__len__()}")

    torch.save(model, model_path)

if __name__ == '__main__':
    train(model_name='a')
    train(model_name='b')
    train(model_name='c')
    train(model_name='d')




