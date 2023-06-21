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
train_batch_size = 256
test_batch_size = 256

top_k = 20

def evaluate(model_name='d'):
    train_dataset = TrainSet()
    test_dataset = TestSet()
    test_dataset.convert_word_to_index(train_dataset.entity_to_index, train_dataset.relation_to_index,
                                       test_dataset.raw_data)
    # train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)

    # 分类任务数据
    test_pos, test_neg = test_dataset.generate_classify_data(train_dataset.entity_num, train_dataset.related_dic)

    if (model_name == 'a'):
        print('loading model_a...')
        model_path = './model/ptranse_a.pth'

    elif (model_name == 'b'):
        print('loading model_b...')
        model_path = './model/ptranse_b.pth'

    elif (model_name == 'c'):
        print('loading model_c...')
        model_path = './model/ptranse_c.pth'

    else:
        print('loading model_d...')
        model_path = './model/ptranse_d.pth'

    model = torch.load(model_path)

    classify_accuracy = model.classify(test_pos, test_neg)
    print(f"===>final results, classify accuracy: {classify_accuracy}")

    # 保存
    # print("保存kge embedding")
    # emtities_embedding = model.entity_embedding.weight.tolist()
    # relation_embedding = model.relation_embedding.weight.tolist()
    # np.savetxt('./model/kge_entities_embedding.txt', np.array(emtities_embedding))
    # np.savetxt('./model/kge_relation_embedding.txt', np.array(relation_embedding))

    corrct_test_1 = 0
    corrct_test_3 = 0
    corrct_test_10 = 0
    mr_list = []
    for data in tqdm(test_loader):
        data = data.to(device)
        # data: [batch_size, 3] => [3, batch_size]
        data = torch.transpose(data, 0, 1)
        out1, out3, out10, mr = model.tail_predict(data[0], data[1], data[2], k=top_k, mr_topk=1000)
        corrct_test_1 += out1
        corrct_test_3 += out3
        corrct_test_10 += out10
        mr_list.append(mr)

    hit1 = corrct_test_1 / test_dataset.__len__()
    hit3 = corrct_test_3 / test_dataset.__len__()
    hit10 = corrct_test_10 / test_dataset.__len__()
    mr = sum(mr_list) / len(mr_list) - 100.0

    print(f"===>final results, hit@1 {hit1}")
    print(f"===>final results, hit@3 {hit3}")
    print(f"===>final results, hit@10 {hit10}")
    print(f"===>final results, mr {mr}")

    # 结果输出
    output = [[classify_accuracy, hit1, hit3, hit10, mr]]
    out_df = pd.DataFrame(output, columns=['分类准确率', 'hit1', 'hit3', 'hit10', 'mr'])
    out_df.to_csv(f'./results/output_{model_name}.csv', index=False)



if __name__ == '__main__':

    # evaluate(model_name='a')
    # evaluate(model_name='b')
    # evaluate(model_name='c')
    evaluate(model_name='d')