# pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
# pip install boto3
# pip install matplotlib
# pip install --upgrade transformers>=4.5

import os
import sys
import time
import re
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

import etri_tokenizer.file_utils
from etri_tokenizer.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from etri_tokenizer.eojeol_tokenization import eojeol_BertTokenizer

from transformers import AutoModel

NER_TAGGED_CORPUS = "NER_tagged_corpus_ETRI_exobrain_team.txt"
KOREAN_RAWTEXT_LIST = "./003_bert_eojeol_pytorch/vocab.korean.rawtext.list"
BERT_EOJEOL_PYTORCH = "./003_bert_eojeol_pytorch"
TOKENIZER = eojeol_BertTokenizer.from_pretrained(KOREAN_RAWTEXT_LIST, do_lower_case=False)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
HIDDEN_SIZE = 768
BATCH_SIZE = 32
MSL = 60
LIFE = 7
LABEL = ["부정", "긍정"]
LABEL_MAP = dict()
NUM_CLASS = len(LABEL)

for i in range(NUM_CLASS) :
    LABEL_MAP[LABEL[i]] = i


class BertModel(nn.Module) :
    def __init__(self) :
        super(BertModel, self).__init__()
        bert = AutoModel.from_pretrained(BERT_EOJEOL_PYTORCH)
        self.bert = bert
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(HIDDEN_SIZE, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, NUM_CLASS)
        self.softmax = nn.LogSoftmax(dim=1)
        return
    
    def forward(self, sent_id, mask) :
        outputs = self.bert(input_ids=sent_id, attention_mask=mask)
        cls_out = outputs.pooler_output # shape: (batch_size, hidden_size)
        x = self.fc1(cls_out)
        x1 = self.relu(x)
        x2 = self.dropout(x1)
        x3 = self.fc2(x2)
        x4 = self.relu(x3)
        x5 = self.dropout(x4)
        x6 = self.fc3(x5)
        x7 = self.softmax(x6)
        return x7


# 로그 출력
def log(msg="", path="./log.txt", output=True) :
    log_file = open(path, "a", encoding="utf-8")
    log_file.write(str(msg) + "\n")
    log_file.close()
    if output :
        print(msg)
    return


def prepareExample(filename="./ratings_all.txt") :
    sentences = []
    target_labels = []

    with open(filename, 'r', encoding='utf-8') as fpr :
        first_line = True
        for line in fpr.readlines() :
            if first_line :
                first_line = False
                continue

            line_splited = line.split('\t')
            sentences.append(line_splited[1])
            lab_idx = line_splited[2][:-1]
            lab_idx = int(lab_idx)
            target_labels.append(lab_idx)
    
    sentences_id = []
    for sentence in sentences :
        token_list = TOKENIZER.tokenize(sentence)
        token_id_list = TOKENIZER.convert_tokens_to_ids(token_list)
        sentences_id.append(token_id_list)
    
    list_tok_seq = []
    list_mask_seq = []

    for token_id_list in sentences_id :
        tok_seq = [2]
        tok_seq.extend(token_id_list)
        mask_seq = [1] * len(tok_seq)

        if len(tok_seq) > MSL-1 :
            tok_seq = tok_seq[:MSL-1]
            mask_seq = mask_seq[:MSL-1]

        tok_seq.append(3)
        mask_seq.append(1)

        tok_seq.extend([0]*(MSL-len(tok_seq)))
        mask_seq.extend([0]*(MSL-len(mask_seq)))

        list_tok_seq.append(tok_seq)
        list_mask_seq.append(mask_seq)

    list_tok_seq = torch.tensor(list_tok_seq, dtype=torch.long)
    list_mask_seq = torch.tensor(list_mask_seq, dtype=torch.long)
    target_labels = torch.tensor(target_labels, dtype=torch.long)

    return list_tok_seq, list_mask_seq, target_labels


def getTrainExample(tok_seq:torch.Tensor, mask_seq:torch.Tensor, label_seq:torch.Tensor) :
    num_examples = tok_seq.shape[0]
    num_tra = int(0.8 * num_examples)
    num_val = num_tra + int(0.1 * num_examples)

    train_tok = tok_seq[:num_tra, :]
    train_mask = mask_seq[:num_tra, :]
    train_y = label_seq[:num_tra]

    train_data = TensorDataset(train_tok, train_mask, train_y)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE, num_workers=4)

    valid_tok = tok_seq[num_tra:num_val, :]
    valid_mask = mask_seq[num_tra:num_val, :]
    valid_y = label_seq[num_tra:num_val]
    
    valid_data = TensorDataset(valid_tok, valid_mask, valid_y)
    valid_sampler = RandomSampler(valid_data)
    valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=BATCH_SIZE, num_workers=4)

    test_tok = tok_seq[num_val:, :]
    test_mask = mask_seq[num_val:, :]
    test_y = label_seq[num_val:]

    test_data = TensorDataset(test_tok, test_mask, test_y)
    test_sampler = RandomSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE, num_workers=4)

    return train_dataloader, valid_dataloader, test_dataloader


def startLearning(train:DataLoader, valid:DataLoader, model:BertModel) :
    epoch = 1
    remain_life = LIFE
    num_training_batches = len(train)
    num_val_batches = len(valid)
    log(f"num of train and validation batches: {num_training_batches} {num_val_batches}")

    model = model.to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=0.5e-5)
    loss_fn = nn.NLLLoss()

    train_loss = []
    valid_loss = []
    min_valid_loss = -1.0
    best_epoch_point = -1

    accuracy = []
    max_acc = -1.0

    while remain_life > 0 :
        # 학습
        model.train()
        train_total_loss = 0
        for step, batch in enumerate(train) :
            batch = [r.to(DEVICE) for r in batch]
            sent_id, mask, label = batch
            model.zero_grad()
            preds = model(sent_id, mask)
            loss = loss_fn(preds, label)
            train_total_loss += loss.item()
            loss.backward()
            optimizer.step()
        train_avg_loss = train_total_loss / num_training_batches
        train_loss.append(train_avg_loss)
        log(f"At epoch = {epoch}, avg loss = {train_avg_loss}")

        # 검증
        model.eval()
        hit_cnt = 0
        total_cnt = 0
        with torch.no_grad() :
            for step, batch in enumerate(valid) :
                batch = [r.to(DEVICE) for r in batch]
                sent_id, mask, label = batch
                preds = model(sent_id, mask)
                preds_label = torch.argmax(preds, dim=1)
                preds_label = preds_label.cpu()
                preds_label = preds_label.numpy()

                label = label.cpu()
                label = label.numpy()

                for i in range(label.shape[0]) :
                    if preds_label[i] == label[i] :
                        hit_cnt += 1
                total_cnt += len(label)
        accuracy.append(hit_cnt / total_cnt)

        if epoch > 1 and max_acc > accuracy[-1] :
            remain_life -= 1
        else :
            max_acc = accuracy[-1]
            best_epoch_point = epoch
            remain_life = LIFE
            torch.save(model.state_dict(), 'saved_weights.pth')

        learningProgress = open("./LearningProgress.txt", 'w', encoding='utf-8')
        learningProgress.write(f"{best_epoch_point}\n{epoch}\n{max_acc}\n{remain_life}\n{train_loss}\n{accuracy}")
        learningProgress.close()

        # model.eval()
        # valid_total_loss = 0
        # with torch.no_grad() :
        #     for step, batch in enumerate(valid) :
        #         batch = [r.to(DEVICE) for r in batch]
        #         sent_id, mask, label = batch
        #         preds = model(sent_id, mask)

        #         loss = loss_fn(preds, label)
        #         valid_total_loss += loss.item()

        # valid_avg_loss = valid_total_loss / num_val_batches
        # valid_loss.append(valid_avg_loss)

        # if epoch > 1 and min_valid_loss < valid_loss[-1] :
        #     remain_life -= 1
        # else :
        #     min_valid_loss = valid_loss[-1]
        #     best_epoch_point = epoch
        #     remain_life = LIFE
        #     torch.save(model.state_dict(), 'saved_weights.pth')
        
        # learningProgress = open("./LearningProgress.txt", 'w', encoding='utf-8')
        # learningProgress.write(f"{best_epoch_point}\n{epoch}\n{min_valid_loss}\n{remain_life}\n{train_loss}\n{valid_loss}")
        # learningProgress.close()

        epoch += 1
    
    x_axis = range(1, epoch)
    plt.plot(x_axis, train_loss, label="training")
    # plt.plot(x_axis, valid_loss, label="validation")
    plt.plot(x_axis, accuracy, label="accuracy")
    plt.axvline(x=best_epoch_point, ymin=0, ymax=1, linestyle="--", label="best epoch point")
    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.savefig("./learning_result.png")

    return model


def startTesting(test:DataLoader, model:BertModel) :
    # 테스트
    model.eval()
    accuracy = 0.0
    hit_cnt = 0
    total_cnt = 0
    with torch.no_grad() :
        for step, batch in enumerate(test) :
            batch = [r.to(DEVICE) for r in batch]
            sent_id, mask, label = batch
            preds = model(sent_id, mask)
            preds_label = torch.argmax(preds, dim=1)
            preds_label = preds_label.cpu()
            preds_label = preds_label.numpy()

            label = label.cpu()
            label = label.numpy()

            for i in range(label.shape[0]) :
                if preds_label[i] == label[i] :
                    hit_cnt += 1
            total_cnt += len(label)
    accuracy = hit_cnt / total_cnt
    log(f"Accuracy = {accuracy}")
    
    return


def main() :
    log(msg="\n------------------------" + str(datetime.now()) + "------------------------\n", output=False)
    log(f"DEVICE : {DEVICE}\n")

    for i in range(torch.cuda.device_count()) :
        print(torch.cuda.get_device_name(device=i))
    
    # 저장된 모델 불러오기
    model_files = list()
    file_list = os.listdir("./")
    for file_name in file_list :
        if os.path.splitext(file_name)[1] == ".pth" :
            model_files.append(file_name)
    
    selectedModelFile = ""
    if len(model_files) > 0 :
        log("------------------------")
        for i in range(len(model_files)) :
            log(f"[{i+1}]   {model_files[i]}")
        log("[N]   사용 안함")
        log("------------------------")
        cmd = input("불러올 모델파일 번호 입력   >> ").upper()
        log(f"불러올 모델파일 번호 입력   >> {cmd}", output=False)
        log("")

        if cmd.isnumeric() and len(model_files) >= int(cmd) and 0 < int(cmd) :
            selectedModelFile = model_files[int(cmd)-1]

    model = BertModel()

    if selectedModelFile != "" :
        # 선택한 모델파일 불러오기
        model.load_state_dict(torch.load("./" + selectedModelFile))
        model = model.to(DEVICE)
        log("")
    else :
        # 학습 예제 생성
        log("학습 예제 생성 중")
        start_time = time.perf_counter()
        x, mask, y = prepareExample()
        train, valid, test = getTrainExample(x, mask, y)
        end_time = time.perf_counter()
        elapse = end_time - start_time
        log(f"└학습 예제 생성 완료 ({elapse:.3f} sec)\n")

        # 모델 훈련
        log("훈련 중")
        start_time = time.perf_counter()
        model = startLearning(train, valid, model)
        end_time = time.perf_counter()
        elapse = end_time - start_time
        log(f"└훈련 완료 ({elapse:.3f} sec)\n")

        # 모델 테스팅
        log("테스팅 중")
        start_time = time.perf_counter()
        startTesting(test, model)
        end_time = time.perf_counter()
        elapse = end_time - start_time
        log(f"└테스팅 완료 ({elapse:.3f} s)\n")
    
    # 실제 입력
    # cmd = "dummy"
    # while True :
    #     cmd = input("문장 입력 >> ")
    #     log(f"문장 입력 >> {cmd}", output=False)

    #     if cmd == "" :
    #         break



    #     log("\n")

    return

if __name__ == "__main__" :
    main()