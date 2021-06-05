import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import argparse
import json
import utils.preprocessing as pp
import utils.data_helper as dh
from transformers import AdamW
from utils import modeling, model_eval


def run_classifier():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_target", type=str, default="trump_hillary")
    parser.add_argument("--model_select", type=str, default="Bertweet", help="BERTweet or BERT model")
    parser.add_argument("--col", type=str, default="Stance1", help="Stance1 or Stance2")
    parser.add_argument("--train_mode", type=str, default="adhoc", help="unified or adhoc")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--dropout", type=float, default=0)
    parser.add_argument("--alpha", type=float, default=0.5)
    args = parser.parse_args()

    random_seeds = [2,3,4,5,6,7]
    target_word_pair = [args.input_target]
    model_select = args.model_select
    col = args.col
    train_mode = args.train_mode
    lr = args.lr
    batch_size = args.batch_size
    total_epoch = args.epochs
    dropout = args.dropout
    alpha = args.alpha
    
    #Creating Normalization Dictionary
    with open("./noslang_data.json", "r") as f:
        data1 = json.load(f)
    data2 = {}
    with open("./emnlp_dict.txt","r") as f:
        lines = f.readlines()
        for line in lines:
            row = line.split('\t')
            data2[row[0]] = row[1].rstrip()
    normalization_dict = {**data1,**data2}

    for target_index in range(len(target_word_pair)):
        best_result, best_val = [], []
        for seed in random_seeds:    
            print("current random seed: ", seed)
            if train_mode == "unified":
                filename1 = '../data/raw_train_all_onecol.csv'
                filename2 = '../data/raw_val_all_onecol.csv'
                filename3 = '../data/raw_test_all_onecol.csv'
                x_train,y_train,x_train_target,y_train2,x_train_target2 = \
                                            pp.clean_all(filename1,'Stance1',normalization_dict)
                x_val,y_val,x_val_target,y_val2,x_val_target2 = \
                                            pp.clean_all(filename2,'Stance1',normalization_dict)
                x_test,y_test,x_test_target,y_test2,x_test_target2 = \
                                            pp.clean_all(filename3,'Stance1',normalization_dict)
            elif train_mode == "adhoc":
                filename1 = '../data/raw_train_'+target_word_pair[target_index]+'.csv'
                filename2 = '../data/raw_val_'+target_word_pair[target_index]+'.csv'
                filename3 = '../data/raw_test_'+target_word_pair[target_index]+'.csv'
                x_train,y_train,x_train_target,y_train2,x_train_target2 = pp.clean_all(filename1,col,normalization_dict)
                x_val,y_val,x_val_target,y_val2,x_val_target2 = pp.clean_all(filename2,col,normalization_dict)
                x_test,y_test,x_test_target,y_test2,x_test_target2 = pp.clean_all(filename3,col,normalization_dict)
                
            num_labels = len(set(y_train))
            # print(x_train_target[0])
            x_train_all = [x_train,y_train,x_train_target,y_train2,x_train_target2]
            x_val_all = [x_val,y_val,x_val_target,y_val2,x_val_target2]
            x_test_all = [x_test,y_test,x_test_target,y_test2,x_test_target2]
            
            # set up the random seed
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed) 

            # prepare for model
            x_train_all,x_val_all,x_test_all = dh.data_helper_bert(x_train_all,x_val_all,x_test_all,\
                                        target_word_pair[target_index],model_select)
            # print(x_test_all[0][0])
            x_train_input_ids, x_train_seg_ids, x_train_atten_masks, y_train, x_train_len, \
                                        trainloader, x_train_input_ids2, y_train2 = \
                                        dh.data_load(x_train_all, batch_size, train_mode)
            x_val_input_ids, x_val_seg_ids, x_val_atten_masks, y_val, x_val_len, valloader, \
                                        x_val_input_ids2, y_val2 = \
                                        dh.data_load(x_val_all, batch_size, train_mode)                            
            x_test_input_ids, x_test_seg_ids, x_test_atten_masks, y_test, x_test_len, testloader, \
                                        x_test_input_ids2, y_test2 = \
                                        dh.data_load(x_test_all, batch_size, train_mode)

            model = modeling.stance_classifier(num_labels,model_select,dropout).cuda()

            for n,p in model.named_parameters():
                if "bert.embeddings" in n:
                    p.requires_grad = False
            optimizer_grouped_parameters = [
                {'params': [p for n, p in model.named_parameters() if n.startswith('bert.encoder')] , 'lr': lr},
                {'params': [p for n, p in model.named_parameters() if n.startswith('bert.pooler')] , 'lr': 1e-3},
                {'params': [p for n, p in model.named_parameters() if n.startswith('linear')], 'lr': 1e-3},
                {'params': [p for n, p in model.named_parameters() if n.startswith('out')], 'lr': 1e-3}
                ]
            
            loss_function = nn.CrossEntropyLoss(reduction='sum')
            optimizer = AdamW(optimizer_grouped_parameters)

            sum_loss = []
            sum_val = []
            val_f1_average = []
            if train_mode == "unified":
                test_f1_average = [[] for i in range(6)]
            elif train_mode == "adhoc":
                test_f1_average = [[]]

            for epoch in range(0, total_epoch):
                print('Epoch:', epoch)
                train_loss, valid_loss = [], []
                model.train()
                for input_ids,seg_ids,atten_masks,target,length,input_ids2,target2 in trainloader:
                    optimizer.zero_grad()
                    output1, output2 = model(input_ids, seg_ids, atten_masks, length, input_ids2)
                    loss1 = loss_function(output1, target)
                    loss2 = loss_function(output2, target2)
                    loss = loss1 + loss2 * alpha
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()
                    train_loss.append(loss.item())
                sum_loss.append(sum(train_loss)/len(x_train))  
                print(sum_loss[epoch])

                # evaluation on dev set 
                model.eval()
                with torch.no_grad():
                    pred1,_ = model(x_val_input_ids, x_val_seg_ids, x_val_atten_masks, x_val_len, x_val_input_ids2)
                    acc, f1_average, precision, recall = model_eval.compute_f1(pred1,y_val)
                    val_f1_average.append(f1_average)
                
                # evaluation on test set
                if train_mode == "unified":
                    x_test_len_list = dh.sep_test_set(x_test_len)
                    y_test_list = dh.sep_test_set(y_test)
                    x_test_input_ids_list = dh.sep_test_set(x_test_input_ids)
                    x_test_seg_ids_list = dh.sep_test_set(x_test_seg_ids)
                    x_test_atten_masks_list = dh.sep_test_set(x_test_atten_masks)
                    x_test_input_ids_list2 = dh.sep_test_set(x_test_input_ids2)
                elif train_mode == "adhoc":
                    x_test_len_list = [x_test_len]
                    y_test_list = [y_test]
                    x_test_input_ids_list, x_test_seg_ids_list, x_test_atten_masks_list, x_test_input_ids_list2 = \
                                    [x_test_input_ids], [x_test_seg_ids], [x_test_atten_masks], [x_test_input_ids2]
                with torch.no_grad():
                    for ind in range(len(y_test_list)):
                        pred1,_ = model(x_test_input_ids_list[ind], x_test_seg_ids_list[ind], \
                                      x_test_atten_masks_list[ind], x_test_len_list[ind], x_test_input_ids_list2[ind])
                        acc, f1_average, precision, recall = model_eval.compute_f1(pred1,y_test_list[ind])
                        test_f1_average[ind].append(f1_average)

            best_epoch = [index for index,v in enumerate(val_f1_average) if v == max(val_f1_average)][-1] 
            best_result.append([f1[best_epoch] for f1 in test_f1_average])

            print("******************************************")
            print("dev results with seed {} on all epochs".format(seed))
            print(val_f1_average)
            best_val.append(val_f1_average[best_epoch])
            print("******************************************")
            print("test results with seed {} on all epochs".format(seed))
            print(test_f1_average)
            print("******************************************")
        
        # model that performs best on the dev set is evaluated on the test set
        print("model performance on the test set: ")
        print(best_result)

if __name__ == "__main__":
    run_classifier()
