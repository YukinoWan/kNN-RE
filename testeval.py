import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from typing import List, Tuple
from decimal import Decimal, ROUND_DOWN
from sklearn.metrics import classification_report
from knn_bert import knn_bert

def compute_f1(preds, labels):
    n_gold = n_pred = n_correct = 0
    for pred, label in zip(preds, labels):
        if pred != 0:
            n_pred += 1
        if label != 0:
            n_gold += 1
        if (pred != 0) and (label != 0) and (pred == label):
            n_correct += 1
    if n_correct == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    else:
        prec = n_correct * 1.0 / n_pred
        recall = n_correct * 1.0 /n_gold
        if prec + recall > 0:
            f1 = 2.0 * prec * recall / (prec + recall)
        else:
            f1 =0.0
        return {"precision": prec, "recall": recall, "f1": f1}
def evaluate(model,
        dev_data_loader,
        device: torch.device,
        knn_train,
        label_dict=dict(),
        embeddings=dict(),
        knn_num=64,
        ratio=0.0,
        ) -> Tuple[float, List]:

    ce_loss = nn.CrossEntropyLoss()
    num_correct, size, total_loss = 0, 0, 0
    #prediction = []

    with torch.no_grad():
        dev_bar = tqdm(dev_data_loader)
        distri = []
        preds = []
        preds1 = []
        preds2 = []
        preds3 = []
        preds4 = []
        preds5 = []
        labels = []
        embed_list = []
        for batch_idx, batch in enumerate(dev_bar):
            batch_size = len(batch['input_ids'])
            batch = {key: value.to(device) for key, value in batch.items()}

            # forward
            output, cls = model(input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    token_type_ids=batch['token_type_ids'])

            predictions = torch.argmax(F.softmax(output, dim=1), dim=1)
            loss = ce_loss(output, batch['label'])

            embed = cls.cpu().numpy()
            if(knn_train):
                distributions = F.softmax(output, dim=1)
                predictions = torch.argmax(distributions, dim=1)
                preds.append(predictions.to('cpu').detach().numpy().copy())
                distri.append(distributions.to('cpu').detach().numpy().copy())
            else:
                predictions1, predictions2, predictions3, predictions4, predictions5 = knn_bert(F.softmax(output, dim=1).to('cpu').detach().numpy().copy(), embed, label_dict, embeddings, knn_num, ratio)
                preds1.append(predictions1)
                preds2.append(predictions2)
                preds3.append(predictions3)
                preds4.append(predictions4)
                preds5.append(predictions5)
            labels.append(batch["label"].to('cpu').detach().numpy().copy())
            #print(classification_report(labels, preds, digits=4))
            #prediction.append(predictions.tolist())
            #num_correct += torch.sum(predictions == batch['label']).item()
            size += batch_size
            total_loss += loss.item() * batch_size
            if(knn_train):
                embed_list.append(embed)


            #result = compute_f1(predictions, batch['label'])
            dev_bar.set_postfix({
                'loss': round(total_loss / (batch_idx + 1), 3)
                })

        labels = np.concatenate(labels, axis=0)
        if(knn_train):
            embed_list = np.concatenate(embed_list, axis=0)
            distri = np.concatenate(distri, axis=0)
            print(embed_list.shape)
            for i in range(len(embed_list)):
                embeddings[i] = embed_list[i]

            for i in range(len(labels)):

                label_dict[i] = labels[i]
            return embed_list, distri, labels


        
        preds1 = np.concatenate(preds1, axis=0)
        preds2 = np.concatenate(preds2, axis=0)
        preds3 = np.concatenate(preds3, axis=0)
        preds4 = np.concatenate(preds4, axis=0)
        preds5 = np.concatenate(preds5, axis=0)

        result = compute_f1(preds1, labels)
        result1 = compute_f1(preds2, labels)
        result2 = compute_f1(preds3, labels)
        result3 = compute_f1(preds4, labels)
        result4 = compute_f1(preds5, labels)
        score = result["f1"]
        score1 = result1["f1"]
        score2 = result2["f1"]
        score3 = result3["f1"]
        score4 = result4["f1"]
        print("f1 score:", score)
        print("f1 score1:", score1)
        print("f1 score2:", score2)
        print("f1 score3:", score3)
        print("f1 score4:", score4)

        print(classification_report(labels, preds1, digits=4))
        print(classification_report(labels, preds2, digits=4))
        print(classification_report(labels, preds3, digits=4))
        print(classification_report(labels, preds4, digits=4))
        print(classification_report(labels, preds5, digits=4))
        #print(f'accuracy = {score:.3f}')
        #print(f'loss = {total_loss / (batch_idx + 1):.3f}')
        #predictions = predictions.tolist()
        #prediction.append(predictions)
        #prediction = sum(prediction, [])

    #return prediction
    if(knn_train):
        return score, embeddings, label_dict
    else:
        return score
