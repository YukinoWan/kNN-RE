import torch
import numpy as np
from numpy.linalg import norm

def knn_bert_single(bert_pred, embed, label_dict, embeddings, knn_num=64, ratio=0.0):
    """
    Combine knn and linear regression
    """
    #count the numver of labels
    label_num = len(bert_pred[0])
    embedding_list = np.array([x for x in embeddings.values()])
    combined_pred1 = np.zeros(bert_pred.shape)
    for i in range(len(embed)):
        similar_dict = {}
        for j in range(len(embeddings)):
            similar_dict[j] = np.dot(embed[i], embedding_list[j])/(norm(embed[i])*norm(embedding_list[j]))

        similar_dict = sorted(similar_dict.items(), key=lambda x: x[1], reverse=True)
        knn_list = [x[0] for x in similar_dict[:knn_num]]
        knn_labels = [label_dict[x] for x in knn_list]
    
        
        #count the number of labels in knn_labels

        label_dict_knn = {}
        for label in range(label_num):
            label_dict_knn[label] = 0
        for x in knn_labels:
            if x in label_dict_knn:
                label_dict_knn[x] += 1
        #compute the probability of each label
        label_prob = {}
        for x in label_dict_knn:
            label_prob[x] = label_dict_knn[x]/knn_num
        #combine bert_pred and label_prob
        for k in range(len(bert_pred[i])):
            combined_pred1[i][k] = ratio * label_prob[k] + (1 - ratio) * bert_pred[i][k]
        print(combined_pred1[i])
        if i ==50:
            break

    final_pred1 = np.argmax(combined_pred1, axis=1)



    return final_pred1
def knn_bert(bert_pred, embed, label_dict, embeddings, knn_num=64, ratio=0.0):
    """
    Combine knn and linear regression
    """
    #count the numver of labels
    label_num = len(bert_pred[0])
    embedding_list = np.array([x for x in embeddings.values()])
    combined_pred1 = np.zeros(bert_pred.shape)
    combined_pred2 = np.zeros(bert_pred.shape)
    combined_pred3 = np.zeros(bert_pred.shape)
    combined_pred4 = np.zeros(bert_pred.shape)
    combined_pred5 = np.zeros(bert_pred.shape)
    for i in range(len(embed)):
        similar_dict = {}
        for j in range(len(embeddings)):
            similar_dict[j] = np.dot(embed[i], embedding_list[j])/(norm(embed[i])*norm(embedding_list[j]))

        similar_dict = sorted(similar_dict.items(), key=lambda x: x[1], reverse=True)
        knn_list = [x[0] for x in similar_dict[:knn_num]]
        knn_labels = [label_dict[x] for x in knn_list]
    
        
        #count the number of labels in knn_labels

        label_dict_knn = {}
        for label in range(label_num):
            label_dict_knn[label] = 0
        for x in knn_labels:
            if x in label_dict_knn:
                label_dict_knn[x] += 1
        #compute the probability of each label
        label_prob = {}
        for x in label_dict_knn:
            label_prob[x] = label_dict_knn[x]/knn_num
        #combine bert_pred and label_prob
        for k in range(len(bert_pred[i])):
            combined_pred1[i][k] = 0 * ratio * label_prob[k] + (1 - 0 *ratio) * bert_pred[i][k]
            combined_pred2[i][k] = 2* ratio * label_prob[k] + (1 - 2 * ratio) * bert_pred[i][k]
            combined_pred3[i][k] = 4* ratio * label_prob[k] + (1 - 4 * ratio) * bert_pred[i][k]
            combined_pred4[i][k] = 6* ratio * label_prob[k] + (1 - 6 * ratio) * bert_pred[i][k]
            combined_pred5[i][k] = 8* ratio * label_prob[k] + (1 - 8 * ratio) * bert_pred[i][k]

    final_pred1 = np.argmax(combined_pred1, axis=1)
    final_pred2 = np.argmax(combined_pred2, axis=1)
    final_pred3 = np.argmax(combined_pred3, axis=1)
    final_pred4 = np.argmax(combined_pred4, axis=1) 
    final_pred5 = np.argmax(combined_pred5, axis=1)



    return final_pred1, final_pred2, final_pred3, final_pred4, final_pred5
