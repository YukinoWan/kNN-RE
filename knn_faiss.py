import numpy as np
import sys
import faiss
from testeval import compute_f1
from scipy.special import softmax
from scipy.special import log_softmax
from sklearn.metrics import classification_report
import pandas as pd


def knn_soft_train(datastore, stored_label, stored_distribution, test_rep, test_distribution, k=4, ratio=0.2):
    ngpus = faiss.get_num_gpus()
    n_labels = len(stored_distribution[0])
    
    
    d = len(datastore[0])
    #print(stored_label.shape)
    
    cpu_index = faiss.IndexFlatL2(d)

    #gpu_index = faiss.index_cpu_to_all_gpus(
    #        cpu_index
    #        )


    cpu_index.add(datastore)

    #print(gpu_index.ntotal)

    D, I = cpu_index.search(test_rep, k)
    neg_D = np.negative(D)

    #print(neg_D)
    norm_D = log_softmax(neg_D, axis=1)
    
    preds = []
    for i in range(len(I)):
        tmp_preds = np.zeros(n_labels)
        #print("--------")
        #print(I[i])
        #np.set_printoptions(precision=5)
        #print(D[i])
        #print(norm_D[i])
        for j in range(len(norm_D[i])):
            #print(stored_label[I[i][j]])
            tmp_preds[int(stored_label[I[i][j]])] += norm_D[i][j]
        
        #print(tmp_preds)
        #print(test_distribution[i])
        #assert False
        combined_preds = (1 - ratio) * test_distribution[i] + ratio * tmp_preds
        #print(combined_preds)
        #assert False
        preds.append(combined_preds)
    
    preds = np.array(preds)
    #preds = np.argmax(preds, axis=1)
    return preds

def knn_faiss_rbf(datastore, stored_label, stored_distribution, test_rep, test_label, test_distribution, k, ratio,temperature):
    ngpus = faiss.get_num_gpus()
    n_labels = len(test_distribution[0])
    
    
    d = len(datastore[0])
    #print(stored_label.shape)
    
    cpu_index = faiss.IndexFlatL2(d)

    gpu_index = faiss.index_cpu_to_all_gpus(
            cpu_index
            )

    #faiss.normalize_L2(datastore)
    #faiss.normalize_L2(test_rep)
    gpu_index.add(datastore)


    #print(gpu_index.ntotal)

    D, I = gpu_index.search(test_rep, k)
    ori_D = D

    T = np.transpose(D)
    T = T / (2.0 * np.var(T, axis=0))
    #temperature = 10.0
    D = np.transpose(T)
    D = D / temperature
    neg_D = np.negative(D)

    #print(neg_D)
    norm_D = softmax(neg_D, axis=1)
    
    preds = []
    for i in range(len(I)):
        
        #if i == 1000:
        #    assert False
            #assert False
        tmp_preds = np.zeros(n_labels)
        #print("--------")
        #print(I[i])
        #np.set_printoptions(precision=5)
        #print(D[i])
        #print(norm_D[i])
        for j in range(len(norm_D[i])):
            #print(stored_label[I[i][j]])
            tmp_preds[int(stored_label[I[i][j]])] += norm_D[i][j]
        
        #print(tmp_preds)
        #print(test_distribution[i])
        #assert False
        combined_preds = (1 - ratio) * test_distribution[i] + ratio * tmp_preds
        #assert False
        preds.append(combined_preds)
        if False:
            if i == 10000:
                assert False
            if int(np.argmax(test_distribution[i])) != 0 and np.argmax(test_distribution[i]) != np.argmax(tmp_preds) and np.argmax(np.array(combined_preds)) == test_label[i]:
                print("test_id:",i)
                print("test_distribution:",test_distribution[i])
                print("knn_ids:",I[i])
                print("knn_distance:",ori_D[i])
                print("knn_variance: ", np.var(ori_D[i]))
                print("devar_distance: ", D[i])
                print("normed_knn_distance:",norm_D[i])
                print("tmp_preds:",tmp_preds)
                print("combined_preds:",combined_preds)
                print([stored_label[x] for x in I[i]])
                print("-----------------")

        #print("processing: %f" %(i/len(I)), end='\r')
    
    preds = np.array(preds)
    #print(preds[:50])
    preds = np.argmax(preds, axis=1)
    return preds


def knn_faiss(datastore, stored_label, stored_distribution, test_rep, test_distribution, k, ratio,temperature):
    ngpus = faiss.get_num_gpus()
    n_labels = len(test_distribution[0])
    
    
    d = len(datastore[0])
    #print(stored_label.shape)
    
    cpu_index = faiss.IndexFlatL2(d)

    gpu_index = faiss.index_cpu_to_all_gpus(
            cpu_index
            )

    #faiss.normalize_L2(datastore)
    #faiss.normalize_L2(test_rep)
    gpu_index.add(datastore)


    print(gpu_index.ntotal)

    D, I = gpu_index.search(test_rep, k)
    #temperature = 10.0
    D = D / temperature
    neg_D = np.negative(D)

    #print(neg_D)
    norm_D = softmax(neg_D, axis=1)
    
    preds = []
    for i in range(len(I)):
        
        #if i == 1000:
        #    assert False
            #assert False
        tmp_preds = np.zeros(n_labels)
        #print("--------")
        #print(I[i])
        #np.set_printoptions(precision=5)
        #print(D[i])
        #print(norm_D[i])
        for j in range(len(norm_D[i])):
            #print(stored_label[I[i][j]])
            tmp_preds[int(stored_label[I[i][j]])] += norm_D[i][j]
        
        #print(tmp_preds)
        #print(test_distribution[i])
        #assert False
        combined_preds = (1 - ratio) * test_distribution[i] + ratio * tmp_preds
        #assert False
        preds.append(combined_preds)
        if False:
            if i == 1000:
                assert False
            if int(np.argmax(test_distribution[i])) != 0 and np.argmax(test_distribution[i]) != np.argmax(tmp_preds) and np.argmax(combined_preds) == test_label[i]:
                print("test_id:",i)
                print("test_distribution:",test_distribution[i])
                print("knn_ids:",I[i])
                print("knn_distance:",D[i])
                print("normed_knn_distance:",norm_D[i])
                print("tmp_preds:",tmp_preds)
                print("combined_preds:",combined_preds)
                print([stored_label[x] for x in I[i]])
                print("-----------------")

        #print("processing: %f" %(i/len(I)), end='\r')
    
    preds = np.array(preds)
    #print(preds[:50])
    preds = np.argmax(preds, axis=1)
    return preds


if __name__ == "__main__":
    name = sys.argv[1]
    #mode =sys.argv[2]
    retrieve_type = sys.argv[2]
    test_type = sys.argv[3]
    train_datastore = np.loadtxt("datastore/{}/{}_{}_datastore/datastore.csv".format(name,name,retrieve_type),dtype='float32',delimiter=",")

    train_label = np.loadtxt("datastore/{}/{}_{}_datastore/label.csv".format(name,name,retrieve_type),dtype='float32',delimiter=",")
    train_logits = np.loadtxt("datastore/{}/{}_{}_datastore/distribution.csv".format(name,name,retrieve_type),dtype='float32',delimiter=",")
    #train_logits = np.loadtxt("ace_train_datastore/knn_logits.csv",dtype='float32',delimiter=",")

    test_datastore = np.loadtxt("datastore/{}/{}_{}_datastore/datastore.csv".format(name,name,test_type),dtype='float32',delimiter=",")
    test_label = np.loadtxt("datastore/{}/{}_{}_datastore/label.csv".format(name,name,test_type),dtype='float32',delimiter=",")
    test_distribution = softmax(np.loadtxt("datastore/{}/{}_{}_datastore/distribution.csv".format(name,name,test_type),dtype='float32',delimiter=","), axis=1)

    bert_pred = test_distribution
    embed = test_datastore
    label_dict = {k:train_label[k] for k in range(len(train_label))}
    embeddings = {k:train_datastore[k] for k in range(len(train_datastore))}

    #preds_cpu = knn_bert_single(bert_pred, embed, label_dict, embeddings, 16, 1.0)
    #result = compute_f1(preds_cpu, test_label, None)

    best_f1 = 0.0
    #results = []
    k_line = []
    ks = [2,4,8,16,32,64,128,256]
    #ks = [4]
    #temperatures = [0.06]
    temperatures = [0.01,0.02,0.03,0.04, 0.05, 0.06, 0.07, 0.08,0.09,0.1,0.2,0.5,1.0]
    ratios = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    #ratios = [1.0]
    #temperatures = [1.0,5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0]
    temperature_line = []
    ratio_line = []
    f1_line = []
    for k in ks:
        for temperature in temperatures:
            for ratio in ratios:
                #temperature = 20.0
                preds = knn_faiss_rbf(train_datastore, train_label, train_logits, test_datastore, test_label, test_distribution, k, ratio,temperature)
    #print(preds[:5])
                result = compute_f1(preds, test_label)
                #results.append(result)
                #print(classification_report(test_label, preds, digits=4))
                #print("k = %d" %k)
                #print("ratio = %f" %ratio)
                #print("temperature = %f" %temperature)
                #print(result)
                if result["f1"] > best_f1:
                    best_f1 = result["f1"]
                    print("best f1: %f" %best_f1)
                    print("-----------------")
                    print("k = %d" %k)
                    print("ratio = %f" %ratio)
                    print("temperature = %f" %temperature)
                    print(result)

                k_line.append(k)
                temperature_line.append(temperature)
                ratio_line.append(ratio)
                f1_line.append(result["f1"])
            #assert False
    #print(results)
    col1 = "k"
    col2 = "temperature"
    col3 = "ratio"
    col4 = "f1"
    data = pd.DataFrame({col1:k_line, col2:temperature_line, col3:ratio_line, col4:f1_line})
    data.to_excel("knn_result/{}_data.xlsx".format(name), sheet_name="{}_{}".format(name,retrieve_type,test_type), index=True)




