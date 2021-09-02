
import numpy as np
import pandas as pd
import math
import re
import statistics
from pprint import pprint

########################  RUN   ###############################################
""""
# Run the code by assigning 1 to one of the parameters in the following lines

"""
# decision tree using Entropy 
DTUE = 0
# decision tree using Gini 
DTUG = 0
#Cross Validation
Cross_Validation = 1
# decision tree depth limited
DTDL = 0

########################  read the file and extract the data   ################
def data_2_df(dat):
    with open(dat, 'r') as f1:
        data = f1.read()

    dataL = re.split('\n',data) #seperate rows of data 
    data1 = [re.split('([+-]?\d*\.?\d+(?:[eE][-+]?\d+)?)',dataL[i]) for i in range(len(dataL))] #sepatate the columns

    K = len(data1)
    #remove the empty spaces and :
    for i in range(K):
        while ('' in data1[i]):
            data1[i].remove('') 
        while (' ' in data1[i]):
            data1[i].remove(' ') 
        while (':' in data1[i]):
            data1[i].remove(':') 
    
    for i in range(K):
        for j in range(len(data1[i])):
            data1[i][j] = int(data1[i][j])
    a = 124
    Data = [[0 for x in range(a)] for y in range(K-1)] 
    for i in range(K-1):
        Data[i][0] = data1[i][0]
        for j in range(1,len(data1[i])-1,2):
            Data[i][data1[i][j]] = data1[i][j+1]
    Data_df = pd.DataFrame(Data)
    
    return Data_df

########################  Entropy and Information Gain defined   ##############
def Entropy(y):
    vals,counts = np.unique(y,return_counts = True)
    Entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(vals))])
    return Entropy

def GI(S,A,target=0):
    Entropy_s = Entropy(S[target])
    values,counts = np.unique(S[A],return_counts = True)
    Entropy_sv = np.sum([(counts[i]/np.sum(counts))*Entropy(S.where(S[A]==values[i]).dropna()[target]) for i in range(len(values))])
    GI = Entropy_s - Entropy_sv
    return GI
########################  Gini and Information Gain defined   #################
def Gini(y):
    vals,counts = np.unique(y,return_counts = True)
    Gini = np.sum([(counts[i]/np.sum(counts))*(1-(counts[i]/np.sum(counts))) for i in range(len(vals))])
    return Gini

def GIG(S,A,target=0):
    Gini_s = Gini(S[target])
    values,counts = np.unique(S[A],return_counts = True)
    Gini_sv = np.sum([(counts[i]/np.sum(counts))*Gini(S.where(S[A]==values[i]).dropna()[target]) for i in range(len(values))])
    GIG = Gini_s - Gini_sv
    return GIG
########################  ID3   ###############################################

def ID3(S,O_S, Att, target_att = 0, parent_node_class = None):
    if len(np.unique(S[target_att])) <= 1:
        return np.unique(S[target_att])[0]
    elif len(S)==0:
        return np.unique(O_S[target_att])[np.argmax(np.unique(O_S[target_att],return_counts=True)[1])]
    elif len(Att) == 0:
        return parent_node_class

    else:

        parent_node_class = np.unique(S[target_att])[np.argmax(np.unique(S[target_att],return_counts=True)[1])]
        GainInfo = [GI(S, A ,target_att) for A in Att]
        best_idx = np.argmax(GainInfo)
        best_A = Att[best_idx]
    
        tree = {best_A:{}}
        Att = [i for i in Att if i != best_A]

        for value in np.unique(S[best_A]):
            value = value
            sub_data = S.where(S[best_A] == value).dropna()
            subtree = ID3(sub_data,O_S,Att, target_att, parent_node_class)

            tree[best_A][value] = subtree
        

    return (tree)
########################  ID3 using Gini   #################

def ID3b(S,O_S, Att, target_att = 0, parent_node_class = None):
    if len(np.unique(S[target_att])) <= 1:
        return np.unique(S[target_att])[0]
    elif len(S)==0:
        return np.unique(O_S[target_att])[np.argmax(np.unique(O_S[target_att],return_counts=True)[1])]
    elif len(Att) == 0:
        return parent_node_class

    else:

        parent_node_class = np.unique(S[target_att])[np.argmax(np.unique(S[target_att],return_counts=True)[1])]
        GainInfo = [GIG(S, A ,target_att) for A in Att]
        best_idx = np.argmax(GainInfo)
        best_A = Att[best_idx]
    
        treeg = {best_A:{}}
        Att = [i for i in Att if i != best_A]

        for value in np.unique(S[best_A]):
            value = value
            sub_data = S.where(S[best_A] == value).dropna()
            subtree = ID3b(sub_data,O_S,Att, target_att, parent_node_class)

            treeg[best_A][value] = subtree
        

    return (treeg)
########################  ID3 with depth limited   ############################
def ID3_D(S,O_S, Att, target_att = 0, parent_node_class = None, depth = 0):
    if len(np.unique(S[target_att])) <= 1:
        return np.unique(S[target_att])[0]
    elif len(S)==0:
        return np.unique(O_S[target_att])[np.argmax(np.unique(O_S[target_att],return_counts=True)[1])]
    elif len(Att) == 0:
        return parent_node_class
    elif depth >= Max_depth:
        return parent_node_class

    else:
        
        parent_node_class = np.unique(S[target_att])[np.argmax(np.unique(S[target_att],return_counts=True)[1])]
        GainInfo = [GI(S, A ,target_att) for A in Att]
        best_idx = np.argmax(GainInfo)
        best_A = Att[best_idx]
    
        tree = {best_A:{}}
        Att = [i for i in Att if i != best_A]

        for value in np.unique(S[best_A]):
            value = value
            sub_data = S.where(S[best_A] == value).dropna()
            subtree = ID3_D(sub_data,O_S,Att, target_att, parent_node_class, depth+1)
            
            tree[best_A][value] = subtree
           
    return (tree) 
########################  test   ############################################## 
def predict(query,tree,default = 1):
    for key in list(query.keys()):
        if key in list(tree.keys()):
            try:
                result = tree[key][query[key]] 
            except:
                return default
            result = tree[key][query[key]]
            if isinstance(result,dict):
                return predict(query,result)

            else:
                return result

def test(data,tree):

    queries = data.iloc[:,1:].to_dict(orient = "records")

    predicted = pd.DataFrame(columns=["predicted"]) 

    for i in range(len(data)):
        predicted.loc[i,"predicted"] = predict(queries[i],tree,1.0) 
        #if i == 250: print('i')
    Acc = (np.sum(predicted["predicted"] == data[0])/len(data))
    return Acc

########################  depth of tree   #####################################
def dict_depth(myDict): 
  
    Ddepth = 1
    obj = [(k, Ddepth + 1) for k in myDict.values() 
                          if isinstance(k, dict)] 
    max_depth = 0
      
    while(obj): 
        n, Ddepth = obj.pop() 
        max_depth = max(max_depth, Ddepth) 
          
        obj = obj + [(k, Ddepth + 1) for k in n.values() 
                                 if isinstance(k, dict)] 
          
    return max_depth      

########################  Inputs   ############################################
Train = data_2_df('a1a.train')
Test = data_2_df('a1a.test')
"""
#Cross Validation
"""               

if Cross_Validation == 1:
    Fold1 = data_2_df('fold1')
    Fold2 = data_2_df('fold2')
    Fold3 = data_2_df('fold3')
    Fold4 = data_2_df('fold4')
    Fold5 = data_2_df('fold5')    
    F1 = pd.concat([Fold2,Fold3,Fold4,Fold5],axis = 0)
    F2 = pd.concat([Fold1,Fold3,Fold4,Fold5],axis = 0)
    F3 = pd.concat([Fold1,Fold2,Fold4,Fold5],axis = 0)
    F4 = pd.concat([Fold1,Fold2,Fold3,Fold5],axis = 0)    
    F5 = pd.concat([Fold1,Fold2,Fold3,Fold4],axis = 0) 
    F = [F1,F2,F3,F4,F5]
    ff = [Fold1,Fold2,Fold3,Fold4,Fold5]
    m_depth = [1,2,3,4,5]
    Accuracy = [[0 for i in range(len(m_depth))] for j in range(len(F))]    
    Mean = [0 for i in range(len(m_depth))]   
    std = [0 for i in range(len(m_depth))]
    for i in range(len(F)):
        training_data = F[i]
        testing_data = ff[i]
        for j in range(len(m_depth)):
            Max_depth = m_depth[j]
            tree = ID3_D(training_data,training_data,training_data.columns[1:])
            Accuracy[i][j] = test(testing_data,tree)
                
    Accuracy = pd.DataFrame(Accuracy)                
    for i in range(len(m_depth)):
        Mean[i] = statistics.mean(Accuracy[:][i])  
        std[i] = statistics.stdev(Accuracy[:][i])
        
        #print('Average prediction accuracy and the standard deviation of depth ', m_depth[i],': ',Mean[i]*100,'%',',', std[i]*100,'%')
    Max_depth = m_depth[np.argmax(Mean)]

"""
# decision tree using Entropy
"""
if DTUE ==1:
    training_data = Train
    testing_data = Test
    tree = ID3(training_data,training_data,training_data.columns[1:])
    pprint(tree)
    Training_accuracy = test(training_data,tree)
    Testing_accuracy = test(testing_data,tree)
    Att_1 = Train.columns[1:]
    entropy = Entropy(training_data[0])
    GainInfo_root = [GI(Train, A ,0) for A in Att_1]

"""
# decision tree using Gini
"""
if DTUG ==1:
    training_data = Train
    testing_data = Test
    tree = ID3b(training_data,training_data,training_data.columns[1:])
    #pprint(tree)
    Training_accuracy = test(training_data,tree)
    Testing_accuracy = test(testing_data,tree)
    Att_1 = Train.columns[1:]
    GainInfo_root = [GIG(Train, A ,0) for A in Att_1]
"""
# decision tree depth limited
"""
if DTDL ==1:
    training_data = Train
    testing_data = Test
    Max_depth = 4
    tree = ID3_D(training_data,training_data,training_data.columns[1:])
    #pprint(tree)
    Training_accuracy = test(training_data,tree)
    Testing_accuracy = test(testing_data,tree)
    Att_1 = Train.columns[1:]
    entropy = Entropy(training_data[0])
    GainInfo_root = [GI(Train, A ,0) for A in Att_1]

########################  Print Output   ######################################  
print('(a) Most common label in the training data is: ',np.unique(training_data[0])[np.argmax(np.unique(training_data[0],return_counts=True)[1])])

if DTUG ==1:
    print('(b) Not applicable ')
elif Cross_Validation ==1:
    print('(b) Not applicable ')
else:
    print('(b) Entropy of the training data is: ', entropy)
print('(c) Best feature and information gain: ', list(tree.keys()),',',max(GainInfo_root))
print('(d) Accuracy on the training set is: ',Training_accuracy*100,'%')
print('(e) Accuracy on the testing set is: ',Testing_accuracy*100,'%')
if Cross_Validation == 1:
    for i in range(5):
        print('(f) Average accuracy and the standard deviation of depth ', m_depth[i],': ',Mean[i]*100,'%',',', std[i]*100,'%')
    print('(g) Best depth is: ',Max_depth//2)
if DTDL ==1:
    print('(g) Best depth is: ',Max_depth//2)
    print('(h) Accuracy on the test set using the best depth is: ',Testing_accuracy*100,'%')
if DTUE ==1:
    print('The training error is: ',(1-Training_accuracy)*100,'%')
    print('The testing error is: ',(1-Testing_accuracy)*100,'%')
    print('The maximum depth of the decision tree is: ',dict_depth(tree)//2)
if DTUG ==1:
    print('The training error is: ',(1-Training_accuracy)*100,'%')
    print('The testing error is: ',(1-Testing_accuracy)*100,'%')
    print('The maximum depth of the decision tree is: ',dict_depth(tree)//2)
