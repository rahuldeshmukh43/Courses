"""
ECE661: hw10 Part-2: ADABOOST
@author: Rahul Deshmukh
email: deshmuk5@purdue.edu
"""
#%% ------------------- Import libraries -------------------------------------#
import numpy as np
import sys
sys.path.append('../../')
import MyCVModule as MyCV
import pickle
import matplotlib.pyplot as plt
#%%---------------------------------------------------------------------------#
# define read path
readpath = '../files/cars/'
# define tasks to be carried out
generate_training_data = 0
generate_testing_data = 0

do_training=0
do_testing=0

#%%---------------- Generate Data---------------------------------------------#
#------------------------- Training Data--------------------------------------#
train_path= readpath+'train/'
#------------read images->Integrate image-> features -------------#
if generate_training_data:
    S,label,N_pos,N_neg = MyCV.Read_img4AdaBoost(train_path) 
    np.save(train_path+'train_data/'+'S.npy',S)
    np.save(train_path+'train_data/'+'label.npy',label)
    np.save(train_path+'train_data/'+'N_pos.npy',N_pos)
    np.save(train_path+'train_data/'+'N_neg.npy',N_neg)    
    train_features=[]
    for i in range(len(S)):
        ifeature = MyCV.get_feature(S[i])
        train_features.append(ifeature)
    np.save(train_path+'train_data/'+'f.npy',train_features)
    
    Num_classifier = len(train_features[1])
    np.save(train_path+'train_data/'+'Num_classifier.npy',Num_classifier)
else:
    S_train = np.load(train_path+'train_data/'+'S.npy')
    train_label = np.load(train_path+'train_data/'+'label.npy')
    train_features = np.load(train_path+'train_data/'+'f.npy')
    Num_classifier=np.load(train_path+'train_data/'+'Num_classifier.npy')

#------------------------ Testing Data----------------------------------------#
test_path =readpath+'test/'
#------------read images->Integrate image-> features -------------#
if generate_testing_data:
    S,label,N_pos,N_neg = MyCV.Read_img4AdaBoost(test_path) 
    np.save(test_path+'test_data/'+'S.npy',S)
    np.save(test_path+'test_data/'+'label.npy',label)
    np.save(test_path+'test_data/'+'N_pos.npy',N_pos)
    np.save(test_path+'test_data/'+'N_neg.npy',N_neg)    
    test_features=[]
    for i in range(len(S)):
        ifeature = MyCV.get_feature(S[i])
        test_features.append(ifeature)
    np.save(test_path+'test_data/'+'f.npy',test_features)
else:
    S_test = np.load(test_path+'test_data/'+'S.npy')
    test_label = np.load(test_path+'test_data/'+'label.npy')
    test_features = np.load(test_path+'test_data/'+'f.npy')

#%%--------------------------- Do Training -----------------------------------#

# set parameters for cascading and adaboost
Smax =10
#adaboost params
Tmax=100
TP_crit=1
FP_crit=0.5
if do_training:    
    Strong_classifiers,TP_s,FP_s = MyCV.do_cascades(train_features,train_label,
                                          Num_classifier,Smax,
                                          Tmax,TP_crit,FP_crit)
    fid = open(train_path+'train_data/'+'Strong_classifiers.pkl','wb')
    pickle.dump(Strong_classifiers,fid)
    fid.close()
    np.save(train_path+'train_data/'+'TP_s.npy',TP_s)
    np.save(train_path+'train_data/'+'FP_s.npy',FP_s)
else:
    fid= open(train_path+'train_data/'+'Strong_classifiers.pkl','rb')
    Strong_classifiers= pickle.load(fid)
    fid.close()
    TP_s=np.load(train_path+'train_data/'+'TP_s.npy')
    FP_s=np.load(train_path+'train_data/'+'FP_s.npy')
#    plt.plot(np.arange(1,len(FP_s)+1,dtype=int),FP_s,c='b',marker='*')
#    plt.xlabel('Stage #')
#    plt.ylabel('FP rate')    
#    plt.show()
print('Trained Strong Classifier with Final TP: '+str(np.prod(TP_s[:-1]))+
      ' FP: '+str(np.prod(FP_s[:-1])))
#%%-------------------------- Do Testing--------------------------------------#

if do_testing:
    print('---------------------Testing-------------------------------------')
    FP_measure,FN_measure=MyCV.Cascade_Testing(test_features,test_label,Strong_classifiers)
    np.save(test_path+'test_data/'+'FP_measure.npy',FP_measure)
    np.save(test_path+'test_data/'+'FN_measure.npy',FN_measure)
else:
    FP_measure = np.load(test_path+'test_data/'+'FP_measure.npy')
    FN_measure = np.load(test_path+'test_data/'+'FN_measure.npy')

# make plot for FP and FN
s=np.arange(1,len(Strong_classifiers)+1,dtype=np.float16)
plt.plot(s,FP_measure,color='r',marker='*',label='FP')
plt.plot(s,FN_measure,color='b',marker='*',label='FN')
plt.xlabel('Stage #')
plt.ylabel('FP or FN')
plt.title('Performance of Strong Classifier')
plt.ylim([0,1])
plt.xlim([1,s[-1]])
plt.legend()
plt.show()

