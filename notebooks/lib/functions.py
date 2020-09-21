#Author:Kavishwar Wagholikar
#waghsk@gmail.com

from sklearn.linear_model import LogisticRegression,LogisticRegressionCV, Ridge
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn import tree

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
import numpy as np
from sklearn.metrics import  roc_auc_score,accuracy_score
import os,sys,re,logging,datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_selection import chi2

'''
Return model trained using Classifier (clsN:classifierName)
on numeriical feature Matrix (X_train)
and binary label vector (y_train)
'''
def getMLModel(clsN,X_train,y_train):
   

    #X=pdf[selFeats].fillna(0).as_matrix()
    if clsN =='LogisticRegression':       
        #alg=LogisticRegression()
        cla=LogisticRegressionCV(penalty="l2",max_iter=10000,n_jobs=1,scoring='roc_auc',cv=3)
    elif clsN == 'RandomForestClassifier':
        cla= RandomForestClassifier(n_estimators=1000)
    elif clsN == 'SVM':
        cla= SVC(gamma='auto')
    elif clsN == 'dummy-stratified':
        cla= DummyClassifier()
    elif clsN == 'dummy-majority':
        cla= DummyClassifier(strategy='most_frequent')
    elif clsN == 'dummy-uniform':
        cla= DummyClassifier(strategy='uniform')
    elif clsN == 'NaiveBayes':
        cla= GaussianNB()
    elif clsN == 'MLP':
        numFeat=int(X_train.shape[1])
        print('numFeat:',numFeat)
        cla = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(10, 2), random_state=1)
    elif clsN == 'DecisionTree':
        cla = tree.DecisionTreeClassifier()
        
        

    model=cla.fit(X_train, y_train)

    #predict
    try:
        if clsN in ['SVM']:   
        
            calibrated = CalibratedClassifierCV(model, method='sigmoid', cv=2)
            calibrated.fit(X_train, y_train)
            #y_scores = calibrated.predict_proba(X_test)[:, 1]
            return calibrated

        else:
            return model
            #y_scores=model.predict_proba(X_test)[:,1]
    except Exception as e:
        raise e



'''
Return Feature Matrix created from numerical feature DataFrame (_numDf)
for subset of patients (idx: index set)
'''

def getFeatM(_numDf,idx):
    _df=_numDf.loc[idx]
    m=np.nan_to_num(_df.values).astype(np.float)
    return (m,_df.columns)

'''
Return index of patients annotated for particular disease/phenotype(label)
'''
def getLabelIdx(_labelDf,label):
    a=_labelDf
    return a[a[label].isin([0.0,1.0])].index.values

'''
Return label vector for selected patients(idx) annotated for particular disease/phenotype(label)
'''

def getLabelV(labelDf,label,idx):
    return labelDf[label][idx].values.astype(np.float)


'''
Return dictionary of score annotations for selected test patients (testIdx)
from their predicted Probabilities (predProb)
Gold Standard is computed from selecting column (label) from dataFrame (_labelDf)
'''

def getScore(testIdx,_labelDf,label,predProb):
    gold=_labelDf.loc[testIdx][label]
    pos,neg=0,0
    for g in gold:
        if g==1.0:
            pos=pos+1
        elif g==0.0:
            neg=neg+1
    posPerc=pos/(pos+neg)

    pred= np.where(predProb > 0.5, 1, 0)

    return {"label":label.replace('_GoldStandardLabel',''),"testSize":len(testIdx),
            "roc_auc":roc_auc_score(gold, predProb)}


'''
For given disease/phenpotype (label), split the patient population (_idx_all) into
test and nonTest sets in the given proportion (frac: fraction)
'''
def getIdx_Test_NonTest(_labelDf,_idx_all,label="label1",frac=0.3):
    idxLabel=getLabelIdx(_labelDf,label)
    testIdx=list(set(np.random.choice(idxLabel, int(len(idxLabel)*frac),replace=False)))
    nonTestIdx=list(set(_idx_all)-set(testIdx))
    debug("TestSet: selected ",len(testIdx),"(",round( len(testIdx)/len(idxLabel),2),") of ",len(idxLabel) ," annotated patients")
    return testIdx,nonTestIdx,label


'''
For given disease/phenpotype (label), 
get ids for the traininig set 
'''
def getIdx_TrainAnno(_labelDf,nonTestIdx=[],label="label1"):
    a=_labelDf.loc[nonTestIdx].copy()
    pos=list(a[a[label].isin([1.0])][label].index.values)
    neg=list(a[a[label].isin([0.0])][label].index.values)
        
    selIdx=list(set(pos).union(set(neg)))
    debug("TrainSet: selected", len(selIdx),"of", len(pos),"pos and ",len(neg),"neg")
    an={'npos':len(pos)}
    return (selIdx,an)

'''
For given disease/phenpotype (label), for given test,
predic the label using simple diagnostic counts
'''

def getPred_dxCount(testDf,trainDf, LabelV, dxCodeName):
    #print(trainDf)
    dxV=testDf[dxCodeName]
    predProb=np.array([ 0.0 if x==0 else 1.0 for x in dxV])
    return predProb


'''
return predicted problabilities and other annotations
for given claassifier (clsN)
'''
def getPred_ML(clsN,testDf,trainDf, trainLabelV):
    startTime=datetime.datetime.now()
    trainX=trainDf.values#as_matrix()
    trainY=trainLabelV
    testX=testDf.values#as_matrix()
    
    from sklearn.feature_selection import chi2, SelectKBest
    import numpy as np
    from sklearn.linear_model import LogisticRegressionCV, LogisticRegression

    #selcols=list(trainXcols)
    model=getMLModel(clsN,trainX,trainY)
    #z=zip(selcols,list(model.coef_[0]))
    #tz=sorted(tuple(z),key=lambda tup: -tup[1])
    predProb=model.predict_proba(testX)[:,1]
    a={}
    a["trainSize"]=len(trainY)
    #a["coef"]=tz
    #print(trainX.shape)
    a["numFeat"]=trainX.shape[1]-1
    a["start_dt"]=startTime
    a["runtime_ms"]=(datetime.datetime.now()-startTime).total_seconds()*1000
    a["cla"]=clsN
    return (predProb,a)

'''
create silver standard and return the correspnding feature set and labels
for given label.
The parameters are training set Size(trainN) and selection Factor (selFactor)
The disease-code name is required (dxCodeName)
'''

def getSilverPolar(trainDf=None, dxCodeName=None,trainN=1000,selFactor=1):
    dxDf=trainDf[[dxCodeName]].copy()
    dxDf["log"]=dxDf[dxCodeName].apply(lambda x:np.log(x+1)).copy()
    
    #potential silver positives
    potPos=dxDf[dxDf["log"]>0.0]
    
    mean=np.mean(potPos["log"])
    sd=np.std(potPos["log"])
    #high pole
    highCut=(mean+(selFactor*sd))
    
    
    pos=dxDf[dxDf["log"]>highCut]
    neg=dxDf[dxDf["log"]==0]
             
    posIdx=pos.index.values
    negIdxR=neg.index.values
    negIdx=np.random.choice(negIdxR,min(int(len(negIdxR)*len(pos)/len(potPos)),len(negIdxR)),replace=False)
    #reducing negs by the same fraction that pos are reduced by
             
    selIdxR=list(set(posIdx).union(set(negIdx)))
    selIdx=np.random.choice(selIdxR,min(trainN,len(selIdxR)),replace=False)
    debug('TrainSet: ','highCut:',highCut,' #tot:',len(dxDf),' #pos:',len(posIdx),' #neg:',len(negIdx),' #for_sel:',len(selIdxR),' #sel:',len(selIdx),' #trainN:',trainN) 
    silverLabelV=np.array([ 0.0 if x<highCut else 1.0 for x in list(dxDf.loc[selIdx]["log"])])
    silverTrainDf=trainDf.loc[selIdx].copy()
    an={'highCut':highCut,'posPot_mean':mean,'posPot_sd':sd,'npos':len(posIdx),'nposPot':len(potPos),'selFactor':selFactor}
    return (silverTrainDf,silverLabelV,an)



'''
Wrapper around getPred_ML, using Silver standard (based on PL) for training.
'''

def getPred_SilverPolar_ML(algN=None,testDf=None,trainDf=None, dxCodeName=None,trainN=1000,selFactor=1):
    #working on non test Rows
    startTime=datetime.datetime.now()
    (silverTrainDf,silverLabelV,an1)=getSilverPolar(trainDf, dxCodeName,trainN,selFactor)             
    (s,an2)=getPred_ML(algN,testDf,silverTrainDf, silverLabelV)
    #overwriting runtime annotation
    an2["start_dt"]=startTime
    an2["runtime_ms"]=(datetime.datetime.now()-startTime).total_seconds()*1000
    an={**an1, **an2}
    return (s,an)

      

#logging.basicConfig(filename='pheno3.log', level=logging.DEBUG)
logFormatter = logging.Formatter(" [%(levelname)-5.5s]  %(message)s")
rootLogger = logging.getLogger()

consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(logging.StreamHandler())
rootLogger.setLevel(logging.ERROR)
#rootLogger.setLevel(logging.DEBUG)

def debug(*msg):
    #logging.debug(' '.join([str(x) for x in msg]))
    print('...',' '.join([str(x) for x in msg]))

def error(*msg):
    logging.error(' '.join([str(x) for x in msg]))

    