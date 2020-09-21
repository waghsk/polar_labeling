#Author:Kavishwar Wagholikar
#waghsk@gmail.com

import os,sys,re,logging,datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Result:
    
    @staticmethod
    def save_result(name,resArrR,asPercent=True):
        dirName=stageDir+"result/"
        fname=dirName+name+".csv"
        import os
        import pathlib
        #pathlib.Path(dirName).mkdir(parents=True, exist_ok=True)
        if resArrR:
            #new_pdf=Result.resultPdf(resArrR,asPercent=asPercent);
            new_pdf=pd.DataFrame(resArrR)
            pdf=None

            try:
                old_pdf=pd.read_csv(fname)
                #print(old_pdf)
                #os.path.exists(fname):
                #print('file size:',str(os.path.getsize(fname)))            
                pdf=pd.concat([new_pdf,old_pdf])#.reset_index(drop=True)
            except Exception as e:
                print('DOES NOT EXIST:',str(fname))
                pdf=new_pdf
        else:
            pdf=pd.DataFrame()
        pdf.to_csv(fname,index=False)
        
        print("saved result for :"+name,' ', len(pdf),fname)
        
    @staticmethod
    def delete_result(name):
        print('Deleting result:',name)
        Result.save_result(name,None,asPercent=False)
    
    @staticmethod
    def read_result(name):
        _df=pd.read_csv(stageDir+"result/"+name+".csv")
        #_df.drop(['Unnamed: 0'], axis=1, inplace=True)
        return _df 