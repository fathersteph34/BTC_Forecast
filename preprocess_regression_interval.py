import argparse
import pandas as pd
import numpy as np
import re
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from tqdm.auto import tqdm
from sklearn.feature_selection import SequentialFeatureSelector

def create_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--csv",help="btc data csv",type=str)

    parser.add_argument("--start",help="start date",type=str)

    parser.add_argument("--end",help="end date",type=str)

    parser.add_argument("--output",help="output path",type=str)

    return parser.parse_args()


if __name__ == "__main__":

    args = create_args()
    
    data = pd.read_csv(args.csv,sep=',')

    data.interpolate(axis=0,inplace=True)

    interval = (data['Date'] >= args.start) & (data['Date'] <= args.end)

    df = data.loc[interval]

    y = df.iloc[:,1:2]
    y = np.ravel(y)
    features_list=[]
    technical_indicators=['sma','ema','wma','trx','mom','std','var','rsi','roc']
    periods=['3','7','14','30','90']
    for i in tqdm( technical_indicators ):
        for j in tqdm( periods,leave=False):
            string=str(j)+str(i)
            X=df.filter(like=string,axis=1 )
            X=SimpleImputer(missing_values=0,strategy='most_frequent').fit_transform(X)
            X=pd.DataFrame(X)
            X.columns=df.filter(like=string,axis=1).columns
            rf1=RandomForestRegressor(random_state=7,n_jobs=-1)
            rfecv=SequentialFeatureSelector(rf1,scoring="r2",n_features_to_select=1)
            rfecv.fit(X,y)
            mask = rfecv.get_support()
            new_features = X.columns[mask]
            features_list.append(str(new_features))

    l1=[]
    for j in range(len(features_list)):
        result1 = re.search("'(.*)'],", features_list[j])
        if result1!=None:
            l1.append(result1.group(1))
    for i in range(len(features_list)):
        result2 = re.search('.*',features_list[i])
        if len(result2.group(0))<33:
            l1.append(result2.group(0))

    l1.sort()

    with open(args.output.split(".")[0] + ".txt","w") as f:
        f.write("\t".join(l1 + ["priceUSD"]) + "\n")

    df[l1 + ["priceUSD"]].to_csv(args.output)
    print("saved btc data to {}".format(args.output))


