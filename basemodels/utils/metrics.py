import sys

import numpy as np

def RSE(pred, true):
    return np.sqrt(np.sum((true-pred)**2)) / np.sqrt(np.sum((true-true.mean())**2))

def CORR(pred, true):
    u = ((true-true.mean(0))*(pred-pred.mean(0))).sum(0) 
    d = np.sqrt(((true-true.mean(0))**2*(pred-pred.mean(0))**2).sum(0))
    return (u/d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred-true))

def MSE(pred, true):
    return np.mean((pred-true)**2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))

def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))

def SMAPE(pred,true):
    smape = np.mean(np.abs((pred - true)) / ((true+pred) * 1/2))
    return smape

def R2_score(pred,true):
    r2 = 1 - np.sum((true - pred) ** 2) / np.sum((true - np.mean(true)) ** 2)
    return r2

def Adjusted_R_Square(pred,true,r2):
    ad_r2 = 1-(1-r2)* (len(true)-1-1)/(len(true)-1)
    return ad_r2

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)

    mspe = MSPE(pred, true)

    r2  = R2_score(pred,true)

    ad_r2 = Adjusted_R_Square(pred,true,r2)
    smape = SMAPE(pred,true)

    return mae,rmse,smape,r2,ad_r2

def var_true(df,index):
    df_ex = df.iloc[:, index:]
    df['var_true'] = 0.00
    df['std_true'] = 0.00
    for i in range(len(df_ex)):
        sum_diff = []
        for c in df_ex.columns.tolist():
            sum_diff.append((float(df['true'][i])-float(df[c][i]))**2)
        df['var_true'][i] = sum(sum_diff)/float(len(sum_diff))
        df['std_true'][i] = (sum(sum_diff)/float(len(sum_diff)))**0.5
    return df


def var_true_multivariate(df,tmp,index,args,j):
    df["std_true_{}".format(args.columns[j + 1])] = 0.00
    df["var_true_{}".format(args.columns[j + 1])] = 0.00
    for k in range(len(tmp[j])):
        sum_diff = []
        for c in tmp[j].iloc[:, index:].columns.tolist():
            sum_diff.append((float(tmp[j].iloc[:, 0][k]) - float(tmp[j][c][k])) ** 2)
        df["var_true_{}".format(args.columns[j + 1])][k] = sum(sum_diff) / float(len(sum_diff))
        df["std_true_{}".format(args.columns[j + 1])][k] = (sum(sum_diff) / float(len(sum_diff))) ** 0.5

    df["std_true_{}".format(args.columns[j + 1])] = round(df["std_true_{}".format(args.columns[j + 1])], 1)
    df["var_true_{}".format(args.columns[j + 1])] = round(df["var_true_{}".format(args.columns[j + 1])], 1)
    return df

def calculate_var(df,args):
    if args.features != 'M':
        index = 0
        if 'true' in df.columns.tolist():
            index = 3
        else:
            index = 2
        std_self = df.iloc[:,index:].std(axis=1)
        var_self = df.iloc[:,index:].var(axis=1,ddof=1)
    
        if index == 3:
            df = var_true(df, index)

        df["std_true"] = round(df["std_true"], 1)
        df["var_true"] = round(df["var_true"], 1)
        df['std_self'] = std_self
        df['var_self'] = var_self
        df["std_self"] = round(df["std_self"], 1)
        df["var_self"] = round(df["var_self"], 1)
    if args.features == 'M':
        tmp = []
        for i in range(args.c_out):
            tmp.append(df[[s for s in df.columns if args.columns[i + 1] in s and "pred" not in s]])
        for j in range(len(tmp)):
            index = 0
            if 'true' in tmp[j].columns.tolist()[0]:
                index = 1
            else:
                index = 0
            std_self = tmp[j].iloc[:, index:].std(axis=1)
            var_self = tmp[j].iloc[:, index:].var(axis=1, ddof=1)
            df['std_self_{}'.format(args.columns[j + 1])] = std_self
            df['var_self_{}'.format(args.columns[j + 1])] = var_self
            df["std_self_{}".format(args.columns[j + 1])] = round(df["std_self_{}".format(args.columns[j + 1])], 1)
            df["var_self_{}".format(args.columns[j + 1])] = round(df["var_self_{}".format(args.columns[j + 1])], 1)
        
            if index == 1:
            
                df["std_true_{}".format(args.columns[j + 1])] = 0.00
                df["var_true_{}".format(args.columns[j + 1])] = 0.00
            
                for k in range(len(tmp[j])):
            
                    sum_diff = []
                    for c in tmp[j].iloc[:, index:].columns.tolist():
                        sum_diff.append((float(tmp[j].iloc[:,0][k]) - float(tmp[j][c][k])) ** 2)
                    df["var_true_{}".format(args.columns[j + 1])][k] = sum(sum_diff) / float(len(sum_diff))
                    df["std_true_{}".format(args.columns[j + 1])][k] = (sum(sum_diff) / float(len(sum_diff))) ** 0.5
        
            df["std_true_{}".format(args.columns[j + 1])] = round(df["std_true_{}".format(args.columns[j + 1])], 1)
            df["var_true_{}".format(args.columns[j + 1])] = round(df["var_true_{}".format(args.columns[j + 1])], 1)
    return df