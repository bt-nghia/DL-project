import os
import sys
import pickle
import argparse
import operator
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity

# No consider about duplicates
def find_index(X, v):
    X = X.tolist()
    l = [X.index(i) for i in v]
    return set(l)


def confusion_matrix_plus(df_g, df_p1, df_p2):
    num_classes = df_g.shape[1]
    cm = np.zeros(num_classes)
    cm_correct = np.zeros(num_classes)
    c_mat = np.zeros((num_classes, num_classes))

    for i in df_g.index:
        
        idx = [
            int(df_g["label"].iloc[i]),  # 提取特定列的值
            int(df_p1["prediction"].iloc[i]),  # 假设 df_p1 和 df_p2 的结构与 df_g 类似
            int(df_p2["prediction"].iloc[i])
        ]
        # print(f"ground_truth:{idx[0]};seed1:{idx[1]};seed2:{idx[2]}")
        if idx[1]==idx[2]:
            cm[idx[1]] += 1
        if idx[0]==idx[1]==idx[2]:
            cm_correct[idx[0]] += 1
        c_mat[idx[1], idx[2]] += 1

    return cm, cm_correct, c_mat


def ER_pearson_correlation(df_p, df_g):
    acc = 0
    for i in df_p.index:

        acc += pearsonr(df_p.iloc[i], df_g.iloc[i])[0]
    return acc/(i+1)


def ER_cosine_similarity(df_p, df_g):
    score = cosine_similarity(df_p, df_g)
    acc = [score[i][i] for i in range(df_p.shape[0])]
    return np.mean(acc)

def ER(df_p, df_g, k=1):
    acc = 0
    for i in df_p.index:
        acc += indicator_kth(df_p.iloc[i], df_g.iloc[i], k)
    return acc/(i+1)/k*100

def ER_new(df_p, df_g, k=1):
    acc = 0
    for i in df_p.index:
        acc += indicator_kth_new(df_p.iloc[i], df_g.iloc[i], k)
    return acc/(i+1)*100


def indicator_kth(A, B, k=1):
    if k==1:
        return 1 if np.argmax(A)==np.argmax(B) else 0
    else:
        P = sorted(A, reverse=True)
        G = sorted(B, reverse=True)
        a, b = set(P[:k]), set(G[:k])
        l_a = find_index(A, a)
        l_b = find_index(B, b)
        return len(l_a.intersection(l_b))
    
def indicator_kth_new(A, B, k=1):
    if k==1:
        return 1 if np.argmax(A)==np.argmax(B) else 0
    else:
        P = sorted(A, reverse=True)
        G = sorted(B, reverse=True)
        a, b = set(P[:k]), set(G[:k])
        l_a = find_index(A, a)
        l_b = find_index(B, b)
        return 1 if len(l_a.intersection(l_b)) else 0


def convert_labels_to_numeric(df, column):
    """自动将 label / prediction 列转换为 0, 1, 2"""
    if pd.api.types.is_numeric_dtype(df[column]):
        return df  # 已经是数值，不需要转换

    # 处理布尔值（True -> 1, False -> 0）
    if df[column].dtype == bool or set(df[column].unique()) == {True, False}:
        df[column] = df[column].astype(int)
        return df

    # 处理字符串类别（自动映射到 0, 1, 2）
    unique_labels = sorted(df[column].unique())
    label_map = {label: idx for idx, label in enumerate(unique_labels)}  # 动态创建映射
    df[column] = df[column].map(label_map).astype(int)
    
    return df

def main():
    parser = argparse.ArgumentParser()

    parser = argparse.ArgumentParser()

    parser.add_argument("-p1", "--predfile1",
                        help="prediction file1", required=True)
    parser.add_argument("-p2", "--predfile2",
                        help="prediction file2", required=True)
    parser.add_argument("-g", "--groundtruth",
                        help="ground truth file", required=True)
    parser.add_argument("-o", "--outputfile", help="output file")

    args = parser.parse_args()

    predfile1 = args.predfile1
    predfile2 = args.predfile2
    groundfile = args.groundtruth
    outputfile = args.outputfile


    # load ground truth file
    df_g = pd.read_csv(groundfile, header=0, sep=",")
    df_p1 = pd.read_csv(predfile1, header=0, sep="\t")
    df_p2 = pd.read_csv(predfile2, header=0, sep="\t")

    print("len(DATA):", len(df_g))
    print("len(pred):", len(df_p1))
    print("len(pred2):", len(df_p2))
    # df_g = convert_labels_to_numeric(df_g, "label")
    df_p1 = convert_labels_to_numeric(df_p1, "prediction")
    df_p2 = convert_labels_to_numeric(df_p2, "prediction")
    # import pdb;pdb.set_trace()

    # compute confusion matrix
    cm, cm_correct, c_mat = confusion_matrix_plus(df_g, df_p1, df_p2)
    er_ea = np.sum(cm_correct)/np.sum(cm)*100
    ea_er = np.sum(cm_correct)/df_g.shape[0]*100
    er = np.sum(cm)/df_g.shape[0]*100
    pearson = ER_pearson_correlation(df_p1, df_p2)
    # pearson = ER_pearson_correlation(df_p1, df_g)

    cosine = ER_cosine_similarity(df_p1, df_p2)

    if 'voting' in predfile1:
        cr, crl = -1, -1
    else:
        cr = ER(df_p1, df_p2, k=2)
        crl = ER_new(df_p1, df_p2, k=2)

    print('{} \t {} \t {} \t {} \t {} \t {} \t{}'.format(er,crl, er_ea, pearson, cosine, ea_er, cr))
    if outputfile:
        f = open(outputfile, 'w')
        res = f"file1:{predfile1}\t file2:{predfile2} \t er: \t {er} \t crl:{crl} \t er_ea \t {er_ea} \t pearson:{pearson}\t cosine:{cosine}\t ea_er: \t {ea_er}\t cr:{cr}"
        # f.write('{},{},{},{},{},{},{},{},{}\n'.format(predfile1, predfile2, er, crl, er_ea, pearson, cosine, ea_er, cr))
        f.write(res)
        f.close()



if __name__ == "__main__":
    main()