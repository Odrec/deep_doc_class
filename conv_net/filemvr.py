import os
import random

def prepare(limit=200):
    limit = limit - 1
    t_path = r"C:\Users\Mats Richter\Documents\GitHub\deep_doc_class\src\features\conv_net\Data\Train"
    v_path = r"C:\Users\Mats Richter\Documents\GitHub\deep_doc_class\src\features\conv_net\Data\Validation"
    pos,neg = get_files()
    v_neg = list()
    v_pos = list()

    p = 0
    n = 0
    #limit = 199
    while(True):
        if n > limit and p > limit:
            break
        if n <= limit:
            while True:
                f = random.choice(neg)
                if f in v_neg:
                    continue
                v_neg.append(f)
                break
            n += 1
        if p <=limit:
            while True:
                f = random.choice(pos)
                if f in v_pos:
                    continue
                v_pos.append(f)
                break
            p += 1
    for pos in v_pos:
        os.rename(t_path+'\\pos\\'+pos,v_path+'\\pos\\'+pos)
    for neg in v_neg:
        os.rename(t_path+'\\neg\\'+neg,v_path+'\\neg\\'+neg)
    return

def undo():
    v_pos,v_neg = get_val_files()

    t_path = r"C:\Users\Mats Richter\Documents\GitHub\deep_doc_class\src\features\conv_net\Data\Train"
    v_path = r"C:\Users\Mats Richter\Documents\GitHub\deep_doc_class\src\features\conv_net\Data\Validation"

    for pos in v_pos:
        os.rename(v_path+'\\pos\\'+pos,t_path+'\\pos\\'+pos)
    for neg in v_neg:
        os.rename(v_path+'\\neg\\'+neg,t_path+'\\neg\\'+neg)
    pass

def get_files():
    neg = os.listdir(r"C:\Users\Mats Richter\Documents\GitHub\deep_doc_class\src\features\conv_net\Data\Train\neg")
    pos = os.listdir(r"C:\Users\Mats Richter\Documents\GitHub\deep_doc_class\src\features\conv_net\Data\Train\pos")
    return pos, neg

def get_val_files():
    neg = os.listdir(r"C:\Users\Mats Richter\Documents\GitHub\deep_doc_class\src\features\conv_net\Data\Validation\neg")
    pos = os.listdir(r"C:\Users\Mats Richter\Documents\GitHub\deep_doc_class\src\features\conv_net\Data\Validation\pos")
    return pos, neg

prepare(200)
#undo()

