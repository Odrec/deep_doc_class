import os
import shutil
import random
import csv
import fnmatch
def setup(csv_file,v_path = ".\\Data\\Validation"):
    t_path = ".\\Data\\Train"
    #pos,neg = get_files()
    v_neg = list()
    v_pos = list()

    p = 0
    n = 0
    #limit = 1
    with open(csv_file,'r') as fp:
        rdr = csv.reader(fp,delimiter=',')
        for row in rdr:
            if row[1] == '1':
                v_pos.append(row[0])
            elif row[1] == '0':
                v_neg.append(row[0])
    for pos in v_pos:
        for file in os.listdir(t_path+'\\pos\\'):
            if fnmatch.fnmatch(file,pos+'*'):
                type(file)
                shutil.move(t_path+'\\pos\\'+file,v_path+'\\pos\\'+file)
    for neg in v_neg:
        for file in os.listdir(t_path+'\\neg\\'):
            if fnmatch.fnmatch(file,neg+'*'):
                shutil.move(t_path+'\\neg\\'+file,v_path+'\\neg\\'+file)
    return

def undo_setup(csv_file,v_path = ".\\Data\\Validation"):
    t_path = ".\\Data\\Train"
    #pos,neg = get_files()
    v_neg = list()
    v_pos = list()

    p = 0
    n = 0
    #limit = 1
    with open(csv_file,'r') as fp:
        rdr = csv.reader(fp,delimiter=',')
        for row in rdr:
            if row[1] == '1':
                v_pos.append(row[0])
            elif row[1] == '0':
                v_neg.append(row[0])
    for pos in v_pos:
        for file in os.listdir(v_path+'\\pos\\'):
            if fnmatch.fnmatch(file,pos+'*'):
                type(file)
                shutil.move(v_path+'\\pos\\'+file,t_path+'\\pos\\'+file)
    for neg in v_neg:
        for file in os.listdir(v_path+'\\neg\\'):
            if fnmatch.fnmatch(file,neg+'*'):
                shutil.move(v_path+'\\neg\\'+file,t_path+'\\neg\\'+file)
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

def get_all_IDs():
    with open("cleaned_manual_class.csv",'r') as fp:
        reader = csv.reader(fp,delimiter=',')
        content = list()
        for row in reader:
            content.append(row)
        return content

def seperate(num=500):
    with open('evaluation_files.csv','w+') as fp:
        writer = csv.writer(fp,delimiter=',',lineterminator='\n')
        content = get_all_IDs()
        c = 0
        new_l = list()
        print(len(content))
        while c < num:
            new = random.choice(content)
            if new in new_l:
                continue
            else:
                content.remove(new)
                new_l.append(new)
                c += 1
        print(len(content))
        print(len(new_l))
        for line in new_l:
            writer.writerow(line)
    cross_val_bin = list()
    bin_size = len(content)/10
    for i in range(10):
        bin = list()
        c = 0
        while c < bin_size:
            if(len(content) == 0):
                break
            new = random.choice(content)
            content.remove(new)
            bin.append(new)
            c += 1
        print("bin length:", len(bin))
        cross_val_bin.append(bin)
    for i,bin in enumerate(cross_val_bin):
        with open('cvset-'+str(i)+'.csv','w+') as fp:
            wr = csv.writer(fp,delimiter=',',lineterminator='\n')
            for line in bin:
                wr.writerow(line)











for i in range(10):
    name = 'cvset'+str(i)
    #os.mkdir('.\\Data\\'+name)
    #os.mkdir('.\\Data\\'+name+'\\pos')
    #os.mkdir('.\\Data\\'+name+'\\neg')
    undo_setup('cvset-'+str(i)+'.csv','.\\Data\\'+name)
    print("Bin",i,"done")


#undo()
#seperate()
#print(len(a))
