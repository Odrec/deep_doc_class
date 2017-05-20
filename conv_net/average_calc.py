import csv
import re


fp = open('classification.csv','r')
fp2 = open('cleaned_manual_class.csv','r')
reader = csv.reader(fp,delimiter=';')
reader2 = csv.reader(fp2,delimiter=',')

rel_tru_pos = 0
rel_fal_pos = 0
rel_tru_neg = 0
rel_fal_neg = 0

t = 0.5
t2 = 0.5
N = 0
first = True
predictions = dict()
page_count = dict()
classes = dict()
names = list()
for row in reader:
    if row == [] or first:
        first = False
        continue
    if float(row[3]) > t:
        row[2] = '1'
    splitted = row[0].split('-')
    if not splitted[0] in page_count:
        if not '.jpg' in splitted[0] and int(splitted[1].split('.')[0]) > 5:
            continue
        names.append(splitted[0])
        page_count[splitted[0]] = 1
        classes[splitted[0]] = int(row[1])
        predictions[splitted[0]] = float(row[3])
    else:
      #  if page_count[splitted[0]]>= 5:
      #      continue
        page_count[splitted[0]] += 1
        predictions[splitted[0]] += float(row[3])
    #if splitted[1] != '2.jpg':
     #   continue
relevance = dict()
type = dict()
for row in reader2:
    relevance[row[0]] = row[2]
    type[row[0]] = row[3]
for name in names:
    predictions[name] /= page_count[name]

    if predictions[name] > t2:
        predictions[name] = 1
    else:
        predictions[name] = 0

for typus in range(1,15):
    N = 0
    tru_pos = 0
    fal_pos = 0
    tru_neg = 0
    fal_neg = 0
    for name in names:
        proto = name.replace('.jpg','')
        if proto in relevance:
            #print(type[proto])
            #if relevance[proto] == '0' :
            #    continue
            if typus != 14 and int(type[proto]) != typus:
                continue


        #print(name)
        N += 1
        if classes[name] == 1 and predictions[name] == 1 :
            tru_pos += 1
        elif classes[name] == 1 and predictions[name] == 0 :
            print(name)
            fal_neg += 1
        elif classes[name] == 0 and predictions[name] == 0 :
            tru_neg += 1
        elif classes[name] == 0 and predictions[name] == 1 :
            fal_pos += 1
        else:
            print(str(classes[name])+' '+str(predictions[name]))
    if True:
        print(N)
        print()
        print('__Confusion Matrix for doc type:',typus,'(14 means all docs)__')
        if(N == 0):
            continue
        print('\t\tPos\tNeg')
        print('True\t'+str(tru_pos)+'\t'+str(tru_neg))
        print('False\t'+str(fal_pos)+'\t'+str(fal_neg))
        print()
        print('\t\tPos\tNeg')
        print('True\t'+str(round(tru_pos/N,3))+'\t'+str(round(tru_neg/N,3)))
        print('False\t'+str(round(fal_pos/N,3))+'\t'+str(round(fal_neg/N,3)))
        print()
        print('accuracy:\t'+str(round((tru_pos+tru_neg)/N,3)))
        try:
            print('critical error:\t'+str(round((fal_neg)/(fal_neg+tru_pos),4)))
        except:
            print('critical error:\t'+str(0.0))