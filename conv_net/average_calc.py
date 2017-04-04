import csv
import re


fp = open('classification.csv','r')
reader = csv.reader(fp,delimiter=';')
tru_pos = 0
fal_pos = 0
tru_neg = 0
fal_neg = 0
t = 0.1
t2 = 0
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
        names.append(splitted[0])
        page_count[splitted[0]] = 1
        classes[splitted[0]] = int(row[1])
        predictions[splitted[0]] = int(row[2])
    else:
        page_count[splitted[0]] += 1
        predictions[splitted[0]] += int(row[2])
    #if splitted[1] != '2.jpg':
     #   continue

for name in names:
    predictions[name] /= page_count[name]

    if predictions[name] > t2:
        predictions[name] = 1
    else:
        predictions[name] = 0
N = 0
for name in names:
    print(name)
    N += 1
    if classes[name] == 1 and predictions[name] == 1 :
        tru_pos += 1
    elif classes[name] == 1 and predictions[name] == 0 :
        #print(row)
        fal_neg += 1
    elif classes[name] == 0 and predictions[name] == 0 :
        tru_neg += 1
    elif classes[name] == 0 and predictions[name] == 1 :
        fal_pos += 1
    else:
        print(str(classes[name])+' '+str(predictions[name]))
print(N)
print()
print('_Confusion Matrix one-strike-rule threshold('+str(t)+')_')
print('\t\tPos\tNeg')
print('True\t'+str(tru_pos)+'\t'+str(tru_neg))
print('False\t'+str(fal_pos)+'\t'+str(fal_neg))
print()
print('\t\tPos\tNeg')
print('True\t'+str(round(tru_pos/N,3))+'\t'+str(round(tru_neg/N,3)))
print('False\t'+str(round(fal_pos/N,3))+'\t'+str(round(fal_neg/N,3)))
print()
print('accuracy:\t'+str(round((tru_pos+tru_neg)/N,3)))
print('critical error:\t'+str(round((fal_neg)/(fal_neg+tru_pos),4)))