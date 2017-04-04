import csv
import re


fp = open('classification.csv','r')
reader = csv.reader(fp,delimiter=';')
tru_pos = 0
fal_pos = 0
tru_neg = 0
fal_neg = 0
t = 0.5
N = 0
first = True
for row in reader:

    if row == [] or first:
        first = False
        continue
    #splitted = row[0].split('-')
    #if splitted[1] != '2.jpg':
     #   continue
    N += 1
    if float(row[3]) > t:
        row[2] = '1'
    if row[1] == '1' and row[2] == '1':
        tru_pos += 1
    if row[1] == '1'and row[2] == '0':
        print(row)
        fal_neg += 1
    if row[1] == '0'and row[2] == '0':
        tru_neg += 1
    if row[1] == '0'and row[2] == '1':
        fal_pos += 1
print()
print()
print('_Confusion Matrix 1st page thresold ('+str(t)+')_')
print('\t\tPos\tNeg')
print('True\t'+str(tru_pos)+'\t'+str(tru_neg))
print('False\t'+str(fal_pos)+'\t'+str(fal_neg))
print()
print('\t\tPos\tNeg')
print('True\t'+str(round(tru_pos/N,3))+'\t'+str(round(tru_neg/N,3)))
print('False\t'+str(round(fal_pos/N,3))+'\t'+str(round(fal_neg/N,3)))
print()
print('accuracy:\t'+str(round((tru_pos+tru_neg)/N,3)))
print('critical error:\t'+str(round((fal_neg)/(fal_neg+tru_pos),3)))