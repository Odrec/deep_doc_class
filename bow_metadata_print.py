__author__ = 'tkgroot'
from bow_metadata import BowMetadata
import pandas as pd
from time import time
"""
 Runnable srcipt for bow_metadata
"""
# Classification Metadata
# c03e30bb9a19d5a24fcd1cc88f245171;4;True
# f1cace3f522a7d7072d46b443336b7f0;4;True
# 3d705ef7bee2de856e545e352a5325ec;4;True
# 189d4bc5378e11884eddeecec9304588;1;False
# b4825922d723e3e794ddd3036b635420;4;True
# 170d3cbd8c0f867b342ac35b29d05ea0;1;False
# 01628a243b61ec0f067b62a8f7ac2f00;1;False
# b93adf5f5c1bd31e26ca7306e8b91a3c;1;False
# a719b36ae0cb229662c706f2482970da;2;True

# Loading files
t0=time()
# Testfiles
# metadata=pd.read_csv("tests/metadataTest.csv", header=0, delimiter=',', quoting=1, encoding='utf-8')
# clf=pd.read_csv("tests/classificationTest.csv", header=0, delimiter=';', quoting=3)

# full metadata/classification/uploader
metadata=pd.read_csv("metadata.csv", header=0, delimiter=',', quoting=1, encoding='utf-8')
author=pd.read_csv('uploader.csv', header=0, delimiter=",", quoting=1)
# clf=pd.read_csv("classification.csv", header=0, delimiter=';', quoting=3)
clf=pd.read_csv('p52-dl/validation.csv', header=0, delimiter=';', quoting=3)

# index shift to document_id
metadata=metadata.set_index(['document_id'])
author=author.set_index(['document_id'])
clf=clf.set_index(['document_id'])

print("metadata loaded %0.3fs" % (time()-t0))

# New Metadata Testing
t1=time()
print('Creating Object of BowMetadata')

author              = BowMetadata('author')
# author2             = BowMetadata('author', 2, 0.5)
filename            = BowMetadata('filename')
title               = BowMetadata('title')
description         = BowMetadata('description')
folder_name         = BowMetadata('folder_name')
folder_description  = BowMetadata('folder_description')

modules = list()
modules.append(author)
modules.append(filename)
modules.append(title)
modules.append(description)
modules.append(folder_name)
modules.append(folder_description)

module_names = ('author', 'filename', 'title', 'description', 'folder_name', 'folder_description')
print('Done in %0.3fs' % (time()-t1))

# General functionality test with validation set
t2=time()

total_result = pd.DataFrame(index=module_names)
m_result_pos, m_result_negT, m_result_negF, m_result_none = list(), list(), list(), list()
for m in modules:
    clf_correct = 0
    clf_false_positive = 0
    clf_false_negative = 0
    clf_none = 0

    for index in clf.index:
        try:
            proba = m.get_function('./files/'+index+'.pdf', metadata.loc[index])
            if proba < 0.5:
                result = False
            elif proba > 0.5:
                result = True
            else:
                result = True
                print("proba: ", proba, "failure: ", index)

            validation_value = clf['published'].loc[index]
            if validation_value == result: clf_correct += 1
            if validation_value != result:
                if validation_value == False and result == True: clf_false_negative += 1
                if validation_value == True and result == False: clf_false_positive += 1
        except:
            clf_none += 1

    m_result_pos.append(clf_correct)
    m_result_negT.append(clf_false_positive)
    m_result_negF.append(clf_false_negative)
    m_result_none.append(clf_none)

total_result['correctly classified'] = pd.Series(m_result_pos, index=module_names)
total_result['classified as false positive'] = pd.Series(m_result_negT, index=module_names)
total_result['classified as false negative'] = pd.Series(m_result_negF, index=module_names)
total_result['not classifiable'] = pd.Series(m_result_none, index=module_names)
total_result['total number of documents'] = pd.Series(len(clf.index), index=module_names)

total_result.to_csv('lib_bow/results.csv')
# with open('lib_bow/results.txt', 'w') as file:
#     for item in result:
#         file.write(str(item)+',')
#         file.write(str(item)+'\n')

print("Done in %0.3fs" % (time()-t2))