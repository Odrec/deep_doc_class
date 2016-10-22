import shutil
from nose.tools import *
from bow_metadata import Bow_Metadata
import pandas as pd
import numpy as np

__author__ = 'tkgroot'

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

# Initializing the different Bag of Words
# bow                 = Bow_Metadata('title', debug=True)
# author              = Bow_Metadata('author', 2, 0.5, debug=True)
# filename            = Bow_Metadata('filename', debug=True)
# title               = Bow_Metadata('title', debug=True)
# description         = Bow_Metadata('description', debug=True)
# folder_name         = Bow_Metadata('folder_name', debug=True)
# folder_description  = Bow_Metadata('folder_description', debug=True)

path = './files'
testData=pd.read_csv("tests/classificationTest.csv", header=0, delimiter=';', quoting=1, encoding='utf-8')
fullData=pd.read_csv("classification.csv", header=0, delimiter=';', quoting=1, encoding='utf-8')

# score with evaluation array
author_scores=np.array([0.2, 0.1, 0.1, 0, 0.1, 0, 0, 0, 0.2])
assert_equal(author_scores.shape, (9,),"array is not of right shape")

def setup_func():
    "set up test fixtures:"
    # metadata=pd.read_csv("tests/metadataTest.csv", header=0, delimiter=',', quoting=1, encoding='utf-8')
    # author=pd.read_csv('tests/uploaderTest.csv', header=0, delimiter=",", quoting=1)
    # clf=pd.read_csv("tests/classificationTest.csv", header=0, delimiter=';', quoting=3)


def teardown_func():
    "tear down test fixtures: remove lib_bow folder"
    # shutil.rmtree('lib_bow')

@with_setup(setup_func(), teardown_func())
def test_bow_metadata():
    bow = Bow_Metadata('title',debug=True)
    assert_equal(bow.of_type, 'title')
    assert_equal(bow.punish_threshold, 10)
    assert_equal(bow.punish_factor, 0.1)

    # Make new Bag of Words
    bow.make_bow()
    bow.bow_author()
    #test if files in path exists

@with_setup(setup_func(), teardown_func())
def test_author_bow():
    author = Bow_Metadata('author',debug=True)
    assert_equal(author.of_type, 'author')
    assert_equal(author.punish_threshold, 10)
    assert_equal(author.punish_factor, 0.1)

    author_punish = Bow_Metadata('author', 2, 0.5, debug=True)
    assert_equal(author_punish.of_type, 'author')
    assert_equal(author_punish.punish_threshold, 2)
    assert_equal(author_punish.punish_factor, 0.5)

    # Takes incredibly long
    # full_author = Bow_Metadata('author')

    # Find Author from Bag of Words, assert score with evaluated scores
    for row in testData.itertuples():
        print("document_id:", row[1], "clf:", row[3], "result:", author_scores[row[0]])
        assert_equal(author.get_function(path+row[1]+".pdf"), author_scores[row[0]], "result doesnt match")

    # Find Author from Bag of Words, assert score without evaliation of the score
    # for row in fullData.itertuples():
    #     print("document_id:", row[1], "clf:", row[3])
    #     assert_equal(full_author.get_function(path+row[1]+".pdf") <= 1, True, "result doesnt match")


    # File doesnt exists
    print("The following file doesnt exists. It should return 0")
    assert_equal(author.get_function("./files/00000000000000000000000000000000.pdf"), 0)

def test_filename_bow():
    filename = Bow_Metadata('filename', debug=True)
    assert_equal(filename.of_type, 'filename')
    assert_equal(filename.punish_threshold, 10)
    assert_equal(filename.punish_factor, 0.1)

    # Finds filename from Bag of Words, asserts score without evaluation of the score
    for row in testData.itertuples():
        print("document_id:", row[1], "clf:", row[3])
        assert_equal(0 <= filename.get_function(path+row[1]+".pdf") <= 1, True, "result is not between 0 and 1")


def test_title_bow():
    title = Bow_Metadata('title', debug=True)
    assert_equal(title.of_type, 'title')
    assert_equal(title.punish_threshold, 10)
    assert_equal(title.punish_factor, 0.1)

    # Finds title from Bag of Words, asserts score without evaluation of the score
    for row in testData.itertuples():
        print("document_id:", row[1], "clf:", row[3])
        assert_equal(0 <= title.get_function(path+row[1]+".pdf") <= 1, True, "result is not between 0 and 1")

@with_setup(setup_func(),teardown_func())
def test_description_bow():
    description = Bow_Metadata('description', debug=True)
    assert_equal(description.punish_threshold, 10)
    assert_equal(description.punish_factor, 0.1)

    # Finds description from Bag of Words, asserts score without evaluation of the score
    for row in testData.itertuples():
        print("document_id:", row[1], "clf:", row[3])
        assert_equal(0 <= description.get_function(path+row[1]+".pdf") <= 1, True, "result is not between 0 and 1")



def test_folder_name_bow():
    folder_name = Bow_Metadata('folder_name', debug=True)
    assert_equal(folder_name.punish_threshold, 10)
    assert_equal(folder_name.punish_factor, 0.1)

    # Finds folder_name from Bag of Words, asserts score without evaluation of the score
    for row in testData.itertuples():
        print("document_id:", row[1], "clf:", row[3])
        assert_equal(0 <= folder_name.get_function(path+row[1]+".pdf") <= 1, True, "result is not between 0 and 1")

def test_folder_description_bow():
    folder_description = Bow_Metadata('folder_description', debug=True)
    assert_equal(folder_description.punish_threshold, 10)
    assert_equal(folder_description.punish_factor, 0.1)

    # Finds folder_description from Bag of Words, asserts score without evaluation of the score
    for row in testData.itertuples():
        print("document_id:", row[1], "clf:", row[3])
        assert_equal(0 <= folder_description.get_function(path+row[1]+".pdf") <= 1, True,
                     "result is not between 0 and 1")

# print(test.get_function("./files/b4825922d723e3e794ddd3036b635420.pdf")) #strangely positive in metadatatest
# print(test.get_function("./files/3d705ef7bee2de856e545e352a5325ec.pdf")) #positive
# print(test.get_function("./files/a719b36ae0cb229662c706f2482970da.pdf")) # positive author with 2 documents marked as positiv
# print(test.get_function("./files/189d4bc5378e11884eddeecec9304588.pdf")) #negative
# print(test.get_function("./files/1c1be8ef8986f848d28280c3444233c7.pdf")) #positive only in metadata