# coding=utf-8

import sys, os, shutil
from os.path import join, realpath, dirname, isdir, basename, isfile
MOD_PATH = dirname(realpath(__file__))
SRC_DIR = os.path.abspath(join(join(realpath(__file__), os.pardir),os.pardir))
if(not(SRC_DIR in sys.path)):
    sys.path.append(SRC_DIR)
FEATURE_DIR = join(SRC_DIR,"features")
if(not(FEATURE_DIR in sys.path)):
    sys.path.append(FEATURE_DIR)

from doc_globals import*

from time import time

import nltk
from nltk.corpus import stopwords
nltk.data.path.append(join(MOD_PATH,'nltk_data'))  # setting path to files

import numpy as np
import pandas as pd
import json, csv, re, math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn import tree

from sklearn.metrics import confusion_matrix

# from langdetect import detect_langs
# from langdetect import detect
# import treetaggerwrapper as ttwp

import pdf_properties
import pdf_text
import pdf_metadata
import pdf_xml_structure


false_docs = ["cc17fb8eaf055e2c8224eaaf3c10145c",
    "43d566f90c24f874781d97905870ed23",
    "96a92f3bcb002f22a698a62a468415e3",
    "9c70eaa380eb60135c5f1953b1f073cf",
    "6aaf86df732ef957de9a48e2b3c589e9",
    "daad7c32072aaa4cdbb9ae2c02c2c9b6",
    "e65752213cebcb6e151a815ed21512f3",
    "5ff6d40b1dc0dde4c51a8436bc57aa20",
    "fc8278775cf12fdf84e88493f8efc645",
    "d047a01a190236e10d6addefe13ed3bc",
    "7e7ff5a990f0e5756e31348ced169854",
    "5268494d6a5b7f3bc27bc2dcbcf79748",
    "10c8f8111c29862458987f77c05f4a39",
    "19e69897d4842c053d781e773e5122bd",
    "da5a198a45c690c3570a7d307597b892",
    "f51d621abeb4c489918d904995bbcec0",
    "ef59be13bd0d88b43fcb779912df7a66",
    "36889863bea6feb43dbe335719aa657a",
    "6c357bc6ddc11a3b5d199d2b1e4e6fac",
    "2ad710735830e95c3e9ce01725552a8b",
    "df59eaa1565f4b3d980c15a00ade3bdc",
    "826c1f6b7d9b2bd7fc50c23a126b71a1",
    "12d0d64889fe56c5789f8b73a8813ce9",
    "9d2cb33abdcfefa9a56bb1f7da4ef263",
    "a5322615e83b66d9b6c5e9a50b3eebb2",
    "57f98112812429c28def64a14020b242",
    "b37ce79bf563844426c7452c38564acc",
    "00a1e2413c919d6e392442533626cb92",
    "acda4e5d66d28ab616cd34fd09055b05",
    "ea6c27a337aff6c262e8316984ff72e5",
    "b2ac2fe5be5547e864f00b2d0f3d8003",
    "7f4038da9e1fb0aa745adcf8cc8cb832",
    "e5b0ee8a5519969ec70f5722a90a8019",
    "a774df02a832585275de3db40f132cdd",
    "580bebc528e28f520565e69ef6de8506",
    "5464ee4ef790013f0caa5d16d72ae7fc",
    "259f6473a159a782cc6f9aaf2ceda6a1",
    "057782ecdfc1f4179d2f98c4d80da89b",
    "06f3ad19c37c9f4460b0c6482505599b",
    "0a32570101750613de0fc032b94f73f4",
    "116eee145a0276280e82e0081bc39130",
    "136d718bee74eb407355cb64f064867d",
    "14c2d5d73ebc31122a07ab587272f82e",
    "170754478dc23a930d22722b43dbf181",
    "1799aad8a5a88b17cd8480fdecdd7b81",
    "3229f885e2668f522c2d64b84a34639b",
    "3981b966fa573e6419054a38660bbdda",
    "44fbbe38e3512475b9e590a937c3e933",
    "49f920671851f6a9010187bdc9743b2c",
    "590ce47a42bf67aa1295d02e85b1a7b5",
    "5c52e2433813f16b46264f07fed83783",
    "5f4616a15b26f79b6190b072dba11aa8",
    "6956ad338f080b4be0710ce32ca12976",
    "73197946354da526a4f08756a9703ddd",
    "74f8f36f354984572718e765d4e85189",
    "8b9b83971a91e40e8a05bd09d0138e49",
    "9100a9452a85c3732784074b41233a9c",
    "a4351807fee95ad4a750546b71b657fb",
    "ad3b3d728c8e579ef31efe61b9739c8a",
    "ae5378982b7227176c3e42dd93c58bd4",
    "c7cda18baa6c465a1a076bc6034b609f",
    "d5763c8901ef9e76bbef5e673145b300",
    "ef82cdd10e45d8184c8779361cce78ef",
    "7daba79637e41d82c9df63465ba2f0a3",
    "f39e2ed04d180820f05d6ff362aedd1c",
    "9dbb018b918a13f12613a063c386e2a7",
    "bf6a3ea6f15c2c9c96a19da3f3ba9e43",
    "38700021b4e8b2b6f34e19c47a096980",
    "fabe694fea65e5b15574f4eaa2a190a9",
    "059600524e5fa4fff9c72854ae7c70dc",
    "6fa54fffe2a2335f5927572e9757ee40",
    "d9eebb93c161bb0131245f80e8ccc0b3",
    "97bb3e396c0396a693f3f8cb2f3654b3",
    "4af0c6020d07663825f905c783828379",
    "3c16342b2169cc662f7f3c4b0e9289a1"
]

xml_error = [
    "4d70b0a1973a94c98b50a9fd43f0ffba",
    "462e37b58d623e5458b8d630e4a4d993",
    "e52c862fe0b09a52c94dee89141925eb",
    "168f489c98b9976bbcce28f628e5078e",
    "dab670f904564b76c3463a4343594126",
    "afe02eb0714555c8176deaab05a5342f",
    "35d1120c963d214c331c3b036377a0cf",
    "e767a496ac211ec94a7f82f905a51393",
    "b98e74c526e1ecfc9ecafe848a21e147",
    "80946afd0a6c882d243ca3d3586a150d",
    "c26eb949d5bd40ded6a0711ef9af3426",
    "aafeaf6ff8f738c537bc8b4ebf012dde",
    "7451990e707f49fc90725036440c4434",
    "0b6b1cb9c9cd832bf02cf67bf8302e2c",
    "f16934cc36ac59d568502ac3a9d34cf3",
    "a4975f6d0d4ff4b8bbd0981ecd0f2285",
    "b8e2d6b46e3cb95bb185b371c1c95e1e",
    "6b5e9e9ce2e29547726ae8f8abd2ca82",
    "477e81c2aac2ff90b327e495366f8f86",
    "07a135fa0d07c49e0e6cd47810003dbc",
    "2312cef18bcf2755f37f1e08c5ce911e",
    "4c4829b9388c25f7ec2a001902e3d648",
    "dc55cb8299cd57bfdab9c7aff817d1eb",
    "81668dba8fec354eeb91024cdd663089",
    "154e7e64bf2119f2a5129991bdf4bf7c",
    "cd977ba21b6b8e6c8f2120e2c2881474",
    "a412dd22dd26aead36dc91026405abbb",
    "7ce1765b88bd1d1a78eca1e14fff62f9",
    "35ea1cd083532790f29aa9975cd6eb45",
    "152c77685b41801858fbcc76c145dab7",
    "3fb837679d044b2cf95a52e3370c2045",
    "f4e06729807af3fd0d162790a4a48ef4",
    "9f5832a1a3eec3ef2266e523f85c4322",
    "f1dfc962076e1f7f89cdaf8efb5cb314",
    "c6237fc82c1c2030e68df38de7d3871a",
    "618452d00a628665901bdc3151265e70",
    "0a78f6074fa498a9a184b3283cde7139",
    "6888a8187c69fc8d91e45828d280bfb7",
    "5ea358146448d79442e42e15233026c9",
    "8df3f8fc7b4c6aa07a0529f441120660",
    "a9e0db397b2e8b95931bffd8a56d5c18",
    "240c7b6e495076afaba00c9889aea082",
    "7b9778f8f7fb1c67783b82fb9bae42ab",
    "e51d2152efe6856a26ddd0adc845d1f0",
    "ca3359a20a73e30b109d0aed1a4b8711",
    "b7d354d689f5052419c45cf4414d8216",
    "8bd58c19720d9cc35d119f965b1ef601",
    "26d3adda46b9c565d327d8d810e6151a",
    "6f188eac23f755fdb358a7ce1136b6e7",
    "dfae946e4721c3470994c90bd5bd393e",
    "1b06d3a26375b06e7b79afb68f80b7da",
    "d47b36df95eb67841ad803256da587de",
    "7262cec0a47a711bb4f34391557219c9",
    "d6c6bdadbbcfe0f16f31ddfa76868d5c",
    "f81c611790ded32869499b298acc9d4e",
    "4fe72e59a686fd5dcddce7e5d8d2c99e",
    "8d072fc27449b5637453d45a9d75e2b0",
    "ee31588463e2469ee6216e9c1a32186d",
    "4a1c0babb340a6bd167a2600ba25e259",
    "5aaaa751412c557f4a1230d48d61614b",
    "56a2018de85943fa30cd4ddce3b9b602",
    "f0140465484930a469c8e08be23b107b",
    "d404503b466c0067fb52ea7fbb1eb99c",
    "7dc8171d1f40cb6d4ab849e0901f9989",
    "3cd73c97de8751d51309ffcca41a2ee5",
    "3848d5e5b76f7aa570105dc0c4e23ca9",
    "4053a8bd67e6b97033b1b1e379eb07b0",
    "efe95fb04937ce282415b3752296037f",
    "998b4efbfcc024e8284aef4f0d9eba87",
    "333c25f776348ecc4c824dfe9318ffe0",
    "1104c1ccb4037a458c7f1306cc8a4b38",
    "ab9cc9945400f56e598c7642120b4ac0",
    "cff2a79f90981ec250bdc460101e92d1",
    "353618076a78e6d84e3b16607693cade",
    "32e90d1459c534087a2daccdfc98453c",
    "f38a244eb0ef01f05e8d9f88d4c15286",
    "6e6a8826597f72e5403a00f1fc43fea2",
    "f677954aab79b144b33a8539a80b5021",
    "e6e55e34f7f10acf95bf72e8f64adc9a",
    "f90fd5e8e62b6216f9575e61f3f8ef45",
    "f25e883b16833c800a297e4d93b0d553",
    "0778080d975a2f66a5bd48b249baab5f",
    "6ce159fb4aee9b927c2157086f32f925",
    "f4e6c77cc5614c657ce54a481fd0f362",
    "ec932748ed8dc56134253c82d61c101d",
    "4443af1116ccc197989be849d9f41aec",
    "2cd9220978624e281e0e3245d9d0fca5",
    "7fde22aa84dcb4b2cc9e52a480507535",
    "95cc939db7100de396da394317e28d1c",
    "e5697f2a78bc48b1eb0252fd2ef7279a",
    "de6d3ae71d8b587fb5217209fac784d8",
    "87245bf63324cf33d4aef86b13c7b0eb",
    "e95621e72ff16ef5048d3626592af65b",
    "7a623019662b3bebf5f76771051dd880",
    "97383cd50190b061553cd73973bce984",
    "f752c531b9c64ee61a33fc4452ecb976",
    "fabcd583132b6eb5d12812726f17bedb",
    "9c93f5b188319d1bddaed4171d0cf7cf",
    "f41d85f790ac6c88d3d66cd876ff0b85",
    "8d196e2c05c7e02d5117302fd7fa6182",
    "170754478dc23a930d22722b43dbf181",
    "0d6f103a9fa711786c797fb1b46e8e22",
    "6edf443e2586f9e0c91d190a9a22f2e5",
    "1ce91be84b2e5643eae607d049c2912d",
    "e3df65c8481ecd9b72f072a065bac4a6",
    "a7f54c58f810e2ea9f95ec28aad08f8b",
    "e84d829645112856c3be0b408e783115",
    "08576fc10f451db0b6161718f7ab9a8d",
    "9b42880245d1b42d5771bc2bffc3b76a",
    "6032b50d503896c3fd7d9cd31d2e0447",
    "cc375cc82ad35b12b14062163670b342",
    "b4610d02c99e9c27742441d36655f1ba",
    "0306420da9829eef44f32ea96d45d568",
    "9a791ce480ed18604f49553e7a992bab",
    "15c4bae8e5f664d9ec481ee6385f8a25",
    "74c3870963becf2b891ef2574bfcb319",
    "1a5d44eb741c693cb626198fc63b50d5",
    "f6d2282e1aa5ff337ec129a553535e67",
    "ad9ade6d7b101edcb4bb9849be69074c",
    "a2835d03b2d55cd13c406106eee9aaf1",
    "c09aaa06323524e14a734b6c711f443f",
    "01ee82f2ec05c0e273d229b5e9412379",
    "6c357bc6ddc11a3b5d199d2b1e4e6fac",
    "11f66b9e33b3773d1039ae28d45183e5",
    "1102816909e4050340e4269631e3a57e",
    "8b5d679b3ad0d2a3bf85a22c4c3fd47c",
    "835508355d5f6284a0c58a1873a90a2a",
    "c7909f905d3376d1d50e306a23a28e60",
    "48c81c9e65935473cbd4c206b023c673",
    "071e9a3c2811f70db3bac220aaa325bb",
    "843b7382bd6ec44dd2bb4c958d4a7751",
    "e4d437e678dc3c9da8b6695d3ff8d7e2",
    "9a5d3349755cdbf1fc71dbee3e36ac37",
    "68ad6882f7cd47437e58108e716cbcad",
    "9a599a19d1e277137e3f083c255a62e3",
    "ddcf3afdc941865b12bdd2af9c055ef6",
    "9cff5d5a04b718f20ca4eecc93e64d8b",
    "9b2b774ef4c39fc30384a0bf0a40f600",
    "8dbff54b538bfe88c04c33122b3bca38",
    "51acd0ea605794b421781e475e322c9a",
    "341343a4fba4b858c1c916cccf9b9cec",
    "d87cc194527e7d6c5e6a8b4037046394",
    "458c3b916b58fbd0d6efe085fb2ab2b0",
    "41d760f50056d2a3bbf086047492c436",
    "6d761eb389d6d0fddf0ee72dc384cf2e",
    "e1cef7747907cecbce8c76026eb2ca06",
    "e33e7c75475d7073160d6058053f83e6",
    "a75d7757a9c31c9dab10c2c8e4c34f89",
    "404d8dce32be7dcb2e4160aa78c44de6",
    "a8c89340f76e1798d1bb9726d91b7cb8",
    "24c9893770471a031e2c5d4619ecc995",
    "13374318ead52e16969c0d0544d7c596",
    "bb88fe6c0328912f0bb90beb4fb87109",
    "206297c4a808111bcee4d769dbb43385",
    "d987c51792fcaa51a8916b6687b09d81",
    "08199c77dbbcd19a24cf4bda154a29da",
    "5ce4ae00d419a61b987b8abfcc931b25",
    "a0240d7f82990009412be04369d5c7d1",
    "e5178ff03bb21e40281c083021dd5d80",
    "9478bb43e1060f6747290d0ef04b806c",
    "203a02d423548260c477a5c1e56c40c6",
    "66fc6ce18b44145964a06d8ff9db3ea1",
    "38d35d6dddf9aa3c5ad647563a1c7b6a",
    "56db3ea9339a2a2d8424804dd3842641",
    "c125a84aa5ac9216987a9e1f725b8e64",
    "325cc5f8fea90ec6455c1fc90805a801",
    "20ed88098c997adade05ebcc5855c20c",
    "e63baff797c0f1755888a2cbb22c4935",
    "f264c2d15410ab35c314b679674c0109",
    "008bd1a8b44ff03dc160552cf0514a92",
    "62c206e1238acf99b33906437504996f",
    "3e039943e16f3eddea67fa71335fe052",
    "70fa622e70974dbf3314adfa4ad68514",
    "c2a992d1b181ce386a342e6324131ac3",
    "9fca5e8d73098cb0bbd103ae77b16030",
    "c35ec2d4552a950069b390101bbbd27d",
    "21f0fb0e872f9cc83ebb69f0890fe1c9",
    "8441e472c36d966dc9d3f0ed9ed767ba",
    "f1abde355dac04bbd21d33574de0c299",
    "664c7b944a1799b494001ebcffd1a312",
    "24549c324d4138b4aabe2e69da3143b9",
    "0880622c454aa613513589c9957b85a9",
    "6487721de7c5bd3f50257e8931020606",
    "14a14bbc72c5ad5c4ef75714fc27a2ca",
    "f2dadc933186ba967ce56853e9242bc5",
    "f7ba2b00060e70cede9eda22f8323e7f",
    "85bf12bd6c668033a3d730f3926b0640",
    "ec5951bec760e3d149092f0cadd71119",
    "7c1b592a0b256f3bb61e42c6a0260022",
    "eee07c5ef8c9d49c5cbb4ae689b76179",
    "a69acaacc655eeadd3f726c1f2fe4dd1",
    "18111af931025656bdf4581bd27b268e",
    "c9e4d8556c070d903cb8b58cfa57e90a",
    "46fe2f38e7e10be55cecc7e87599e88a"
]

class BowClassifier():
    """
    BowClassifier is a container for a trained Bow-Model whose main purpose is to map a input string to a value. The value is the likelyhood for the input string to represent a copyright pdf protected document. The input string can be of different kinds of origins like pdf properties or some metadata or the content of the pdf. For the classification it uses trained models of a Countvectorizer and a set of different classifiers from the sklearn librabry. The BowClassifier provides means for training, storing and crossvalidating those models as well.
    """
    def __init__(self, name):
        """
        Initializes a BowClassifier.

        @param name: Identifier for what kind of data is mapped in the Analyzer.
        @dtype name: str
        @param max_features: The maximum amount of words in the Countvectorizer.
        @dtype max_features: int
        @param n_estimators: The amount of trees in the RandomForest.
        @dtype n_estimators: str
        """
        self.name = name
        self.data_origin = None
        self.csvmeta_options = ["title", "folder_name", "description", "folder_description", "filename"]
        self.pdfinfo_options = ["author", "producer", "creator"]
        self.structure_options = ["bold_text", "first_100_words", "last_100_words"]

        # check if name is a viable string
        if(name in self.csvmeta_options):
            self.data_origin = "csvmeta"
        elif(name in self.pdfinfo_options):
            self.data_origin = "pdfinfo"
        elif(name in self.structure_options):
            self.data_origin = "pdf_structure"
        elif(name=="text"):
            self.data_origin = "pdfcontent"

        # if it was not print usage and exit
        if(self.data_origin is None):
            print("%s is not a valid input argument!!\nUse either one of: %s\nOr one of %s or text!!!"%(self.name,
                str(self.csvmeta_options),str(self.pdfinfo_options)))
            sys.exit(1)

        self.vectorizer = None
        self.classifier = None

    def load_vectorizer_model(self, modelpath):
        '''
        Loads the trained vectorizer models for this classifier.

        @param modelpath: The full path to a modelfile (.pkl) file
        @dtype modelpath: str
        '''

        if(modelpath == "default"):
            modelpath = join(MOD_PATH,"vectorizer",self.name+'.pkl')

        try:
            self.vectorizer = joblib.load(modelpath)
        except FileNotFoundError:
            print("File %s does not exist!!!" %(modelpath,))
            sys.exit(1)
        except:
            print("File %s could not be loaded with sklearn.ensemble.joblib!!!" %(modelpath,))
            sys.exit(1)

    def load_prediction_model(self, modelpath):
        '''
        Loads the trained forest models for this classifier.

        @param modelpath: The full path to a modelfile (.pkl) file
        @dtype modelpath: str
        '''
        def_cls = ["custom_forest", "custom_log_reg", "vectorizer_forest", "vectorizer_log_reg", "word_vec"]

        if(modelpath in def_cls):
            modelpath = join(MOD_PATH,modelpath,self.name+'.pkl')

        try:
            self.model = joblib.load(modelpath)
        except FileNotFoundError:
            print("File %s does not exist!!!" %(modelpath,))
            sys.exit(1)
        except:
            print("File %s could not be loaded with sklearn.ensemble.joblib!!!" %(modelpath,))
            sys.exit(1)

    def load_custom_words_vectorizer(self, wordspath):
        if(wordspath == "default"):
            wordspath = join(MOD_PATH,"words",self.name+'.txt')

        try:
            f = open(wordspath, 'r')
            words = f.read()
            f.close()
        except FileNotFoundError:
            print("File %s does not exist!!!" %(wordspath,))
            sys.exit(1)
        except:
            print("Error while reading file %s." %(modelpath,))
            sys.exit(1)
        vocab = re.split("\s",words)
        vocab = re.sub("\s"," ",words)
        vocab = vocab.split()
        self.vectorizer = CountVectorizer(analyzer='word', encoding="utf-8", vocabulary=vocab)

    def load_clean_data(self, doc_ids):
        if(self.data_origin == "csvmeta"):
            meta_path = join(DATA_PATH,"classified_metadata.csv")
            clean_data = pdf_metadata.load_single_metafield(doc_ids,self.name,meta_path)
            for i in range(len(clean_data)):
                clean_data[i] = preprocess_pdf_metadata_string(clean_data[i])
        elif(self.data_origin == "pdfinfo"):
            prop_path = join(PRE_EXTRACTED_DATA_PATH,"pdf_properties.json")
            clean_data = pdf_properties.load_single_property(doc_ids,prop_path,self.name)
            for i in range(len(clean_data)):
                clean_data[i] = preprocess_pdf_property_string(clean_data[i])
        elif(self.data_origin == "pdf_structure"):
            struc_path = join(PRE_EXTRACTED_DATA_PATH,"xml_text_structure.json")
            clean_data = pdf_xml_structure.load_single_property(doc_ids, struc_path, self.name)
            for i in range(len(clean_data)):
                clean_data[i] = preprocess_pdf_text_string(clean_data[i])
        else:
            clean_data = pdf_text.get_pdf_texts_json(doc_ids,PDF_PATH,TXT_PATH)
            for i in range(len(clean_data)):
                clean_data[i] = preprocess_pdf_text_string(clean_data[i])
        return clean_data

    def fit_vectorizer(self, vectorizer, clean_data, save_path=None):
        self.vectorizer = vectorizer
        self.vectorizer.fit(clean_data)

        if(not(save_path is None)):
            if(not(isdir(os.path.dirname(save_path)))):
                os.makedirs(join(save_path))

            joblib.dump(self.model, save_path)

    def train_classifier(self, model, data, classes, save_path=None):
        self.model = model
        self.model.fit(data, classes)

        if(not(save_path is None)):
            if(not(isdir(os.path.dirname(save_path)))):
                os.makedirs(join(save_path))

            joblib.dump(self.model, save_path)

    def get_function(self, input_string):
        '''
        Copmputes the prediction probability for the input string.

        @param input_string: The string which is to be classified
        @dtype input_string: str
        '''
        # check if the input is of type string
        if(not(type(input_string)==str or input_string is None)):
            print("Input has to be of type string! It is of type %s" %(str(type(input_string)),))
            sys.exit(1)

        # switch string cleaning according to input origin
        if(self.data_origin == "csvmeta"):
            clean_test_data = preprocess_pdf_metadata_string(input_string)
        elif(self.data_origin == "pdfinfo"):
            clean_test_data = preprocess_pdf_property_string(input_string)
        elif(self.data_origin == "pdf_structure"):
            clean_test_data = preprocess_pdf_text_string(input_string)
        else:
            clean_test_data = preprocess_pdf_text_string(input_string)

        # get vector for the input
        test_data_feature = self.vectorizer.transform([clean_test_data]).toarray()

        f_val = self.model.predict_proba(test_data_feature)[0][1]

        # return predictiSon
        return f_val, self.name

    def predict_probs(self, data, classifier="forest", t=0.25):

        if(classifier=="log_reg"):
            probs = self.model.predict_proba(data)[:,1]
            probs[np.logical_and(probs<(1-t), probs>t)]=0.5
        elif(classifier=="forest"):
            probs = self.model.predict_proba(data)[:,1]
            probs[np.logical_and(probs<(1-t), probs>t)]=0.5
        elif(classifier=="word_vector"):
            probs = data>0
        else:
            print("<classifier> has to be one of [log_reg, forest, custom]. It is %s!!!"%(classifier,))
            sys.exit(1)

        return probs

    def crossvalidate(self, doc_ids, labels, vectorizer, classifier, n_folds=10):

        seed=7
        np.random.seed(seed)

        print("Crossvalidating " + self.name+":")
        kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        iteration = 0

        thres = [0.25]

        scores = np.zeros((len(thres),n_folds))
        scores2 = np.zeros((len(thres),n_folds))
        predicted = np.zeros((len(thres),n_folds))
        clean_data = self.load_clean_data(doc_ids)
        labels = np.array(labels)

        for train, test in kfold.split(doc_ids,labels):

            # split the data
            train_data = [clean_data[t] for t in train]
            test_data = [clean_data[t] for t in test]
            train_labels = [labels[t] for t in train]
            test_labels = [labels[t] for t in test]

            for  i,t in enumerate(thres):
                # print("%s: iteration %d/%d"%(classifier, iteration+1, n_folds))
                # get the model
                self.train2(train_data, train_labels, vectorizer, classifier)
                # predict
                probs, clf = self.predict_probs2(test_data, test_labels, classifier, t)
                preds = probs>=0.5
                score = float(np.sum([preds==clf]))/len(clf)
                print("accuracy: %.4f" %(score,))
                pos_dev = np.sum(np.abs(probs-0.5)[preds==clf])/np.sum([preds==clf])
                neg_dev = np.sum(np.abs(probs-0.5)[preds!=clf])/np.sum([preds!=clf])
                # print("correct_std_0.5: %.4f" %(pos_dev,))
                # print("flase_std_0.5: %.4f" %(neg_dev,))
                preds = probs[probs!=0.5]
                clf2 = clf[probs!=0.5]
                preds = preds>=0.5
                if(len(clf2)>0):
                    score2 = float(np.sum([preds==clf2]))/len(clf2)
                else:
                    score2=0
                print("accuracy2: %.4f" %(score2,))
                print("predicted: %d/%d" %(len(clf2),len(clf)))
                print('\n')
                scores[i,iteration] = score
                scores2[i,iteration] = score2
                predicted[i,iteration] = len(clf2)

            iteration += 1

        for j in range(len(thres)):
            print('\n')
            print("%s overall accuracy on %s: %.4f" %("thres:" + str(thres[j]), self.name, np.mean(scores[j])))
            print("%s overall accuracy2 on %s: %.4f" %("thres:" + str(thres[j]), self.name, np.mean(scores2[j])))
            print("%s overall predicted on %s: %d/%d" %("thres:" + str(thres[j]), self.name, np.mean(predicted[j]),len(clf)))

        # print("Overfitting result:")
        # self.train(doc_ids, labels)
        # self.predict_probs(doc_ids, labels, False)
        # print('\n')

    def analyze_word_distribution(self,clean_data,clf):
            pos_vectorizer = CountVectorizer(analyzer='word',
                encoding="utf-8",
                max_features=20000,
                min_df=0.001,
                max_df=1.0)
            pos_vectorizer.fit([cd for i,cd in enumerate(clean_data) if clf[i]])
            pos_words = pos_vectorizer.get_feature_names()
            pos_features = pos_vectorizer.transform([cd for i,cd in enumerate(clean_data) if clf[i]]).toarray()
            pos_features = pos_features>0

            neg_vectorizer = CountVectorizer(analyzer='word',
                encoding="utf-8",
                max_features=10000,
                min_df=0.001,
                max_df=1.0)
            neg_vectorizer.fit([cd for i,cd in enumerate(clean_data) if not(clf[i])])
            neg_words = neg_vectorizer.get_feature_names()
            neg_features = neg_vectorizer.transform([cd for i,cd in enumerate(clean_data) if not(clf[i])]).toarray()
            neg_features = neg_features>0

            shared_words = [pw for pw in pos_words if(pw in neg_words)]

            print("pos_words: %d" %(len(pos_words),))
            print("pos_docs: %d" %(sum(clf==1),))
            print("neg_words: %d" %(len(neg_words),))
            print("neg_docs: %d" %(sum(clf==0),))
            print("shared_words: %d" %(len(shared_words),))

            pos_ziped = zip(pos_words,np.asarray(pos_features.sum(axis=0)).ravel())
            sorted_pos = sorted(pos_ziped,key=lambda idx: idx[0],reverse=True)

            neg_ziped = zip(neg_words,np.asarray(neg_features.sum(axis=0)).ravel())
            sorted_neg = sorted(neg_ziped,key=lambda idx: idx[0],reverse=True)

            j = 0
            print("Shared")
            for i in range(len(sorted_pos)):
                word = sorted_pos[i][0]
                if(word in shared_words):
                    while(sorted_neg[j][0]!=word):
                        j+=1
                    pos_perc = float(sorted_pos[i][1])/sum(clf==1)
                    neg_perc = float(sorted_neg[j][1])/sum(clf==0)
                    rel = max(pos_perc,neg_perc)/min(pos_perc,neg_perc)
                    rel1 = rel>8 and (sorted_pos[i][1]+sorted_neg[j][1])>8
                    rel2 = rel>6 and (sorted_pos[i][1]+sorted_neg[j][1])>20
                    if(rel1 or rel2):
                        print("%35s\t%.3f\t%.3f\t%.3f\t%d\t%d"%(word,
                            pos_perc,
                            neg_perc,
                            rel,
                            sorted_pos[i][1],
                            sorted_neg[j][1]))

            pos_ziped = zip(pos_words,np.asarray(pos_features.sum(axis=0)).ravel())
            sorted_pos = sorted(pos_ziped,key=lambda idx: idx[1],reverse=True)
            print("pos")
            print(len(sorted_pos))
            for i in range(len(sorted_pos)):
                word = sorted_pos[i][0]
                if(not(word in shared_words)):
                    pos_perc = float(sorted_pos[i][1])/sum(clf==1)
                    print("%35s\t%.3f\t%d"%(word,
                        pos_perc,
                        sorted_pos[i][1]))
                    if(pos_perc<0.01):
                        break

            neg_ziped = zip(neg_words,np.asarray(neg_features.sum(axis=0)).ravel())
            sorted_neg = sorted(neg_ziped,key=lambda idx: idx[1],reverse=True)
            print("neg")
            print(len(sorted_neg))
            for i in range(len(sorted_neg)):
                word = sorted_neg[i][0]
                if(not(word in shared_words)):
                    neg_perc = float(sorted_neg[i][1])/sum(clf==0)
                    print("%35s\t%.3f\t%d"%(word,
                        neg_perc,
                        sorted_neg[i][1]))
                    if(neg_perc<0.01):
                        break


##### Crossvalidation #####
def ddc_cat_crossvalidation_bin(model, doc_ids, labels, results_path, kfolds=10, shuffle=True, seed=None):

    # get a Feature_Extractor
    fe = setup_feature_extractor()
    # get pathes to train and test files
    DATA_PATH = "/home/kai/Workspace/deep_doc_class/deep_doc_class/data/"
    train_file = join(DATA_PATH, "feature_values", "cross_eval_train.csv")
    test_file = join(DATA_PATH, "feature_values", "cross_eval_test.csv")

    # the seed makes sure to get the same random distribution every time
    if(not(seed is None)):
        np.random.seed(seed)
    kfold = StratifiedKFold(n_splits=kfolds, shuffle=shuffle, random_state=seed)

    # count the itrations
    kfold_iter = 0
    # reserve space for results
    train_results = np.ones((len(doc_ids),kfolds))*-1
    test_results = np.ones((len(doc_ids),kfolds))*-1

    # evaluate each split
    for train, test in kfold.split(doc_ids,labels):
        # make sure the model is untrained
        model.re_initialize()
        # get train and test document ids
        train_ids = [doc_ids[i] for i in train]
        test_ids = [doc_ids[i] for i in test]
        train_labels = labels[train]
        test_labels = labels[test]

        # let the feature extractor train the bow classifiers on the data only
        fe.train_bow_classifier(train_ids, train_labels, "custom", "log_reg")

        #  extract training and testing data
        fe.extract_features(doc_input=zip(train_ids,train_labels) ,feature_file=train_file)
        fe.extract_features(doc_input=zip(test_ids,test_labels) ,feature_file=test_file)

        # load that data from file
        train_data, train_labels, train_docs, column_names = load_data(train_file)
        test_data, test_labels, test_docs, column_names = load_data(test_file)

        # train the model
        model.train(train_data, train_labels)
        # get a ranking of the feature importance and save a txt
        analyse_feature_ranking(classifier=model, feature_names=column_names, results_path=join(results_path,"feature_ranking_%d.txt"%(kfold_iter,)))

        # predict the training set
        train_probs = model.predict_proba_binary(train_data)[:]
        train_results[train,kfold_iter] = train_probs

        # predict the testing set
        test_probs = model.predict_proba_binary(test_data)
        test_results[test,kfold_iter] = test_probs

        kfold_iter += 1
        print("%d. iteration done!%"(kfold_iter, ))

    return train_results, test_results

def analyse_crossvalidation_results_doc_categories(results, labels, doc_ids, results_path, filename, thres=[0.1,0.2,0.3,0.4,0.5], boxplots=True):

    # id description dict
    cat_id_to_des = {
        1:"scanned",
        2:"scanned_max2p",
        3:"paper_style",
        4:"book_style",
        5:"official",
        6:"powerpoint_slides",
        7:"other_slides",
        8:"course_material",
        9:"course_material_scanned",
        10:"official_style_course_mater",
        11:"handwritten",
        12:"pw_protected",
        13:"unsure"
    }

    legend = "Copyright protected documents have a positive label.\n"+\
        "Not copyright protected documents have a negative label.\n"+\
        "False positives are documents that have falsely been predicted as copyrights!\n"+\
        "False negatives are copyright protected documents which were not found!\n\n"+\
        "Precision: true positives / predicted positives\n"+\
        "Recall: true positives / real positives\n"+\
        "F1: 2* precision*recall/(precision+recall)\n\n"

    if(not(isdir(results_path))):
        os.makedirs(results_path)

    # those are the names of the overall scores comouted during each run
    score_names = ["accuracy", "precision", "recall", "f1", "tn", "fp", "fn", "tp"]
    # evaluate the results considering different thresholds
    test_size = len(labels)/len(results[0])

    for t in thres:
        # category evaluation
        category_eval = np.zeros((13,5))
        # get the average scores
        average_scores = np.zeros((8,))

        false_neg_ids = {
            1:[],
            2:[],
            3:[],
            4:[],
            5:[],
            6:[],
            7:[],
            8:[],
            9:[],
            10:[],
            11:[],
            12:[],
            13:[]
        }

        # create a tables
        category_eval_table = PrettyTable(["category","count","false_negative", "false_positive", "not_important","maybe_wrong"])
        results_table = PrettyTable(score_names)
        for k in range(0,len(results[0])):
            indices = results[:,k]>=0
            probs = results[indices,k]
            preds = probs>t
            target = labels[indices]
            test_files = list(compress(doc_ids,indices))

            results_row = []
            results_row.append(accuracy_score(target, preds))
            results_row.append(precision_score(target, preds, average="binary"))
            results_row.append(recall_score(target, preds, average="binary"))
            results_row.append(f1_score(target, preds, average="binary"))
            results_row.extend(confusion_matrix(target, preds).ravel())

            average_scores += np.array(results_row)
            results_row[0:4] = ["%.3f"%(val,) for val in results_row[0:4]]
            results_row[4:] = ["%d   %.3f"%(val,val/test_size) for val in results_row[4:]]
            results_table.add_row(results_row)

            category_eval = update_cat_eval(category_eval, false_neg_ids, test_files, preds)

        results_table.add_row(['']*len(score_names))
        average_scores /= len(results[0])
        average_scores = list(average_scores)
        average_scores[0:4] = ["%.3f"%(val,) for val in average_scores[0:4]]
        average_scores[4:] = ["%d   %.3f"%(int(val),val/test_size) for val in average_scores[4:]]
        results_table.add_row(average_scores)

        with open(join(results_path, filename+"_pred_eval_thres_%.2f.txt"%(t,)), 'w') as f:
            f.write(("Crossvalidation results for threshold %.2f"%(t,)).center(80) + "\n")
            f.write("\n"+legend+"\n\n")
            f.write(results_table.get_string())

        for i in range(len(category_eval)):
            row = [cat_id_to_des[i+1]]
            row.extend(category_eval[i,:])
            row[2:] = ["%d   %.3f"%(int(val),val/row[1]) for val in row[2:]]
            category_eval_table.add_row(row)
        a = ["total"]
        a.extend(np.sum(category_eval, axis=0))
        a[2:] = ["%d   %.3f"%(int(val),val/row[1]) for val in a[2:]]
        category_eval_table.add_row(a)

        with open(join(results_path, filename+"_category_eval_thres_%.2f.txt"%(t,)),"w") as cat_f:
            cat_f.write(("Crossvalidation results").center(80)+"\n")
            cat_f.write(("for different document categories").center(80)+"\n")
            cat_f.write(("for threshold %.2f"%(t,)).center(80) + "\n")
            cat_f.write("\n"+legend+"\n\n")
            cat_f.write(category_eval_table.get_string())
            cat_f.write("\n\n\n")
            cat_f.write("False negative Ids:\n\n")
            for key,vals in false_neg_ids.items():
                cat_f.write(cat_id_to_des[key]+"\n")
                for d_id_t in vals:
                    if(d_id_t[1]==1):
                        cat_f.write(d_id_t[0] + "\txml_error\n")
                    else:
                        cat_f.write(d_id_t[0] + "\n")
                cat_f.write("\n")

def compare_parameter_combinations_bin(classifier,
    comb_params,
    train_data,
    train_labels,
    test_data,
    test_labels,
    results_file,
    save_model=False):

    p_vals = []
    p_names = []
    for k,vals in comb_params.items():
        p_vals.append(vals)
        p_names.append(k)

    combinations = list(itertools.product(*p_vals))
    comb_names = []

    header = ["model", "eval_set", "accuracy", "precision", "recall", "f1", "tn", "fp", "fn", "tp"]
    results_table = PrettyTable(header)

    for i in range(len(combinations)):
        c_name = ""
        for j,n in enumerate(p_names):
            c_name+=(n+"-"+str(combinations[i][j])+"_")
            classifier.set_attribute(n,combinations[i][j])
        c_name = c_name[:-1]
        classifier.re_initialize()
        classifier.train(train_data, train_labels)

        for eval_set in [(train_data,train_labels,"train"), (test_data,test_labels,"test")]:

            preds = classifier.predict_binary(eval_set[0])
            results_row = [c_name,eval_set[2]]
            # add results
            results_row.append(accuracy_score(eval_set[1], preds))
            results_row.append(precision_score(eval_set[1], preds, average="binary"))
            results_row.append(recall_score(eval_set[1], preds, average="binary"))
            results_row.append(f1_score(eval_set[1], preds, average="binary"))
            results_row.extend(confusion_matrix(eval_set[1], preds).ravel())
            # format results
            results_row[2:6] = ["%.3f"%(val,) for val in results_row[2:6]]
            results_row[6:] = ["%d   %.3f"%(val,val/len(eval_set[1])) for val in results_row[6:]]
            results_table.add_row(results_row)
        results_table.add_row(['']*len(header))

        if(save_model):
            if(not(isdir("models"))):
                os.makedirs("models")
            classifier.save_model(join("models",c_name))

    # write results to file
    with open(results_file, 'w') as f:
        f.write(("Parameter Combination Results:").center(80) + "\n\n")
        f.write(("parameter order:").center(80) + "\n")
        for n in p_names:
            f.write((n).center(80) + "\n")
        f.write(results_table.get_string())

def update_cat_eval(category_eval, false_neg_ids, doc_ids, pred):

    false_docs = ["cc17fb8eaf055e2c8224eaaf3c10145c",
        "43d566f90c24f874781d97905870ed23",
        "96a92f3bcb002f22a698a62a468415e3",
        "9c70eaa380eb60135c5f1953b1f073cf",
        "6aaf86df732ef957de9a48e2b3c589e9",
        "daad7c32072aaa4cdbb9ae2c02c2c9b6",
        "e65752213cebcb6e151a815ed21512f3",
        "5ff6d40b1dc0dde4c51a8436bc57aa20",
        "fc8278775cf12fdf84e88493f8efc645",
        "d047a01a190236e10d6addefe13ed3bc",
        "7e7ff5a990f0e5756e31348ced169854",
        "5268494d6a5b7f3bc27bc2dcbcf79748",
        "10c8f8111c29862458987f77c05f4a39",
        "19e69897d4842c053d781e773e5122bd",
        "da5a198a45c690c3570a7d307597b892",
        "f51d621abeb4c489918d904995bbcec0",
        "ef59be13bd0d88b43fcb779912df7a66",
        "36889863bea6feb43dbe335719aa657a",
        "6c357bc6ddc11a3b5d199d2b1e4e6fac",
        "2ad710735830e95c3e9ce01725552a8b",
        "df59eaa1565f4b3d980c15a00ade3bdc",
        "826c1f6b7d9b2bd7fc50c23a126b71a1",
        "12d0d64889fe56c5789f8b73a8813ce9",
        "9d2cb33abdcfefa9a56bb1f7da4ef263",
        "a5322615e83b66d9b6c5e9a50b3eebb2",
        "57f98112812429c28def64a14020b242",
        "b37ce79bf563844426c7452c38564acc",
        "00a1e2413c919d6e392442533626cb92",
        "acda4e5d66d28ab616cd34fd09055b05",
        "ea6c27a337aff6c262e8316984ff72e5",
        "b2ac2fe5be5547e864f00b2d0f3d8003",
        "7f4038da9e1fb0aa745adcf8cc8cb832",
        "e5b0ee8a5519969ec70f5722a90a8019",
        "a774df02a832585275de3db40f132cdd",
        "580bebc528e28f520565e69ef6de8506",
        "5464ee4ef790013f0caa5d16d72ae7fc",
        "259f6473a159a782cc6f9aaf2ceda6a1",
        "057782ecdfc1f4179d2f98c4d80da89b",
        "06f3ad19c37c9f4460b0c6482505599b",
        "0a32570101750613de0fc032b94f73f4",
        "116eee145a0276280e82e0081bc39130",
        "136d718bee74eb407355cb64f064867d",
        "14c2d5d73ebc31122a07ab587272f82e",
        "170754478dc23a930d22722b43dbf181",
        "1799aad8a5a88b17cd8480fdecdd7b81",
        "3229f885e2668f522c2d64b84a34639b",
        "3981b966fa573e6419054a38660bbdda",
        "44fbbe38e3512475b9e590a937c3e933",
        "49f920671851f6a9010187bdc9743b2c",
        "590ce47a42bf67aa1295d02e85b1a7b5",
        "5c52e2433813f16b46264f07fed83783",
        "5f4616a15b26f79b6190b072dba11aa8",
        "6956ad338f080b4be0710ce32ca12976",
        "73197946354da526a4f08756a9703ddd",
        "74f8f36f354984572718e765d4e85189",
        "8b9b83971a91e40e8a05bd09d0138e49",
        "9100a9452a85c3732784074b41233a9c",
        "a4351807fee95ad4a750546b71b657fb",
        "ad3b3d728c8e579ef31efe61b9739c8a",
        "ae5378982b7227176c3e42dd93c58bd4",
        "c7cda18baa6c465a1a076bc6034b609f",
        "d5763c8901ef9e76bbef5e673145b300",
        "ef82cdd10e45d8184c8779361cce78ef",
        "7daba79637e41d82c9df63465ba2f0a3",
        "f39e2ed04d180820f05d6ff362aedd1c",
        "9dbb018b918a13f12613a063c386e2a7",
        "bf6a3ea6f15c2c9c96a19da3f3ba9e43",
        "38700021b4e8b2b6f34e19c47a096980",
        "fabe694fea65e5b15574f4eaa2a190a9",
        "059600524e5fa4fff9c72854ae7c70dc",
        "6fa54fffe2a2335f5927572e9757ee40",
        "d9eebb93c161bb0131245f80e8ccc0b3",
        "97bb3e396c0396a693f3f8cb2f3654b3",
        "4af0c6020d07663825f905c783828379",
        "3c16342b2169cc662f7f3c4b0e9289a1"
    ]

    xml_error = [
        "4d70b0a1973a94c98b50a9fd43f0ffba",
        "462e37b58d623e5458b8d630e4a4d993",
        "e52c862fe0b09a52c94dee89141925eb",
        "168f489c98b9976bbcce28f628e5078e",
        "dab670f904564b76c3463a4343594126",
        "afe02eb0714555c8176deaab05a5342f",
        "35d1120c963d214c331c3b036377a0cf",
        "e767a496ac211ec94a7f82f905a51393",
        "b98e74c526e1ecfc9ecafe848a21e147",
        "80946afd0a6c882d243ca3d3586a150d",
        "c26eb949d5bd40ded6a0711ef9af3426",
        "aafeaf6ff8f738c537bc8b4ebf012dde",
        "7451990e707f49fc90725036440c4434",
        "0b6b1cb9c9cd832bf02cf67bf8302e2c",
        "f16934cc36ac59d568502ac3a9d34cf3",
        "a4975f6d0d4ff4b8bbd0981ecd0f2285",
        "b8e2d6b46e3cb95bb185b371c1c95e1e",
        "6b5e9e9ce2e29547726ae8f8abd2ca82",
        "477e81c2aac2ff90b327e495366f8f86",
        "07a135fa0d07c49e0e6cd47810003dbc",
        "2312cef18bcf2755f37f1e08c5ce911e",
        "4c4829b9388c25f7ec2a001902e3d648",
        "dc55cb8299cd57bfdab9c7aff817d1eb",
        "81668dba8fec354eeb91024cdd663089",
        "154e7e64bf2119f2a5129991bdf4bf7c",
        "cd977ba21b6b8e6c8f2120e2c2881474",
        "a412dd22dd26aead36dc91026405abbb",
        "7ce1765b88bd1d1a78eca1e14fff62f9",
        "35ea1cd083532790f29aa9975cd6eb45",
        "152c77685b41801858fbcc76c145dab7",
        "3fb837679d044b2cf95a52e3370c2045",
        "f4e06729807af3fd0d162790a4a48ef4",
        "9f5832a1a3eec3ef2266e523f85c4322",
        "f1dfc962076e1f7f89cdaf8efb5cb314",
        "c6237fc82c1c2030e68df38de7d3871a",
        "618452d00a628665901bdc3151265e70",
        "0a78f6074fa498a9a184b3283cde7139",
        "6888a8187c69fc8d91e45828d280bfb7",
        "5ea358146448d79442e42e15233026c9",
        "8df3f8fc7b4c6aa07a0529f441120660",
        "a9e0db397b2e8b95931bffd8a56d5c18",
        "240c7b6e495076afaba00c9889aea082",
        "7b9778f8f7fb1c67783b82fb9bae42ab",
        "e51d2152efe6856a26ddd0adc845d1f0",
        "ca3359a20a73e30b109d0aed1a4b8711",
        "b7d354d689f5052419c45cf4414d8216",
        "8bd58c19720d9cc35d119f965b1ef601",
        "26d3adda46b9c565d327d8d810e6151a",
        "6f188eac23f755fdb358a7ce1136b6e7",
        "dfae946e4721c3470994c90bd5bd393e",
        "1b06d3a26375b06e7b79afb68f80b7da",
        "d47b36df95eb67841ad803256da587de",
        "7262cec0a47a711bb4f34391557219c9",
        "d6c6bdadbbcfe0f16f31ddfa76868d5c",
        "f81c611790ded32869499b298acc9d4e",
        "4fe72e59a686fd5dcddce7e5d8d2c99e",
        "8d072fc27449b5637453d45a9d75e2b0",
        "ee31588463e2469ee6216e9c1a32186d",
        "4a1c0babb340a6bd167a2600ba25e259",
        "5aaaa751412c557f4a1230d48d61614b",
        "56a2018de85943fa30cd4ddce3b9b602",
        "f0140465484930a469c8e08be23b107b",
        "d404503b466c0067fb52ea7fbb1eb99c",
        "7dc8171d1f40cb6d4ab849e0901f9989",
        "3cd73c97de8751d51309ffcca41a2ee5",
        "3848d5e5b76f7aa570105dc0c4e23ca9",
        "4053a8bd67e6b97033b1b1e379eb07b0",
        "efe95fb04937ce282415b3752296037f",
        "998b4efbfcc024e8284aef4f0d9eba87",
        "333c25f776348ecc4c824dfe9318ffe0",
        "1104c1ccb4037a458c7f1306cc8a4b38",
        "ab9cc9945400f56e598c7642120b4ac0",
        "cff2a79f90981ec250bdc460101e92d1",
        "353618076a78e6d84e3b16607693cade",
        "32e90d1459c534087a2daccdfc98453c",
        "f38a244eb0ef01f05e8d9f88d4c15286",
        "6e6a8826597f72e5403a00f1fc43fea2",
        "f677954aab79b144b33a8539a80b5021",
        "e6e55e34f7f10acf95bf72e8f64adc9a",
        "f90fd5e8e62b6216f9575e61f3f8ef45",
        "f25e883b16833c800a297e4d93b0d553",
        "0778080d975a2f66a5bd48b249baab5f",
        "6ce159fb4aee9b927c2157086f32f925",
        "f4e6c77cc5614c657ce54a481fd0f362",
        "ec932748ed8dc56134253c82d61c101d",
        "4443af1116ccc197989be849d9f41aec",
        "2cd9220978624e281e0e3245d9d0fca5",
        "7fde22aa84dcb4b2cc9e52a480507535",
        "95cc939db7100de396da394317e28d1c",
        "e5697f2a78bc48b1eb0252fd2ef7279a",
        "de6d3ae71d8b587fb5217209fac784d8",
        "87245bf63324cf33d4aef86b13c7b0eb",
        "e95621e72ff16ef5048d3626592af65b",
        "7a623019662b3bebf5f76771051dd880",
        "97383cd50190b061553cd73973bce984",
        "f752c531b9c64ee61a33fc4452ecb976",
        "fabcd583132b6eb5d12812726f17bedb",
        "9c93f5b188319d1bddaed4171d0cf7cf",
        "f41d85f790ac6c88d3d66cd876ff0b85",
        "8d196e2c05c7e02d5117302fd7fa6182",
        "170754478dc23a930d22722b43dbf181",
        "0d6f103a9fa711786c797fb1b46e8e22",
        "6edf443e2586f9e0c91d190a9a22f2e5",
        "1ce91be84b2e5643eae607d049c2912d",
        "e3df65c8481ecd9b72f072a065bac4a6",
        "a7f54c58f810e2ea9f95ec28aad08f8b",
        "e84d829645112856c3be0b408e783115",
        "08576fc10f451db0b6161718f7ab9a8d",
        "9b42880245d1b42d5771bc2bffc3b76a",
        "6032b50d503896c3fd7d9cd31d2e0447",
        "cc375cc82ad35b12b14062163670b342",
        "b4610d02c99e9c27742441d36655f1ba",
        "0306420da9829eef44f32ea96d45d568",
        "9a791ce480ed18604f49553e7a992bab",
        "15c4bae8e5f664d9ec481ee6385f8a25",
        "74c3870963becf2b891ef2574bfcb319",
        "1a5d44eb741c693cb626198fc63b50d5",
        "f6d2282e1aa5ff337ec129a553535e67",
        "ad9ade6d7b101edcb4bb9849be69074c",
        "a2835d03b2d55cd13c406106eee9aaf1",
        "c09aaa06323524e14a734b6c711f443f",
        "01ee82f2ec05c0e273d229b5e9412379",
        "6c357bc6ddc11a3b5d199d2b1e4e6fac",
        "11f66b9e33b3773d1039ae28d45183e5",
        "1102816909e4050340e4269631e3a57e",
        "8b5d679b3ad0d2a3bf85a22c4c3fd47c",
        "835508355d5f6284a0c58a1873a90a2a",
        "c7909f905d3376d1d50e306a23a28e60",
        "48c81c9e65935473cbd4c206b023c673",
        "071e9a3c2811f70db3bac220aaa325bb",
        "843b7382bd6ec44dd2bb4c958d4a7751",
        "e4d437e678dc3c9da8b6695d3ff8d7e2",
        "9a5d3349755cdbf1fc71dbee3e36ac37",
        "68ad6882f7cd47437e58108e716cbcad",
        "9a599a19d1e277137e3f083c255a62e3",
        "ddcf3afdc941865b12bdd2af9c055ef6",
        "9cff5d5a04b718f20ca4eecc93e64d8b",
        "9b2b774ef4c39fc30384a0bf0a40f600",
        "8dbff54b538bfe88c04c33122b3bca38",
        "51acd0ea605794b421781e475e322c9a",
        "341343a4fba4b858c1c916cccf9b9cec",
        "d87cc194527e7d6c5e6a8b4037046394",
        "458c3b916b58fbd0d6efe085fb2ab2b0",
        "41d760f50056d2a3bbf086047492c436",
        "6d761eb389d6d0fddf0ee72dc384cf2e",
        "e1cef7747907cecbce8c76026eb2ca06",
        "e33e7c75475d7073160d6058053f83e6",
        "a75d7757a9c31c9dab10c2c8e4c34f89",
        "404d8dce32be7dcb2e4160aa78c44de6",
        "a8c89340f76e1798d1bb9726d91b7cb8",
        "24c9893770471a031e2c5d4619ecc995",
        "13374318ead52e16969c0d0544d7c596",
        "bb88fe6c0328912f0bb90beb4fb87109",
        "206297c4a808111bcee4d769dbb43385",
        "d987c51792fcaa51a8916b6687b09d81",
        "08199c77dbbcd19a24cf4bda154a29da",
        "5ce4ae00d419a61b987b8abfcc931b25",
        "a0240d7f82990009412be04369d5c7d1",
        "e5178ff03bb21e40281c083021dd5d80",
        "9478bb43e1060f6747290d0ef04b806c",
        "203a02d423548260c477a5c1e56c40c6",
        "66fc6ce18b44145964a06d8ff9db3ea1",
        "38d35d6dddf9aa3c5ad647563a1c7b6a",
        "56db3ea9339a2a2d8424804dd3842641",
        "c125a84aa5ac9216987a9e1f725b8e64",
        "325cc5f8fea90ec6455c1fc90805a801",
        "20ed88098c997adade05ebcc5855c20c",
        "e63baff797c0f1755888a2cbb22c4935",
        "f264c2d15410ab35c314b679674c0109",
        "008bd1a8b44ff03dc160552cf0514a92",
        "62c206e1238acf99b33906437504996f",
        "3e039943e16f3eddea67fa71335fe052",
        "70fa622e70974dbf3314adfa4ad68514",
        "c2a992d1b181ce386a342e6324131ac3",
        "9fca5e8d73098cb0bbd103ae77b16030",
        "c35ec2d4552a950069b390101bbbd27d",
        "21f0fb0e872f9cc83ebb69f0890fe1c9",
        "8441e472c36d966dc9d3f0ed9ed767ba",
        "f1abde355dac04bbd21d33574de0c299",
        "664c7b944a1799b494001ebcffd1a312",
        "24549c324d4138b4aabe2e69da3143b9",
        "0880622c454aa613513589c9957b85a9",
        "6487721de7c5bd3f50257e8931020606",
        "14a14bbc72c5ad5c4ef75714fc27a2ca",
        "f2dadc933186ba967ce56853e9242bc5",
        "f7ba2b00060e70cede9eda22f8323e7f",
        "85bf12bd6c668033a3d730f3926b0640",
        "ec5951bec760e3d149092f0cadd71119",
        "7c1b592a0b256f3bb61e42c6a0260022",
        "eee07c5ef8c9d49c5cbb4ae689b76179",
        "a69acaacc655eeadd3f726c1f2fe4dd1",
        "18111af931025656bdf4581bd27b268e",
        "c9e4d8556c070d903cb8b58cfa57e90a",
        "46fe2f38e7e10be55cecc7e87599e88a"
    ]

    cat_path = "/home/kai/Workspace/deep_doc_class/deep_doc_class/data/classification_with_category.csv"
    cat_dict = {}

    with open(cat_path,"r") as cat_file:
        reader = csv.DictReader(cat_file)
        for row in reader:
            cat_dict[row["doc_id"]]=row

    for i in range(len(doc_ids)):
        d_id = doc_ids[i]
        # if(d_id in false_docs):
        #     continue
        label = int(cat_dict[d_id]["class"])
        cat = int(cat_dict[d_id]["category"])-1
        imp = int(cat_dict[d_id]["important"])
        p = pred[i]
        category_eval[cat,0]+=1
        if(label and not(p)):
            false_neg_ids[cat+1].append([d_id,0])
            category_eval[cat,1] +=1
            if(not(imp)):
                category_eval[cat,3]+=1
            if(d_id in xml_error or d_id in false_docs):
                false_neg_ids[cat+1][-1][1]=1
                category_eval[cat,4]+=1
        elif(not(label) and p):
            category_eval[cat,2] +=1
            if(not(imp)):
                category_eval[cat,3]+=1
            if(d_id in xml_error):
                category_eval[cat,4]+=1
        else:
            continue

    return category_eval


##### Preprocess different kinds of input #####
def preprocess_pdf_property_string(text):
    if(text is None):
        return"None"
    else:
        text = text.lower()
        text = " ".join(re.findall("[a-z]{2,}",text))
        #text = clean_string_regex(text, regex='[^\ba-z]', sub="")
        return text

def preprocess_pdf_text_string(text):
    text = remove_whitespace(text)
    # words = find_regex(text)
    words = remove_stopwords(text.split())
    text =  " ".join(words)
    return text

def preprocess_pdf_metadata_string(text, lang=['german','english']):
    if(text is None or (type(text) is float and math.isnan(text))):
        return""
    else:
        words = find_regex(text, regex=r'(?u)\b\w\w\w+\b')
        words = remove_stopwords(words)
        text = " ".join(words)
        text = text.replace("_", " ").lower()
        return text

def clean_string_regex(txt, regex=';|-|\.|,|\"|[0-9]', sub=""):
    txt = txt.lower()
    txt = re.sub(regex, sub, txt)
    return txt

def remove_whitespace(txt):
    txt = re.sub("\s", " ", txt)
    return txt

def find_regex(txt, regex=r'(?u)\b\w\w\w+\b|©'):
    words = re.findall(regex,txt)
    return words

def remove_stopwords(words):
    languages = ["english", "german", "french"]
    for language in languages:
        stop_words=set(stopwords.words(language))
        words=[w for w in words if not w in stop_words]
    return words

# unused helper functions
def lemmatizer(txt, taggerdir, taggerlang):
    pass
    # tt = ttwp.TreeTagger(TAGDIR=taggerdir, TAGLANG=taggerlang)
    # taglist = tt.tag_text(txt)
    # lemmalist = []
    # for tag in taglist:
    #     lemmalist.append(tag.split('\t')[2])
    # return lemmalist

def get_lang(txt, get_prob=False):
    if(get_prob):
        langs = detect_langs(txt)
        language = langs[0]
        return language.lang, language.prob
    else:
        language = detect(txt)
        return language

def remove_bad_doc_ids(doc_ids, classes, bad_ids):
    cnt = 0
    pos_cnt = 0
    neg_cnt = 0

    for i in range(len(doc_ids)-1,-1,-1):
        if(doc_ids[i] in bad_ids):
            cnt += 1
            del doc_ids[i]
            if(classes[i]):
                pos_cnt += 1
            else:
                neg_cnt += 1
            del classes[i]

    print(cnt,pos_cnt,neg_cnt)
    return doc_ids, classes

def remove_bad_strings(doc_ids, classes, clean_strings, bad_strings):
    cnt = 0
    pos_cnt = 0
    neg_cnt = 0

    for i in range(len(doc_ids)-1,-1,-1):
        if(clean_strings[i] in bad_strings):
            cnt += 1
            del clean_strings[i]
            del doc_ids[i]
            if(classes[i]):
                pos_cnt += 1
            else:
                neg_cnt += 1
            del classes[i]

    print(cnt,pos_cnt,neg_cnt)
    return doc_ids, classes, clean


if __name__ == "__main__":
    sys.path.append("/home/kai/Workspace/deep_doc_class/deep_doc_class/src")

    args = sys.argv
    train_file = args[1]
    test_file = args[2]

    train = pd.read_csv(train_file, delimiter=',', header=0, quoting=1)
    train.columns = ["document_id", "class"]
    doc_ids = list(train["document_id"])
    classes = list(train["class"])

    test = pd.read_csv(test_file, delimiter=',', header=0, quoting=1)
    test.columns = ["document_id", "class"]
    doc_ids += list(test["document_id"])
    classes += list(test["class"])

    for i in range(len(doc_ids)-1,-1,-1):
        if(doc_ids[i] in false_docs):
            del doc_ids[i]
            del classes[i]

    features = []
    features.append(BowClassifier("filename", None, None))
    features.append(BowClassifier("folder_name", None, None))
    features.append(BowClassifier("creator", None, None))
    features.append(BowClassifier("producer", None, None))
    features.append(BowClassifier("bold_text", None, None))
    features.append(BowClassifier("first_100_words", None, None))
    features.append(BowClassifier("last_100_words", None, None))
    features.append(BowClassifier("text", None, None))

    # for f in features:
    #     print(f.name)
    #     for vect in ["vectorizer", "custom"]:
    #         print(vect)
    #         for classifier in ["forest"]:
    #             print(classifier)
    #             f.crossvalidate(doc_ids, classes, vect, classifier)
    #             # print(f.vectorizer.get_feature_names())
    #             # print(len(f.vectorizer.get_feature_names()))
    print("Starting crossvalidation")
    for f in features:
        # f.analyze_word_distribution(f.load_clean_data(doc_ids), np.array(classes))
        f.crossvalidate(doc_ids, classes, "custom", "forest")
        pause()
