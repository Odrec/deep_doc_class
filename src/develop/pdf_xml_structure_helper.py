import sys, os, json, shutil, codecs, csv
from os.path import join, realpath, dirname, isdir, basename, isfile, splitext

SRC_DIR = os.path.abspath(join(join(realpath(__file__), os.pardir),os.pardir))
if(not(SRC_DIR in sys.path)):
    sys.path.append(SRC_DIR)

from time import time
import subprocess
from PIL import Image as PI

import features.pdf_xml_structure
import pdf_images

from doc_globals import*
import numpy as np
np.seterr(all='raise')

import pyocr
import pyocr.builders
import io

from shutil import copy

TMP_DIR = join(DATA_PATH,"tmp2")
FNULL = open(os.devnull, 'w')

def pause():
    """
    Pause the execution until Enter gets pressed
    """
    input("Press Enter to continue...")
    return

###### Image Stuff #######
def get_text_from_pil_img(pil_image, lang="deu"):
    if(not(lang in ["eng", "deu", "fra"])):
        print("Not the right language!!!\n Languages are: deu, eng, fra")

    tool = pyocr.get_available_tools()[0]

    txt = tool.image_to_string(
        pil_image,
        lang=lang)

    return txt

def img_stuff(img_list):
    entropy = []
    color = False
    for img_path in img_list:
        if(not(isfile(img_path))):
            continue
        pil_image = PI.open(img_path)
        # with PI.open(io.BytesIO(page)) as pil_image:
        gs_image = pil_image.convert("L")
        hist = np.array(gs_image.histogram())
        hist = np.divide(hist,np.sum(hist))
        hist[hist==0] = 1
        e = -np.sum(np.multiply(hist, np.log2(hist)))
        entropy.append(e)

        # if(not(color)):
        #     col_image = pil_image.convert('RGB')
        #     np_image = np.array(col_image)
        #     if((np_image[:,:,0]==np_image[:,:,1])==(np_image[:,:,1]==np_image[:,:,2])):
        #         color = True

        os.remove(img_path)

    if(len(entropy)==0):
        print("Zero images loaded. Either pdf is empty or Ghostscript didn't create images correctly.")
        mean_entropy = np.nan
    else:
        mean_entropy = np.mean(entropy)
    return mean_entropy, color

def checkDictForNeg(check_dict):
    neg_feat = []
    for feat,val in check_dict.items():
        try:
            num_val = float(val)
            if(num_val<0):
                neg_feat.append(feat)
                continue
        except ValueError:
            continue
        except Exception as e:
            print(e)
            continue
    return neg_feat

def load_single_property(doc_ids, properties_path, field):

    # get pdfinfo dict information
    structure_data = None
    properties = []
    with open(properties_path,"r") as f:
        structure_data = json.load(f)
    for doc_id in doc_ids:
        try:
            properties.append(structure_data[doc_id][field])
        except:
            print("No structure data: " + doc_id)
            properties.append("")

    return properties

def show_document_page(filename, page):
    page_arg = "--page-label=" + str(page)
    args = ["evince", "--fullscreen", page_arg, filename]
    plot = subprocess.Popen(args, stdout=FNULL, stderr=subprocess.STDOUT)
    return

# def extract_training_images(doc_dir, img_dir, doc_ids):
#     ## want to find:
#     # clear textblocks : 1/0
#     # multiple columns : 1/0
#     # blockformatted : 1/0
#     # table : 1/0
#     # list: 1/0
#     # pagenumbers : 1/0
#     # clear artifacts : 1/0
#     # handwritten : 1/0
#
#     # telling copyright frontpage : 1/0
#     # telling university frontpage : 1/0
#     # closing references : 1/0
#
#     c = 0
#     pred_data = {}
#     files = []
#     if isdir(doc_dir):
#         if(doc_ids is None):
#             for root, dirs, fls in os.walk(doc_dir):
#                 for name in fls:
#                     if splitext(basename(name))[1] == '.pdf':
#                         files.append(join(root,name))
#     for pdfFile in files:
#         c+=1
#         #pdfFile = join(doc_dir,d_id+".pdf")
#         d_id = splitext(basename(pdfFile))[0]
#         doc = Document(pdfFile)
#         err_message = doc.process_xml(img_flag=False)
#         show_document_page(pdfFile, 1)
#         if(not(len(doc.pages)>0)):
#             pred_data[d_id] = "pw_protected"
#             continue
#         print(doc.pages[0].width)
#         print(doc.pages[0].height)
#         for key,val in doc.stat_lists.items():
#             if(key in ["textbox_columns","blockstyle_columns","lines_pbs","lines_ptb"]):
#                 print(key)
#                 print(val)
#         most_text_page = np.argmax(doc.stat_lists["words_pp"])
#         last_page = len(doc.stat_lists["words_pp"])
#         show_document_page(pdfFile, 1)
#
#         doc_data = {}
#         a = "1"
#         while(not(len(a)==2)):
#             a = input()
#             if(a=="n"):
#                 pred_data[d_id] = c
#                 with open('new_data.json', 'w') as outfile:
#                     json.dump(pred_data, outfile)
#                 sys.exit(1)
#         doc_data["clear_cp_titlepage"] =a[0]
#         doc_data["clear_ncp_titlepage"] =a[1]
#
#         # for i in range(2,last_page):
#
#         show_document_page(pdfFile, most_text_page)
#             a = "1"
#             while(not(len(a)==5)):
#                 a = input()
#                 if(a=="n"):
#                     pred_data[d_id] = c
#                     with open('new_data.json', 'w') as outfile:
#                         json.dump(pred_data, outfile)
#                     sys.exit(1)
#
#             doc_data["clear_textblocks"] =a[0]
#             doc_data["multiple_columns"] =a[1]
#             doc_data["blockformatted"] =a[2]
#             doc_data["list"] =a[3]
#             doc_data["table"] =a[4]
#             doc_data["pagenumbers"] =a[5]
#             doc_data["clear artifacts"] =a[6]
#             doc_data["handwritten"] =a[7]
#
#
#         show_document_page(pdfFile, last_page)
#         a = "1"
#         while(not(len(a)==1)):
#             a = input()
#             if(a=="n"):
#                 pred_data[d_id] = c
#                 with open('new_data.json', 'w') as outfile:
#                     json.dump(pred_data, outfile)
#                 sys.exit(1)
#         doc_data["closing_references"] =a[0]
#
#         pred_data[d_id] = doc_data

def generate_blockstyle_training_images():
    categories = {1:"scanned",
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
    13:"unsure"}

    allowed = [3,4,5,8,10,13]
    header = ["doc_id","page","block"]

    with open(join(SRC_DIR,"features","time.log"), 'r') as fp1:
        time_data = json.load(fp1, encoding="utf-8")
    with open(join(SRC_DIR,"features","already_classified.log"), 'r') as class_log:
        class_log_data = json.load(class_log)
    with open(join(DATA_PATH,"classification_with_category.csv")) as csvfile:
        reader = csv.reader(csvfile)
        class_cat_data = []
        next(reader)
        for row in reader:
            class_cat_data.append(row)
    with open(join(DATA_PATH,"blockstyle_class.csv")) as csvfile2:
        reader = csv.reader(csvfile2)
        blockstyle_data = []
        for row in reader:
            blockstyle_data.append(row)
    for _,doc_id,_,_,cat,_ in class_cat_data:
        cat = int(cat)
        block_processed = doc_id in class_log_data and "block" in class_log_data[doc_id]
        if(not(cat in allowed) or doc_id in time_data or block_processed):
            continue
        abs_filepath = join(PDF_PATH,doc_id+".pdf")
        blockstyles = extract_blockstyle_information(abs_filepath, doc_id, categories[cat])
        class_log_data[doc_id] = ["block"]
        # generate_images(abs_filepath, doc_id)
        for i,bs in enumerate(blockstyles):
            blockstyle_data.append((doc_id,i+1,bs))
        with open(join(DATA_PATH,"blockstyle_class.csv"), 'w') as csvfile3:
            writer = csv.writer(csvfile3, delimiter=",")
            writer.writerows(blockstyle_data)
        with open(join(SRC_DIR,"features","already_classified.log"), 'w') as class_log:
            json.dump(class_log_data, class_log)

def extract_blockstyle_information(abs_filepath, doc_id, cat):
    (f_dict, abs_filepath, doc) = get_structure_features(abs_filepath, False)
    print("class: {} \t id: {} \t pages: {}".format(cat,doc_id,len(doc.pages)))
    blockstyles = (np.ones(len(doc.pages))*-1)
    if(len(doc.pages)<=3):
        show_document_page(abs_filepath, 1)
        for i in range(len(doc.pages)):
            confirm = input("none?")
            blockstyles[i] = 0
            if(confirm=="u"):
                blockstyles[i] = -1
                print("undefined!")
            elif(confirm!=""):
                blockstyles[i] = 1
                print("block!")
        doc.clean_files()
        return blockstyles.astype(int)

    show_document_page(abs_filepath, 1)
    print("EASY")
    show_none = input("show none?")==""
    show_block = input("show block?")==""

    none_easy_ones = []
    for i in range(len(doc.pages)):
        p_num = i+1
        guess = "unsure"
        if(doc.stat_lists["blockstyles_lines_pp"][i] >  min(6,doc.stat_lists["not_blockstyles_lines_pp"][i])):
            guess = "block"
        elif(doc.stat_lists["blockstyles_lines_pp"][i] <= doc.stat_lists["not_blockstyles_lines_pp"][i]/4):
            guess = "none"
        print("page: {} \t block: {} \t none: {} \t guess: {}".format(p_num,doc.stat_lists["blockstyles_lines_pp"][i], doc.stat_lists["not_blockstyles_lines_pp"][i], guess))

        if(guess=="block"):
            blockstyles[i] = 1
            if(show_block):
                show_document_page(abs_filepath, p_num)
                confirm = input("block?")
                if(confirm=="u"):
                    blockstyles[i] = -1
                    print("undefined!")
                elif(confirm=="all none"):
                    blockstyles[i:] = 0
                    break
                elif(confirm=="all undef"):
                    blockstyles[i:] = -1
                    break
                elif(confirm!=""):
                    blockstyles[i] = 0
                    print("none!")
        elif(guess=="none"):
            blockstyles[i] = 0
            if(show_none):
                show_document_page(abs_filepath, p_num)
                confirm = input("none?")
                if(confirm=="u"):
                    blockstyles[i] = -1
                    print("undefined!")
                elif(confirm=="all none"):
                    blockstyles[i:] = 0
                    break
                elif(confirm=="all undef"):
                    blockstyles[i:] = -1
                    break
                elif(confirm!=""):
                    blockstyles[i] = 0
                    print("block!")
        else:
            blockstyles[i] = -1
            none_easy_ones.append(i)
    if(len(none_easy_ones)==0):
        doc.clean_files()
        return blockstyles.astype(int)

    print("\n\n\n {} TRICKY CASES".format(len(none_easy_ones)))
    if(input("Judge?")!=""):
        doc.clean_files()
        return blockstyles.astype(int)
    show = input("Show?")==""
    default = int(input("default? Enter for None")!="")
    input_val = "block?" if default else "none?"
    changed_val = "none!" if default else "block!"

    for i in none_easy_ones:
        print("page: {} \t block: {} \t none: {}".format(i+1,doc.stat_lists["blockstyles_lines_pp"][i], doc.stat_lists["not_blockstyles_lines_pp"][i]))
        blockstyles[i] = default
        if(show):
            show_document_page(abs_filepath, i+1)
            confirm = input(input_val)
            if(confirm!=""):
                blockstyles[i] = int(not(default))
                print("changed!")
    doc.clean_files()
    return blockstyles.astype(int)

def find_neg_values(json_file):
    neg_dict = {}

    with open(json_file, 'r') as fp:
        structure_data = json.load(fp, encoding="utf-8")
    for key,doc in structure_data.items():
        for feat,val in doc.items():
            try:
                num_val = float(val)
                if(num_val<0):
                    if(not(feat in neg_dict)):
                        neg_dict[feat] = []
                    neg_dict[feat].append(key)
            except ValueError:
                continue
            except Exception as e:
                print(e)
                input()
                continue
    with open("../../data/neg_dict.json", 'w') as fp:
        json.dump(neg_dict, fp)
    return neg_dict

def find_blockstyle(json_file):
    neg_dict = {}

    with open(json_file, 'r') as fp:
        structure_data = json.load(fp, encoding="utf-8")
    for key,doc in structure_data.items():
        for feat,val in doc.items():
            try:
                num_val = float(val)
                if(num_val<0):
                    if(not(feat in neg_dict)):
                        neg_dict[feat] = []
                    neg_dict[feat].append(key)
            except ValueError:
                continue
            except Exception as e:
                print(e)
                input()
                continue
    with open("../../data/neg_dict.json", 'w') as fp:
        json.dump(neg_dict, fp)
    return neg_dict

def find_neg_doc_ids(json_file):
    neg_list = []
    with open(join(SRC_DIR,"features","time.log"), 'r') as fp1:
        time_data = json.load(fp1, encoding="utf-8")

    with open(json_file, 'r') as fp:
        structure_data = json.load(fp, encoding="utf-8")
    for key,doc in structure_data.items():
        for feat,val in doc.items():
            try:
                num_val = float(val)
                if(num_val<0):
                    t_key = key+".pdf"
                    if((t_key in time_data) and time_data[t_key]>1000):
                        print(t_key, time_data[t_key], feat, val)
                    else:
                        neg_list.append(key)
                    break
            except ValueError:
                continue
            except Exception as e:
                print(e)
                input()
                continue
    return neg_list

def check_extraction_neg(neg_dict):
    # with open("../../data/neg_dict.json", 'w') as fp:
    #     neg_dict  = json.load(fp)
    # for key, id_list in neg_dict.items():
    # print(neg_dict.keys())
    key = "min_image_space_pp"
    # id_list = neg_dict[key]
    # take_id = id_list[0]
    take_id = "8441e472c36d966dc9d3f0ed9ed767ba"
    print(take_id)
    print(key)
    if(not(isdir(TMP_DIR))):
        os.makedirs(TMP_DIR)
    if(not(isdir(join(DATA_PATH,"files_test_neg")))):
        os.makedirs(join(DATA_PATH,"files_test_neg"))
    copy(join(DATA_PATH,"pdf_files",take_id+".pdf"),join(DATA_PATH,"files_test_neg"))
    (f_dict, abs_filepath) = get_structure_features(join(DATA_PATH,"files_test_neg",take_id+".pdf"))
    print(f_dict[key])
    # input()
    # print(f_dict)

if __name__ == "__main__":
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

    doc_dict = {
        "powerpoint1" : "76ae7c120910a7830a1c0e0262d8cc5e.pdf",
        "powerpoint2" : "15a25c0553f25d4edcc288028b384cba.pdf",
        "slide_overview1" : "1e733d496d75352df67daefe269c1e88.pdf",
        "slide_overview2" : "5e9f1f979fff677b72e6228ded542a97.pdf",
        "lecture_mats1" : "1fd731e88a30612291de1923d5fa5263.pdf",
        "lecture_mats2" : "26286ffa140c615eb9a5a4ab46eb30be.pdf",
        "lecture_mats3" : "185422bf26436452aa0b3b8247e322af.pdf",
        "book_style1" : "37cf51662d8385b47ad00d36070766b0.pdf",
        "scan1" : "1569e3de486040aaaf71653c8e4bee6d.pdf",
        "scan2" : "c92b478470f9147ea02229ac7de22adc.pdf",
        "scan3" : "0b561e7ffe8da5a589c7e33e55203de6.pdf",
        "table1" : "1945835eac5a4162cc00ba89c30e6a90.pdf",
        "paper_style1" : "0f6b08591d82390c4ad1a590266f92bb.pdf",
        "long_doc" : "c41effe246dd564d7c72416faca33c21.pdf",
        "pw_protect" : "26013961e27e976cf7dff7b5bc6086c6.pdf",
        "warning" : "db8f79a0bbe5bb518a54eb92f5b9b499.pdf",
        "image_problem" : "d659934e59b496a11b7bffb22eabbba9.pdf"
    }

    # pdf1 = "ca13bf1246f3d4f1eac8362729a17c3e"
    # pdf2 = "353d6d426b03277d82f5b1f526101a86"
    # pdf3 = "ea9549a004b66ffa18cd892456adf43c"
    pdf4 = "b713d099025ca4e80a7cf23fc37fb84e"
    get_structure_features(join(PDF_PATH,pdf4+".pdf"))
    # generate_blockstyle_training_images()

    # find_neg_values("../../data/pre_extracted_data/xml_text_structure_new.json")
    # neg_list = find_neg_doc_ids("../../data/pre_extracted_data/xml_text_structure_new.json")
    # check_extraction_neg(neg_list)

    # test_path = join(DATA_PATH,"files_test_html")
    #
    # new_doc = "0f6b08591d82390c4ad1a590266f92bb"
    #
    # # show_document_page(join(test_path,new_doc+".pdf"),3)
    #
    # extract_training_images(test_path, None, None)

    #get_structure_features(join(test_path,new_doc+".pdf"))

    # for xe in xml_error:
    #     print(xe)
    #     get_structure_features(join(test_path,xe+".pdf"))

    # # docs = [lecture_mats1, lecture_mats2, book_style1, table1, paper_style1]
    # docs = [doc_dict["pw_protect"]]
    #
    # doc_ids = [splitext(d)[0] for d in doc_dict.values()]
    #
    # # print(get_structure_features("../../data/pdf_files/37cf51662d8385b47ad00d36070766b0.pdf"))
    #
    # neg_list = ["8441e472c36d966dc9d3f0ed9ed767ba"]
    # s1 = time()
    # pre_extract_pdf_structure_data_to_file(
    #     doc_dir="../../data/pdf_files",
    #     text_dir="../../data/xml_text_files_new",
    #     structure_file="../../data/pre_extracted_data/xml_text_structure_new.json",
    #     doc_ids=neg_list,
    #     num_cores=None,
    #     batch_size=10)
    # print(time()-s1)


    # print(img_stuff(image_pathes))
    # # show time discrepancy
    # st = time()
    # features.pdf_structure.process_file(doc)
    # print(time()-st)

    # st = time()
    # print(develop.pdf_images.get_grayscale_entropy_tmpfile(doc))
    # print(time()-st)
