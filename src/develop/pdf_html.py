import sys, os
from collections import Counter
from os.path import join, realpath, dirname, isdir, basename, isfile, splitext
import warnings

SRC_DIR = os.path.abspath(join(join(realpath(__file__), os.pardir),os.pardir))
sys.path.append(SRC_DIR)
from time import time
import subprocess
import xml.etree.ElementTree as ET
from PIL import Image as PI

# import features.pdf_structure
import develop.pdf_images
from doc_globals import*
import numpy as np

import pyocr
import pyocr.builders
import io
import re

bcolors = {
    "HEADER" : '\033[95m',
    "OKBLUE" : '\033[94m',
    "OKGREEN" : '\033[92m',
    "WARNING" : '\033[93m',
    "FAIL" : '\033[91m',
    "ENDC" : '\033[0m',
    "BOLD" : '\033[1m',
    "UNDERLINE" : '\033[4m'
}

def print_bcolors(formats, text):
	"""
	Add console formatting identifer to strings.

	@param formats: a list of formats for the string (has to be in the dict bcolors)
	@dtype formats: list(String)

	@param text: the string should be formatted
	@dtype text: string

	@return formated_text: the formatted string
	@dtype formated_text: string
	"""
	formated_text = ''
	for format in formats:
		formated_text += bcolors[format]
	formated_text += text + bcolors["ENDC"]
	return formated_text

def pause():
    """
    Pause the execution until Enter gets pressed
    """
    input("Press Enter to continue...")
    return

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

def process_page_content(xml_page, fontspecs, images):
    page_lines = []
    tops = {}
    lefts = {}
    rights = {}
    page_line_index = 0

    for elem in page:
        if(elem.tag=="fontspec"):
            fontspecs["famlilies"].add(elem.attrib["family"])
            fontspecs["colors"].add(elem.attrib["color"])
            fontspecs["spec_counts"].append(0)
            fontspecs["sizes"].append(int(elem.attrib["size"]))
            prev_text = None
        elif(elem.tag=="image"):
            images["pathes"].append(elem.attrib["src"])
            images["sizes"].append(int(elem.attrib["height"])*int(elem.attrib["width"]))
            prev_text = None
        elif(elem.tag=="text"):
            # create a text tuple containing the location, font and text information

            text_iter = elem.iter()
            text_elem = next(text_iter)
            text_list = [
                int(text_elem.attrib["top"]),
                int(text_elem.attrib["left"]),
                int(text_elem.attrib["width"]),
                int(text_elem.attrib["height"]),
                int(text_elem.attrib["font"]),
                (text_elem.text + " ") if not(text_elem.text is None) else '',
                0]

            format_set = set()
            for format_elem in text_iter:
                if(not(format_elem.text is None)):
                    text_list[5] += format_elem.text + " "
                    format_set.add(format_elem.tag)

            if(len(format_set)>0):
                if(len(format_set)>1):
                    text_list[6] = 4
                else:
                    formatter = format_set.pop()
                    if(formatter=="b"):
                        text_list[6] = 1
                    elif(formatter=="a"):
                        text_list[6] = 2
                    elif(formatter=="i"):
                        text_list[6] = 3
                    else:
                        warnstring = "Unexpected text formatter found, namely %s"%(format_set[0],)
                        print(print_bcolors(["WARNING"], warnstring))

            text_list[5] = text_list[5][:-1]
            if(len(text_list[5])>0):
                if(text_list[5]==" "):
                    continue
                text_list[5] = text_list[5]
                fontspecs["spec_counts"][text_list[4]] += 1
                page_lines.append(text_list)

                try:
                    tops[text_list[0]].append(page_line_index)
                except KeyError:
                    tops[text_list[0]] = [page_line_index]
                try:
                    lefts[text_list[1]].append(page_line_index)
                except KeyError:
                    lefts[text_list[1]] = [page_line_index]
                try:
                    rights[text_list[1]+text_list[2]].append(page_line_index)
                except KeyError:
                    rights[text_list[1]+text_list[2]] = [page_line_index]

                page_line_index += 1
        else:
            warnstring = "Unexpected tag found, namely %s"%(elem.tag,)
            print(print_bcolors(["WARNING"], warnstring))
            print(elem.tag,elem.attrib,elem.text)
    return page_lines, tops, lefts, rights

def get_xml_structure(filepath, img_flag=True):
    args = ["pdftohtml"]
    if(img_flag):
        args += ["-i"]
    args += ["-xml", filepath]
    output = subprocess.Popen(args, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE).communicate()[0].decode(errors='ignore')

    xml = splitext(filepath)[0] + ".xml"
    tree = ET.parse(xml)
    return tree

def check_multiple_lines(line_idc, text_lines, tops, lefts, rights):
    lines = [(text_lines[l_id],l_id)for l_id in line_idc]
    sorted_lines = sorted(lines, key=lambda x: x[0][1], reverse=False)
    merges = []
    for pos in range(1,len(sorted_lines)):
        formatted = sorted_lines[pos-1][0][6]>0 or sorted_lines[pos][0][6]>0 or not(sorted_lines[pos-1][0][5]==sorted_lines[pos][0][5])
        v_dist = sorted_lines[pos][0][1]-(sorted_lines[pos-1][0][1]+sorted_lines[pos-1][0][2])
        outlier = len(lefts[sorted_lines[pos][0][1]])<=2
        if(formatted and v_dist<5 and outlier):
            merges.append((sorted_lines[pos-1][1],sorted_lines[pos][1]))
    for m in reversed(merges):
        print(text_lines[m[0]])
        print(text_lines[m[1]])
        merge_lines(m, text_lines, tops, lefts, rights)
        print(text_lines[m[0]])
        print(text_lines[m[1]])
        pause()

def merge_lines(merge_tuple, text_lines, tops, lefts, rights):
    # get the line index of the lines which shall be merged
    left_id = merge_tuple[0]
    right_id = merge_tuple[1]

    # update the entries in the dictionaries
    # delete the tops entry of the right line
    tops[text_lines[right_id][0]].remove(right_id)
    # delete the lefts entry of the right line
    lefts[text_lines[right_id][1]].remove(right_id)
    # update the rights entry
    # delete the right id an put the left id in its place
    rights[text_lines[right_id][1]+text_lines[right_id][2]].remove(right_id)
    rights[text_lines[right_id][1]+text_lines[right_id][2]].append(left_id)
    # remove the left id
    rights[text_lines[left_id][1]+text_lines[left_id][2]].remove(left_id)

    # compute new width = (right offset + right width) - left_offset
    text_lines[left_id][2] = (text_lines[right_id][1]+text_lines[right_id][2]) - text_lines[left_id][1]
    # compute new text = left text + right text
    text_lines[left_id][5] += (" " + text_lines[right_id][5])
    # compute new formatter as 4 if there are two different ones and the maximum otherwise
    lf = text_lines[left_id][6]
    rf = text_lines[right_id][6]
    if(lf>0 and rf>0 and not(lf==rf)):
        text_lines[left_id][6] = 4
    else:
        text_lines[left_id][6] = max(lf,rf)

    # delete the right line
    text_lines[right_id] = None


st = time()
test_path = join(DATA_PATH,"files_test_html")

powerpoint1 = "76ae7c120910a7830a1c0e0262d8cc5e.pdf"
powerpoint2 = "15a25c0553f25d4edcc288028b384cba.pdf"
slide_overview1 = "1e733d496d75352df67daefe269c1e88.pdf"
slide_overview2 = "5e9f1f979fff677b72e6228ded542a97.pdf"
lecture_mats1 = "1fd731e88a30612291de1923d5fa5263.pdf"
lecture_mats2 = "26286ffa140c615eb9a5a4ab46eb30be.pdf"
lecture_mats3 = "185422bf26436452aa0b3b8247e322af.pdf"
book_style1 = "37cf51662d8385b47ad00d36070766b0.pdf"
scan1 = "1569e3de486040aaaf71653c8e4bee6d.pdf"
scan2 = "c92b478470f9147ea02229ac7de22adc.pdf"
table1 = "1945835eac5a4162cc00ba89c30e6a90.pdf"
paper_style1 = "16350525a1fe0971fb02d009b1e3af72.pdf"
long_doc = "c41effe246dd564d7c72416faca33c21.pdf"
docs = [lecture_mats1, lecture_mats2, book_style1, table1, paper_style1]
docs = [book_style1]

for doc_str in docs:
    tree = get_xml_structure(join(test_path,doc_str), img_flag=True)
    root = tree.getroot()

    images = {"pathes":[],
    "sizes":[]}
    fontspecs = {"famlilies":set(),
    "colors":set(),
    "sizes":[],
    "spec_counts":[],
    "count":0}
    page_contents = []
    outline = False
    outline_items = 0

    page_counter = 0
    print(doc_str)
    prev_text = None
    for page in root:
        page_counter += 1
        if(page.tag=="page"):
            # parse the text structure information
            text_lines, tops, lefts, rights = process_page_content(page, fontspecs, images)

            # get the distances of consecutive lines and the number of lines
            h_dists = []
            v_dists = []
            mult_counter = 0
            sorted_tops = sorted(tops.items(), key=lambda x: x[0], reverse=False)
            for i in range(1,len(sorted_tops)):
                # TODO: fonts need  to be updated
                if(len(sorted_tops[i][1])>1):
                    check_multiple_lines(sorted_tops[i][1], text_lines, tops, lefts, rights)
                    lines = [text_lines[l_id] for l_id in sorted_tops[i][1]]
                    sorted_lines = sorted(lines, key=lambda x: x[1], reverse=False)
                    for pos in range(1,len(sorted_lines)):
                        v_dists.append(sorted_lines[pos][1]-(sorted_lines[pos-1][1]+sorted_lines[pos-1][2]))
                        # print(sorted_lines[pos-1][5])
                        # print(sorted_lines[pos][5])
                        # print(v_dists[-1])
                    mult_counter += 1
                h_dists.append(sorted_tops[i][0] -(sorted_tops[i-1][0]+text_lines[sorted_tops[i-1][1][0]][3]))

            print("multiple_lines: %d\n"%(mult_counter,))

            v_dists = Counter(v_dists)
            v_dists = v_dists.most_common()
            print("vertical_distances: " + str(v_dists))
            print("\n")

            h_dists = Counter(h_dists)
            h_dists = h_dists.most_common()
            print("horizontal_distances: " + str(h_dists))
            print("\n")

            # get the hists of left_margins
            left_counts = [(key,len(val)) for key,val in lefts.items() if(len(val)>1)]
            print(sorted(left_counts, key=lambda x: x[1], reverse=True))
            print("\n")
            # get the hists of right_margins
            right_counts = [(key,len(val)) for key,val in rights.items()]
            print(sorted(right_counts, key=lambda x: x[1], reverse=True))
            print("\n")
            pause()
            '''
            Kriterien paper_style:
                2-3 hohe left werte
                ähnliche anzahlen für right werte
                ähnlich hohe zahl von multiple entries
                genügend abstand der left werte

            Kriterien Book style:
                1 hoher left wert
                1 ähnlich hoher right wert
                hoher left wert ist der kleinste (ausnahmen berücksichtigen)


            Kriterien Table style:
                2 und mehr hohe left werte
                unterschiedliche right werte
                ähnlich hohe zahl von multiple entries
                eventuell geringer abstand der left werte

            Kriterien Block code
                Block code ähnlich hohe left wie right werte
            '''

            # page_contents.append(page_texts)
        elif(page.tag=="outline"):
            outline = True
            outline_iter = page.iter()
            next(outline_iter)
            for elem in outline_iter:
                if(elem.tag=="outline"):
                    continue
                elif(elem.tag=="item"):
                    outline_items += 1
                else:
                    warnstring = "Unexpected outline tag found, namely %s"%(elem.tag,)
                    print(print_bcolors(["WARNING"], warnstring))
                    print(elem.tag,elem.attrib,elem.text)
        else:
            warnstring = "First child is not a page but %s"%(page.tag,)
            print(print_bcolors(["WARNING"], warnstring))
            print(page.tag,page.attrib,page.text)
            for elem in page.iter():
                print(elem.tag,elem.attrib,elem.text)

        if(page_counter==5):
            break


    # print(images["sizes"])
    # print(fontspecs)


# print(img_stuff(image_pathes))
# # show time discrepancy
# st = time()
# features.pdf_structure.process_file(doc)
# print(time()-st)

# st = time()
# print(develop.pdf_images.get_grayscale_entropy_tmpfile(doc))
# print(time()-st)

# 2. write a parser with at least following features:
"""
stats_dict = {
"pages":0,
"color_cnt":0,
"font_cnt":0,

"word_cnt":0,
"main_font_word_cnt":0,
"other_fonts_word_cnt":0,

"bold_cnt":0,
"annotation_cnt":0,
"italic_cnt":0,
"mult_formats_cnt":0,

"image_cnt":0,
"line_cnt":0,
"textbox_cnt":0,
"images_pp":0,
"line_pp":0,
"textbox_pp":0,

"image_space_pp":0,
"dev_image_space_pp":0,
"text_space_pp":0,
"dev_text_space_pp":0,
"free_space_pp":0,
"dev_free_space_pp":0,

"size_ptb":0,
"dev_size_ptb":0,
"lines_ptb":0,
"dev_lines_ptb":0,
"words_pl":0,
"dev_words_pl":0,
"size_pib":0,
"dev_size_pib":0,

"md_offset":0,
"md_offset_perc":0,
"md_end":0,
"md_end_perc":0,

"column_cnt":0,

"pred_book_style":0
"pred_paper_style":0
"pred_table_style":0
"pred_enum_style":0

"text":""
"bold_text":""
"copyright_symbol":""
"paragraphs":0
"color_img_cnt":0
"gray_img_cnt":0
}
"""
