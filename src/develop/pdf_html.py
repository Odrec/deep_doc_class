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

###### Helpers #######
bcolors = {
    "reset" : '\033[0m',
    "bold" : '\033[1m',
    "underline" : '\033[4m',
    "concealed" : '\033[8m',

    "black" : '\033[30',
    "red" : '\033[31m',
    "green" : '\033[32m',
    "yellow" : '\033[33m',
    "blue" : '\033[34m',
    "magenta" : '\033[35m',
    "cyan" : '\033[36m',
    "white" : '\033[37',

    "b_black" : '\033[40',
    "b_red" : '\033[41m',
    "b_green" : '\033[42m',
    "b_yellow" : '\033[43m',
    "b_blue" : '\033[44m',
    "b_magenta" : '\033[45m',
    "b_cyan" : '\033[46m',
    "b_white" : '\033[47'
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
	formated_text += text + bcolors["reset"]
	return formated_text

def pause():
    """
    Pause the execution until Enter gets pressed
    """
    input("Press Enter to continue...")
    return

###### Parsing Classes #######
class Text_Box(object)

    def __init__(self, text_line, line_idx):
        self.top_pos = text_line[0]
        self.bot_pos = text_line[0]+text_line[3]
        self.left_pos = text_line[1]
        self.line_ids = [line_idx]
        self.text = text_line[5]

    def __str__(self):
        return "top:" + str(self.top_pos) + " bot:" + str(self.bot_pos) + " left:" + str(self.left_pos) + " lines:" + str(len(self.line_ids))

    def add_line(self, text_line, line_idx, norm_dist, dist_dev=1, margin_dev=1):
        line_dist = text_line[0] - self.bot_pos
        dist_criterion = line_dist<=norm_dist+dist_dev and line_dist>=norm_dist-dist_dev

        if(dist_criterion and right_criteron):
            self.bot_pos = self.bot_pos + line_dist + text_line[3]
            self.line_ids.append(line_idx)
            self.text += "\n" + text_line[5]
            # print(self)
            # print(self.text)
            # print()
            return True
        # print("line:" + str(text_line))
        # print("bot_pos: " + str(self.bot_pos))
        # print("right_pos: " + str(self.right_pos))
        # print("right_diff: " + str(right_dif))
        # print("line_dist: " + str(line_dist))

        return False

class Text_Box_Block(object):

    def __init__(self, text_line, line_idx):
        self.top_pos = text_line[0]
        self.bot_pos = text_line[0]+text_line[3]
        self.left_pos = text_line[1]
        self.right_pos = text_line[1]+text_line[2]
        self.line_ids = [line_idx]
        self.text = text_line[5]

    def __str__(self):
        return "top:" + str(self.top_pos) + " bot:" + str(self.bot_pos) + " left:" + str(self.left_pos) + " right:" + str(self.right_pos) + " lines:" + str(len(self.line_ids))

    def add_line(self, text_line, line_idx, norm_dist, dist_dev=1, margin_dev=1):
        line_dist = text_line[0] - self.bot_pos
        dist_criterion = line_dist<=norm_dist+dist_dev and line_dist>=norm_dist-dist_dev

        right_dif = self.right_pos - (text_line[1]+text_line[2])
        right_criteron = right_dif<=margin_dev and right_dif>=(-margin_dev)

        if(dist_criterion and right_criteron):
            self.bot_pos = self.bot_pos + line_dist + text_line[3]
            self.line_ids.append(line_idx)
            self.text += "\n" + text_line[5]
            # print(self)
            # print(self.text)
            # print()
            return True
        # print("line:" + str(text_line))
        # print("bot_pos: " + str(self.bot_pos))
        # print("right_pos: " + str(self.right_pos))
        # print("right_diff: " + str(right_dif))
        # print("line_dist: " + str(line_dist))

        return False

class Page(object):

    def __init__(self, xml_page):
        self.xml_page = xml_page
        self.page_lines = None
        self.tops = None
        self.lefts = None
        self.rights = None
        self.sorted_tops = None
        self.h_dists = None
        self.v_dists = None

    def structure_text_coordinates(self, doc):
        self.page_lines = []
        self.tops = {}
        self.lefts = {}
        self.rights = {}
        page_line_index = 0

        for elem in page:
            if(elem.tag=="fontspec"):
                doc.fontspecs["famlilies"].add(elem.attrib["family"])
                doc.fontspecs["colors"].add(elem.attrib["color"])
                doc.fontspecs["spec_counts"].append(0)
                doc.fontspecs["sizes"].append(int(elem.attrib["size"]))
                doc.fontspecs["heights"].append({})
                doc.fontspecs["best_heights"].append(-1)
            elif(elem.tag=="image"):
                doc.images["pathes"].append(elem.attrib["src"])
                doc.images["sizes"].append(int(elem.attrib["height"])*int(elem.attrib["width"]))
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

                if(text_list[3]!=doc.fontspecs["best_heights"][text_list[4]]):
                    h_dict = doc.fontspecs["heights"][text_list[4]]
                    try:
                        h_dict[text_list[3]] += 1
                    except KeyError:
                        h_dict[text_list[3]] = 1

                    best_height = max(h_dict.items(), key=lambda h_dict: h_dict[1])
                    doc.fontspecs["best_heights"][text_list[4]] = best_height[0]
                    if(text_list[3]!=best_height[0]):
                        prev_line = self.page_lines[-1]
                        print("strange height")
                        print(prev_line)
                        print(text_list)
                        h_dist = text_list[1]-(prev_line[1]+prev_line[2])
                        if(h_dist>=0 and h_dist<10):
                            print("changing")
                            # update the rights entry
                            # delete the the prev line id
                            self.rights[prev_line[1]+prev_line[2]].remove(page_line_index-1)
                            # delete whole entry if became emtpy
                            if(len(self.rights[prev_line[1]+prev_line[2]])==0):
                                del self.rights[prev_line[1]+prev_line[2]]
                            # put the prev line id in the new rights place
                            try:
                                self.rights[text_list[1]+text_list[2]].append(page_line_index-1)
                            except:
                                self.rights[text_list[1]+text_list[2]] = [page_line_index-1]

                            # compute new width = (new offset + new width) - left_offset
                            prev_line[2] = (text_list[1]+text_list[2]) - prev_line[1]
                            # compute new text = left text + right text
                            prev_line[5] += (" " + text_list[5])
                            # compute new formatter as 4 if there are two different ones and the maximum otherwise
                            lf = prev_line[6]
                            rf = text_list[6]
                            if(lf>0 and rf>0 and not(lf==rf)):
                                prev_line[6] = 4
                            else:
                                prev_line[6] = max(lf,rf)
                            print(print_bcolors(["red"], str(prev_line)))
                            continue

                        else:
                            # TODO: heights need to be completely evaluated first otherwise divergences from the real hight at the beginning make some good heights worse
                            pass
                            # print("changing height")
                            # text_list[3] = best_height[0]
                            # if(text_list[4]==2):
                            #     text_list[3] = 14
                            # print(print_bcolors(["blue"], str(text_list)))


                    else:
                        h_dict[text_list[3]] = 1
                else:
                    doc.fontspecs["heights"][text_list[4]][text_list[3]] += 1

                if(len(text_list[5])>0):
                    if(text_list[5]==" "):
                        continue
                    text_list[5] = text_list[5]
                    doc.fontspecs["spec_counts"][text_list[4]] += 1
                    self.page_lines.append(text_list)

                    try:
                        self.tops[text_list[0]].append(page_line_index)
                    except KeyError:
                        self.tops[text_list[0]] = [page_line_index]
                    try:
                        self.lefts[text_list[1]].append(page_line_index)
                    except KeyError:
                        self.lefts[text_list[1]] = [page_line_index]
                    try:
                        self.rights[text_list[1]+text_list[2]].append(page_line_index)
                    except KeyError:
                        self.rights[text_list[1]+text_list[2]] = [page_line_index]

                    page_line_index += 1
            else:
                warnstring = "Unexpected tag found, namely %s"%(elem.tag,)
                print(print_bcolors(["WARNING"], warnstring))
                print(elem.tag,elem.attrib,elem.text)

    def structure_text_alignment(self):

        # get the distances of consecutive lines and the number of lines
        self.v_dists = []
        self.h_dists = []
        mult_counter = 0
        self.sorted_tops = sorted(self.self.tops.items(), key=lambda x: x[0], reverse=False)
        for i in range(1,len(self.sorted_tops)):
            # TODO: fonts need  to be updated
            if(len(self.sorted_tops[i][1])>1):
                self.check_multiple_lines(self.sorted_tops[i][1])
                lines = [self.text_lines[l_id] for l_id in self.sorted_tops[i][1]]
                sorted_lines = sorted(lines, key=lambda x: x[1], reverse=False)
                for pos in range(1,len(sorted_lines)):
                    h_dists.append(sorted_lines[pos][1]-(sorted_lines[pos-1][1]+sorted_lines[pos-1][2]))
                    # print(sorted_lines[pos-1][5])
                    # print(sorted_lines[pos][5])
                    # print(h_dists[-1])
                mult_counter += 1
            vert_dist = -1
            forward = 0
            while(vert_dist<0 and len(self.sorted_tops)>i+forward):
                vert_dist = self.sorted_tops[i+forward][0] -(self.sorted_tops[i-1][0]+text_lines[self.self.sorted_tops[i-1][1][0]][3])
                forward += 1
            self.v_dists.append(vert_dist)

            # v_dists.append(sorted_tops[i][0] -(sorted_tops[i-1][0]+text_lines[sorted_tops[i-1][1][0]][3]))
            # TODO: think about how to get the best vertical distance
            # consider one vertical distance per font type

        print("multiple_lines: %d\n"%(mult_counter,))

        self.h_dists = Counter(self.h_dists)
        self.h_dists = self.h_dists.most_common()
        print("horizontal_distances: " + str(self.h_dists))
        print("\n")

        self.v_dists = Counter(self.self.v_dists)
        self.v_dists = self.v_dists.most_common()
        print("vertical_distances: " + str(self.v_dists))
        print("\n")

        # get the hists of left_margins
        left_counts = [(key,len(val)) for key,val in self.lefts.items() if(len(val)>1)]
        sorted_left_counts = sorted(left_counts, key=lambda x: x[1], reverse=True)
        print("lefts: " + str(sorted_left_counts))
        print("\n")
        # get the hists of right_margins
        right_counts = [(key,len(val)) for key,val in self.rights.items()]
        sorted_right_counts = sorted(right_counts, key=lambda x: x[1], reverse=True)
        print("rights: " + str(sorted_right_counts))
        print("\n")

    def check_multiple_lines(line_idc):
        lines = [(text_lines[l_id],l_id)for l_id in line_idc]
        sorted_lines = sorted(lines, key=lambda x: x[0][1], reverse=False)
        merges = []
        for pos in range(1,len(sorted_lines)):
            formatted = sorted_lines[pos-1][0][6]>0 or sorted_lines[pos][0][6]>0 or not(sorted_lines[pos-1][0][5]==sorted_lines[pos][0][5])

            v_dist = sorted_lines[pos][0][1]-(sorted_lines[pos-1][0][1]+sorted_lines[pos-1][0][2])
            outlier = len(lefts[sorted_lines[pos][0][1]])<=2
            if(v_dist<10 and (formatted or outlier)):
                merges.append((sorted_lines[pos-1][1],sorted_lines[pos][1]))
            # else:
            #     print(sorted_lines[pos-1][0])
            #     print(sorted_lines[pos][0])
            #     print(v_dist<5,formatted,outlier)
        for m in reversed(merges):
            # print(text_lines[m[0]])
            # print(text_lines[m[1]])
            merge_lines(m, text_lines, tops, lefts, rights)
            # print(text_lines[m[0]])
            # print(text_lines[m[1]])
            # pause()

    def merge_lines(merge_tuple):
        # get the line index of the lines which shall be merged
        left_id = merge_tuple[0]
        right_id = merge_tuple[1]

        # update the entries in the dictionaries
        # delete the tops entry of the right line
        tops[text_lines[right_id][0]].remove(right_id)
        # delete the lefts entry of the right line
        lefts[text_lines[right_id][1]].remove(right_id)
        if(len(lefts[text_lines[right_id][1]])==0):
            del lefts[text_lines[right_id][1]]
        # update the rights entry
        # delete the right id an put the left id in its place
        rights[text_lines[right_id][1]+text_lines[right_id][2]].remove(right_id)
        rights[text_lines[right_id][1]+text_lines[right_id][2]].append(left_id)
        # remove the left id
        rights[text_lines[left_id][1]+text_lines[left_id][2]].remove(left_id)
        if(len(rights[text_lines[left_id][1]+text_lines[left_id][2]])==0):
            del rights[text_lines[left_id][1]+text_lines[left_id][2]]

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


class Document(object):

    def __init__(self, doc_path):
        doc_path = doc_path
        doc_id = splitext(basename(doc_path))[0]
        images = {"pathes":[],
        "sizes":[]}
        fontspecs = {"famlilies":set(),
        "colors":set(),
        "sizes":[],
        "spec_counts":[],
        "count":0,
        "heights":[],
        "best_heights":[]}
        pages = []
        outline = False
        outline_items = 0

    def process_xml():
        # get the xml data in a tree structure
        tree = get_xml_structure(join(test_path,doc_str), img_flag=True)
        # get the head of the tree
        root = tree.getroot()

        # go through all pages
        for root_obj in root:
            if(root_obj.tag=="page"):
                self.process_page(root_obj)
            elif(root_obj.tag=="outline"):
                self.process_outline(root_obj)
            else:
                warnstring = "First child is not a page or outline but %s"%(root_obj.tag,)
                print(print_bcolors(["yellow"], warnstring))
                print(root_obj.tag,root_obj.attrib,root_obj.text)
                for elem in root_obj.iter():
                    print(elem.tag,elem.attrib,elem.text)

    def process_page(xml_page):
        page = Page(xml_page)
        self.pages.append(page)
        # parse the text structure information
        page.structure_text_coordinates(self)
        page.structure_text_alignment(self)

        # min 10% of the lines end at the same right pos
        tbox_blockstyle_left = {}
        tbox_blockstyle_top = []

        line_cnt = len(sorted_tops)
        max_rights = sorted_right_counts[0][1]
        print(line_cnt)
        print(max_rights)
        if(sorted_right_counts[0][0]+1 in rights):
            max_rights += len(rights[sorted_right_counts[0][0]+1])
        if(sorted_right_counts[0][0]-1 in rights):
            max_rights += len(rights[sorted_right_counts[0][0]-1])
        print(max_rights)
        block_thres = int(np.log(np.power(len(sorted_tops)-1,3)))
        print(block_thres)
        block_style = sorted_right_counts[0][1] >= block_thres

        print("block_style: " + str(block_style))
        pause()
        s1 = time()
        if(block_style):
            #TODO improve performance by going sorting lefts and adding them
            norm_dist = v_dists[0][0]
            print(norm_dist)
            for (top_lvl, line_ids) in sorted_tops:
                # if(root_obj.attrib["number"]=="2"):
                    # print("skip first page")
                    # pause()
                for l_id in line_ids:
                    # print(l_id,text_lines[l_id])
                    added = False
                    if(text_lines[l_id][1] in tbox_blockstyle_left):
                        for tbox in tbox_blockstyle_left[text_lines[l_id][1]]:
                            if(tbox.add_line(text_line=text_lines[l_id], line_idx=l_id, norm_dist=norm_dist, dist_dev=1, margin_dev=5)):
                                added = True
                                break
                        if(not(added)):
                            nb = Text_Box_Block(text_lines[l_id],l_id)
                            tbox_blockstyle_left[text_lines[l_id][1]].append(nb)
                            tbox_blockstyle_top.append(nb)
                    else:
                        if(not(added)):
                            nb = Text_Box_Block(text_lines[l_id],l_id)
                            tbox_blockstyle_left[text_lines[l_id][1]] =  [nb]
                            tbox_blockstyle_top.append(nb)
            for i in range(len(tbox_blockstyle_top)-1,-1,-1):
                if(len(tbox_blockstyle_top[i].line_ids)<2):
                    continue
                else:
                    print(print_bcolors(["red"],"big_box:\t" + str(tbox_blockstyle_top[i])))
                    print(tbox_blockstyle_top[i].text)
                    big_block = tbox_blockstyle_top[i]
                    top_line = big_block.top_pos - norm_dist
                    bottom_line = big_block.bot_pos + norm_dist
                    left_margin = big_block.left_pos
                    right_margin = big_block.right_pos

                    # get possible belowbox
                    k=i+1
                    while(k<len(tbox_blockstyle_top)):
                        bb = tbox_blockstyle_top[k]
                        if(len(bb.line_ids)>1):
                            k+=1
                            continue
                        print(print_bcolors(["green"],"below_box:\t" + str(bb)))
                        bb = tbox_blockstyle_top[k]
                        dist_crit = bb.top_pos>=bottom_line-1 and bb.top_pos<=bottom_line+1
                        left_crit = bb.left_pos>=left_margin-1 and bb.left_pos<=left_margin+1
                        right_crit = bb.right_pos<right_margin+1
                        print(dist_crit,left_crit,right_crit)
                        if(dist_crit and left_crit and right_crit):
                            print(print_bcolors(["magenta"],bb.text))
                            break
                        elif(bb.top_pos>bottom_line+1):
                            break
                        else:
                            k+=1

                    # get possible topbox
                    j=i-1
                    while(j>=0):
                        tb = tbox_blockstyle_top[j]
                        if(len(tb.line_ids)>1):
                            j-=1
                            continue
                        print(print_bcolors(["blue"],"top_top:\t" + str(tb)))
                        dist_crit = tb.bot_pos>=top_line-1 and tb.bot_pos<=top_line+1
                        left_crit = tb.left_pos>=left_margin+10 and tb.left_pos<=left_margin+25
                        right_crit = tb.right_pos>=right_margin-1 and tb.right_pos<=right_margin+1
                        print(dist_crit,left_crit,right_crit)
                        if(dist_crit and left_crit and right_crit):
                            print(print_bcolors(["magenta"],tb.text))
                            break
                        elif(tb.bot_pos<top_line-1):
                            break
                        else:
                            j-=1
        pause()
        print("---------NEW PAGE-------------------")

        # paper_style =
        # book_style =
        # table_style =
        # structured_information =


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

    def process_outline(xml_outline):
        self.outline = True
        outline_iter = xml_outline.iter()
        next(outline_iter)
        for elem in outline_iter:
            if(elem.tag=="outline"):
                continue
            elif(elem.tag=="item"):
                self.outline_items += 1
            else:
                warnstring = "Unexpected outline tag found, namely %s"%(elem.tag,)
                print(print_bcolors(["WARNING"], warnstring))
                print(elem.tag,elem.attrib,elem.text)


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

###### XML Parsing Functions #######
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
paper_style1 = "0f6b08591d82390c4ad1a590266f92bb.pdf"
long_doc = "c41effe246dd564d7c72416faca33c21.pdf"
docs = [lecture_mats1, lecture_mats2, book_style1, table1, paper_style1]
docs = [lecture_mats3]

for doc_str in docs:
    # create a new document
    doc = Document(join(test_path,doc_str))
    doc.process_xml()

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
