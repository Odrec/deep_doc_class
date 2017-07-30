# TODO: consider parsing the document manually

# TODO: think about how to get the best vertical distance consider one vertical distance per font type

# TODO: heights need to be completely evaluated first otherwise divergences from the real hight at the beginning make some good heights worse

import sys, os, json, shutil, codecs
from collections import Counter
from os.path import join, realpath, dirname, isdir, basename, isfile, splitext
import warnings

SRC_DIR = os.path.abspath(join(join(realpath(__file__), os.pardir),os.pardir))
if(not(SRC_DIR in sys.path)):
    sys.path.append(SRC_DIR)

from time import time
import subprocess
import xml.etree.ElementTree as ET
from PIL import Image as PI

# import features.pdf_structure
import develop.pdf_images
from doc_globals import*
import numpy as np
np.seterr(all='raise')
from scipy import stats

from multiprocessing import Pool

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



###### Parsing Classes #######
class Text_Box(object):

    def __init__(self, block_box):
        self.block_id = block_box.block_id
        self.top_pos = block_box.top_pos
        self.bot_pos = block_box.bot_pos
        self.left_pos = block_box.left_pos
        self.right_pos = block_box.right_pos
        self.line_ids = block_box.line_ids
        self.text = block_box.text

    def __str__(self):
        return "top:" + str(self.top_pos) + " bot:" + str(self.bot_pos) + " left:" + str(self.left_pos) + " lines:" + str(len(self.line_ids))

    def add_block(self, block):
        self.bot_pos = block.bot_pos
        self.left_pos = self.left_pos
        self.right_pos = max(self.right_pos,block.right_pos)
        self.line_ids.extend(block.line_ids)
        self.text += ("\n" + block.text)


class Text_Box_Block(object):

    def __init__(self, text_line, block_id, line_idx):
        self.block_id = block_id
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

    def add_block(self, block):
        self.bot_pos = block.bot_pos
        self.left_pos = min(self.left_pos, block.left_pos)
        self.right_pos = max(self.right_pos, block.right_pos)
        self.line_ids.extend(block.line_ids)
        self.text += ("\n" + block.text)


class Page(object):

    def __init__(self, xml_page):
        self.xml_page = xml_page
        self.width = int(xml_page.attrib["width"])
        self.height = int(xml_page.attrib["height"])
        self.space = self.width*self.height
        self.image_space = 0

        self.text_lines = []
        self.tops = {}
        self.lefts = {}
        self.rights = {}

        self.sorted_tops = None
        self.h_dists = None
        self.v_dists = None
        self.norm_dist = None

        self.tbox_blockstyle_left = {}
        self.tbox_blockstyle_top = []
        self.tbox_top = []
        self.tbox_left = {}

    def structure_content_elements(self, doc):
        page_line_index = 0
        for elem in self.xml_page:
            if(elem.tag=="fontspec"):
                doc.fontspecs["count"]+=1
                doc.fontspecs["famlilies"].add(elem.attrib["family"])
                doc.fontspecs["colors"].add(elem.attrib["color"])
                doc.fontspecs["spec_counts"].append(0)
                doc.fontspecs["sizes"].append(int(elem.attrib["size"]))
                doc.fontspecs["heights"].append({})
                doc.fontspecs["best_heights"].append(-1)
            elif(elem.tag=="image"):
                doc.images["pathes"].append(elem.attrib["src"])
                img_size = int(elem.attrib["height"])*int(elem.attrib["width"])
                doc.images["sizes"].append(img_size)
                self.image_space += img_size
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

                for format_elem in text_iter:
                    if(not(format_elem.text is None)):
                        text_list[5] += format_elem.text + " "
                        if(format_elem.tag=="b"):
                            if(text_list[6]==0 or text_list[6]==1):
                                text_list[6] = 1
                            else:
                                text_list[6] = 4
                            doc.bold_text += format_elem.text + " "
                            doc.counts["bold_words"] += len(format_elem.text.split())
                        elif(format_elem.tag=="a"):
                            if(text_list[6]==0 or text_list[6]==2):
                                text_list[6] = 2
                            else:
                                text_list[6] = 4
                            doc.counts["annotations"] += 1
                        elif(format_elem.tag=="i"):
                            if(text_list[6]==0 or text_list[6]==3):
                                text_list[6] = 3
                            else:
                                text_list[6] = 4
                        else:
                            warnstring = "Unexpected text formatter found, namely %s"%(format_elem.tag,)
                            print(print_bcolors(["red"], warnstring))

                doc.text += text_list[5]
                text_list[5] = text_list[5][:-1]

                if(text_list[3]!=doc.fontspecs["best_heights"][text_list[4]]):
                    h_dict = doc.fontspecs["heights"][text_list[4]]
                    try:
                        h_dict[text_list[3]] += 1
                    except KeyError:
                        h_dict[text_list[3]] = 1

                    best_height = max(h_dict.items(), key=lambda h_dict: h_dict[1])
                    doc.fontspecs["best_heights"][text_list[4]] = best_height[0]
                    if(text_list[3]!=best_height[0] and len(self.text_lines)>0):
                        prev_line = self.text_lines[-1]

                        h_dist = text_list[1]-(prev_line[1]+prev_line[2])
                        if(h_dist>=0 and h_dist<10):

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
                            # print(print_bcolors(["red"], str(prev_line)))
                            continue

                        else:
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
                    doc.fontspecs["spec_counts"][text_list[4]] += len(text_list[5].split())
                    self.text_lines.append(text_list)

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
        self.sorted_tops = sorted(self.tops.items(), key=lambda x: x[0], reverse=False)
        # print(len(self.sorted_tops))
        for i in range(1,len(self.sorted_tops)):

            if(len(self.sorted_tops[i][1])>1):
                self.check_multiple_lines(self.sorted_tops[i][1])
                lines = [self.text_lines[l_id] for l_id in self.sorted_tops[i][1]]
                sorted_lines = sorted(lines, key=lambda x: x[1], reverse=False)
                for pos in range(1,len(sorted_lines)):
                    self.h_dists.append(sorted_lines[pos][1]-(sorted_lines[pos-1][1]+sorted_lines[pos-1][2]))
                    # print(sorted_lines[pos-1][5])
                    # print(sorted_lines[pos][5])
                    # print(h_dists[-1])
                mult_counter += 1
            vert_dist = -1
            forward = 0
            while(vert_dist<0 and len(self.sorted_tops)>i+forward):
                vert_dist = self.sorted_tops[i+forward][0] -(self.sorted_tops[i-1][0]+self.text_lines[self.sorted_tops[i-1][1][0]][3])
                same_font = self.text_lines[self.sorted_tops[i+forward][1][0]][4] == self.text_lines[self.sorted_tops[i-1][1][0]][4]
                forward += 1
            if(same_font):
                self.v_dists.append(vert_dist)

            # v_dists.append(sorted_tops[i][0] -(sorted_tops[i-1][0]+text_lines[sorted_tops[i-1][1][0]][3]))

        # print("multiple_lines: %d\n"%(mult_counter,))

        self.h_dists = Counter(self.h_dists)
        self.h_dists = self.h_dists.most_common()
        # print("horizontal_distances: " + str(self.h_dists))
        # print("\n")

        self.v_dists = Counter(self.v_dists)
        self.v_dists = self.v_dists.most_common()
        # print("vertical_distances: " + str(self.v_dists))
        # print("\n")
        if(len(self.v_dists)>0):
            self.norm_dist = self.v_dists[0][0]

        # get the hists of left_margins
        left_counts = [(key,len(val)) for key,val in self.lefts.items() if(len(val)>1)]
        self.sorted_left_counts = sorted(left_counts, key=lambda x: x[1], reverse=True)
        # print("lefts: " + str(self.sorted_left_counts))
        # print("\n")
        # get the hists of right_margins
        right_counts = [(key,len(val)) for key,val in self.rights.items()]
        self.sorted_right_counts = sorted(right_counts, key=lambda x: x[1], reverse=True)
        # print("rights: " + str(self.sorted_right_counts))
        # print("\n")

    def check_multiple_lines(self, line_idc):
        lines = [(self.text_lines[l_id],l_id)for l_id in line_idc]
        sorted_lines = sorted(lines, key=lambda x: x[0][1], reverse=False)
        merges = []
        for pos in range(1,len(sorted_lines)):
            # formatted = sorted_lines[pos-1][0][6]>0 or sorted_lines[pos][0][6]>0 or not(sorted_lines[pos-1][0][5]==sorted_lines[pos][0][5])

            formatted = sorted_lines[pos-1][0][6]>0 or sorted_lines[pos][0][6]>0

            v_dist = sorted_lines[pos][0][1]-(sorted_lines[pos-1][0][1]+sorted_lines[pos-1][0][2])
            outlier = len(self.lefts[sorted_lines[pos][0][1]])<=2
            if(v_dist<10 and (formatted or outlier)):
                merges.append((sorted_lines[pos-1][1],sorted_lines[pos][1]))
            # else:
            #     print(sorted_lines[pos-1][0])
            #     print(sorted_lines[pos][0])
            #     print(v_dist<5,formatted,outlier)
        for m in reversed(merges):
            # print(text_lines[m[0]])
            # print(text_lines[m[1]])
            self.merge_lines(m)
            # print(text_lines[m[0]])
            # print(text_lines[m[1]])
            # pause()

    def merge_lines(self, merge_tuple):
        # get the line index of the lines which shall be merged
        left_id = merge_tuple[0]
        right_id = merge_tuple[1]

        # update the entries in the dictionaries
        # delete the tops entry of the right line
        self.tops[self.text_lines[right_id][0]].remove(right_id)
        # delete the lefts entry of the right line
        self.lefts[self.text_lines[right_id][1]].remove(right_id)
        if(len(self.lefts[self.text_lines[right_id][1]])==0):
            del self.lefts[self.text_lines[right_id][1]]
        # update the rights entry
        # delete the right id an put the left id in its place
        self.rights[self.text_lines[right_id][1]+self.text_lines[right_id][2]].remove(right_id)
        self.rights[self.text_lines[right_id][1]+self.text_lines[right_id][2]].append(left_id)
        # remove the left id
        self.rights[self.text_lines[left_id][1]+self.text_lines[left_id][2]].remove(left_id)
        if(len(self.rights[self.text_lines[left_id][1]+self.text_lines[left_id][2]])==0):
            del self.rights[self.text_lines[left_id][1]+self.text_lines[left_id][2]]

        # compute new width = (right offset + right width) - left_offset
        self.text_lines[left_id][2] = (self.text_lines[right_id][1]+self.text_lines[right_id][2]) - self.text_lines[left_id][1]
        # compute new text = left text + right text
        self.text_lines[left_id][5] += (" " + self.text_lines[right_id][5])
        # compute new formatter as 4 if there are two different ones and the maximum otherwise
        lf = self.text_lines[left_id][6]
        rf = self.text_lines[right_id][6]
        if(lf>0 and rf>0 and not(lf==rf)):
            self.text_lines[left_id][6] = 4
        else:
            self.text_lines[left_id][6] = max(lf,rf)

        # delete the right line
        self.text_lines[right_id] = None

    def check_blockstyle_evidence(self, doc):
        line_cnt = len(self.sorted_tops)
        max_rights = self.sorted_right_counts[0][1]
        # print(line_cnt)
        # print(max_rights)
        if(self.sorted_right_counts[0][0]+1 in self.rights):
            max_rights += len(self.rights[self.sorted_right_counts[0][0]+1])
        if(self.sorted_right_counts[0][0]-1 in self.rights):
            max_rights += len(self.rights[self.sorted_right_counts[0][0]-1])
        # print(max_rights)
        block_thres = int(np.log(np.power(len(self.sorted_tops)-1,3)))
        # print(block_thres)
        block_style = self.sorted_right_counts[0][1] >= block_thres
        return block_style

    def merge_text_blockstyle(self):
        if(len(self.v_dists)>0):
            norm_dist = self.v_dists[0][0]
            # print(norm_dist)
            for (top_lvl, line_ids) in self.sorted_tops:
                # if(root_obj.attrib["number"]=="2"):
                    # print("skip first page")
                    # pause()
                for l_id in line_ids:
                    # print(l_id,text_lines[l_id])
                    added = False
                    if(self.text_lines[l_id][1] in self.tbox_blockstyle_left):
                        for tbox_id in self.tbox_blockstyle_left[self.text_lines[l_id][1]]:
                            tbox = self.tbox_blockstyle_top[tbox_id]
                            if(tbox.add_line(text_line=self.text_lines[l_id], line_idx=l_id, norm_dist=norm_dist, dist_dev=1, margin_dev=5)):
                                added = True
                                break
                        if(not(added)):
                            block_id = len(self.tbox_blockstyle_top)
                            nb = Text_Box_Block(self.text_lines[l_id], block_id, l_id)
                            self.tbox_blockstyle_left[self.text_lines[l_id][1]].append(block_id)
                            self.tbox_blockstyle_top.append(nb)
                    else:
                        if(not(added)):
                            block_id = len(self.tbox_blockstyle_top)
                            nb = Text_Box_Block(self.text_lines[l_id], block_id, l_id)
                            self.tbox_blockstyle_left[self.text_lines[l_id][1]] =  [block_id]
                            self.tbox_blockstyle_top.append(nb)

    def check_blockstyle_boundaries(self):

        norm_dist = self.v_dists[0][0]
        for i in range(len(self.tbox_blockstyle_top)-1,-1,-1):
            if(self.tbox_blockstyle_top[i] is None or len(self.tbox_blockstyle_top[i].line_ids)<=2):
                continue
            else:
                # print(print_bcolors(["red"],"big_box:\t" + str(self.tbox_blockstyle_top[i])))
                # print(self.tbox_blockstyle_top[i].text)
                big_block = self.tbox_blockstyle_top[i]
                top_line = big_block.top_pos - norm_dist
                bottom_line = big_block.bot_pos + norm_dist
                left_margin = big_block.left_pos
                right_margin = big_block.right_pos

                # get possible belowbox
                k=i+1
                while(k<len(self.tbox_blockstyle_top)):
                    bb = self.tbox_blockstyle_top[k]
                    if(bb is None or len(bb.line_ids)>1):
                        k+=1
                        continue
                    # print(print_bcolors(["green"],"below_box:\t" + str(bb)))
                    bb = self.tbox_blockstyle_top[k]
                    dist_crit = bb.top_pos>=bottom_line-1 and bb.top_pos<=bottom_line+1
                    left_crit = bb.left_pos>=left_margin-1 and bb.left_pos<=left_margin+1
                    right_crit = bb.right_pos<right_margin+1
                    # print(dist_crit,left_crit,right_crit)
                    if(dist_crit and left_crit and right_crit):
                        # print(print_bcolors(["magenta"],bb.text))
                        self.tbox_blockstyle_left[bb.left_pos].remove(k)
                        self.tbox_blockstyle_left[big_block.left_pos].remove(i)
                        self.tbox_blockstyle_left[min(bb.left_pos,big_block.left_pos)].append(i)
                        if(len(self.tbox_blockstyle_left[bb.left_pos])==0):
                            del self.tbox_blockstyle_left[bb.left_pos]
                        if(len(self.tbox_blockstyle_left[big_block.left_pos])==0):
                            del self.tbox_blockstyle_left[big_block.left_pos]
                        big_block.add_block(bb)
                        self.tbox_blockstyle_top[k] = None
                    elif(bb.top_pos>bottom_line+1):
                        break
                    else:
                        k+=1

                # get possible topbox
                j=i-1
                while(j>=0):
                    tb = self.tbox_blockstyle_top[j]
                    if(tb is None or len(tb.line_ids)>1):
                        j-=1
                        continue
                    # print(print_bcolors(["blue"],"top_top:\t" + str(tb)))
                    dist_crit = tb.bot_pos>=top_line-1 and tb.bot_pos<=top_line+1
                    left_crit = tb.left_pos>=left_margin+10 and tb.left_pos<=left_margin+25
                    right_crit = tb.right_pos>=right_margin-1 and tb.right_pos<=right_margin+1
                    # print(dist_crit,left_crit,right_crit)
                    if(dist_crit and left_crit and right_crit):
                        # print(print_bcolors(["magenta"],tb.text))
                        self.tbox_blockstyle_top[i] = None
                        self.tbox_blockstyle_left[big_block.left_pos].remove(i)
                        self.tbox_blockstyle_left[tb.left_pos].remove(j)
                        self.tbox_blockstyle_left[min(tb.left_pos,big_block.left_pos)].append(j)
                        if(len(self.tbox_blockstyle_left[big_block.left_pos])==0):
                            del self.tbox_blockstyle_left[big_block.left_pos]
                        if(len(self.tbox_blockstyle_left[tb.left_pos])==0):
                            del self.tbox_blockstyle_left[tb.left_pos]
                        tb.add_block(big_block)
                        break
                    elif(tb.bot_pos<top_line-1):
                        break
                    else:
                        j-=1

    def merge_text_not_blocked(self):
        norm_dist = self.v_dists[0][0]
        i = 0
        while(i < len(self.tbox_blockstyle_top)):
            box = self.tbox_blockstyle_top[i]
            if(not(box is None) and len(box.line_ids)<=2):
                # print(print_bcolors(["red"],"big_box:\t" + str(box)))
                # print(box.text)
                current_block = Text_Box(box)
                top_line = current_block.top_pos - norm_dist
                bottom_line = current_block.bot_pos + norm_dist
                left_margin = current_block.left_pos

                # get possible belowbox
                box_index = i
                k=i+1
                while(k<len(self.tbox_blockstyle_top)):
                    bb = self.tbox_blockstyle_top[k]
                    if(bb is None):
                        k+=1
                        continue
                    elif(bb.top_pos>bottom_line+1):
                        try:
                            self.tbox_left[current_block.left_pos].append(len(self.tbox_top))
                        except KeyError:
                            self.tbox_left[current_block.left_pos] = [len(self.tbox_top)]
                        self.tbox_top.append(current_block)

                        # print(current_block)
                        # print(current_block.text)
                        # print(self.tbox_blockstyle_left[current_block.left_pos])
                        # print(i)
                        # print(box_index)
                        self.tbox_blockstyle_left[current_block.left_pos].remove(box_index)
                        if(len(self.tbox_blockstyle_left[current_block.left_pos])==0):
                            del self.tbox_blockstyle_left[current_block.left_pos]
                        self.tbox_blockstyle_top[box_index] = None
                        break

                    elif(len(bb.line_ids)>2):
                        k+=1
                    else:
                        # print(print_bcolors(["green"],"below_box:\t" + str(bb)))
                        bb = self.tbox_blockstyle_top[k]
                        dist_crit = bb.top_pos>=bottom_line-1 and bb.top_pos<=bottom_line+1
                        left_crit = bb.left_pos>=left_margin-1 and bb.left_pos<=left_margin+1
                        # print(dist_crit,left_crit)
                        if(dist_crit and left_crit):
                            # print(print_bcolors(["magenta"],bb.text))
                            current_block.add_block(bb)
                            top_line = current_block.top_pos - norm_dist
                            bottom_line = current_block.bot_pos + norm_dist
                            left_margin = current_block.left_pos

                            self.tbox_blockstyle_top[k] = None
                            self.tbox_blockstyle_left[bb.left_pos].remove(k)
                            if(len(self.tbox_blockstyle_left[bb.left_pos])==0):
                                del self.tbox_blockstyle_left[bb.left_pos]
                        k+=1
                if(k>=len(self.tbox_blockstyle_top)):
                    try:
                        self.tbox_left[current_block.left_pos].append(len(self.tbox_top))
                    except KeyError:
                        self.tbox_left[current_block.left_pos] = [len(self.tbox_top)]
                    self.tbox_top.append(current_block)

                    # print(current_block)
                    # print(current_block.text)
                    # print(self.tbox_blockstyle_left[current_block.left_pos])
                    # print(i)
                    # print(box_index)
                    self.tbox_blockstyle_left[current_block.left_pos].remove(box_index)
                    if(len(self.tbox_blockstyle_left[current_block.left_pos])==0):
                        del self.tbox_blockstyle_left[current_block.left_pos]
                    self.tbox_blockstyle_top[box_index] = None
                i += 1
            else:
                i+=1

    def create_text_blocks(self):
        box_index = 0
        for i in range(len(self.text_lines)):
            c_line = self.text_lines[i]
            if(not(c_line is None)):
                current_block = Text_Box(Text_Box_Block(c_line, box_index, i))
                try:
                    self.tbox_left[current_block.left_pos].append(len(self.tbox_top))
                except KeyError:
                    self.tbox_left[current_block.left_pos] = [len(self.tbox_top)]
                self.tbox_top.append(current_block)
                box_index += 1


class Document(object):

    def __init__(self, doc_path):
        self.doc_path = doc_path
        self.doc_id = splitext(basename(doc_path))[0]

        self.text = ""
        self.bold_text = ""

        self.images = {
            "pathes":[],
            "sizes":[]
        }
        self.fontspecs = {
            "famlilies":set(),
            "colors":set(),
            "sizes":[],
            "spec_counts":[],
            "count":0,
            "heights":[],
            "best_heights":[]
        }
        self.counts = {
            "pages":0,
            "outline_items":0,
            "words":0,
            "bold_words":0,
            "annotations":0,
            "lines":0,
            "textboxes":0,
            "blockstyles":0,
        }

        self.stat_lists = {
            "image_space_pp":[],
            "page_space_pp":[],
            "free_space_pp":[],
            "words_pp":[],
            "lines_pp":[],
            "textboxes_pp":[],
            "blockstyles_pp":[],
            "textbox_space_pp":[],
            "blockstyle_space_pp":[],
            "word_pl":[],
            "lines_pbs":[],
            "lines_ptb":[],
            "column_counts":[],
            "sufficient_rights":[],
            "textbox_columns":[],
            "blockstyle_columns":[]
        }

        self.rights = {}
        self.lefts = {}

    def process_xml(self,img_flag=False):
        # get the xml file
        err = self.get_xml_structure(img_flag=img_flag)
        # try to open the file
        content = ""
        # pdftohtml seems create not utf-8 encoded chars even when utf-8 requested. Decode the file using latin one in case of those few exceptions.
        try:
            with codecs.open(self.xml_path, "r", "utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            with codecs.open(self.xml_path, "r", "latin1") as f:
                content = f.read()
        # parse the xml
        try:
            tree = ET.fromstring(content)
        # if no file was created the pdf could not be accessedd
        except FileNotFoundError:
            if(err.split()[0]=="Permission"):
                return "permission_error"
            return "pw_protected"
        # if the xml parser found an error
        except ET.ParseError as e:
            print(e)
            # the error description has usually two parts
            # error type : position of the error
            e_parts = str(e).split(":")
            # remove invalid xml tokens
            if(e_parts[0]=="not well-formed (invalid token)"):
                content = self.remove_invalid_xml_char(content)
            # remove not matching tags
            elif(e_parts[0]=="mismatched tag"):
                content = self.remove_invalid_xml_form(content)
            # no idea otherwise
            else:
                print("Unknown XML ParseError!")
                print(e)
                sys.exit(1)
            # try again to parse the xml
            try:
                tree = ET.fromstring(content)
            # if errors persist no idea
            except ET.ParseError:
                print("XML-ERROR in: " + self.doc_id)
                sys.exit(1)
                return "xml_error"

        # get the head of the tree
        #root = tree.getroot()
        root = tree
        # go through all pages
        for root_obj in root:
            if(root_obj.tag=="page"):
                self.counts["pages"] += 1
                self.process_page(root_obj)
            elif(root_obj.tag=="outline"):
                self.counts["outline_items"] += 1
                self.process_outline(root_obj)
            else:
                warnstring = "First child is not a page or outline but %s"%(root_obj.tag,)
                print(print_bcolors(["yellow"], warnstring))
                print(root_obj.tag,root_obj.attrib,root_obj.text)
                for elem in root_obj.iter():
                    print(elem.tag,elem.attrib,elem.text)

        return ""

    def get_xml_structure(self, img_flag=True):
        args = ["pdftohtml"]
        # if(img_flag):
        #     args += ["-i"]
        #args += ["-xml","-enc","Latin1", self.doc_path]
        args += ["-xml","-enc","UTF-8", self.doc_path]
        err = subprocess.Popen(args, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE).communicate()[1].decode(errors="ignore")
        self.xml_path = splitext(self.doc_path)[0] + ".xml"
        return err

    def process_page(self, xml_page):
        # print("---------NEW PAGE-------------------")
        page = Page(xml_page)
        # parse the text structure information
        # if(len(xml_page)==0):
        #     print(xml_page)
        #     self.images["pathes"].append(xml_page.attrib["number"])
        #     img_size = int(xml_page.attrib["height"])*int(xml_page.attrib["width"])
        #     self.images["sizes"].append(img_size)
        #     page.image_space += img_size
        #     print("yep")
        # else:
        page.structure_content_elements(self)
        if(len(page.text_lines)>0):
            page.structure_text_alignment()
            # block_style = page.check_blockstyle_evidence(self)
            if(not(page.norm_dist is None)):
                page.merge_text_blockstyle()
                page.check_blockstyle_boundaries()
                page.merge_text_not_blocked()
            else:
                page.create_text_blocks()

        self.update_page_stats(page)

        # pause()

    def process_outline(self, xml_outline):
        outline_iter = xml_outline.iter()
        next(outline_iter)
        for elem in outline_iter:
            if(elem.tag=="outline"):
                continue
            elif(elem.tag=="item"):
                self.counts["outline_items"] += 1
            else:
                warnstring = "Unexpected outline tag found, namely %s"%(elem.tag,)
                print(print_bcolors(["WARNING"], warnstring))
                print(elem.tag,elem.attrib,elem.text)

    def update_page_stats(self, page):
        self.stat_lists["lines_pp"].append(0)
        self.stat_lists["words_pp"].append(0)
        for line in page.text_lines:
            if(not(line is None)):
                self.text += line[5] + " "
                word_cnt = len(line[5].split())
                self.counts["words"] += word_cnt
                self.stat_lists["words_pp"][-1] += word_cnt
                self.stat_lists["word_pl"].append(word_cnt)
                self.counts["lines"] += 1
                self.stat_lists["lines_pp"][-1] += 1

        self.stat_lists["textboxes_pp"].append(0)
        self.stat_lists["textbox_space_pp"].append(0)
        for tb in page.tbox_top:
            if(not(tb is None)):
                # print(tb.text)
                # print("")
                self.counts["textboxes"] += 1
                self.stat_lists["textboxes_pp"][-1] += 1
                self.stat_lists["textbox_space_pp"][-1] += (tb.right_pos-tb.left_pos) * (tb.bot_pos-tb.top_pos)
                self.stat_lists["lines_ptb"].append(len(tb.line_ids))

        self.stat_lists["blockstyles_pp"].append(0)
        self.stat_lists["blockstyle_space_pp"].append(0)
        for tb in page.tbox_blockstyle_top:
            if(not(tb is None) and len(tb.line_ids)>=3):
                # print(tb.text)
                # print("")
                self.counts["blockstyles"] += 1
                self.stat_lists["blockstyles_pp"][-1] += 1
                self.stat_lists["blockstyle_space_pp"][-1] += (tb.right_pos-tb.left_pos) * (tb.bot_pos-tb.top_pos)
                self.stat_lists["lines_pbs"].append(len(tb.line_ids))

        self.stat_lists["page_space_pp"].append(page.space)
        self.stat_lists["image_space_pp"].append(page.image_space)
        self.stat_lists["free_space_pp"].append(page.space-page.image_space-self.stat_lists["blockstyle_space_pp"][-1]-self.stat_lists["textbox_space_pp"][-1])

        for key, vals in page.lefts.items():
            try:
                self.lefts[key] += len(vals)
            except KeyError:
                self.lefts[key] = len(vals)

        for key, vals in page.rights.items():
            try:
                self.rights[key] += len(vals)
            except KeyError:
                self.rights[key] = len(vals)

        # TODO: make column functions page functions since they are just accessing page attributes
        self.stat_lists["blockstyle_columns"].append(self.get_blockstyle_columns(page))
        self.stat_lists["textbox_columns"].append(self.get_textbox_columns(page))

    def get_blockstyle_columns(self, page):
        block_columns = 0
        if(self.stat_lists["blockstyles_pp"][-1]>0):
            block_columns = 1
            sorted_left_blockstyles = [(key,vals) for key,vals in page.tbox_blockstyle_left.items()]
            sorted_left_blockstyles = sorted(sorted_left_blockstyles, key=lambda x: x[0], reverse=False)
            # print(sorted_left_blockstyles)
            min_right_per_left_boxes = []
            for key,vals in sorted_left_blockstyles:
                min_box = None
                min_right = page.width
                for val in vals:
                    box = page.tbox_blockstyle_top[val]
                    # print(box.text)
                    # pause()
                    if(len(box.line_ids)>=2 and box.right_pos<min_right):
                        min_box = box
                        min_right = box.right_pos
                if(not(min_box is None)):
                    min_right_per_left_boxes.append((key,min_box))
                    # print(key,min_box.right_pos)
            i = 0
            while(i<len(min_right_per_left_boxes)):
                key, box = min_right_per_left_boxes[i]
                i+=1
                right_pos = box.right_pos
                # print(key,right_pos)
                min_right = page.width
                found = False
                for j in range(i,len(min_right_per_left_boxes)):
                    next_box = min_right_per_left_boxes[j][1]
                    if(right_pos<=min_right_per_left_boxes[j][0]):
                        if(next_box.right_pos<min_right):
                            min_right = next_box.right_pos
                            # print(min_right_per_left_boxes[j][0],min_right)
                            i=j
                            found = True
                if(found):
                    block_columns += 1
        return block_columns

    def get_textbox_columns(self, page):
        textbox_columns = 0
        if(self.stat_lists["textboxes_pp"][-1]>0):
            textbox_columns = 1
            sorted_left_tbox = [(key,vals) for key,vals in page.tbox_left.items()]
            sorted_left_tbox = sorted(sorted_left_tbox, key=lambda x: x[0], reverse=False)
            # print(sorted_left_tbox)
            min_right_per_left_boxes = []
            for key,vals in sorted_left_tbox:
                min_box = None
                min_right = page.width
                for val in vals:
                    box = page.tbox_top[val]
                    # print(box.text)
                    # pause()
                    if(len(box.line_ids)>=2 and box.right_pos<min_right):
                        min_box = box
                        min_right = box.right_pos
                if(not(min_box is None)):
                    min_right_per_left_boxes.append((key,min_box))
                    # print(key,min_box.right_pos)
            i = 0
            while(i<len(min_right_per_left_boxes)):
                key, box = min_right_per_left_boxes[i]
                i+=1
                right_pos = box.right_pos
                # print(key,right_pos)
                min_right = page.width
                found = False
                for j in range(i,len(min_right_per_left_boxes)):
                    next_box = min_right_per_left_boxes[j][1]
                    if(right_pos<=min_right_per_left_boxes[j][0]):
                        if(next_box.right_pos<min_right):
                            min_right = next_box.right_pos
                            # print(min_right_per_left_boxes[j][0],min_right)
                            i=j
                            found = True
                if(found):
                    textbox_columns += 1
        return textbox_columns

    def get_feature_dict(self):
        features = {
            "count_pages":0,

            "count_outline_items":0,

            # FONT STUFF
            "count_fonts":0,
            "count_font_colors":0,
            "count_font_families":0,
            "max_font_size":0,
            "min_font_size":0,
            "main_font_size":0,
            "perc_main_font_word":0,
            # "other_font_word_perc":0,

            # IMAGE STUFF (space always as ratio)
            "count_images":0,                # total count
            "total_image_space":0,          # percentage of total image space
            # "mean_image_space_pp":0,        # mean space per page
            "dev_image_space_pp":0,         # std of space per page
            "max_image_space_pp":0,            # maximum space per page
            "min_image_space_pp":0,            # minimum space per page
            "biggest_image":0,              # biggest image
            "samllest_image":0,             # smallest image
            # "mean_image_size":0,            # mean size of the images
            # "dev_image_size":0,             # std of the size of the images

            # TEXT STUFF
            "text":"",
            "bold_text":"",
            "first_100_words":"",
            "last_100_words":"",

            "count_words":0,
            "count_bold_words":0,
            "count_annotations":0,
            "count_lines":0,
            "count_textboxes":0,
            "count_blockstyles":0,

            # "mean_words_pp":0,
            # "mean_bold_words_pp":0,
            # "mean_lines_pp":0,
            # "mean_textboxes_pp":0,
            # "mean_blockstyles_pp":0,
            # "mean_textbox_space_pp":0,
            # "mean_blockstyle_space_pp":0,

            "dev_words_pp":0,
            "dev_lines_pp":0,
            "dev_textboxes_pp":0,
            "dev_blockstyles_pp":0,
            "dev_textbox_space_pp":0,
            "dev_blockstyle_space_pp":0,

            "max_words_pp":0,
            # "max_bold_words_pp":0,
            "max_lines_pp":0,
            "max_textboxes_pp":0,
            "max_blockstyles_pp":0,
            "max_textbox_space_pp":0,
            "max_blockstyle_space_pp":0,

            "min_words_pp":0,
            # "min_bold_words_pp":0,
            "min_lines_pp":0,
            "min_textboxes_pp":0,
            "min_blockstyles_pp":0,
            "min_textbox_space_pp":0,
            "min_blockstyle_space_pp":0,

            "mean_words_per_line":0,
            "dev_words_per_line":0,
            "mean_lines_per_blockstyle":0,
            "dev_lines_per_blockstyle":0,
            "max_lines_per_blockstyle":0,

            #STRUCTURE STUFF
            "modal_right":0,
            "perc_modal_right":0,
            "max_right":0,
            "modal_left":0,
            "perc_modal_left":0,
            "max_lefts":0,

            "modal_textbox_columns_pp":0,
            "perc_modal_textbox_columns_pp":0,
            "min_textbox_columns_pp":0,
            "max_textbox_columns_pp":0,
            "modal_blockstyle_columns_pp":0,
            "perc_modal_blockstyle_columns_pp":0,
            "min_blockstyle_columns_pp":0,
            "max_blockstyle_columns_pp":0,

            #FREE STUFF
            "total_free_space":0,
            "dev_free_space_pp":0,
            "max_free_space_pp":0,
            "min_free_space_pp":0
        }

        # pages
        features["count_pages"] = self.counts["pages"]
        page_spaces = self.stat_lists["page_space_pp"]
        total_page_space = sum(page_spaces)

        features["count_outline_items"] = self.counts["outline_items"]

        # font stuff
        if(self.fontspecs["count"]>0):
            features["count_fonts"] = self.fontspecs["count"]
            features["count_font_colors"] = len(self.fontspecs["colors"])
            features["count_font_families"] = len(self.fontspecs["famlilies"])
            features["max_font_size"] = max(self.fontspecs["sizes"])
            features["min_font_size"] = min(self.fontspecs["sizes"])
            if(max(self.fontspecs["spec_counts"])>0):
                main_font_index = np.argmax(self.fontspecs["spec_counts"])
                features["main_font_size"] = self.fontspecs["sizes"][main_font_index]
                features["perc_main_font_word"] = self.fontspecs["spec_counts"][main_font_index]/np.sum(self.fontspecs["spec_counts"])

            # print("FONT STUFF:\n")
            # print("count_fonts: " + str(features["count_fonts"]))
            # print("count_font_colors: " + str(features["count_font_colors"]))
            # print("count_font_families: " + str(features["count_font_families"]))
            # print("max_font_size: " + str(features["max_font_size"]))
            # print("min_font_size: " + str(features["min_font_size"]))
            # print("main_font_size: " + str(features["main_font_size"]))
            # print("perc_main_font_word: " + str(features["perc_main_font_word"]))

        # image stuff (space always as ratio)
        if(len(self.images["pathes"])>0):
            image_spaces = self.stat_lists["image_space_pp"]
            total_image_space = sum(image_spaces)
            image_page_ratios = [image_spaces[i]/page_spaces[i] for i in range(len(image_spaces))]

            features["count_images"] = len(self.images["pathes"])
            features["total_image_space"] = total_image_space/total_page_space
            features["dev_image_space_pp"] = np.std(image_page_ratios)
            features["max_image_space_pp"] = max(image_page_ratios)
            features["min_image_space_pp"] = min(image_page_ratios)
            features["biggest_image"] = max(self.images["sizes"])
            features["samllest_image"] = min(self.images["sizes"])

            # print("IMAGE STUFF:\n")
            # print("count_images: " + str(features["count_images"]))
            # print("total_image_space: " + str(features["total_image_space"]))
            # print("dev_image_space_pp: " + str(features["dev_image_space_pp"]))
            # print("max_image_space_pp: " + str(features["max_image_space_pp"]))
            # print("min_image_space_pp: " + str(features["min_image_space_pp"]))
            # print("biggest_image: " + str(features["biggest_image"]))
            # print("samllest_image: " + str(features["samllest_image"]))

        if(len(self.text)>0):
            # TEXT STUFF
            features["text"] = self.text
            features["bold_text"] = self.bold_text
            text_list = self.text.split()
            features["first_100_words"] = " ".join(text_list[0:100])
            features["last_100_words"] = " ".join(text_list[-100:])

            features["count_words"] = self.counts["words"]
            features["count_bold_words"] = self.counts["bold_words"]
            features["count_annotations"] = self.counts["annotations"]
            features["count_lines"] = self.counts["lines"]
            features["count_textboxes"] = self.counts["textboxes"]
            features["count_blockstyles"] = self.counts["blockstyles"]

            features["dev_words_pp"] = np.std(self.stat_lists["words_pp"])
            features["dev_lines_pp"] = np.std(self.stat_lists["lines_pp"])
            features["max_words_pp"] = max(self.stat_lists["words_pp"])
            features["max_lines_pp"] = max(self.stat_lists["lines_pp"])
            # features["max_bold_words_pp"] = max(self.stat_lists["words"])
            features["min_words_pp"] = min(self.stat_lists["words_pp"])
            features["min_lines_pp"] = min(self.stat_lists["lines_pp"])
            # features["min_bold_words_pp"] = min(self.stat_lists["words"])

        if(len(self.stat_lists["word_pl"])>0):
            features["mean_words_per_line"] = np.mean(self.stat_lists["word_pl"])
            features["dev_words_per_line"] = np.std(self.stat_lists["word_pl"])

            # print("TEXT STUFF:\n")
            # print("text: " + str(features["text"]))
            # print("bold_text: " + str(features["bold_text"]))
            # print("count_words: " + str(features["count_words"]))
            # print("count_bold_words: " + str(features["count_bold_words"]))
            # print("count_annotations: " + str(features["count_annotations"]))
            # print("count_lines: " + str(features["count_lines"]))
            # print("count_textboxes: " + str(features["count_textboxes"]))
            # print("count_blockstyles: " + str(features["count_blockstyles"]))
            # print("dev_words_pp: " + str(features["dev_words_pp"]))
            # print("dev_lines_pp: " + str(features["dev_lines_pp"]))
            # print("max_words_pp: " + str(features["max_words_pp"]))
            # print("max_lines_pp: " + str(features["max_lines_pp"]))
            # print("min_words_pp: " + str(features["min_words_pp"]))
            # print("min_lines_pp: " + str(features["min_lines_pp"]))
            # print("mean_words_per_line: " + str(features["mean_words_per_line"]))
            # print("dev_words_per_line: " + str(features["dev_words_per_line"]))

            if(self.counts["textboxes"]>0):
                textbox_spaces = self.stat_lists["textbox_space_pp"]
                total_textbox_space = sum(textbox_spaces)
                textbox_page_ratios = [textbox_spaces[i]/page_spaces[i] for i in range(len(textbox_spaces))]
                features["total_textbox_space"] = total_textbox_space/total_page_space

                features["dev_textboxes_pp"] = np.std(self.stat_lists["textboxes_pp"])
                features["dev_textbox_space_pp"] = np.std(textbox_page_ratios)
                features["max_textboxes_pp"] = max(self.stat_lists["textboxes_pp"])
                features["max_textbox_space_pp"] = max(textbox_page_ratios)
                features["min_textboxes_pp"] = min(self.stat_lists["textboxes_pp"])
                features["min_textbox_space_pp"] = min(textbox_page_ratios)
                features["mean_lines_per_textbox"] = np.mean(self.stat_lists["lines_ptb"])
                features["dev_lines_per_textbox"] = np.std(self.stat_lists["lines_ptb"])
                features["max_lines_per_textbox"] = max(self.stat_lists["lines_ptb"])

                textbox_columns = self.stat_lists["textbox_columns"]
                features["modal_textbox_columns_pp"] = stats.mode(textbox_columns)[0][0]
                features["perc_modal_textbox_columns_pp"] = stats.mode(textbox_columns)[1][0]/self.counts["pages"]
                features["min_textbox_columns_pp"] = min(textbox_columns)
                features["max_textbox_columns_pp"] = max(textbox_columns)

                # print("TEXTBOX STUFF:\n")
                # print("total_textbox_space: " + str(features["total_textbox_space"]))
                # print("dev_textboxes_pp: " + str(features["dev_textboxes_pp"]))
                # print("dev_textbox_space_pp: " + str(features["dev_textbox_space_pp"]))
                # print("max_textboxes_pp: " + str(features["max_textboxes_pp"]))
                # print("max_textbox_space_pp: " + str(features["max_textbox_space_pp"]))
                # print("min_textboxes_pp: " + str(features["min_textboxes_pp"]))
                # print("min_textbox_space_pp: " + str(features["min_textbox_space_pp"]))
                # print("mean_lines_per_textbox: " + str(features["mean_lines_per_textbox"]))
                # print("dev_lines_per_textbox: " + str(features["dev_lines_per_textbox"]))
                # print("max_lines_per_textbox: " + str(features["max_lines_per_textbox"]))
                # print("modal_textbox_columns_pp: " + str(features["modal_textbox_columns_pp"]))
                # print("perc_modal_textbox_columns_pp: " + str(features["perc_modal_textbox_columns_pp"]))
                # print("min_textbox_columns_pp: " + str(features["min_textbox_columns_pp"]))
                # print("max_textbox_columns_pp: " + str(features["max_textbox_columns_pp"]))

            if(self.counts["blockstyles"]>0):
                blockstyle_spaces = self.stat_lists["blockstyle_space_pp"]
                total_blockstyle_space = sum(blockstyle_spaces)
                blockstyle_page_ratios = [blockstyle_spaces[i]/page_spaces[i] for i in range(len(blockstyle_spaces))]
                features["total_blockstyle_space"] = total_blockstyle_space/total_page_space

                features["dev_blockstyles_pp"] = np.std(self.stat_lists["blockstyles_pp"])
                features["dev_blockstyle_space_pp"] = np.std(blockstyle_page_ratios)
                features["max_blockstyles_pp"] = max(self.stat_lists["blockstyles_pp"])
                features["max_blockstyle_space_pp"] = max(blockstyle_page_ratios)
                features["min_blockstyles_pp"] = min(self.stat_lists["blockstyles_pp"])
                features["min_blockstyle_space_pp"] = min(blockstyle_page_ratios)
                features["mean_lines_per_blockstyle"] = np.mean(self.stat_lists["lines_pbs"])
                features["dev_lines_per_blockstyle"] = np.std(self.stat_lists["lines_pbs"])
                features["max_lines_per_blockstyle"] = max(self.stat_lists["lines_pbs"])

                blockstyle_columns = self.stat_lists["blockstyle_columns"]
                features["modal_blockstyle_columns_pp"] = stats.mode(textbox_columns)[0][0]
                features["perc_modal_blockstyle_columns_pp"] = stats.mode(textbox_columns)[1][0]/self.counts["pages"]
                features["min_blockstyle_columns_pp"] = min(textbox_columns)
                features["max_blockstyle_columns_pp"] = max(textbox_columns)

                # print("BLOCKSTYLE STUFF:\n")
                # print("total_blockstyle_space: " + str(features["total_blockstyle_space"]))
                # print("dev_blockstyles_pp: " + str(features["dev_blockstyles_pp"]))
                # print("dev_blockstyle_space_pp: " + str(features["dev_blockstyle_space_pp"]))
                # print("max_blockstyles_pp: " + str(features["max_blockstyles_pp"]))
                # print("max_blockstyle_space_pp: " + str(features["max_blockstyle_space_pp"]))
                # print("min_blockstyles_pp: " + str(features["min_blockstyles_pp"]))
                # print("min_blockstyle_space_pp: " + str(features["min_blockstyle_space_pp"]))
                # print("mean_lines_per_blockstyle: " + str(features["mean_lines_per_blockstyle"]))
                # print("dev_lines_per_blockstyle: " + str(features["dev_lines_per_blockstyle"]))
                # print("max_lines_per_blockstyle: " + str(features["max_lines_per_blockstyle"]))
                # print("modal_blockstyle_columns_pp: " + str(features["modal_blockstyle_columns_pp"]))
                # print("perc_modal_blockstyle_columns_pp: " + str(features["perc_modal_blockstyle_columns_pp"]))
                # print("min_blockstyle_columns_pp: " + str(features["min_blockstyle_columns_pp"]))
                # print("max_blockstyle_columns_pp: " + str(features["max_blockstyle_columns_pp"]))


            #STRUCTURE STUFF
            if(len(self.rights)>0):
                max_right, max_right_count = max(self.rights.items(), key=lambda x: x[1])
                total_rights = np.sum(list(self.rights.values()))
                features["modal_right"] = max_right
                features["perc_modal_right"] = max_right_count/total_rights
                features["max_right"] = max(self.rights.keys())
            if(len(self.lefts)>0):
                max_left, max_left_count = max(self.lefts.items(), key=lambda x: x[1])
                total_lefts = np.sum(list(self.lefts.values()))
                features["modal_left"] = max_left
                features["perc_modal_left"] = max_left_count/total_lefts
                features["max_lefts"] = max(self.lefts.keys())

            # print("MARGIN STUFF:\n")
            # print("modal_right: " + str(features["modal_right"]))
            # print("perc_modal_right: " + str(features["perc_modal_right"]))
            # print("max_right: " + str(features["max_right"]))
            # print("modal_left: " + str(features["modal_left"]))
            # print("perc_modal_left: " + str(features["perc_modal_left"]))
            # print("max_left: " + str(features["max_left"]))

        #FREE STUFF
        free_spaces = self.stat_lists["free_space_pp"]
        total_free_space = sum(free_spaces)
        free_page_ratios = [free_spaces[i]/page_spaces[i] for i in range(len(free_spaces))]
        features["total_free_space"] = total_free_space/total_page_space
        features["dev_free_space_pp"] = np.std(free_page_ratios)
        features["max_free_space_pp"] = max(free_page_ratios)
        features["min_free_space_pp"] = min(free_page_ratios)

        # print("FREE STUFF:\n")
        # print("total_free_space: " + str(features["total_free_space"]))
        # print("dev_free_space_pp: " + str(features["dev_free_space_pp"]))
        # print("max_free_space_pp: " + str(features["max_free_space_pp"]))
        # print("min_free_space_pp: " + str(features["min_free_space_pp"]))

        for k,v in features.items():
            if(not(type(v)==str)):
                features[k] = np.float64(v)
        return features

    def get_error_features(self, error_code):
        features = {
            "count_pages":0,

            "count_outline_items":0,

            # FONT STUFF
            "count_fonts":0,
            "count_font_colors":0,
            "count_font_families":0,
            "max_font_size":0,
            "min_font_size":0,
            "main_font_size":0,
            "perc_main_font_word":0,
            # "other_font_word_perc":0,

            # IMAGE STUFF (space always as ratio)
            "count_images":0,                # total count
            "total_image_space":0,          # percentage of total image space
            # "mean_image_space_pp":0,        # mean space per page
            "dev_image_space_pp":0,         # std of space per page
            "max_image_space_pp":0,            # maximum space per page
            "min_image_space_pp":0,            # minimum space per page
            "biggest_image":0,              # biggest image
            "samllest_image":0,             # smallest image
            # "mean_image_size":0,            # mean size of the images
            # "dev_image_size":0,             # std of the size of the images

            # TEXT STUFF
            "text":error_code,
            "bold_text":error_code,
            "first_100_words":error_code,
            "last_100_words":error_code,

            "count_words":0,
            "count_bold_words":0,
            "count_annotations":0,
            "count_lines":0,
            "count_textboxes":0,
            "count_blockstyles":0,

            # "mean_words_pp":0,
            # "mean_bold_words_pp":0,
            # "mean_lines_pp":0,
            # "mean_textboxes_pp":0,
            # "mean_blockstyles_pp":0,
            # "mean_textbox_space_pp":0,
            # "mean_blockstyle_space_pp":0,

            "dev_words_pp":0,
            "dev_lines_pp":0,
            "dev_textboxes_pp":0,
            "dev_blockstyles_pp":0,
            "dev_textbox_space_pp":0,
            "dev_blockstyle_space_pp":0,

            "max_words_pp":0,
            # "max_bold_words_pp":0,
            "max_lines_pp":0,
            "max_textboxes_pp":0,
            "max_blockstyles_pp":0,
            "max_textbox_space_pp":0,
            "max_blockstyle_space_pp":0,

            "min_words_pp":0,
            # "min_bold_words_pp":0,
            "min_lines_pp":0,
            "min_textboxes_pp":0,
            "min_blockstyles_pp":0,
            "min_textbox_space_pp":0,
            "min_blockstyle_space_pp":0,

            "mean_words_per_line":0,
            "dev_words_per_line":0,
            "mean_lines_per_blockstyle":0,
            "dev_lines_per_blockstyle":0,
            "max_lines_per_blockstyle":0,

            #STRUCTURE STUFF
            "modal_right":0,
            "perc_modal_right":0,
            "max_right":0,
            "modal_left":0,
            "perc_modal_left":0,
            "max_lefts":0,

            "modal_textbox_columns_pp":0,
            "perc_modal_textbox_columns_pp":0,
            "min_textbox_columns_pp":0,
            "max_textbox_columns_pp":0,
            "modal_blockstyle_columns_pp":0,
            "perc_modal_blockstyle_columns_pp":0,
            "min_blockstyle_columns_pp":0,
            "max_blockstyle_columns_pp":0,

            #FREE STUFF
            "total_free_space":0,
            "mean_free_space_pp":0,
            "dev_free_space_pp":0,
            "max_free_space_pp":0,
            "min_free_space_pp":0
        }

        for k,v in features.items():
            if(not(type(v)==str)):
                features[k] = np.float64(v)
        return features

    def clean_files(self):
        for img_path in self.images["pathes"]:
            if(isfile(img_path)):
                os.remove(img_path)
            else:
                print("Image not found: " + str(img_path))
        xml_file = splitext(self.doc_path)[0] + ".xml"
        if(isfile(xml_file)):
            os.remove(xml_file)
        else:
            print("Password protected: " + self.doc_id)

    def remove_invalid_xml_char(self, content):
        content = list(content)
        for i in range(len(content)):
            content[i] = remove_invalid_xml_char(content[i])


        content = "".join(content)
        return content

    def remove_invalid_xml_form(self, content):
        # split text into lines
        lines = content.split("\n")
        # remove invalid lines until no err
        error = True
        while(error):
            try:
                text = "\n".join(lines)
                tree = ET.fromstring(text)
                error = False
            except ET.ParseError as e:
                e_parts = str(e).split(":")
                if(e_parts[0]=="mismatched tag"):
                    line,column = e_parts[1].split(",")
                    line = int(line.split(" line ")[1])
                    column = int(column.split(" column ")[1])
                    # print(e)
                    # print(lines[line-1])
                    del lines[line-1]
                else:
                    error = False

        content = "\n".join(lines)
        return content


def get_structure_features(abs_filepath):
    doc = Document(abs_filepath)
    err_message = doc.process_xml(img_flag=False)
    if(not(err_message)):
        f_dict = doc.get_feature_dict()
    else:
        f_dict = doc.get_error_features(err_message)
    doc.clean_files()
    doc = None
    return (f_dict, abs_filepath)

def pre_extract_pdf_structure_data(doc_dir, doc_ids=None, num_cores=None):
    files = []

    if isdir(doc_dir):
        if(doc_ids is None):
            for root, dirs, fls in os.walk(doc_dir):
                for name in fls:
                    if splitext(basename(name))[1] == '.pdf':
                        files.append(join(root,name))
        else:
            for d_id in doc_ids:
                files.append(join(doc_dir,d_id+".pdf"))

    else:
        print("Error: You need to specify a path to the folder containing all files.")
        sys.exit(1)

    if(not(num_cores is None) and num_cores>1):
        pool = Pool(num_cores)
        res = pool.map(get_structure_features, batch_files)
    else:
        res = []
        for f in files:
            res.append(get_structure_features(f))
    res_fix={}
    for x in res:
        d_id = splitext(basename(x[1]))[0]
        doc_features = x[0]
        res_fix[d_id] = doc_features

    return res_fix

def pre_extract_pdf_structure_data_to_file(doc_dir, text_dir, structure_file, doc_ids=None, num_cores=1, batch_size=None):
    files = []

    if isdir(doc_dir):
        if(doc_ids is None):
            for root, dirs, fls in os.walk(doc_dir):
                for name in fls:
                    if splitext(basename(name))[1] == '.pdf':
                        files.append(join(root,name))
        else:
            for d_id in doc_ids:
                files.append(join(doc_dir,d_id+".pdf"))

    else:
        print("Error: You need to specify a path to the folder containing all files.")
        sys.exit(1)

    if(not(isdir(text_dir))):
        os.makedirs(text_dir)
    if(batch_size is None):
        batch_size = len(files)

    for i in range(0,len(files),batch_size):
        batch_files = files[i:min(i+batch_size,len(files))]

        if(not(num_cores is None) and num_cores>1):
            pool = Pool(num_cores)
            res = pool.map(get_structure_features, batch_files)
        else:
            res = []
            for f in files:
                res.append(get_structure_features(f))
        res_fix={}
        for x in res:
            d_id = splitext(basename(x[1]))[0]
            doc_features = x[0]
            with open(join(text_dir,d_id+".txt"),"w") as f:
                f.write(x[0]["text"])
            del doc_features["text"]
            res_fix[d_id] = doc_features

        if(isfile(structure_file)):
            with open(structure_file, 'r') as fp:
                structure_data = json.load(fp, encoding="utf-8")
            structure_data.update(res_fix)
        else:
            structure_data = res_fix

        with open(structure_file, 'w') as fp:
            json.dump(structure_data, fp, indent=4)
        print("%.1f%% done!"%((i+batch_size)/len(files)*100,))

        structure_data = None
        res_fix = None

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

def remove_invalid_xml_char(c):
    illegal_unichrs = [ (0x00, 0x08), (0x0B, 0x1F), (0x7F, 0x84), (0x86, 0x9F),
                    (0xD800, 0xDFFF), (0xFDD0, 0xFDDF), (0xFFFE, 0xFFFF),
                    (0x1FFFE, 0x1FFFF), (0x2FFFE, 0x2FFFF), (0x3FFFE, 0x3FFFF),
                    (0x4FFFE, 0x4FFFF), (0x5FFFE, 0x5FFFF), (0x6FFFE, 0x6FFFF),
                    (0x7FFFE, 0x7FFFF), (0x8FFFE, 0x8FFFF), (0x9FFFE, 0x9FFFF),
                    (0xAFFFE, 0xAFFFF), (0xBFFFE, 0xBFFFF), (0xCFFFE, 0xCFFFF),
                    (0xDFFFE, 0xDFFFF), (0xEFFFE, 0xEFFFF), (0xFFFFE, 0xFFFFF),
                    (0x10FFFE, 0x10FFFF) ]

    illegal_ranges = ["%s-%s" % (chr(low), chr(high))
                  for (low, high) in illegal_unichrs
                  if low < sys.maxunicode]

    illegal_xml_re = re.compile(u'[%s]' % u''.join(illegal_ranges))
    if illegal_xml_re.search(c) is not None:
        #Replace with space
        return ' '
    else:
        return c

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

    spec_doc = "8bd58c19720d9cc35d119f965b1ef601" #f1dfc962076e1f7f89cdaf8efb5cb314
    test_path = join(DATA_PATH,"pdf_files")
    # test_path = join(DATA_PATH,"files_test_html")

    # for xe in xml_error:
    #     print(xe)
    #     get_structure_features(join(test_path,xe+".pdf"))

    # get_structure_features(join(test_path,spec_doc+".pdf"))

    # doc_dict = {
    #     "powerpoint1" : "76ae7c120910a7830a1c0e0262d8cc5e.pdf",
    #     "powerpoint2" : "15a25c0553f25d4edcc288028b384cba.pdf",
    #     "slide_overview1" : "1e733d496d75352df67daefe269c1e88.pdf",
    #     "slide_overview2" : "5e9f1f979fff677b72e6228ded542a97.pdf",
    #     "lecture_mats1" : "1fd731e88a30612291de1923d5fa5263.pdf",
    #     "lecture_mats2" : "26286ffa140c615eb9a5a4ab46eb30be.pdf",
    #     "lecture_mats3" : "185422bf26436452aa0b3b8247e322af.pdf",
    #     "book_style1" : "37cf51662d8385b47ad00d36070766b0.pdf",
    #     "scan1" : "1569e3de486040aaaf71653c8e4bee6d.pdf",
    #     "scan2" : "c92b478470f9147ea02229ac7de22adc.pdf",
    #     "scan3" : "0b561e7ffe8da5a589c7e33e55203de6.pdf",
    #     "table1" : "1945835eac5a4162cc00ba89c30e6a90.pdf",
    #     "paper_style1" : "0f6b08591d82390c4ad1a590266f92bb.pdf",
    #     "long_doc" : "c41effe246dd564d7c72416faca33c21.pdf",
    #     "pw_protect" : "26013961e27e976cf7dff7b5bc6086c6.pdf",
    #     "warning" : "db8f79a0bbe5bb518a54eb92f5b9b499.pdf",
    #     "image_problem" : "d659934e59b496a11b7bffb22eabbba9.pdf"
    # }
    # # docs = [lecture_mats1, lecture_mats2, book_style1, table1, paper_style1]
    # docs = [doc_dict["pw_protect"]]
    #
    # doc_ids = [splitext(d)[0] for d in doc_dict.values()]
    #
    # # print(get_structure_features("../../data/pdf_files/37cf51662d8385b47ad00d36070766b0.pdf"))
    #
    s1 = time()
    pre_extract_pdf_structure_data_to_file(
        doc_dir="../../data/pdf_files",
        text_dir="../../data/xml_text_files",
        structure_file="../../data/pre_extracted_data/xml_text_structure.json",
        doc_ids=xml_error,
        num_cores=4,
        batch_size=10)
    print(time()-s1)


    # print(img_stuff(image_pathes))
    # # show time discrepancy
    # st = time()
    # features.pdf_structure.process_file(doc)
    # print(time()-st)

    # st = time()
    # print(develop.pdf_images.get_grayscale_entropy_tmpfile(doc))
    # print(time()-st)
