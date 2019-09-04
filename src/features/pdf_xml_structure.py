# TODO: consider parsing the document manually

# TODO: think about how to get the best vertical distance consider one vertical distance per font type

# TODO: heights need to be completely evaluated first otherwise divergences from the real hight at the beginning make some good heights worse

import sys, os, json, codecs, gc
from collections import Counter
from os.path import join, realpath, dirname, basename, isfile, splitext
import logging

SRC_DIR = os.path.abspath(join(join(realpath(__file__), os.pardir),os.pardir))
if(not(SRC_DIR in sys.path)):
    sys.path.append(SRC_DIR)

from time import time
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
np.seterr(all='raise')
from scipy import stats

import re

FNULL = open(os.devnull, 'w')

logging.config.fileConfig(fname='log.conf', disable_existing_loggers=False)
# Get the logger specified in the file
logger = logging.getLogger("copydocLogger")
debuglogger = logging.getLogger("debugLogger")

num_docs=0
FEATURES = {
    "count_pages":0,
    "count_outline_items":0,

    # FONTS
    "count_fonts":0,
    "count_font_colors":0,
    "count_font_families":0,
    "max_font_size":0,
    "min_font_size":0,
    "main_font_size":0,
    "perc_main_font_word":0,

    # CONTENT
    "text":"empty",
    "bold_text":"empty",
    "first_100_words":"empty",
    "last_100_words":"empty",
    "copyright_symbol":0,

    # WORDS
    "mean_words_pp":0,
    "mean_lines_pp":0,
    "mean_words_per_line":0,
    "mean_bold_words_pp":0,
    "mean_annotations_pp":0,

    #IMAGES AND FREE SPACE
    "image_error":0,
    "mean_image_space_pp":0,
    "mean_images_pp":0,
    "max_image_page_ratio":0,
    "mean_free_space_pp":0,

    # TEXT BOXES
    "mean_textboxes_pp":0,
    "mean_lines_per_textbox":0,
    "mean_blockstyles_pp":0,
    "mean_lines_per_blockstyle":0,

    "modal_textbox_columns_pp":0,
    "perc_modal_textbox_columns_pp":0,
    "modal_blockstyle_columns_pp":0,
    "perc_modal_blockstyle_columns_pp":0,

    "mean_textbox_space_pp":0,
    "mean_blockstyle_space_pp":0,

    # MARGINS
    "modal_right":0,
    "perc_modal_right":0,
    "modal_left":0,
    "perc_modal_left":0,
    "max_right":0,
    "max_left":0,
    "min_left":0,
}

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
        return "top:" + str(self.top_pos) + " bot:" + str(self.bot_pos) + \
        " left:" + str(self.left_pos) + " lines:" + str(len(self.line_ids))

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
        return "top:" + str(self.top_pos) + " bot:" + str(self.bot_pos) +\
        " left:" + str(self.left_pos) + " right:" + str(self.right_pos) + " lines:" + str(len(self.line_ids))

    def add_line(self, text_line, line_idx, norm_dist, dist_dev=1, margin_dev=1):
        line_dist = text_line[0] - self.bot_pos
        dist_criterion = line_dist <= norm_dist+dist_dev and line_dist >= norm_dist-dist_dev

        right_dif = self.right_pos - (text_line[1]+text_line[2])
        right_criteron = right_dif <= margin_dev and right_dif >= (-margin_dev)

        if(dist_criterion and right_criteron):
            self.bot_pos = self.bot_pos + line_dist + text_line[3]
            self.line_ids.append(line_idx)
            self.text += "\n" + text_line[5]
            return True

        return False

    def add_block(self, block):
        self.bot_pos = block.bot_pos
        self.left_pos = min(self.left_pos, block.left_pos)
        self.right_pos = max(self.right_pos, block.right_pos)
        self.line_ids.extend(block.line_ids)
        self.text += ("\n" + block.text)

class Page(object):

    def __init__(self, xml_page, number):
        self.number = number
        self.xml_page = xml_page
        self.width = int(xml_page.attrib["width"])
        self.height = int(xml_page.attrib["height"])
        self.space = self.width*self.height
        self.image_space = 0
        self.img_rects = []

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

        self.text_img_overlap = 0
        self.img_img_overlap = 0

    def structure_content_elements(self, doc):
        page_line_index = 0
        invisible_fonts = []
        image_counter = np.zeros(50)
        for elem in self.xml_page:
            if elem.tag == "fontspec":
                if(int(elem.attrib["size"])<=0):
                    invisible_fonts.append(elem.attrib["id"])
                doc.fontspecs["count"]+=1
                doc.fontspecs["families"].add(elem.attrib["family"])
                doc.fontspecs["colors"].add(elem.attrib["color"])
                doc.fontspecs["spec_counts"].append(0)
                doc.fontspecs["sizes"].append(int(elem.attrib["size"]))
                doc.fontspecs["heights"].append({})
                doc.fontspecs["best_heights"].append(-1)
            elif elem.tag == "image":
                doc.images["patches"].append(elem.attrib["src"])
                # get image rect positions
                top = int(elem.attrib["top"])
                left = int(elem.attrib["left"])
                width = int(elem.attrib["width"])
                height = int(elem.attrib["height"])
                # make sure values are positive (page overlap is cut off)
                # sometimes width and height are negative because the startpoint is bottom right
                if width < 0:
                    left += width
                    width = abs(width)
                if height < 0:
                    top += height
                    height = abs(height)
                if top < 0:
                    top = 0
                if left < 0:
                    left = 0
                if (top+height) > self.height:
                    height = self.height-top
                if (left+width) > self.width:
                    width = self.width-left
                img_rect = (left, top, left + width, top + height)
                size = height * width
                #avoid negative size
                if size < 0: size = size * -1
                if size < 50:
                    image_counter[size] += 1
                    if image_counter[size] > 50:
                        raise ValueError("Too many images", "ImageError")
                if not(img_rect in self.img_rects) and size > 1:
                    overlap = 0
                    for r in self.img_rects:
                        o = rectOverlap(img_rect,r)
                        if o:
                            overlap += o
                            self.img_img_overlap += o
                    self.img_rects.append(img_rect)
                    doc.images["sizes"].append(size)
                    self.image_space += max([(size-overlap),0])
            elif elem.tag == "text":
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
                    if not(format_elem.text is None):
                        text_list[5] += format_elem.text + " "
                        if format_elem.tag == "b":
                            if(text_list[6]==0 or text_list[6]==1):
                                text_list[6] = 1
                            else:
                                text_list[6] = 4
                            doc.bold_text += format_elem.text + " "
                            doc.counts["bold_words"] += len(format_elem.text.split())
                        elif format_elem.tag == "a":
                            if(text_list[6]==0 or text_list[6]==2):
                                text_list[6] = 2
                            else:
                                text_list[6] = 4
                            doc.counts["annotations"] += 1
                        elif format_elem.tag == "i":
                            if text_list[6] == 0 or text_list[6] == 3:
                                text_list[6] = 3
                            else:
                                text_list[6] = 4
                        else:
                            debuglogger.error("Unexpected text formatter found, namely %s"%(format_elem.tag,))

                doc.text += text_list[5]
                text_list[5] = text_list[5][:-1]
                if text_list[3] != doc.fontspecs["best_heights"][text_list[4]]:
                    h_dict = doc.fontspecs["heights"][text_list[4]]
                    try:
                        h_dict[text_list[3]] += 1
                    except KeyError:
                        h_dict[text_list[3]] = 1
                    best_height = max(h_dict.items(), key=lambda h_dict: h_dict[1])
                    doc.fontspecs["best_heights"][text_list[4]] = best_height[0]
                    if text_list[3] != best_height[0] and len(self.text_lines) > 0:
                        prev_line = self.text_lines[-1]

                        h_dist = text_list[1]-(prev_line[1]+prev_line[2])
                        if h_dist >= 0 and h_dist<10:

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

                    else:
                        h_dict[text_list[3]] = 1
                else:
                    doc.fontspecs["heights"][text_list[4]][text_list[3]] += 1

                if len(text_list[5]) > 0:
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
                debuglogger.error("Unexpected tag found %s", elem.tag)
                debuglogger.Debug("Tag, attribute and text %sv %s %s", elem.tag,elem.attrib,elem.text)

    def structure_text_alignment(self):
        # get the distances of consecutive lines and the number of lines
        self.v_dists = []
        self.h_dists = []
        mult_counter = 0
        self.sorted_tops = sorted(self.tops.items(), key=lambda x: x[0], reverse=False)
        for i in range(1,len(self.sorted_tops)):
            if len(self.sorted_tops[i][1]) > 1:
                self.check_multiple_lines(self.sorted_tops[i][1])
                lines = [self.text_lines[l_id] for l_id in self.sorted_tops[i][1]]
                sorted_lines = sorted(lines, key=lambda x: x[1], reverse=False)
                for pos in range(1,len(sorted_lines)):
                    self.h_dists.append(sorted_lines[pos][1]-(sorted_lines[pos-1][1]+sorted_lines[pos-1][2]))
                mult_counter += 1
            possible_successor = False
            forward = 0
            current_text_line = self.text_lines[self.sorted_tops[i-1][1][0]]
            while not(possible_successor) and len(self.sorted_tops) > i+forward:
                next_text_line = self.text_lines[self.sorted_tops[i+forward][1][0]]
                height = current_text_line[3]
                vert_dist = self.sorted_tops[i+forward][0] - (self.sorted_tops[i-1][0]+height)
                if vert_dist > 0:
                    possible_successor = True
                    same_font = next_text_line[4] == current_text_line[4]
                elif (abs(vert_dist) / height) < 0.15:
                    similar_left = abs(current_text_line[1] - next_text_line[1])<=2
                    current_right = current_text_line[1]+current_text_line[2]
                    next_right = next_text_line[1]+next_text_line[2]
                    similar_right = abs(current_right - next_right)<=2
                    fit_inside = np.argmin([current_text_line[1],next_text_line[1]]) == np.argmax([current_right,next_right])
                    if similar_left or  similar_right or fit_inside:
                        possible_successor = True
                        same_font = next_text_line[4] == current_text_line[4]
                    else:
                        forward += 1
                else:
                    forward += 1
            if possible_successor and same_font:
                self.v_dists.append(vert_dist)

        self.h_dists = Counter(self.h_dists)
        self.h_dists = self.h_dists.most_common()

        self.v_dists = Counter(self.v_dists)
        self.v_dists = self.v_dists.most_common()

        if len(self.v_dists) > 0:
            self.norm_dist = self.v_dists[0][0]

        # get the hists of left_margins
        left_counts = [(key,len(val)) for key,val in self.lefts.items() if(len(val)>1)]
        self.sorted_left_counts = sorted(left_counts, key=lambda x: x[1], reverse=True)
        # get the hists of right_margins
        right_counts = [(key,len(val)) for key,val in self.rights.items()]
        self.sorted_right_counts = sorted(right_counts, key=lambda x: x[1], reverse=True)

    def check_multiple_lines(self, line_idc):
        lines = [(self.text_lines[l_id],l_id)for l_id in line_idc]
        sorted_lines = sorted(lines, key=lambda x: x[0][1], reverse=False)
        merges = []
        for pos in range(1,len(sorted_lines)):
            formatted = sorted_lines[pos-1][0][6]>0 or sorted_lines[pos][0][6]>0
            v_dist = sorted_lines[pos][0][1]-(sorted_lines[pos-1][0][1]+sorted_lines[pos-1][0][2])
            outlier = len(self.lefts[sorted_lines[pos][0][1]])<=2
            if v_dist < 10 and (formatted or outlier):
                merges.append((sorted_lines[pos-1][1],sorted_lines[pos][1]))
        for m in reversed(merges):
            self.merge_lines(m)

    def merge_lines(self, merge_tuple):
        # get the line index of the lines which shall be merged
        left_id = merge_tuple[0]
        right_id = merge_tuple[1]
        # update the entries in the dictionaries
        # delete the tops entry of the right line
        self.tops[self.text_lines[right_id][0]].remove(right_id)
        # delete the lefts entry of the right line
        self.lefts[self.text_lines[right_id][1]].remove(right_id)
        if len(self.lefts[self.text_lines[right_id][1]]) == 0:
            del self.lefts[self.text_lines[right_id][1]]
        # update the rights entry
        # delete the right id an put the left id in its place
        self.rights[self.text_lines[right_id][1]+self.text_lines[right_id][2]].remove(right_id)
        self.rights[self.text_lines[right_id][1]+self.text_lines[right_id][2]].append(left_id)
        # remove the left id
        self.rights[self.text_lines[left_id][1]+self.text_lines[left_id][2]].remove(left_id)
        if len(self.rights[self.text_lines[left_id][1] + self.text_lines[left_id][2]]) == 0:
            del self.rights[self.text_lines[left_id][1]+self.text_lines[left_id][2]]
        # compute new width = (right offset + right width) - left_offset
        self.text_lines[left_id][2] = (self.text_lines[right_id][1]+self.text_lines[right_id][2]) - self.text_lines[left_id][1]
        # compute new text = left text + right text
        self.text_lines[left_id][5] += (" " + self.text_lines[right_id][5])
        # compute new formatter as 4 if there are two different ones and the maximum otherwise
        lf = self.text_lines[left_id][6]
        rf = self.text_lines[right_id][6]
        if lf > 0 and rf > 0 and not(lf == rf):
            self.text_lines[left_id][6] = 4
        else:
            self.text_lines[left_id][6] = max(lf,rf)
        # delete the right line
        self.text_lines[right_id] = None

    def check_blockstyle_evidence(self, doc):
        max_rights = self.sorted_right_counts[0][1]
        if self.sorted_right_counts[0][0] + 1 in self.rights:
            max_rights += len(self.rights[self.sorted_right_counts[0][0]+1])
        if self.sorted_right_counts[0][0] - 1 in self.rights:
            max_rights += len(self.rights[self.sorted_right_counts[0][0]-1])
        block_thres = int(np.log(np.power(len(self.sorted_tops)-1,3)))
        block_style = self.sorted_right_counts[0][1] >= block_thres
        return block_style

    def merge_text_blockstyle(self):
        if len(self.v_dists) > 0:
            norm_dist = self.v_dists[0][0]
            for (top_lvl, line_ids) in self.sorted_tops:
                for l_id in line_ids:
                    added = False
                    if self.text_lines[l_id][1] in self.tbox_blockstyle_left:
                        for tbox_id in self.tbox_blockstyle_left[self.text_lines[l_id][1]]:
                            tbox = self.tbox_blockstyle_top[tbox_id]
                            if tbox.add_line(text_line=self.text_lines[l_id], \
                                             line_idx=l_id, norm_dist=norm_dist, \
                                             dist_dev=1, margin_dev=5):
                                added = True
                                break
                        if not(added):
                            block_id = len(self.tbox_blockstyle_top)
                            nb = Text_Box_Block(self.text_lines[l_id], block_id, l_id)
                            self.tbox_blockstyle_left[self.text_lines[l_id][1]].append(block_id)
                            self.tbox_blockstyle_top.append(nb)
                    else:
                        if not(added):
                            block_id = len(self.tbox_blockstyle_top)
                            nb = Text_Box_Block(self.text_lines[l_id], block_id, l_id)
                            self.tbox_blockstyle_left[self.text_lines[l_id][1]] =  [block_id]
                            self.tbox_blockstyle_top.append(nb)

    def check_blockstyle_boundaries(self):
        norm_dist = self.v_dists[0][0]
        for i in range(len(self.tbox_blockstyle_top)-1,-1,-1):
            if self.tbox_blockstyle_top[i] is None or len(self.tbox_blockstyle_top[i].line_ids) <= 2:
                continue
            else:
                big_block = self.tbox_blockstyle_top[i]
                top_line = big_block.top_pos - norm_dist
                bottom_line = big_block.bot_pos + norm_dist
                left_margin = big_block.left_pos
                right_margin = big_block.right_pos

                # get possible belowbox
                k=i+1
                while k < len(self.tbox_blockstyle_top):
                    bb = self.tbox_blockstyle_top[k]
                    if bb is None or len(bb.line_ids) > 1:
                        k+=1
                        continue
                    bb = self.tbox_blockstyle_top[k]
                    dist_crit = bb.top_pos>=bottom_line-1 and bb.top_pos<=bottom_line+1
                    left_crit = bb.left_pos>=left_margin-1 and bb.left_pos<=left_margin+1
                    right_crit = bb.right_pos<right_margin+1
                    if dist_crit and left_crit and right_crit:
                        self.tbox_blockstyle_left[bb.left_pos].remove(k)
                        self.tbox_blockstyle_left[big_block.left_pos].remove(i)
                        self.tbox_blockstyle_left[min(bb.left_pos,big_block.left_pos)].append(i)
                        if len(self.tbox_blockstyle_left[bb.left_pos]) == 0:
                            del self.tbox_blockstyle_left[bb.left_pos]
                        if len(self.tbox_blockstyle_left[big_block.left_pos]) == 0:
                            del self.tbox_blockstyle_left[big_block.left_pos]
                        big_block.add_block(bb)
                        self.tbox_blockstyle_top[k] = None
                    elif bb.top_pos > bottom_line+1:
                        break
                    else:
                        k+=1

                # get possible topbox
                j=i-1
                while j >= 0:
                    tb = self.tbox_blockstyle_top[j]
                    if tb is None or len(tb.line_ids) > 1:
                        j-=1
                        continue
                    dist_crit = tb.bot_pos>=top_line-1 and tb.bot_pos<=top_line+1
                    left_crit = tb.left_pos>=left_margin+10 and tb.left_pos<=left_margin+25
                    right_crit = tb.right_pos>=right_margin-1 and tb.right_pos<=right_margin+1
                    if dist_crit and left_crit and right_crit:
                        self.tbox_blockstyle_top[i] = None
                        self.tbox_blockstyle_left[big_block.left_pos].remove(i)
                        self.tbox_blockstyle_left[tb.left_pos].remove(j)
                        self.tbox_blockstyle_left[min(tb.left_pos,big_block.left_pos)].append(j)
                        if len(self.tbox_blockstyle_left[big_block.left_pos]) == 0:
                            del self.tbox_blockstyle_left[big_block.left_pos]
                        if len(self.tbox_blockstyle_left[tb.left_pos]) == 0:
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
        while i < len(self.tbox_blockstyle_top):
            box = self.tbox_blockstyle_top[i]
            if not (box is None) and len(box.line_ids) <= 2:
                current_block = Text_Box(box)
                bottom_line = current_block.bot_pos + norm_dist
                left_margin = current_block.left_pos
                # get possible belowbox
                box_index = i
                k=i+1
                while k < len(self.tbox_blockstyle_top):
                    bb = self.tbox_blockstyle_top[k]
                    if bb is None:
                        k+=1
                        continue
                    elif bb.top_pos > bottom_line+1:
                        try:
                            self.tbox_left[current_block.left_pos].append(len(self.tbox_top))
                        except KeyError:
                            self.tbox_left[current_block.left_pos] = [len(self.tbox_top)]
                        self.tbox_top.append(current_block)

                        self.tbox_blockstyle_left[current_block.left_pos].remove(box_index)
                        if len(self.tbox_blockstyle_left[current_block.left_pos]) == 0:
                            del self.tbox_blockstyle_left[current_block.left_pos]
                        self.tbox_blockstyle_top[box_index] = None
                        break

                    elif len(bb.line_ids) > 2:
                        k+=1
                    else:
                        bb = self.tbox_blockstyle_top[k]
                        dist_crit = bb.top_pos>=bottom_line-1 and bb.top_pos<=bottom_line+1
                        left_crit = bb.left_pos>=left_margin-1 and bb.left_pos<=left_margin+1
                        if dist_crit and left_crit:
                            current_block.add_block(bb)
                            bottom_line = current_block.bot_pos + norm_dist
                            left_margin = current_block.left_pos
                            self.tbox_blockstyle_top[k] = None
                            self.tbox_blockstyle_left[bb.left_pos].remove(k)
                            if len(self.tbox_blockstyle_left[bb.left_pos]) == 0:
                                del self.tbox_blockstyle_left[bb.left_pos]
                        k+=1
                if k >= len(self.tbox_blockstyle_top):
                    try:
                        self.tbox_left[current_block.left_pos].append(len(self.tbox_top))
                    except KeyError:
                        self.tbox_left[current_block.left_pos] = [len(self.tbox_top)]
                    self.tbox_top.append(current_block)

                    self.tbox_blockstyle_left[current_block.left_pos].remove(box_index)
                    if len(self.tbox_blockstyle_left[current_block.left_pos]) == 0:
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
        self.pages = []
        self.text = ""
        self.bold_text = ""

        self.images = {
            "patches":[],
            "sizes":[]
        }
        self.fontspecs = {
            "families":set(),
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
            "bold_words":0,
            "annotations":0,
            "lines":0,
            "textboxes":0,
            "blockstyles":0,
            "words":0
        }

        self.stat_lists = {
            "image_space_pp":[],
            "page_space_pp":[],
            "free_space_pp":[],
            "img_img_overlap_pp":[],
            "text_img_overlap_pp":[],
            "words_pp":[],
            "lines_pp":[],
            "textboxes_pp":[],
            "blockstyles_pp":[],
            "textbox_space_pp":[],
            "blockstyle_space_pp":[],
            "word_pl":[],
            "lines_pbs":[],
            "lines_ptb":[],
            "textbox_columns":[],
            "blockstyle_columns":[],
            "blockstyles_words_pp":[],
            "not_blockstyles_words_pp":[],
            "not_blockstyles_lines_pp":[],
            "blockstyles_lines_pp":[]
        }

        self.rights = {}
        self.lefts = {}
        self.xml_path = ""

    def process_xml(self):
        #get the xml file
        err = self.get_xml_structure()
        content = ""
        global num_docs
        from time import time
        t_parse_0 = time()

        #try to open the file
        #pdftohtml seems create not utf-8 encoded 
        #chars even when utf-8 requested. Decode the file 
        #using latin one in case of those few exceptions.
        try:
            with codecs.open(self.xml_path, "r", "utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            with codecs.open(self.xml_path, "r", "latin1") as f:
                content = f.read()
        #if no file was created the pdf could not be accessedd
        except FileNotFoundError:
            num_docs = num_docs + 1
            logger.error("Error %s -- documents with errors %s. Document: %s."%(err,num_docs,join(dirname(self.xml_path),self.doc_id)))
            debuglogger.error("Error %s -- documents with errors %s. Document: %s."%(err,num_docs,join(dirname(self.xml_path),self.doc_id)))
            e=""
            if err: e = err.split()[0] 
            if e == "Permission":    
                return err
            elif e == "Syntax":
                return err
            elif e == "Command":
                return err
            return "pw_protected"
        
        clean_files(dirname(self.xml_path),self.doc_id)
        gc.collect()
        #parse the xml
        try:
            amount_content = len(list(content))
            debuglogger.info("Starting to get element tree from file %s. Size %s."%(self.doc_id, amount_content))
            tree = ET.fromstring(content)
            debuglogger.info("Finished getting element tree from file %s"%(self.doc_id))
        #if the xml parser found an error
        except ET.ParseError as e:
            #the error description has usually two parts
            #error type : position of the error
            logger.info("Problem parsing xml file %s %s"%(self.doc_id, e))
            debuglogger.info("Problem parsing xml file %s %s"%(self.doc_id, e))
            logger.info("Trying to fix the problem.")
            e_parts = str(e).split(":")
            #remove invalid xml tokens
            if e_parts[0] == "not well-formed (invalid token)":
                debuglogger.info("Removing invalid characters from file %s"%(self.doc_id))
                content = self.remove_invalid_xml_char(content)
            #remove not matching tags
            elif e_parts[0] == "mismatched tag":
                debuglogger.info("Removing invalid xml forms from file %s"%(self.doc_id))
                content = self.remove_invalid_xml_form(content)

            #try again to parse the xml
            try:
                debuglogger.info("Starting to get element tree from file %s"%(self.doc_id))
                tree = ET.fromstring(content)
                debuglogger.info("Finished getting element tree from file %s"%(self.doc_id))
            #if errors persist no idea
            except ET.ParseError:
                num_docs = num_docs + 1
                logger.info("Problem could not get fixed.")
                t_parse_1 = time() - t_parse_0
                debuglogger.error("XML parsing error in: %s. Time it took: %s. Number of docs that have failed: %s"%(self.doc_id, t_parse_1, num_docs))
                logger.error("Problem parsing xml file %s. Time it took: %s. Number of docs that have failed: %s"%(self.doc_id, t_parse_1, num_docs))
                clean_files(dirname(self.xml_path),self.doc_id)
                return "xml_error"
        #get the head of the tree
        root = tree
        debuglogger.info("Navigating the pages for file %s."%self.doc_id)
        #go through all pages
        for root_obj in root:
            if root_obj.tag == "page":
                self.counts["pages"] += 1
                self.process_page(root_obj, self.counts["pages"])
            elif root_obj.tag == "outline":
                self.counts["outline_items"] += 1
                self.process_outline(root_obj)
            else:
                debuglogger.error("First child is not a page or outline but %s",root_obj.tag)
                debuglogger.debug("Tag attribute and text %s %s",root_obj.tag,root_obj.attrib,root_obj.text)
                for elem in root_obj.iter():
                    debuglogger.debug("In case of problem all elements are printed %s %s %s",\
                                        elem.tag,elem.attrib,elem.text)
        t_parse_1 = time() - t_parse_0
        debuglogger.info("Finished navigating the pages. Time it took: %s. Number of pages for file %s: %s."%(t_parse_1, self.doc_id, self.counts['pages']))
        return ""

    def get_xml_structure(self):
        args = ["pdftohtml"]
        args += ["-xml","-enc","UTF-8", self.doc_path]
        self.xml_path = " "
        try:
            sprocess = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            err = sprocess.communicate(timeout=90)[1].decode(errors="ignore")
            self.xml_path = splitext(self.doc_path)[0] + ".xml"
        except:
            logger.error("Timeout during structure extraction for file %s"%self.doc_path)
            debuglogger.error("Timeout during structure extraction for file %s"%self.doc_path)
            err = "timeout"
        sprocess.terminate()
        return err

    def process_page(self, xml_page, number):
        page = Page(xml_page, number)
        page.structure_content_elements(self)
        if len(page.text_lines) > 0:
            page.structure_text_alignment()
            if(not(page.norm_dist is None)):
                page.merge_text_blockstyle()
                page.check_blockstyle_boundaries()
                page.merge_text_not_blocked()
            else:
                page.create_text_blocks()
        self.update_page_stats(page)
        self.pages.append(page)

    def process_outline(self, xml_outline):
        outline_iter = xml_outline.iter()
        next(outline_iter)
        for elem in outline_iter:
            if elem.tag == "outline":
                continue
            elif elem.tag == "item":
                self.counts["outline_items"] += 1
            else:
                debuglogger.error("Unexpected outline tag found %s",elem.tag)
                debuglogger.debug("Tag, attribute and text %s %s %s",elem.tag,elem.attrib,elem.text)

    def update_page_stats(self, page):
        self.stat_lists["lines_pp"].append(0)
        self.stat_lists["words_pp"].append(0)
        self.stat_lists["not_blockstyles_words_pp"].append(0)
        self.stat_lists["blockstyles_words_pp"].append(0)
        self.stat_lists["not_blockstyles_lines_pp"].append(0)
        self.stat_lists["blockstyles_lines_pp"].append(0)
        for line in page.text_lines:
            if not(line is None):
                self.text += line[5] + " "
                word_cnt = len(line[5].split())
                self.counts["words"] += word_cnt
                self.stat_lists["words_pp"][-1] += word_cnt
                self.stat_lists["word_pl"].append(word_cnt)
                self.counts["lines"] += 1
                self.stat_lists["lines_pp"][-1] += 1

        self.stat_lists["textboxes_pp"].append(0)
        self.stat_lists["textbox_space_pp"].append(0)
        text_img_overlap = 0
        tb_tb_o = 0
        tb_rects = []
        for tb in page.tbox_top:
            if not(tb is None):
                left = min(max(0,tb.left_pos),page.width)
                top = min(max(0,tb.top_pos),page.height)
                right = min(tb.right_pos,page.width)
                bot = min(tb.bot_pos,page.height)
                tbox_rect = (left,top,right,bot)
                size = (right-left) * (bot-top)
                if size > 0:
                    self.counts["textboxes"] += 1
                    self.stat_lists["textboxes_pp"][-1] += 1
                    self.stat_lists["lines_ptb"].append(len(tb.line_ids))
                    self.stat_lists["not_blockstyles_words_pp"][-1] += len(tb.text.split())
                    self.stat_lists["not_blockstyles_lines_pp"][-1] += len(tb.line_ids)
                    if not(tbox_rect in tb_rects):
                        self.stat_lists["textbox_space_pp"][-1] += (right-left) * (bot-top)
                        for r in page.img_rects:
                            o = rectOverlap(tbox_rect,r)
                            if(o):
                                text_img_overlap += o
                        for r in tb_rects:
                            ot = rectOverlap(tbox_rect,r)
                            if(ot):
                                tb_tb_o += ot
                        tb_rects.append(tbox_rect)

        self.stat_lists["blockstyles_pp"].append(0)
        self.stat_lists["blockstyle_space_pp"].append(0)
        for tb in page.tbox_blockstyle_top:
            if not(tb is None) and len(tb.line_ids) >= 3:
                left = min(max(0,tb.left_pos),page.width)
                top = min(max(0,tb.top_pos),page.height)
                right = min(tb.right_pos,page.width)
                bot = min(tb.bot_pos,page.height)
                tbox_rect = (left,top,right,bot)
                size = (right-left) * (bot-top)
                if size > 0:
                    self.counts["blockstyles"] += 1
                    self.stat_lists["blockstyles_pp"][-1] += 1
                    self.stat_lists["lines_pbs"].append(len(tb.line_ids))
                    self.stat_lists["blockstyles_words_pp"][-1] += len(tb.text.split())
                    self.stat_lists["blockstyles_lines_pp"][-1] += len(tb.line_ids)
                    if not(tbox_rect in tb_rects):
                        self.stat_lists["blockstyle_space_pp"][-1] += (right-left) * (bot-top)
                        for r in page.img_rects:
                            o = rectOverlap(tbox_rect,r)
                            if(o):
                                text_img_overlap += o
                        for r in tb_rects:
                            ot = rectOverlap(tbox_rect,r)
                            if(ot):
                                tb_tb_o += ot
                        tb_rects.append(tbox_rect)
        self.stat_lists["img_img_overlap_pp"].append(page.img_img_overlap)
        self.stat_lists["text_img_overlap_pp"].append(text_img_overlap)
        self.stat_lists["page_space_pp"].append(page.space)
        self.stat_lists["image_space_pp"].append(page.image_space)
        self.stat_lists["free_space_pp"].append(page.space-page.image_space-self.stat_lists["blockstyle_space_pp"][-1]-self.stat_lists["textbox_space_pp"][-1]+text_img_overlap+tb_tb_o)

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
        if self.stat_lists["blockstyles_pp"][-1] > 0:
            block_columns = 1
            sorted_left_blockstyles = [(key,vals) for key,vals in page.tbox_blockstyle_left.items()]
            sorted_left_blockstyles = sorted(sorted_left_blockstyles, key=lambda x: x[0], reverse=False)
            min_right_per_left_boxes = []
            for key,vals in sorted_left_blockstyles:
                min_box = None
                min_right = page.width
                for val in vals:
                    box = page.tbox_blockstyle_top[val]
                    if len(box.line_ids) >= 2 and box.right_pos<min_right:
                        min_box = box
                        min_right = box.right_pos
                if not(min_box is None):
                    min_right_per_left_boxes.append((key,min_box))
            i = 0
            while i < len(min_right_per_left_boxes):
                key, box = min_right_per_left_boxes[i]
                i+=1
                right_pos = box.right_pos
                min_right = page.width
                found = False
                for j in range(i,len(min_right_per_left_boxes)):
                    next_box = min_right_per_left_boxes[j][1]
                    if right_pos <= min_right_per_left_boxes[j][0]:
                        if next_box.right_pos<min_right:
                            min_right = next_box.right_pos
                            i=j
                            found = True
                if found:
                    block_columns += 1
        return block_columns

    def get_textbox_columns(self, page):
        textbox_columns = 0
        if self.stat_lists["textboxes_pp"][-1] > 0:
            textbox_columns = 1
            sorted_left_tbox = [(key,vals) for key,vals in page.tbox_left.items()]
            sorted_left_tbox = sorted(sorted_left_tbox, key=lambda x: x[0], reverse=False)
            min_right_per_left_boxes = []
            for key,vals in sorted_left_tbox:
                min_box = None
                min_right = page.width
                for val in vals:
                    box = page.tbox_top[val]
                    if(len(box.line_ids)>=2 and box.right_pos<min_right):
                        min_box = box
                        min_right = box.right_pos
                if not(min_box is None):
                    min_right_per_left_boxes.append((key,min_box))
            i = 0
            while i < len(min_right_per_left_boxes):
                key, box = min_right_per_left_boxes[i]
                i+=1
                right_pos = box.right_pos
                min_right = page.width
                found = False
                for j in range(i, len(min_right_per_left_boxes)):
                    next_box = min_right_per_left_boxes[j][1]
                    if right_pos <= min_right_per_left_boxes[j][0]:
                        if next_box.right_pos < min_right:
                            min_right = next_box.right_pos
                            i=j
                            found = True
                if found:
                    textbox_columns += 1
        return textbox_columns

    def get_feature_dict(self, image_error):
        features = FEATURES.copy()

        # PAGES
        pages = self.counts["pages"]
        features["count_pages"] = pages
        features["count_outline_items"] = self.counts["outline_items"]

        # FONTS
        if self.fontspecs["count"]>0:
            features["count_fonts"] = self.fontspecs["count"]
            features["count_font_colors"] = len(self.fontspecs["colors"])
            features["count_font_families"] = len(self.fontspecs["families"])
            np_sizes = np.array(self.fontspecs["sizes"])
            features["max_font_size"] = np.max(np_sizes)
            features["min_font_size"] = np.min(np_sizes[np_sizes>0])
            if(max(self.fontspecs["spec_counts"])>0):
                main_font_index = np.argmax(self.fontspecs["spec_counts"])
                features["main_font_size"] = self.fontspecs["sizes"][main_font_index]
                features["perc_main_font_word"] = \
                self.fontspecs["spec_counts"][main_font_index] / np.sum(self.fontspecs["spec_counts"])

        # IMAGES
        features["image_error"] = int(image_error)
        page_spaces = self.stat_lists["page_space_pp"]
        total_page_space = sum(page_spaces)
        #avoid division by 0
        if total_page_space == 0: total_page_space = 0.01
        if len(self.images["sizes"]) > 0:
            image_spaces = self.stat_lists["image_space_pp"]
            total_image_space = sum(image_spaces)
            image_page_ratios = [image_spaces[i]/page_spaces[i] for i in range(len(image_spaces))]
            #if it's empty put at least an element
            if not image_page_ratios: image_page_ratios = [0]
            features["mean_images_pp"] = len(self.images["sizes"])/pages
            features["mean_image_space_pp"] = total_image_space/total_page_space
            features["max_image_page_ratio"] = max(image_page_ratios)
        free_spaces = self.stat_lists["free_space_pp"]
        total_free_space = sum(free_spaces)
        features["mean_free_space_pp"] = total_free_space/total_page_space

        if len(self.text)>0:
            # TEXT STUFF
            features["text"] = self.text
            features["bold_text"] = self.bold_text
            text_list = self.text.split()
            features["first_100_words"] = " ".join(text_list[0:100])
            features["last_100_words"] = " ".join(text_list[-100:])
            features["copyright_symbol"] = text_contains_copyright_symbol(self.text)

            features["mean_words_pp"] = np.mean(self.stat_lists["words_pp"])
            features["mean_lines_pp"] = np.mean(self.stat_lists["lines_pp"])
            features["mean_bold_words_pp"] = self.counts["bold_words"]/pages
            features["mean_annotations_pp"] = self.counts["annotations"]/pages
        if len(self.stat_lists["word_pl"]) > 0:
            features["mean_words_per_line"] = np.mean(self.stat_lists["word_pl"])

        if self.counts["textboxes"] > 0:
            features["mean_textboxes_pp"] = self.counts["textboxes"]/pages
            features["mean_textbox_space_pp"]  = np.mean(self.stat_lists["textbox_space_pp"])
            features["mean_lines_per_textbox"] = np.mean(self.stat_lists["lines_ptb"])
            textbox_columns = self.stat_lists["textbox_columns"]
            features["modal_textbox_columns_pp"] = stats.mode(textbox_columns)[0][0]
            features["perc_modal_textbox_columns_pp"] = stats.mode(textbox_columns)[1][0]/self.counts["pages"]

        if self.counts["blockstyles"] > 0:
            features["mean_blockstyles_pp"] = self.counts["blockstyles"]/pages
            features["mean_blockstyle_space_pp"]  = np.mean(self.stat_lists["blockstyle_space_pp"])
            features["mean_lines_per_blockstyle"] = np.mean(self.stat_lists["lines_pbs"])
            blockstyle_columns = self.stat_lists["blockstyle_columns"]
            features["modal_blockstyle_columns_pp"] = stats.mode(blockstyle_columns)[0][0]
            features["perc_modal_blockstyle_columns_pp"] = stats.mode(blockstyle_columns)[1][0]/self.counts["pages"]

        # MARGINS
        if len(self.rights) > 0:
            max_right, max_right_count = max(self.rights.items(), key=lambda x: x[1])
            total_rights = np.sum(list(self.rights.values()))
            features["modal_right"] = max_right
            features["perc_modal_right"] = max_right_count/total_rights
            features["max_right"] = max(self.rights.keys())
        if len(self.lefts) > 0:
            max_left, max_left_count = max(self.lefts.items(), key=lambda x: x[1])
            total_lefts = np.sum(list(self.lefts.values()))
            features["modal_left"] = max_left
            features["perc_modal_left"] = max_left_count/total_lefts
            features["max_left"] = max(self.lefts.keys())
            features["min_left"] = min(self.lefts.keys())

        for k,v in features.items():
            if not(type(v) == str):
                features[k] = np.float64(v)
        return features

    def get_error_features(self, error_code):
        features = FEATURES.copy()
        features["text"] = error_code
        features["bold_text"] = error_code
        features["first_100_words"] = error_code
        features["last_100_words"] = error_code
        for k,v in features.items():
            if not(type(v) == str):
                features[k] = np.float64(v)
        return features

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
        while error:
            try:
                text = "\n".join(lines)
                ET.fromstring(text)
                error = False
            except ET.ParseError as e:
                e_parts = str(e).split(":")
                if e_parts[0] == "mismatched tag":
                    line,column = e_parts[1].split(",")
                    line = int(line.split(" line ")[1])
                    column = int(column.split(" column ")[1])
                    if len(lines) >= line: 
                        del lines[line-1]
                    else: break
                else:
                    error = False
        content = "\n".join(lines)
        return content

def get_structure_features(file, log=False, clean=False):
    debuglogger.debug("Getting structure for file %s",basename(file))
    s=time()
    doc = Document(file)
    err_message = False
    img_err = False
    try:
        err_message = doc.process_xml()
    except ValueError as err:
        debuglogger.error("(Xml structure): %s",err)
        logger.error("(Xml structure): %s",err)

    if not err_message:
        f_dict = doc.get_feature_dict(img_err)
    else:
        f_dict = doc.get_error_features(err_message)
    if log:
        extract_time = time()-s
        log_time(basename(file), extract_time)
    if not os.path.isfile(file):
        debuglogger.debug("Something happened, file is gone %s", basename(file))
        sys.exit(1)
    doc = None
    return (f_dict, file)

def log_time(doc_id, extract_time):
    log_file = "time.log"
    if isfile(log_file):
        with open(log_file, 'r') as fp:
            time_data = json.load(fp, encoding="utf-8")
        time_data.update({doc_id:extract_time})
    else:
        time_data = {doc_id:extract_time}

    with open(log_file, 'w') as fp:
        json.dump(time_data, fp, indent=4)

def rectOverlap(a, b):
    dx = min(a[2], b[2]) - max(a[0], b[0])
    dy = min(a[3], b[3]) - max(a[1], b[1])
    if (dx >= 0) and (dy >= 0):
        return dx*dy

def pre_extract_pdf_structure_data(batch_files, pool):
    res=[]
#    sys.exit(1)
#    for f in batch_files:
#        res.append(get_structure_features(f))
    res = pool.map(get_structure_features, batch_files)
    res_fix={}
    for x in res:
        d_id = splitext(basename(x[1]))[0]
        doc_features = x[0]
        res_fix[d_id] = doc_features
    clean_files(dirname(batch_files[0]))
    return res_fix

def clean_files(path, idd=""):
    '''
    Cleans all the files produced by pdftohtml.
    
    @param path: the path where the files to clean are located
    @dtype path: str
    '''
    from glob import glob
    xml_files = glob(join(path,idd+"*.{}".format('xml')))
    for x in xml_files: os.remove(x)
    image_files = glob(join(path,idd+"*.{}".format('jpg')))
    for i in image_files: os.remove(i)
    image_files = glob(join(path,idd+"*.{}".format('png')))
    for i in image_files: os.remove(i)

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

def text_contains_copyright_symbol(text):
    symbols = re.findall(r'(©|[^a-z]doi[^a-z]|[^a-z]isbn[^a-z])', text.lower())
    return int(len(symbols)>0)