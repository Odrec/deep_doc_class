#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 08:17:27 2019

This file holds the names of the features to be used for training and classification.

@author: odrec
"""

#These lists are redundant since they can be extracted from the preprocessing
#of the metadata, properties and structure but maybe is a good idea to keep it here 
#for consistency checking.

#list of all bow features
bow_text_features = [
    "text",
    "first_100_words",
    "last_100_words",
    "bold_text"
]

bow_prop_features = [
    "producer",
    "creator"
]

bow_meta_features = [
    "file_name",
    "folder_name"
]

#list of all numeric features
numeric_features = [
    #FILE
    "file_size",
    "page_rot",
    
    #PAGES
    "count_pages",
    "count_outline_items",
    "page_size_x",
    "page_size_y",

    # FONTS
    "count_fonts",
    "count_font_colors",
    "count_font_families",
    "max_font_size",
    "min_font_size",
    "main_font_size",
    "perc_main_font_word",
    
    #CONTENT
    "copyright_symbol",

    # WORDS
    "mean_words_pp",
    "mean_lines_pp",
    "mean_words_per_line",
    "mean_bold_words_pp",
    "mean_annotations_pp",

    # IMAGES AND FREE
    "image_error",
    "mean_image_space_pp",
    "mean_images_pp",
    "max_image_page_ratio",
    "mean_free_space_pp",

    # TEXT BOXES
    "mean_textboxes_pp",
    "mean_lines_per_textbox",
    "mean_blockstyles_pp",
    "mean_lines_per_blockstyle",

    "modal_textbox_columns_pp",
    "perc_modal_textbox_columns_pp",
    "modal_blockstyle_columns_pp",
    "perc_modal_blockstyle_columns_pp",

    "mean_textbox_space_pp",
    "mean_blockstyle_space_pp",

    # MARGINS
    "modal_right",
    "perc_modal_right",
    "modal_left",
    "perc_modal_left",
    "max_right",
    "max_left",
    "min_left",
]