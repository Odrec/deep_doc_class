def pdf_images_handler(content_dict, entropy_path, doc_id):
    if(isfile(entropy_path)):
        with open(entropy_path,"r") as f:
            entropy_dict = json.load(f)
        if(doc_id in entropy_dict):
            # (entropy,color) = entropy_dict[doc_id]
            entropy = entropy_dict[doc_id]
        else:
            entropy, color = get_grayscale_entropy_tmpfile(filepath,1,5)
            entropy_dict[doc_id] = (entropy,color)
            with open(entropy_path,"w") as f:
                json.dump(entropy_dict, f, indent=4)
    else:
        entropy, color = get_grayscale_entropy_tmpfile(filepath,1,5)
        entropy_dict[doc_id] = (entropy,color)

    content_dict["entropy"]=entropy
    # values_dict["color"]=color
    return content_dict

def get_grayscale_entropy(fp, first_page=-1, last_page=-1):
    args = ["gs", "-dNOPAUSE", "-dBATCH", "-sDEVICE=jpeg", "-sOutputFile=-"]
    if(not(first_page==-1)):
        args.append("-dFirstPage=%d"%(first_page,))
    if(not(last_page==-1)):
        args.append("-dLastPage=%d"%(last_page,))
    args.append("-r200")
    args.append(fp)

    output = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()[0]
    
    page_break_regex = b'Page [0-9]+\n'
    sub_regex = b'Substituting (.)+\n'
    loading_regex = b'Loading (.)+\n'
    query_regex = b'Querying (.)+\n'
    cant_find_regex = b'Can\'t find \(or can\'t open\) (.)+\n'
    didnt_find_regex = b'Didn\'t find (.)+\n'

    pages = re.split(page_break_regex, output)

    entropy = []
    for i in range(1, len(pages)):
        page = pages[i]
        page = page.split(b'done.\n')[-1]
        page = re.sub(sub_regex, b'', page)
        page = re.sub(loading_regex, b'', page)
        page = re.sub(query_regex, b'', page)

        pil_image = PI.open(io.BytesIO(page))
        # with PI.open(io.BytesIO(page)) as pil_image:
        gs_image = pil_image.convert("L")
        hist = np.array(gs_image.histogram())
        hist = np.divide(hist,np.sum(hist))
        hist[hist==0] = 1
        e = -np.sum(np.multiply(hist, np.log2(hist)))
        entropy.append(e)

    return np.mean(entropy)

def get_grayscale_entropy_tmpfile(fp, first_page=-1, last_page=-1):
    args = ["gs", "-dNOPAUSE", "-dBATCH", "-sDEVICE=jpeg", "-sOutputFile=tmp-%03d.jpeg"]
    if(not(first_page==-1)):
        args.append("-dFirstPage=%d"%(first_page,))
    if(not(last_page==-1)):
        args.append("-dLastPage=%d"%(last_page,))
    args.append("-r200")
    args.append(fp)

    output = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()[0]

    invalid_file_regex = r'Error: /invalidfileaccess in pdf_process_Encrypt'

    if(re.search(invalid_file_regex, output.decode())):
        return np.nan

    entropy = []
    color = False
    for i in range(first_page,last_page+1):
        imgfile = "tmp-%03d.jpeg"%(i,)
        if(not(isfile(imgfile))):
            continue
        pil_image = PI.open(imgfile)
        # with PI.open(io.BytesIO(page)) as pil_image:
        gs_image = pil_image.convert("L")
        hist = np.array(gs_image.histogram())
        hist = np.divide(hist,np.sum(hist))
        hist[hist==0] = 1
        e = -np.sum(np.multiply(hist, np.log2(hist)))
        entropy.append(e)

        if(not(color)):
            col_image = pil_image.convert('RGB')
            np_image = np.array(col_image)
            if((np_image[:,:,0]==np_image[:,:,1])==(np_image[:,:,1]==np_image[:,:,2])):
                color = True

        os.remove(imgfile)

    if(len(entropy)==0):
        print(print_bcolors(["WARNING","BOLD"],
            "Zero images loaded. Either pdf is empty or Ghostscript didn't create images correctly."))
        mean_entropy = np.nan
    else:
        mean_entropy = np.mean(entropy)
    return mean_entropy, color

def get_pil_image_from_pdf(fp, page=1):
    args = ["gs", "-dNOPAUSE", "-dBATCH", "-sDEVICE=jpeg", "-sOutputFile=-", "-r200"]
    args.append("-dFirstPage=%d"%(page,))
    args.append("-dLastPage=%d"%(page,))
    args.append(fp)

    output = subprocess.Popen(args, stdout=subprocess.PIPE).communicate()[0]
    page_break_regex = b'Page [0-9]+\n'
    page = re.split(page_break_regex, output)[-1]
    page = page.split(b'done.\n')[-1]
    pil_image = PI.open(io.BytesIO(page))
    return pil_image

def get_text_from_pil_img(pil_image, lang="deu"):
    if(not(lang in ["eng", "deu", "fra"])):
        print("Not the right language!!!\n Languages are: deu, eng, fra")

    tool = pyocr.get_available_tools()[0]
    txt = tool.image_to_string(
        pil_image,
        lang=lang,
    )
    return txt