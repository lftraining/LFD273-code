from io import BytesIO
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import requests
import torch
import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET

def save_images(folder, search_term, count=10):
    if not os.path.exists(folder):
        os.mkdir(folder)

    SEARCH_URL = "https://huggingface.co/api/experimental/images/search"

    params = {"q": search_term, "license": "public", "imageType": "photo", "count": count}

    resp = requests.get(SEARCH_URL, params=params)
    if resp.status_code == 200:
        content = resp.json()['value']
        urls = [img['thumbnailUrl'] for img in content]

        folder = os.path.join(folder, search_term)
        if not os.path.exists(folder):
            os.mkdir(folder)

        i = 0
        for url in urls:
            try:
                img = get_image_from_url(url)
                fname = os.path.join(folder, f'{i}.jpg')
                img.save(fname)
                i += 1
            except Exception:
                pass
        print(f'Retrieved {i} images for {search_term}')
    else:
        print(f'Failed to retrieve URLs for {search_term}')

def show(imgs, titles=None):
    import torchvision.transforms.functional as tF
    
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        if isinstance(img, torch.Tensor):
            img = img.detach()
            img = tF.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        if titles is not None:
            if i < len(titles):
                axs[0, i].set_title(titles[i])
    return fig

def get_image_from_url(url, headers=None):
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    img = Image.open(BytesIO(resp.content))
    return img

def xml_to_csv(path):
    """Iterates through all .xml files (generated by labelImg) in a given directory and combines
    them in a single Pandas dataframe.

    Parameters:
    ----------
    path : str
        The path containing the .xml files
    Returns
    -------
    Pandas DataFrame
        The produced dataframe
    """

    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        filename = root.find('filename').text
        width = int(root.find('size').find('width').text)
        height = int(root.find('size').find('height').text)
        for member in root.findall('object'):
            bndbox = member.find('bndbox')
            value = (filename,
                     width,
                     height,
                     member.find('name').text,
                     int(bndbox.find('xmin').text),
                     int(bndbox.find('ymin').text),
                     int(bndbox.find('xmax').text),
                     int(bndbox.find('ymax').text),
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height',
                   'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df