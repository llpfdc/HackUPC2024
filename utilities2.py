import torch

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import requests
from io import BytesIO
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader

def preprocess_dataframe(dataframe, size=100):
  dataframe = dataframe.dropna()

  row_images = pd.concat([dataframe['IMAGE_VERSION_1'],
                          dataframe['IMAGE_VERSION_2'],
                          dataframe['IMAGE_VERSION_3']],
                         ignore_index=True)
  dataframe = pd.DataFrame({'images': row_images})

  dataframe = dataframe[:size]

  return dataframe

def url_to_Image(url):
  ''' Convert from url to image '''
  try:
    response = requests.get(url)
    response.raise_for_status()
    img = Image.open(BytesIO(response.content))
    return img
  except Exception as e:
    return None

def links_to_data_and_labels(data_links):
  return_data = []
  labels = []
  prefix = "https://sttc-stage-zaraphr.inditex.com"
  for index, row in data_links.iterrows():
    link = row['images']
    labels.append(get_labels(link))
    try:
      image = url_to_Image(link)
      tensor = image_to_tensor(image)
      return_data.append(tensor)
    except:
      parts = link.split("/")
      half_link = parts.index("photos")
      new_url = "/".join(parts[half_link:])
      new_link = prefix + '/' + new_url
      image = url_to_Image(new_link)
      if image:
        tensor = image_to_tensor(image)
        return_data.append(tensor)
  dataloader = DataLoader(return_data, batch_size=32, shuffle=True)
  return return_data, [label[2] for label in labels]

def get_labels(link):
  parts = link.split("/")
  photos_index = parts.index("photos")
  link_fields = parts[photos_index:]
  link_list = [x for x in link_fields if x != '']
  season = 0
  if link_list[2] == 'W':
    season = 1
  elif link_list[2] == 'I':
    season = 2
  elif link_list[2] == 'V':
    season = 3
  try:
    return [0, 0, int(link_list[3]), 0]
  except:
    return [0, 0, 0, 0]
  #return [int(link_list[1]), season, int(link_list[3]), int(link_list[4])]


def preprocess_image(img):
    ''' Resize image '''
    base_width = 64
    wpercent = (base_width / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))

    return img.resize((base_width, hsize), Image.Resampling.LANCZOS)


def image_to_tensor(image):
    ''' Convert image to tensor feasible by ResNet-50 '''
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocess(image)

    return image.unsqueeze(0)


def generate_embedding(image):
    ''' Generate embedding from image '''
    image = preprocess_image(image)
    image_tensor = image_to_tensor(image)
    with torch.no_grad():
         features = resnet_embed(image_tensor)
    embedding = features.squeeze().numpy()
    result = [float(emb) for emb in embedding]

    return result


def print_images(list_urls):
  ''' print images from url list '''
  fig, axs = plt.subplots(1, len(list_urls), figsize=(10, 4))

  for i, link in enumerate(list_urls):
    response = requests.get(link)
    img = Image.open(BytesIO(response.content))
    axs[i].imshow(img)
    axs[i].axis('off')

  plt.show()
