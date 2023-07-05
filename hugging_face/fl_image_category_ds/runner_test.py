from fl_image_category_ds import FLImageCategory
from datasets import load_dataset, Image, Dataset
import urllib
import PIL
import requests
from io import BytesIO

# url = 'https://fabletics-us-cdn.justfab.com/media/images/products/JT2044322-5886/JT2044322-5886-1_998x1498.jpg'
# response = urllib.request.urlopen(urllib.request.Request(url))
# code = response.getcode()
# image = PIL.Image.open(BytesIO(response.read()))

dataset = load_dataset('../fl_image_category_ds')

# ds_generator = FLImageCategory()