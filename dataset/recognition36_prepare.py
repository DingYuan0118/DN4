import os
import csv
import numpy as np
import random
from PIL import Image
import pdb
import json

data_dir = './dataset/recognition36_crop'                # the path of the download dataset
save_dir = './dataset/recognition36_crop'
with open(os.path.join(data_dir, "novel_all.json"), "r") as f:
    meta_data = json.load(f)

image_names = meta_data["image_names"]
image_labels = meta_data["image_labels"]
label_names = meta_data["label_names"]

for i, name in enumerate(image_names):
    name = name.split("/")[-1]
    image_names[i] = name

test_data = []
for image, label in zip(image_names, image_labels):
    test_data.extend([[image, label_names[label]]])


with open(os.path.join(save_dir, 'test.csv'), 'w') as csvfile:
	writer = csv.writer(csvfile)

	writer.writerow(['filename', 'label'])
	writer.writerows(test_data)

print() 