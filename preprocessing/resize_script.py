import os
from PIL import Image
import torchvision.transforms as transforms

img_dir = './data/leftImg8bit'
seq_dir = './data/leftImg8bit_sequence'
fine_dir = './data/gtFine'

cities = [c for c in os.listdir(img_dir)]
for city in cities:
    img_paths = [os.path.join(img_dir, city, p) for p in os.listdir(os.path.join(img_dir, city))]
    seq_paths = [os.path.join(seq_dir, city, p) for p in os.listdir(os.path.join(seq_dir, city))]
    fine_paths = [os.path.join(fine_dir, city, p) for p in os.listdir(os.path.join(fine_dir, city))]

    paths = img_paths + seq_paths + fine_paths

    paths.sort()
    print(city, len(paths))
    for path in paths:
        if path[-5:] == '.json':
            continue

        try:
            t = transforms.Resize((224,448))
            t(Image.open(path)).save(path)
        except:
            print(f'encountered for path: {path}')
