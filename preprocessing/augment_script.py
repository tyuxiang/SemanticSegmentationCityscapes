import os
import random
from tqdm import tqdm
from PIL import Image

from torchvision import transforms

def augment_image(path, gamma, saturation):
    img = Image.open(path)
    img = transforms.ToTensor()(img)
    img = transforms.functional.adjust_gamma(img, gamma)
    img = transforms.functional.adjust_saturation(img, saturation)
    img = transforms.ToPILImage()(img)
    return img

def get_augmented_filename(path: str):
    dirpath = os.path.dirname(path)
    basename = os.path.basename(path)
    city, seq, frame, _ = basename.split('_')
    aug_name = '{}_{:06d}_{:06d}_leftImg8bit.png'.format(city, int(seq)+900000, int(frame))
    aug_path = os.path.join(dirpath, aug_name)
    return aug_path


def main():
    img_dir = './data/leftImg8bit'
    seq_dir = './data/leftImg8bit_sequence'

    cities = [c for c in os.listdir(img_dir)]
    for city in cities:
        to_augment = []

        ref_dir = os.path.join(img_dir, city)
        img_paths = os.listdir(ref_dir)
        img_paths.sort()
        
        for img_path in img_paths:
            augment_set = []
            img_name = img_path.split('_')
            seq = int(img_name[1])
            ref_frame = int(img_name[2])

            if seq >= 900000:
                continue

            augment_set.append(os.path.join(img_dir, city, img_path))

            for frame in range(ref_frame-3, ref_frame+1):
                path = '{}_{:06d}_{:06d}_leftImg8bit.png'.format(city, seq, frame)
                path = os.path.join(seq_dir, city, path)
                augment_set.append(path)
            to_augment.append(augment_set)
        
        print(city, len(img_paths), len(to_augment))
        for s in tqdm(to_augment):
            # print(f'{idx}: {len(s)}, {s[0]}')
            gamma = random.uniform(0.5, 1.5)
            saturation = random.uniform(0.5, 1.5)
            for idx, img_path in enumerate(s):
                if idx == 4:
                    img = ref_img
                else:
                    img = augment_image(img_path, gamma, saturation)
                
                if idx == 1:
                    ref_img = img
                    
                aug_path = get_augmented_filename(img_path)
                # print(img_path, aug_path)
                img.save(aug_path)


# main()

