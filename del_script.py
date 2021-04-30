import os

img_dir = './data/leftImg8bit'
seq_dir = './data/leftImg8bit_sequence'
mode = 'train'

mode_path = os.path.join(img_dir, mode)
cities = [c for c in os.listdir(mode_path)]
for city in cities:
    ref_dir = os.path.join(mode_path, city)
    img_paths = os.listdir(ref_dir)
    img_paths.sort()

    for img_path in img_paths:
        img_name = img_path.split('_')
        seq = int(img_name[1])
        ref_frame = int(img_name[2])

        to_be_deleted_dir = os.path.join(seq_dir, mode, city)

        for frame in range(0, ref_frame-3):
            try:
                path = '{}_{:06d}_{:06d}_leftImg8bit.png'.format(city, seq, frame)
                os.remove(os.path.join(to_be_deleted_dir, path))
            except:
                print(f'{path} does not exist')
        for frame in range(ref_frame+1, 30):
            try:
                path = '{}_{:06d}_{:06d}_leftImg8bit.png'.format(city, seq, frame)
                os.remove(os.path.join(to_be_deleted_dir, path))
            except:
                print(f'{path} does not exist')

print('cleansed')
