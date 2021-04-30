import os

img_dir = './data/leftImg8bit'
seq_dir = './data/leftImg8bit_sequence'

cities = [c for c in os.listdir(img_dir)]
for city in cities:
    to_keep = []

    ref_dir = os.path.join(img_dir, city)
    img_paths = os.listdir(ref_dir)
    img_paths.sort()
    print(city, len(img_paths))

    for img_path in img_paths:
        img_name = img_path.split('_')
        seq = int(img_name[1])
        ref_frame = int(img_name[2])

        for frame in range(ref_frame-3, ref_frame+1):
            path = '{}_{:06d}_{:06d}_leftImg8bit.png'.format(city, seq, frame)
            to_keep.append(path)

    seq_city_dir = os.path.join(seq_dir, city)
    img_paths = os.listdir(seq_city_dir)
    img_paths.sort()

    for img_path in img_paths:
        # print(img_path, img_path in to_keep)
        if img_path in to_keep:
            continue
        else:
            try:
                os.remove(os.path.join(seq_city_dir, img_path))
            except:
                print(f'issue with {img_path}')

print('cleansed')
