import os

to_be_deleted_dir = './data/leftImg8bit_sequence/train/ulm'

for seq in range(95):
    for frame in range(0, 16):
        try:
            path = 'ulm_{:06d}_{:06d}_leftImg8bit.png'.format(seq, frame)
            os.remove(os.path.join(to_be_deleted_dir, path))
        except:
            print(f'{path} does not exist')
    for frame in range(20, 30):
        try:
            path = 'ulm_{:06d}_{:06d}_leftImg8bit.png'.format(seq, frame)
            os.remove(os.path.join(to_be_deleted_dir, path))
        except:
            print(f'{path} does not exist')

print('cleansed')
