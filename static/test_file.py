import shutil
import os
import glob
source = 'Images/'
dest1 = 'img'


files = os.listdir(source)

for f in files:
    print(f)
    for img_path in sorted(glob.glob("Images/" + f + "/*.jpg")):
        # print(img_path)
        print(source+img_path)
        shutil.move(source+f, dest1)

