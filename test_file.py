import glob
import os
import pickle
features=[]
img_paths=[]
for feature_path in glob.glob("static/feature_2/*"):
    features.append(pickle.load(open(feature_path, 'rb')))
    x=os.path.splitext(os.path.basename(feature_path))
    # print(x)
    img_paths.append('static/img_2/' + os.path.splitext(os.path.basename(feature_path))[0])

print(img_paths)