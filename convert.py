# import package
import os
import labelme2coco


# set directory that contains labelme annotations and image files
labelme_folder = "./mask"

# set path for coco json to be saved
save_json_path = "./mask/test_coco.json"
os.remove(save_json_path) if os.path.exists(save_json_path) else None
# convert labelme annotations to coco
labelme2coco.convert(labelme_folder, save_json_path)
