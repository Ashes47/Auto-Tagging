import pickle
from constants import CUSTOM_CLASS_LIST, temp_file, model_name
import cv2
import os
import shutil
from ultralytics import YOLO

def save_custom_info():
    with open('CUSTOM_CLASS_LIST.txt', "wb") as fp:  
        pickle.dump(CUSTOM_CLASS_LIST, fp)


def pixel_box_to_yolobox(pixel_box, image_width, image_height):
    x1, y1, x2, y2 = pixel_box[0], pixel_box[1], pixel_box[2], pixel_box[3]
    x = abs(x2-x1)/(2*image_width)
    y = abs(y1 -y2)/(2*image_height)
    w = abs(x2-x1)/image_width
    h = abs(y1 -y2)/image_height
    yolo_box = [x, y, w, h]
    return yolo_box


def define_new_class(custom_class):
    custom_class_count = CUSTOM_CLASS_LIST["next_class_counter"]
    CUSTOM_CLASS_LIST["next_class_counter"] += 1
    CUSTOM_CLASS_LIST[custom_class] = {
        "class_count": custom_class_count,
        "class_split_count": 0,
        "train": 0,
        "valid": 0,
        "split_index": 0
    }


def update_class_details(custom_class, folder_to_save):
    CUSTOM_CLASS_LIST[custom_class]["split_index"] = (CUSTOM_CLASS_LIST[custom_class]["split_index"] + 1) % 10
    CUSTOM_CLASS_LIST[custom_class][folder_to_save] += 1
    
    if custom_class not in CUSTOM_CLASS_LIST["class_list"]:
        CUSTOM_CLASS_LIST["class_list"].append(custom_class)
    save_custom_info()


def get_unique_name(custom_class, folder_to_save):
    return str(CUSTOM_CLASS_LIST[custom_class][folder_to_save])


def save_image(folder_to_save, unique_name, image):
    cv2.imwrite('custom_dataset/' + folder_to_save  + '/images/' + unique_name + '.jpg', image)


def save_label(folder_to_save, unique_name, class_count, yolo_box):
    file = open('custom_dataset/' + folder_to_save  + '/labels/' + unique_name + ".txt", "w")
    file.writelines(str(class_count) + " ")
    for x in yolo_box:
        file.writelines(str(x) + " ")
    file.close()


def write_yaml():
    # Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
    # dataset root dir
    file = open("data.yaml", "w")
    file.writelines("path: custom_dataset/ \n")
    file.writelines("train: train/images/ \n") # train images (relative to 'path')
    file.writelines("val: valid/images/ \n") # val images (relative to 'path')

    # number of classes
    file.writelines(f"nc: {CUSTOM_CLASS_LIST['next_class_counter']} \n")

    # class names
    #names: ['0', '1', '2']
    file.writelines(f"names: {CUSTOM_CLASS_LIST['class_list']} \n")
    file.close()


def save_image_info(folder_to_save, unique_name, image, class_count, yolo_box):
    save_image(folder_to_save, unique_name, image)
    save_label(folder_to_save, unique_name, class_count, yolo_box)
    write_yaml()


async def add_class(custom_class, pixel_box):
    image = cv2.imread(temp_file)
    image_height, image_width, _ = image.shape
    yolo_box = pixel_box_to_yolobox(pixel_box, image_width, image_height)
    if CUSTOM_CLASS_LIST.get(custom_class) is None:
        define_new_class(custom_class)
    
    class_details = CUSTOM_CLASS_LIST[custom_class]
    class_count = class_details["class_count"]
    split_index = class_details["split_index"]
    folder_to_save = CUSTOM_CLASS_LIST["split"][split_index]

    update_class_details(custom_class, folder_to_save)
    unique_name = get_unique_name(custom_class, folder_to_save)
    save_image_info(folder_to_save, unique_name, image, class_count, yolo_box)

def train_custom_object_detection(epochs):
    model = YOLO(model_name)  # load a pretrained model (recommended for training)
    # Use the model
    model.train(data="/content/data.yaml", epochs=epochs, imgsz=640)
    os.rename("runs/detect/train/weights/best.pt", "/models/custom_model.pt")
    shutil.rmtree("runs")
    CUSTOM_CLASS_LIST["training_status"] = False