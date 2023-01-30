import pickle
from constants import CUSTOM_CLASS_LIST, temp_file, model_name
import cv2
import os
import pandas
import shutil
from ultralytics import YOLO
from face_recog import mtcnn
from object_detection import show_crop
from PIL import Image


def save_custom_info():
    with open('CUSTOM_CLASS_LIST.txt', "wb") as fp:  
        pickle.dump(CUSTOM_CLASS_LIST, fp)


def pixel_box_to_yolobox(pixel_box, image_width, image_height):
    x1, y1, x2, y2 = pixel_box[0], pixel_box[1], pixel_box[2], pixel_box[3]
    x = abs(x1 + x2)/(2*image_width)
    y = abs(y1 + y2)/(2*image_height)
    w = abs(x1 - x2)/image_width
    h = abs(y1 - y2)/image_height
    yolo_box = [x, y, w, h]
    return yolo_box


def define_new_class(custom_class):
    custom_class_count = CUSTOM_CLASS_LIST["next_class_counter"]
    CUSTOM_CLASS_LIST["next_class_counter"] += 1
    CUSTOM_CLASS_LIST[custom_class] = {
        "class_count": custom_class_count,
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


def get_unique_name():
    CUSTOM_CLASS_LIST["next_unique_name"] += 1
    return str(CUSTOM_CLASS_LIST["next_unique_name"])


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
    if os.path.exists("data.yaml"):
        os.remove("data.yaml") 
    file = open("data.yaml", "w")
    file.writelines("path: " + os.getcwd() + "/custom_dataset/ \n")
    file.writelines("train: train/images/ \n") # train images (relative to 'path')
    file.writelines("val: valid/images/ \n") # val images (relative to 'path')

    # number of classes
    file.writelines(f"nc: {len(CUSTOM_CLASS_LIST['class_list'])} \n")

    # class names
    #names: ['0', '1', '2']
    file.writelines(f"names: {CUSTOM_CLASS_LIST['class_list']} \n")
    file.close()


def save_image_info(folder_to_save, unique_name, image, class_count, yolo_box):
    save_image(folder_to_save, unique_name, image)
    save_label(folder_to_save, unique_name, class_count, yolo_box)
    write_yaml()


def add_classes(custom_classes, pixel_boxes):
    for custom_class, pixel_box in zip(custom_classes, pixel_boxes):
        image = cv2.imread(temp_file)
        cropped = show_crop(image, pixel_box)
        face = mtcnn(Image.fromarray(cropped))
        if face is None:
            add_class(custom_class, pixel_box)


def add_class(custom_class, pixel_box):
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
    unique_name = get_unique_name()
    save_image_info(folder_to_save, unique_name, image, class_count, yolo_box)

def train_custom_object_detection(epochs):
    if os.path.exists('./custom_dataset/train/labels.cache'):
        os.remove('./custom_dataset/train/labels.cache')
    if os.path.exists('./custom_dataset/valid/labels.cache'):
        os.remove('./custom_dataset/valid/labels.cache')
    if os.path.exists("./models/custom_model.pt"):
        os.remove("./models/custom_model.pt")

    model = YOLO(model_name)  # load a pretrained model (recommended for training)
    # Use the model
    model.train(data="data.yaml", epochs=epochs, imgsz=640)
    CUSTOM_CLASS_LIST["training_status"] = False
    os.rename("./runs/detect/train/weights/best.pt", "./models/custom_model.pt")
    shutil.rmtree("runs")


def delete_class(folder, class_number):
    path = "./custom_dataset/" + folder + "/labels/"
    image_path = "./custom_dataset/" + folder + "/images/"
    for txt in os.listdir(path):
        if txt[-4:] == ".txt":
            file = pandas.read_csv(path + txt)
            if int(file.columns[0].split(' ')[0]) == class_number:
                print('Removing data')
                os.remove(path+txt)
                os.remove(image_path + txt[:-4] + '.jpg')


def clear_custom_class(customClass):
    if customClass in CUSTOM_CLASS_LIST["class_list"]:
        index = CUSTOM_CLASS_LIST["class_list"].index(customClass)
        classCount = CUSTOM_CLASS_LIST[customClass]["class_count"]
        print(f"{customClass} with class Count {classCount} is being deleted")
        CUSTOM_CLASS_LIST["class_list"] = CUSTOM_CLASS_LIST["class_list"][:index] + CUSTOM_CLASS_LIST["class_list"][index+1:]
        CUSTOM_CLASS_LIST.pop(customClass)
        delete_class("train", classCount)
        delete_class("valid", classCount)
        save_custom_info()
        write_yaml()
