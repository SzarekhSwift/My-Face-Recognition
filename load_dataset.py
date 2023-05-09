import os
import sys
import numpy as np
import cv2
 
IMAGE_SIZE = 64
 
# Отрегулируйте размер в соответствии с указанным размером изображения
def resize_image(image, height = IMAGE_SIZE, width = IMAGE_SIZE):
    top, bottom, left, right = (0, 0, 0, 0)
    
    # Получить размер изображения
    h, w, _ = image.shape
    
    # Для картинок разной длины и ширины найдите самую длинную сторону
    longest_edge = max(h, w)    
    
    # При вычислении короткой стороны необходимо увеличить ширину в пикселях, чтобы она была такой же длины, как и длинная сторона
    if h < longest_edge:
        dh = longest_edge - h
        top = dh // 2
        bottom = dh - top
    elif w < longest_edge:
        dw = longest_edge - w
        left = dw // 2
        right = dw - left
    else:
        pass 
    
    # Цвет RGB
    BLACK = [0, 0, 0]
    
    # Добавить рамку к изображению, которая является длиной и шириной изображения, и cv2.BORDER_CONSTANT определяет цвет границы по значению
    constant = cv2.copyMakeBorder(image, top , bottom, left, right, cv2.BORDER_CONSTANT, value = BLACK)
    
    # Настроить размер изображения и вернуть
    return cv2.resize(constant, (height, width))
 
# Читать данные обучения
images = []
labels = []
def read_path(path_name):    
    for dir_item in os.listdir(path_name):
        # Начать с исходного пути для наложения, объединить в узнаваемый рабочий путь
        full_path = os.path.abspath(os.path.join(path_name, dir_item))
        
        if os.path.isdir(full_path):    # Если это папка, продолжить рекурсивный вызов
            read_path(full_path)
        else:   #файл
            if dir_item.endswith('.jpg'):
                image = cv2.imread(full_path)                
                image = resize_image(image, IMAGE_SIZE, IMAGE_SIZE)
                
                # Оставьте этот код, вы можете увидеть фактический эффект вызова функции resize_image ()
                #cv2.imwrite('1.jpg', image)
                
                images.append(image)                
                labels.append(path_name)                                
                    
    return images,labels
    
 
# Считать данные обучения по указанному пути
def load_dataset(path_name):
    images,labels = read_path(path_name)    
    
    # Преобразуем все входные изображения в четырехмерный массив, размер (количество изображений * IMAGE_SIZE * IMAGE_SIZE * 3)
    # Всего 1000 изображений, а IMAGE_SIZE - 64, поэтому размер для меня 1200 * 64 * 64 * 3
    # Картинка 64 * 64 пикселя, у одного пикселя 3 значения цвета (RGB)
    images = np.array(images)
    print(images.shape)    
    
    # Данные аннотации, в папке 'data' находятся все мои изображения лиц, все обозначены как 0, другая папка находится под одноклассником, все обозначены как 1
    labels = np.array([0 if label.endswith('Mark') else 1 for label in labels])
    #labels = np.array(labels)   
    
    return images, labels
 
if __name__ == '__main__':
    if len(sys.argv) != 1:
        print("Usage:%s path_name\r\n" % (sys.argv[0]))    
    else:
        images, labels = load_dataset('E:/Doc/GitClone/My-Face-Recognition/data') #"../new/data"