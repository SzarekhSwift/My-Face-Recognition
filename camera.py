import cv2
import sys
 
from PIL import Image
 
def CatchPICFromVideo(window_name, camera_idx, catch_pic_num, path_name):
    cv2.namedWindow(window_name)
    
    # Источник видео, это может быть сохраненное видео или прямо с USB-камеры
    cap = cv2.VideoCapture(camera_idx)                
    
    # Сообщите OpenCV использовать классификатор распознавания лиц
    classfier = cv2.CascadeClassifier('E:/Doc/GitClone/My-Face-Recognition/haarcascades/haarcascade_frontalface_alt2.xml')
    
    # Цвет рамки, которая будет нарисована после распознавания лица, формат RGB
    color = (0, 255, 0)
    
    num = 0    
    while cap.isOpened():
        ok, frame = cap.read() # Прочитать фрейм данных
        if not ok:            
            break                
    
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Преобразовать изображение текущего кадра в изображение в оттенках серого 
        
        # Обнаружение лица, 1,2 и 2 - это коэффициент масштабирования изображения и эффективное количество обнаруживаемых точек соответственно.
        faceRects = classfier.detectMultiScale(grey, scaleFactor = 1.2, minNeighbors = 3, minSize = (32, 32))
        if len(faceRects) > 0:          # Больше 0, распознается человеческое лицо 
            for faceRect in faceRects:  # Создайте рамку для каждого лица отдельно
                x, y, w, h = faceRect                        
                
                # Сохранить текущий кадр как картинку
                img_name = '%s/%d.jpg'%(path_name, num)                
                image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                cv2.imwrite(img_name, image)                                
                                
                num += 1                
                if num > (catch_pic_num):   # Выход из цикла, если он превышает указанное максимальное количество сохранений
                    break
                
                # Нарисуйте прямоугольник
                cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)
                
                # Покажите, сколько изображений лица было снято на данный момент
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame,'num:%d' % (num),(x + 30, y + 30), font, 1, (255,0,255),4)                
        
        # Превышение указанного максимального количества сохранений для завершения программы
        if num > (catch_pic_num): break                
                       
        # Display image
        cv2.imshow(window_name, frame)        
        c = cv2.waitKey(10)
        if c & 0xFF == ord('q'):
            break        
    
    # Отпустить камеру и уничтожить все окна
    cap.release()
    cv2.destroyAllWindows() 
    
if __name__ == '__main__':
    if len(sys.argv) != 1:
        print("Usage:%s camera_id face_num_max path_name\r\n" % (sys.argv[0]))
    else:
        CatchPICFromVideo("Face capture", 0, 1000, './data/Mark')