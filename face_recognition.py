#-*- coding: utf-8 -*-

import cv2
import sys
import gc
from face_train import Model

if __name__ == '__main__':
    if len(sys.argv) != 1:
        print("Usage:%s camera_id\r\n" % (sys.argv[0]))
        sys.exit(0)
        
    #загрузка модели
    model = Model()
    model.load_model(file_path = './model/me.face.model.h5')    
              
    #цвет границы прямоугольника      
    color = (0, 255, 0)
    
    #захват видеопотока
    cap = cv2.VideoCapture(0)
    
    #путь хранения каскаов Хаара
    cascade_path = "./haarcascades/haarcascade_frontalface_alt2.xml"  
    
    #обнаружение лица
    while True:
        ret, frame = cap.read()   
        
        if ret is True:
            
            #задать градиент
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            continue
        #Используем классификатор распознавания лиц
        cascade = cv2.CascadeClassifier(cascade_path)                

        #Используем классификатор, чтобы определить, какая область является лицом
        faceRects = cascade.detectMultiScale(frame_gray, scaleFactor = 1.2, minNeighbors = 3, minSize = (32, 32))        
        if len(faceRects) > 0:                 
            for faceRect in faceRects: 
                x, y, w, h = faceRect
                
                image = frame[y - 10: y + h + 10, x - 10: x + w + 10]
                faceID = model.face_predict(image)   
                
                if faceID == 0:                                                        
                    cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, thickness = 2)
                    
                    #Кто изображен
                    cv2.putText(frame,'Mark', 
                                (x + 30, y + 30),                      #координаты
                                cv2.FONT_HERSHEY_SIMPLEX,              #шрифт
                                1,                                     #размер шрифта
                                (255,0,255),                           #цвет
                                2)                                     #ширина строки
                else:
                    pass
                            
        cv2.imshow("Me Identification", frame)
        
        k = cv2.waitKey(10)
        if k & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()