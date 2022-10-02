import random
 
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.optimizers import gradient_descent_v2
from keras.utils import np_utils
from keras.models import load_model
from keras import backend as K

 
from load_dataset import load_dataset, resize_image, IMAGE_SIZE
 
 
 
class Dataset:
    def __init__(face, path_name):
        #Обучающий набор
        face.train_images = None
        face.train_labels = None
        
        #Проверочный набор
        face.valid_images = None
        face.valid_labels = None
        
        # Тестовый набор
        face.test_images  = None            
        face.test_labels  = None
        
        # Путь загрузки набора данных
        face.path_name    = path_name
        
        # Порядок размеров, используемых текущей библиотекой
        face.input_shape = None
        
    # Загрузить набор данных и разделить набор данных по принципу перекрестной проверки и выполнить соответствующую предварительную обработку
    def load(face, img_rows = IMAGE_SIZE, img_cols = IMAGE_SIZE, img_channels = 3, nb_classes = 2):
        # Загрузить набор данных в память
        images, labels = load_dataset(face.path_name)        
        
        train_images, valid_images, train_labels, valid_labels = train_test_split(images, labels, test_size = 0.3, random_state = random.randint(0, 100))        
        _, test_images, _, test_labels = train_test_split(images, labels, test_size = 0.5, random_state = random.randint(0, 100))                
        
        # Если текущий порядок измерений - это 'th', порядок ввода данных изображения следующий: каналы, строки, столбцы, иначе: строки, столбцы, каналы
        # Эта часть кода предназначена для реорганизации набора обучающих данных в соответствии с порядком измерений, требуемым библиотекой keras
        if K.image_data_format() == 'th':
            train_images = train_images.reshape(train_images.shape[0], img_channels, img_rows, img_cols)
            valid_images = valid_images.reshape(valid_images.shape[0], img_channels, img_rows, img_cols)
            test_images = test_images.reshape(test_images.shape[0], img_channels, img_rows, img_cols)
            face.input_shape = (img_channels, img_rows, img_cols)            
        else:
            train_images = train_images.reshape(train_images.shape[0], img_rows, img_cols, img_channels)
            valid_images = valid_images.reshape(valid_images.shape[0], img_rows, img_cols, img_channels)
            test_images = test_images.reshape(test_images.shape[0], img_rows, img_cols, img_channels)
            face.input_shape = (img_rows, img_cols, img_channels)            
            
            # Вывести номер обучающего набора, проверочного набора, тестового набора
            print(train_images.shape[0], 'train samples')
            print(valid_images.shape[0], 'valid samples')
            print(test_images.shape[0], 'test samples')
        
            # Наша модель использует category_crossentropy в качестве функции потерь, поэтому она должна быть основана на количестве категорий nb_classes
            #Category метка векторизуется методом однократного кодирования. Здесь всего две категории. После преобразования данные метки становятся двумерными.
            train_labels = np_utils.to_categorical(train_labels, nb_classes)                        
            valid_labels = np_utils.to_categorical(valid_labels, nb_classes)            
            test_labels = np_utils.to_categorical(test_labels, nb_classes)                        
        
            # Плавающие данные в пикселях для нормализации
            train_images = train_images.astype('float32')            
            valid_images = valid_images.astype('float32')
            test_images = test_images.astype('float32')
            
            # Нормализуем его, и значение пикселя изображения нормализуется до интервала 0 ~ 1
            train_images /= 255
            valid_images /= 255
            test_images /= 255            
        
            face.train_images = train_images
            face.valid_images = valid_images
            face.test_images  = test_images
            face.train_labels = train_labels
            face.valid_labels = valid_labels
            face.test_labels  = test_labels
            
#CNN
class Model:
    def __init__(face):
        face.model = None 
        
    # Моделирование
    def build_model(face, dataset, nb_classes = 2):
        # Создайте пустую сетевую модель, это линейная модель с накоплением, каждый слой нейронной сети будет добавлен последовательно, профессиональное название - последовательная модель или линейная модель с накоплением
        face.model = Sequential() 
        
        # Следующий код будет последовательно добавлять уровни, необходимые для сети CNN, добавление - это сетевой уровень
        face.model.add(Conv2D(filters=32, kernel_size=3, padding='same', input_shape = dataset.input_shape, data_format='channels_last'))    # 1 2-мерный сверточный слой
        face.model.add(Activation('relu'))                                  # 2 слой функции активации
        
        face.model.add(Conv2D(filters=32, kernel_size=3))                             # 3 2-мерный сверточный слой 
        face.model.add(Activation('relu'))                                  # 4 слой функции активации
        
        face.model.add(MaxPooling2D(pool_size=2))                      # 5 Уровень объединения
        face.model.add(Dropout(0.25))                                       # 6 Выпадающий слой
 
        face.model.add(Conv2D(filters=64, kernel_size=3))             # 7 2-мерный сверточный слой
        face.model.add(Activation('relu'))                                  # 8 Функциональный слой активации
        
        face.model.add(Conv2D(filters=64, kernel_size=3))                             # 9 2-мерный сверточный слой
        face.model.add(Activation('relu'))                                  # 10 слой функции активации
        
        face.model.add(MaxPooling2D(pool_size=2))                      # 11 объединяющий слой
        face.model.add(Dropout(0.25))                                       # 12 Выпадающий слой
 
        face.model.add(Flatten())                                           # 13 Свернуть слой
        face.model.add(Dense(512))                                          # 14 Плотный слой, также известный как полностью связанный слой
        face.model.add(Activation('relu'))                                  # 15 Функциональный слой активации 
        face.model.add(Dropout(0.5))                                        # 16 Выпадающий слой
        face.model.add(Dense(nb_classes))                                   # 17 Плотный слой
        face.model.add(Activation('softmax'))                               # 18 Слой классификации, вывести окончательный результат
        
        # Обзор выходной модели
        face.model.summary()
        
    # Тренировочная модель
    def train(face, dataset, batch_size = 20, epochs = 10, data_augmentation = True):        
        sgd = gradient_descent_v2.SGD(learning_rate = 0.01, decay = 1e-6, 
                  momentum = 0.9, nesterov = True) # Используя оптимизатор SGD + импульс для обучения, сначала сгенерируйте объект оптимизатора 
        face.model.compile(loss='categorical_crossentropy',
                           optimizer=sgd,
                           metrics=['accuracy'])   # Завершить фактическую работу по настройке модели
        
        # Не используйте продвижение данных, так называемое продвижение заключается в создании новых из обучающих данных, которые мы предоставляем, с использованием таких методов, как вращение, переворачивание и шум
        # Обучающие данные, сознательно увеличиваем масштаб обучающих данных и увеличиваем объем обучения модели
        if not data_augmentation:            
            face.model.fit(dataset.train_images,
                           dataset.train_labels,
                           batch_size = batch_size,
                           epochs = epochs,
                           validation_data = (dataset.valid_images, dataset.valid_labels),
                           shuffle = True)
        # Используйте данные в реальном времени для улучшения
        else:            
            # Определить генератор данных для продвижения данных, он возвращает datagen объекта генератора, datagen вызывается каждый раз
            # Во-вторых, он генерирует набор данных (последовательная генерация) для экономии памяти. Фактически, это генератор данных Python
            datagen = ImageDataGenerator(
                featurewise_center = False,             # Будет ли децентрализовать входные данные (среднее значение 0),
                samplewise_center  = False,             # Будет ли каждая выборка входных данных означать 0
                featurewise_std_normalization = False,  # Стандартизированы ли данные (входные данные делятся на стандартное отклонение набора данных)
                samplewise_std_normalization  = False,  # Делить ли данные каждой выборки на собственное стандартное отклонение
                zca_whitening = False,                  # Применять ли отбеливание ZCA к исходным данным
                rotation_range = 30,                    # Угол, на который изображение поворачивается случайным образом во время продвижения данных (диапазон 0 ～ 180)
                width_shift_range  = 0.3,               # Степень горизонтального смещения изображения при продвижении данных (единица измерения - пропорция ширины изображения, число с плавающей запятой между 0 и 1)
                height_shift_range = 0.3,               # То же, что и выше, но здесь вертикальное
                horizontal_flip = True,                 # Выполнять ли случайный переворот по горизонтали
                vertical_flip = True)                  # Выполнять ли случайный вертикальный переворот
 
            # Вычислить количество всего набора обучающих выборок для нормализации значений характеристик, отбеливания ZCA и т. Д.
            datagen.fit(dataset.train_images)                        
 
            # Используйте генератор, чтобы начать обучение модели
            face.model.fit(datagen.flow(dataset.train_images, dataset.train_labels,
                                                batch_size = batch_size),
                                                #steps_per_epoch = dataset.train_images.shape[0],
                                                epochs = epochs,
                                                validation_data = (dataset.valid_images, dataset.valid_labels))    
    
    MODEL_PATH = './my.face.model.h5'
    def save_model(face, file_path = MODEL_PATH):
         face.model.save(file_path)
 
    def load_model(face, file_path = MODEL_PATH):
         face.model = load_model(file_path)
 
    def evaluate(face, dataset):
         score = face.model.evaluate(dataset.test_images, dataset.test_labels, verbose = 1)
         print("%s: %.2f%%" % (face.model.metrics_names[1], score[1] * 100))
 
    # Узнай лицо
    def face_predict(face, image):    
        # Порядок измерения по-прежнему зависит от внутренней системы
        if K.image_data_format() == 'th' and image.shape != (1, 3, IMAGE_SIZE, IMAGE_SIZE):
            image = resize_image(image)                             # Размер должен совпадать с размером обучающего набора. Оба должны быть IMAGE_SIZE x IMAGE_SIZE
            image = image.reshape((1, 3, IMAGE_SIZE, IMAGE_SIZE))   # В отличие от обучения модели, на этот раз прогнозируется только для 1 изображения 
        elif K.image_data_format() == 'tf' and image.shape != (1, IMAGE_SIZE, IMAGE_SIZE, 3):
            image = resize_image(image)
            image = image.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))                    
        
        #плавать и нормализовать
        image = image.astype('float32')
        image /= 255
        
        #Учитывая вероятность того, что входное изображение принадлежит каждой категории, мы являемся бинарной категорией, тогда функция даст вероятность того, что входное изображение принадлежит 0 и 1
        result = face.model.predict(image)
        print('result:', result)
        
        #дать прогнозы класса: 0 или 1
        result = face.model.predict_classes(image)        

        #Возврат результатов прогнозирования категории
        return result[0]
    

if __name__ == '__main__':
    dataset = Dataset('./data/')    
    dataset.load()
    
    
    model = Model()
    model.build_model(dataset)
    model.train(dataset)
    model.save_model(file_path = './model/me.face.model.h5')
    '''
    model = Model()
    model.load_model(file_path = './model/me.face.model.h5')
    model.evaluate(dataset)'''