import cv2
import numpy as np
import os
from sklearn.metrics.pairwise import euclidean_distances
import sys
import time
from tempfile import mkdtemp
import os.path as path


class Comercial_detector:
    """Objeto que dada una cantidad de videos de comerciales, detecta los comerciales en otro video
    """

    def __init__(self, comercials_path, detections_file, fps=30, x_cell=1, y_cell=1,
                 option='grey', hist_option='color', sobel_shape=3, cat_magnitud=False, treshold=0,
                 approx=True, dir_abs=False, bins=256, histogram_desc=False):
        """Constructor de la clase

        Args:
            comercials_path (str): La ruta a la carpeta de videos de comerciales
            detections_file (str): La ruta y nombre del archivo donde escribir las detecciones obtenidas
            fps (int): Los frames leidos en cada segundo
            x_cell (int): La cantidad de celdas a dividir la imagen a lo ancho
            y_cell (int): La cantidad de celdas a dividir la imagen a lo alto
            option (str): La opcion de conversion de la imagen, puede ser grey o rgb
            hist_option (str): La opcion del tipo de histograma, 'color' para histograma de color, 'gradient' para
                histograma de gradiente, 'both' para ambos y 'fourier' para la transformada de fourier
            sobel_shape (int): (es ignorado con hist_option='color') Tamano del filtro de sobel, puede 1, 3, 5 o 7
            cat_magnitude (bool): (es ignorado con hist_option='color') Si concatenar el histograma de magnitudes
                o desechar
            treshold (float): (es ignorado con hist_option='color') El umbral de conteo sobre el gradiente
            approx (bool): (es ignorado con hist_option='color') Si aproximar el calculo de la magnitud del gradiente
            dir_abs (bool): (es ignorado con hist_option='color') Si aplicar valor absoluto a las direcciones
                del gradiente
            bins (int): Cantidad de bines del histograma
            histogram_desc (bool): Si se quiere usar histogramas o solo la imagen plana de menor tamano
        """
        self.comercials_path = comercials_path
        self.detections_file = detections_file
        self.fps = fps
        self.x_cell = x_cell
        self.y_cell = y_cell
        self.option = option
        self.hist_option = hist_option
        self.sobel_shape = sobel_shape
        self.cat_magnitud = cat_magnitud
        self.treshold = treshold
        self.approx = approx
        self.dir_abs = dir_abs
        self.minimum_value_x = 1e-10
        self.bins = bins
        self.histogram_desc = histogram_desc
        self.files_path = {}
        self.nearest_frames = 5
        self.radius = 3
        self.detections_tresh = 0.5    #0.3
        self.com_durations = {}
        self.max_overlap = 1

    def fit(self):
        """Convierte todos los videos de comerciales a matrices de caracteristicas

        Raises:
            RuntimeError: El video no se pudo abrir
            ValueError: Opcion de conversion de imagen no reconocido
            ValueError: Opcion de transformada invalida
        """
        # Por cada archivo dentro de la carpeta de comerciales
        directory = os.listdir(self.comercials_path)
        for file in directory:
            # Concatena la ruta de la carpeta con el nombre del archivo
            file_path = self.comercials_path + '/' + file
            print('Convirtiendo video: ' + file_path)

            # Calcula la matriz de caracteristicas del comercial
            self.read_mp4(file_path, file.split('.')[0])
            print('-' * 60)
            print()

        return self

    def detect(self, video_path):
        '''Detecta los comerciales de un video, crea un archivo con las apariciones de cada comercial y una tasa de credibilidad
        
        Args:
            video_path (str): La ruta del video donde aparecen los comerciales
            
        Raises:
            RuntimeError: El video no se pudo abrir
            ValueError: Opcion de conversion de imagen no reconocido
            ValueError: Opcion de transformada invalida
        '''
        # Calcula la matriz de caracteristicas de los frames del video
        video_matrix = self.read_mp4(video_path)
        print("-"*60)
        print("Calculando distancias")
        for index, file in enumerate(self.files_path):
            # Concatena el nombre del comercial con la carpeta y lee el archivo
            comercial_matrix = np.memmap(file, dtype='float32', mode='r', shape=self.files_path[file])
            # Calcula distancia euclidiana entre matrices
            dist_matrix = euclidean_distances(video_matrix, comercial_matrix)
            # Crea vector de distancias maximas
            max_vector = np.max(dist_matrix, axis=1, keepdims=True)+1
            for i in range(self.nearest_frames):
                if i == 0:
                    # Crea matrices de indices y distancias minimas
                    argminim_matrix = np.argmin(dist_matrix, axis=1).reshape(-1, 1)
                    minimum_matrix = np.min(dist_matrix, axis=1).reshape(-1, 1)
                else:
                    # Reemplaza los minimos encontrados
                    maximum = np.max(minimum_matrix, axis=1, keepdims=True)
                    max_min = np.where(dist_matrix<=maximum, max_vector, dist_matrix)
                    # Encuentra los siguientes minimos e indices correspondientes y los concatena a la matriz
                    argmin = np.argmin(max_min, axis=1).reshape(-1, 1)
                    minim = np.min(max_min, axis=1, keepdims=True)
                    argminim_matrix = np.concatenate((argminim_matrix, argmin), axis=1)
                    minimum_matrix = np.concatenate((minimum_matrix, minim), axis=1)
            if index == 0:
                # Crea matrices de distancias minimas con los indices y nombres de comerciales correspondientes
                nearest_com = np.array([[file]*self.nearest_frames for i in range(dist_matrix.shape[0])], dtype=object)
                nearest_dist = minimum_matrix
                nearest_ind = argminim_matrix
            else:
                # Compara las distancias y retorna matriz del mismo tamano con las distancias minimas de ambas matrices con
                # sus respectivos indices y nombres de comerciales
                nearest_dist, nearest_ind, nearest_com = self.compare(nearest_dist, 
                                                                      nearest_ind,
                                                                      nearest_com,
                                                                      minimum_matrix, 
                                                                      argminim_matrix, 
                                                                      file)
        print("Detectando comerciales")
        detections, com_detections = self.detect_comercials(nearest_ind, nearest_com)
        # Extrae el nombre del video y realiza las detecciones
        video = video_path.replace('/', '.').split('.')[-2]
        self.write_detections(detections, com_detections, video)
    
    
    def write_detections(self, detections, com_detections, video):
        """Escribe las detecciones en el archivo que sobrepasan un umbral de decision
        
        Args:
            detections (numpy.ndarray): Vector de conteos de secuencia de cada frame del video
            com_detections (numpy.ndarray): Vector de nombres de los comerciales a los que pertenecen los conteos de las detecciones
            video (str): Nombre del video sobre el cual se estan realizando las detecciones de comerciales
        """
        print('\n' + '-'*60)
        with open(self.detections_file, 'w+') as the_file:
            last_com = ''
            last_start = 0
            last_duration = 0
            last_score = -1
            for i in range(len(detections)):
                # Si el conteo en el frame actual es mayor que el umbral
                if detections[i] > self.detections_tresh:
                    # Se concatena el nombre del video, el segundo en que comienza el comercial y la duracion junto al nombre del comercial y al score
                    comercial_path = com_detections[i][0]
                    # Extrae solo el nombre del comercial
                    comercial_name = comercial_path.replace('.', '\\').split('\\')[-2]
                    starts = i/self.fps
                    duration = self.com_durations[comercial_name]
                    score = detections[i][0]
                    # Verifica si existe superposicion de las detecciones y retorna el comercial con mejor score de entre los dos
                    overlapping, best_com, best_start, best_duracion, best_score = self.overlap(last_com, comercial_name, 
                                                                                                last_start, starts, 
                                                                                                last_duration, duration, 
                                                                                                last_score, score)
                    #self.write_row(the_file, video, starts, duration, comercial_name, score)
                    # Si no existe superposicion escribe la deteccion en el archivo y almacena en variables la actual
                    if not overlapping:
                        if last_score >= 0:
                            self.write_row(the_file, video, last_start, last_duration, last_com, last_score)
                        last_com = comercial_name
                        last_start = starts
                        last_duration = duration
                        last_score = score
                    # Si existe superposicion reemplaza el guardado por el de mejor score entre este y el actual
                    else:
                        last_com = best_com
                        last_start = best_start
                        last_duration = best_duracion
                        last_score = best_score
            self.write_row(the_file, video, last_start, last_duration, last_com, last_score)
                        
                        
    def write_row(self, file, video, start, duration, comercial, score):
        """Concatena una fila que se escribe en un archivo file, en la que contiene el nombre del video, el segundo que comienza el comercial, la duracion, el nombre y el score de este
        
        Args:
            file (TextIOWrapper): El archivo en el que se esta escribiendo
            video (str): El nombre del video donde se detecto el comercial
            start (float): El segundo en que comienza el comercial
            duration (float): La duracion del comercial
            comercial (str): El nombre del comercial
            score (float): El score del comercial o certeza de que el comercial aparecio
        """
        det = video + '\t' + str(np.round(start, 1)) + '\t' + str(np.round(duration, 1)) + '\t' + comercial + '\t' + str(np.round(score, 1))
        #if len(comercial) <= 8:
        #    det = video + '\t' + str(np.round(start, 1)) + '\t' + str(np.round(duration, 1)) + '\t' + comercial + '\t\t' + str(np.round(score, 1))
        print(det)
        file.write(det + '\n')
    
    def overlap(self, nombre_com1, nombre_com2, start1, start2, duracion1, duracion2, score1, score2):
        """Verifica si existe traslape entre dos videos dados, retorna los parametros del video que tiene mejor score
        
        Args:
            nombre_com1 (str): Nombre del comercial 1
            nombre_com2 (str): Nombre del comercial 2
            start1 (float): El segundo en que inicia el comercial 1
            start2 (float): El segundo en que inicia el comercial 2
            duration1 (float): La duracion del comercial 1
            duration2 (float): La duracion del comercial 2
            score1 (float): El score del comercial 1
            score2 (float): El score del comercial 2
            
        Returns:
            overlapping (bool): Si hay superposicion o no
            best_com (str): El nombre del comercial con mejor score
            best_start (float): El segundo de comienzo del comercial con mejor score
            best_duracion (float): La duracion del comercial con mejor score
            best_score (float): El mejor score de entre los dos comerciales
        """
        assert(start1 <= start2)
        overlapping = False
        # start2 siempre mayor que start1 si start1 es anterior
        # Si el video 2 comienza antes de que termine el primero
        if start2 < start1 + duracion1 - self.max_overlap:
            overlapping = True
        # Guarda el comercial con mejor score
        best_com = nombre_com2
        best_start = start2
        best_duracion = duracion2
        best_score = score2
        if score1 > score2:
            best_com = nombre_com1
            best_start = start1
            best_duracion = duracion1
            best_score = score1
        
        return overlapping, best_com, best_start, best_duracion, best_score
    
    def detect_comercials(self, nearest_ind, nearest_com):
        """Dada una matriz de indices de los comerciales mas cercanos y los comerciales a lo que pertenecen esos indices hace un
        conteo de los frames que pertenecen a una secuencia y calcula el comercial que mas se parece, es decir, el que tiene un 
        mayor conteo
        
        Args:
            nearest_ind (numpy.ndarray): La matriz de indices de los comerciales mas parecidos
            nearest_com (numpy.ndarray): La matriz de los nombres de los comerciales a los que pertenecen los indices
            
        Returns:
            detections (numpy.ndarray): Vector de conteos de secuencia de cada frame del video
            com_detections (numpy.ndarray): Vector de nombres de los comerciales a los que pertenecen los conteos de las detecciones
        """
        # Inicializa el vector de conteos
        detections = np.zeros((nearest_ind.shape[0], 1), dtype=np.float32)
        # Inicializa el vector de nombres de comerciales
        com_detections = np.array([[''] for i in range(nearest_ind.shape[0])], dtype=object)
        for i in range(nearest_ind.shape[0]):
            sys.stdout.write("\rProcesando el frame {}/{}".format(i, nearest_ind.shape[0]))
            # Si existe un indice que sea menor que un radio, detecta el inicio de un comercial
            if any(nearest_ind[i, :] < self.radius):
                # Extrae los nombres de los comerciales que podrian comenzar en el frame actual
                comercials = nearest_com[i, :][nearest_ind[i, :]<self.radius]                
                # Inicializa conteo maximo y el comercial al que pertenece el conteo maximo
                max_count = 0
                max_com = ''
                for com in comercials:
                    # Hace un conteo de los frames siguientes que estan en un cierto rango
                    count = self.sequence(nearest_ind, nearest_com, i, com)
                    # Si el conteo del comercial es mayor que el del anterior lo reemplaza
                    if count > max_count:
                        max_count = count
                        max_com = com
                # Guarda el comercial que obtuvo un mayor conteo y su conteo
                detections[i] = max_count
                com_detections[i] = max_com
        return detections, com_detections
    
    def sequence(self, nearest_ind, nearest_com, index, comercial):
        """Parado en un frame del video, cuenta la cantidad de frames siguientes que se parecen al comercial dado
        
        Args:
            nearest_ind (numpy.ndarray): La matriz de indices de los comerciales mas parecidos
            nearest_com (numpy.ndarray): La matriz de los nombres de los comerciales a los que pertenecen los indices
            index (int): La posicion del frame desde la que se empieza el conteo
            comercial (str): El nombre del comercial que se encontrar en la secuencia
        
        Returns:
            percentage (float): El porcentaje de frames que se parecen entre el comercial y parte del video
        """
        # Cantidad de frames del comercial
        com_frames = self.files_path[comercial][0]
        count = 1
        if index + com_frames < nearest_ind.shape[0]:
	        # Lista de indices del comercial (entre 0 y n-1 donde n es el tamano del comercial)
	        indexes = np.arange(com_frames).reshape(com_frames, 1)
	        # Matriz donde tiene True si el indice encontrado esta cerca del indice correspondiente al comercial y pertenece al mismo comercial
	        near = np.abs(nearest_ind[index:index+com_frames, :] - indexes) < self.radius
	        same_com = nearest_com[index:index+com_frames, :] == comercial
	        aux = near * same_com
	        # Transforma la matriz en vector y coloca un 1 si en la fila hay uno o mas frames pertenecientes al mismo comercial de indice cercano al de la secuencia
	        com_vector = np.sum(aux, axis=1, keepdims=True) > 0
	        # Cuenta la cantidad de indices que parecen secuencia del comercial
	        count = np.sum(com_vector)
        # Normaliza el conteo por la cantidad de frames del comercial
        percentage = count/com_frames
        return percentage
    
    
    def compare(self, nearest_dist, nearest_ind, nearest_com, minimum_matrix, argminim_matrix, file):
        """Compara dos matrices de distancias, retorna la matriz minima junto a las matrices de indices y 
        comercial al cual pertenecen las distancias modificadas
        
        Args:
            nearest_dist (numpy.ndarray): Matriz de distancias 1
            nearest_ind (numpy.ndarray): Matriz de indices a los que pertenecen las distancias
            nearest_com (numpy.ndarray): Matriz de nombres de comerciales a los que pertenecen las distancias
            minimum_matrix (numpy.ndarray): Matriz de distancias 2
            argminim_matrix (numpy.ndarray): Matriz de indices a los que pertenecen las distancias 2
            file (str): Nombre video al que pertenecen las distancias 2
        
        Returns:
            nearest_dist2 (numpy.ndarray): Matriz de distancias minimas de entre la 1 y 2
            nearest_ind2 (numpy.ndarray): Matriz de indices a las que pertenecen las distancias minimas
            nearest_com2 (numpy.ndarray): Matriz de nombres de videos a los que pertenecen las distancias minimas
        """
        # Copia las matrices de ditancias, indices y nombre comerciales
        nearest_dist2 = nearest_dist.copy()
        nearest_ind2 = nearest_ind.copy()
        nearest_com2 = nearest_com.copy()
        # Concatena matrices de distancia, indices y nombres
        concatenate_dist = np.concatenate((nearest_dist, minimum_matrix), axis=1)
        concatenate_ind = np.concatenate((nearest_ind, argminim_matrix), axis=1)
        file_matrix = np.array([[file]*self.nearest_frames for i in range(nearest_dist.shape[0])], dtype=object)
        concatenate_com = np.concatenate((nearest_com, file_matrix), axis=1)
        # Calcula vector de distancias maximas
        max_vector = np.max(concatenate_dist, axis=1, keepdims=True)+1
        for i in range(nearest_dist.shape[1]):
            if i == 0:
                # Crea la matriz de minimos
                minimums = np.min(concatenate_dist, axis=1, keepdims=True)
            else:
                # Reemplaza los minimos ya visitados por el maximo y calcula los siguientes minimos
                max_min = np.where(concatenate_dist > minimums, concatenate_dist, max_vector)
                minimums = np.min(max_min, axis=1, keepdims=True)
            # Identifica los minimos y los coloca en la matriz de distancias minimas, junto a sus indices y nombre de comercial correspondientes
            minimums_ind = np.where(concatenate_dist == minimums, True, False)
            nearest_dist2[:, i] = concatenate_dist[minimums_ind][:nearest_dist.shape[0]]
            nearest_ind2[:, i] = concatenate_ind[minimums_ind][:nearest_dist.shape[0]]
            nearest_com2[:, i] = concatenate_com[minimums_ind][:nearest_dist.shape[0]]
        return nearest_dist2, nearest_ind2, nearest_com2

    def calcular_histograma_color(self, cell, channel=0):
        """Calcula el histograma de gradientes de una celda.

        Args:
            cell (numpy.ndarray): La celda de la imagen
            channel (int): El canal de la celda sobre el cual se calcula el histograma

        Returns:
            hist (numpy.ndarray): El vector de caracteristicas de la celda
        """
        # Se calcula el histograma de color de la celda en el canal channel y se normaliza
        hist = cv2.calcHist([cell], [channel], None, [self.bins], [0, 256]).astype(np.float)
        hist = hist / (cell.shape[0] * cell.shape[1])
        return hist

    def calcular_histograma_transf(self, cell, channel=0, transform='gradient'):
        """Calcula el histograma de gradientes o de la transformada de fourier de una celda.

        Args:
            cell (numpy.ndarray): La celda de la imagen
            channel (int): El canal de color sobre el que se realiza la transformada
            transform (str): 'gradient' para calcular histograma de gradiente, 'fourier' para el histograma de la
                trasformada de fourier

        Raises:
            ValueError: Opcion de transformada invalida

        Returns:
            directions_hist (numpy.ndarray): El vector de caracteristicas de la celda
        """
        # Si transform es 'gradient', calcula las derivadas en x e y con el filtro de sobel con pesos gausianos,
        # se realiza un padding de reflejo
        if transform == 'gradient':
            transf_x = cv2.Sobel(cell, cv2.CV_64F, 1, 0, ksize=self.sobel_shape,
                                 borderType=cv2.BORDER_REFLECT_101).astype(
                np.float32)  # BORDER_REPLICATE, BORDER_REFLECT, BORDER_REFLECT_101, BORDER_WRAP, BORDER_CONSTANT
            transf_y = cv2.Sobel(cell, cv2.CV_64F, 0, 1, ksize=self.sobel_shape,
                                 borderType=cv2.BORDER_REFLECT_101).astype(np.float32)

        # Si transform es 'fourier', calcula las componentes real (x) e imaginario (y) de la transformada de fourier
        elif transform == 'fourier':
            # Se calcula la fft en dos dimensiones de la imagen
            f = np.fft.fft2(cell)
            # Se separa la transformada en componente real y componente imaginario
            transf_y = np.imag(f).astype(np.float32)
            transf_x = np.real(f).astype(np.float32)
        else:
            raise ValueError("Opcion de transformada invalida")

        # Se calcula magnitud del gradiente
        if self.approx:
            magnitude = np.abs(transf_x) + np.abs(transf_y)
        else:
            magnitude = np.sqrt(np.power(transf_x, 2) + np.power(transf_y, 2))

        # Se crea la matriz que pone un 255 donde la magnitud es mayor al umbral, 0 en caso contrario
        mask = np.where(cell > self.treshold, 255, cell)
        mask = np.where(mask != 255, 0, mask).astype(np.uint8)
        if len(mask.shape) > 2:
            mask = mask[:, :, channel]

        # Se calcula matriz de direcciones del gradiente
        minimum_matrix = np.ones(transf_x.shape, dtype=np.float32) * self.minimum_value_x
        transf_x = np.maximum(transf_x, minimum_matrix)
        directions = np.arctan(transf_y / transf_x)

        # Se calcula el histograma de la matriz de direcciones y se normaliza
        if self.dir_abs:
            directions_hist = cv2.calcHist([np.abs(directions)], [channel], mask, [self.bins], [0, np.pi]).astype(
                np.float)
        else:
            directions_hist = cv2.calcHist([directions], [channel], mask, [self.bins], [-np.pi, np.pi]).astype(np.float)
        # Se normaliza la matriz de direcciones
        directions_hist = directions_hist / (cell.shape[0] * cell.shape[1])

        # Si esta activa la opcion de concatenar magnitud, se calcula el histograma de magnitudes
        if self.cat_magnitud:
            magnitud_hist = cv2.calcHist([magnitude], [channel], mask, [self.bins])
            magnitud_hist = magnitud_hist / (cell.shape[0] * cell.shape[1])
            directions_hist = np.concatenate((directions_hist, magnitud_hist), axis=None)

        return directions_hist

    def hist_descriptor(self, img):
        """Calcula el histograma por zonas de una imagen, puede ser con la imagen a color (por cada canal) o
        convertirla a escala de grises.
        Tambien calcula histograma de color y/o de gradiente.

        Args:
            img (numpy.ndarray): La imagen en BGR

        Raises:
            ValueError: Opcion de conversion de imagen no reconocido
            ValueError: Opcion de transformada invalida

        Returns:
            feature (numpy.ndarray): El vector de caracteristicas de la imagen
        """
        # Se calcula alto y ancho de imagen
        height = img.shape[0]
        width = img.shape[1]
        # Se calcula alto y ancho de la parte de 1s de la mascara
        mask_height = height // self.y_cell
        mask_width = width // self.x_cell

        # Si la opcion channel es grey se convierte la imagen a escala de grises
        new_img = img.copy()
        if self.option == 'grey':
            new_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        elif self.option != 'rgb':
            raise ValueError("Opcion de conversion de imagen no reconocido")
        # Si la imagen no tiene tercera dimension le asigna 1 a channels, caso contrario le asigna la tercera dimension
        if len(new_img.shape) < 3:
            channels = 1
        else:
            channels = new_img.shape[2]
        # Crea un vector de caracteristicas vacio
        feature = np.array([])

        # Por cada celda en x
        for j in range(self.x_cell):
            # Por cada celda en y
            for k in range(self.y_cell):
                # Se calculan los indices donde comienza y termina el filtro en x e y
                x1 = j * mask_width
                x2 = (j + 1) * mask_width
                y1 = k * mask_height
                y2 = (k + 1) * mask_height
                # Se extrae la celda de la imagen, tambien se utilizan pixeles del borde cuando el alto/y_cell
                # no es entero, lo mismo para x
                if j == self.x_cell - 1 and k == self.y_cell - 1:
                    cell = new_img[y1:, x1:]
                elif j == self.x_cell - 1:
                    cell = new_img[y1:y2, x1:]
                elif k == self.y_cell - 1:
                    cell = new_img[y1:, x1:x2]
                else:
                    cell = new_img[y1:y2, x1:x2]
                # Por cada canal
                for i in range(channels):
                    # Si la opcion es both, se calculan los histogramas de gradiente y color, y se concatenan
                    if self.hist_option == 'both':
                        hist1 = self.calcular_histograma_color(cell, i)
                        hist2 = self.calcular_histograma_transf(cell, i, 'gradient')
                        hists = np.concatenate((hist1, hist2), axis=None)
                    # Si la opcion es color, se calcula el histograma de colores
                    elif self.hist_option == 'color':
                        hists = self.calcular_histograma_color(cell, i)
                    # Si la opcion es color, gradient o fourier, se calcula el histograma normalizado de la celda
                    else:
                        hists = self.calcular_histograma_transf(cell, i, self.hist_option)
                    # Se concatena el histograma de la celda al vector de caracteristicas
                    feature = np.concatenate((feature, hists), axis=None)
        return feature

    def grey_descriptor(self, img):
        # Se convierte la imagen a escala de grises
        new_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.resize(new_img, (self.x_cell, self.y_cell), interpolation=cv2.INTER_LINEAR).reshape(1, -1)

    def read_mp4(self, video_path, feature_path=''):
        """Lee un archivo de video mp4, calcula el vector de caracteristicas de sus frames y lo guarda en un archivo.

        Args:
            video_path (str): La ruta al video
            feature_path (str): Nombre del archivo donde guardar los vectores de caracteristicas

        Raises:
            RuntimeError: El video no se pudo abrir
            ValueError: Opcion de conversion de imagen no reconocido
            ValueError: Opcion de transformada invalida
        """
        # Lee el video
        video = cv2.VideoCapture(video_path)
        total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
        print("Total de frames del video", int(total_frames))
        # Lee alto y ancho de los frames del video
        video_height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
        video_width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
        print("Ancho de los frames: {}".format(video_width))
        print("Alto de los frames: {}".format(video_height))
        # Calcula la cantidad de frames que se debe saltar
        video_fps = min(self.fps, video.get(cv2.CAP_PROP_FPS))
        skip_frames = int(video.get(cv2.CAP_PROP_FPS) // video_fps + 1)
        print("Sampleando cada: {}".format(skip_frames))
        # Guarda duracion del video
        self.com_durations[video_path.replace('/', '.').split('.')[-2]] = total_frames/video.get(cv2.CAP_PROP_FPS)
        # Si el video no se logro abrir arroja una exepcion
        if not video.isOpened():
            raise RuntimeError("El video no se pudo abrir")
        # Mientras el video este abierto
        while video.isOpened():
            actual_frame = video.get(cv2.CAP_PROP_POS_FRAMES)
            # Si el frame actual es el ultimo, rompe loop
            if int(actual_frame) == int(total_frames):
                break
            sys.stdout.write("\rProcesando el frame {}/{}".format(int(actual_frame), int(total_frames)))
            # Extrae un frame del video, si ret es False, entonces no hay mas frames disponibles
            ret, frame = video.read()
            # Si no quedan frames por extraer rompe loop
            if not ret:
                break
            # Calcula el descriptor de la imagen
            # Si el descriptor es por histogramas
            if self.histogram_desc:
                descriptor = self.hist_descriptor(frame)
            # Si el descriptor es por escala de grises
            else:
                descriptor = self.grey_descriptor(frame)
            # Si el frame actual es el primero, se crea la matriz de caracteristicas
            if actual_frame == 0:
                features_matrix = descriptor.reshape(1, -1)
            # Sino, se concatena el descriptor a la matriz de caracteristicas
            else:
                features_matrix = np.concatenate((features_matrix, descriptor.reshape(1, -1)), axis=0)
            # Salta skip_frames imagenes del video
            video.set(cv2.CAP_PROP_POS_FRAMES, actual_frame + skip_frames)

        print("\nDimensiones del vector de caracteristicas: {}".format(features_matrix.shape))
        if feature_path != '':
            # Guarda la matriz en un archivo
            filename = path.join(mkdtemp(), feature_path + '.dat')
            self.files_path[filename] = features_matrix.shape
            fp = np.memmap(filename, dtype='float32', mode='w+', shape=features_matrix.shape)
            fp[:] = features_matrix
            del fp

        return features_matrix


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso: {} [television/video.mp4] [comerciales/]".format(sys.argv[0]))
        sys.exit(1)
    video_path = sys.argv[1]
    comercial_path = sys.argv[2]
    detections_path = 'detect.txt'
    #comercial_path = 'comerciales'
    #video_path = 'television/mega-2014_04_10.mp4'
    #video_path = 'television/mega-2014_04_11.mp4'
    #video_path = 'television/mega-2014_04_25.mp4'
    tic = time.time()
    com = Comercial_detector(comercial_path, detections_path, fps=3, x_cell=4, y_cell=4,
                             option='grey', hist_option='color', sobel_shape=3,
                             cat_magnitud=False, treshold=100, approx=True, bins=8,
                             histogram_desc=True)
    com.fit().detect(video_path)
    dt = time.time() - tic
    print('Tiempo de ejecucion total: {} min {} s'.format(dt//60, np.round((dt%60), 0)))
