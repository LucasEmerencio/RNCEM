import os
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms
import subprocess
from matplotlib import pyplot as plt
import cv2
import json
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS



app = Flask(__name__)
CORS(app)

app.config['UPLOAD_FOLDER'] = os.path.join(os.getcwd(), 'uploads')
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])



# Configurações iniciais
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Parâmetros de entrada
model_name = "vgg16"
weights_path = "/home/okura/Documents/tcc/RNCEM/Rede_classificadora/vgg16_aug_pt.pth" #SUBSTITUIR PELO SEU CAMINHO DE PESOS
quiet_mode = True

# Carregar o modelo
if model_name == "vgg16":
    model = models.vgg16(weights=None)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, 6)
else:
    raise ValueError(f"Modelo '{model_name}' não suportado.")

model = torch.nn.DataParallel(model)
model.to(device)

# Carregar pesos com `weights_only=True` para evitar avisos de segurança
model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
model.eval()

# Transformações de pré-processamento
data_transforms = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

# Identificar classes a partir dos nomes das pastas no diretório de teste
class_names = ["Aedes_albopictus", "Aedes_fulvus", "Aedes_scapularis", "Aedes_serratus", "Anopheles_bellator", "Mansonia_humeralis"]
classes = {indice : name  for indice, name in enumerate(class_names)}

############## HELPER FUNCTIONS #################################

def get_coordinates_json(result_file):
    coordinates = []

    with open(result_file, 'r') as file:
        data = json.load(file)  # Load the JSON data

        for frame in data:
            frame_id = frame["frame_id"]
            for obj in frame["objects"]:
                relative_coords = obj["relative_coordinates"]

                # Extract the relative coordinates
                center_x = relative_coords["center_x"]
                center_y = relative_coords["center_y"]
                width = relative_coords["width"]
                height = relative_coords["height"]

                # Add the coordinates and frame ID to the list
                coordinates.append({
                    "frame_id": frame_id,
                    "center_x": center_x,
                    "center_y": center_y,
                    "width": width,
                    "height": height
                })

    return coordinates

def crop_and_resize_image(path, cx, cy, range_x, range_y, crop_path = None):
    # Load image
    image = cv2.imread(path)
    if image is None:
        print("Error: Image not found at the specified path.")
        return

    img_height, img_width = image.shape[:2]

    #Coordenadas Absolutas
    cx = int(cx * img_width)
    cy = int(cy * img_height)

    range_x = int(range_x * img_width)
    range_y = int(range_y * img_height)

    # Calculo do vértice superior esquerdo da caixa de crop
    x_start = int(cx - range_x / 2)
    y_start = int(cy - range_y / 2)

    # Assegurando que as dimensões de crop respeitam os limites da imagem
    x_start = max(0, x_start)
    y_start = max(0, y_start)
    x_end = min(img_width, x_start + range_x)
    y_end = min(img_height, y_start + range_y)

    # Crop the image
    cropped_image = image[y_start:y_end, x_start:x_end]

    # Resize cropped image to 512x512
    resized_image = cv2.resize(cropped_image, (224, 224), interpolation=cv2.INTER_CUBIC)

    #Salvamos a imagem com crop no diretório especificado se o caminho for passado para função
    if crop_path is not None:
      jpeg_filename = os.path.join(crop_path, f"{os.path.splitext(os.path.basename(path))[0]}.jpg")
      cv2.imwrite(jpeg_filename, resized_image)

    return resized_image

def detect_and_crop(path):

  original_dir = os.getcwd()
  os.chdir('/home/okura/Documents/tcc/RNCEM/Rede_detectora/darknet_mosquito') #SUBSTITUIR PELO SEU CAMINHO DA DARKNET

  try:
    result = subprocess.run(["./darknet", "detector", "test", "data/obj.data", "cfg/yolov3_custom2.cfg", "yolov3_custom2_last.weights", path, "-thresh", "0.2", "-ext_output", "-out", "result.txt"], shell = False, check = True)
  except subprocess.CalledProcessError as e:
    print(f"Error during batch processing: {e}")

  coordinates = get_coordinates_json("result.txt")
  coordinates = coordinates[0]

  os.chdir(original_dir)
  if coordinates["frame_id"] is None:
    return None
  else:
    cx, cy, width, height = (
      coordinates["center_x"],
      coordinates["center_y"],
      coordinates["width"],
      coordinates["height"]
    )
    #crop_path é facultativo, se for omitido as images não serão salvas em outro diretório
    imagem = crop_and_resize_image(path, cx, cy, width, height)

    return imagem

def imShow(path):

  image = cv2.imread(path)
  height, width = image.shape[:2]
  resized_image = cv2.resize(image,(3*width, 3*height), interpolation = cv2.INTER_CUBIC)

  fig = plt.gcf()
  fig.set_size_inches(5, 5)
  plt.axis("off")
  plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
  plt.show()
#################### MAIN FUNCTION ###############################
@app.route('/classify', methods=['POST'])

def classify():
  if 'file' not in request.files:
          return jsonify({'error': 'Nenhum arquivo enviado'}), 400

  file = request.files['file']
  if file.filename == '':
        return jsonify({'error': 'Nenhum arquivo selecionado'}), 400
  valid_extensions = ['.jpg', '.jpeg', '.png']
  file.filename = secure_filename(file.filename)
  image_path = os.path.abspath(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
  file_extension = os.path.splitext(file.filename)[1].lower()

  if file_extension not in valid_extensions:
        return jsonify({'error': 'Arquivo com extensão inválida. Envie uma imagem com extensão .jpg, .jpeg ou .png.'}), 400
  
  print(f"Image Path: {image_path}")
  file.save(image_path)
  try:
      if os.path.exists(image_path):
        print(f"Caminho do arquivo encontrado")

        
        if (any(image_path.lower().endswith(ext) for ext in valid_extensions)):
          print(f"Arquivo é uma imagem")
          imagem = detect_and_crop(image_path)

          if imagem is None:
            print(f"Algum erro no processo, confira a imagem antes de colocar o caminho")
            return jsonify({'error': 'Erro ao processar a imagem'}), 400
          else:
            cv2.imwrite("resultado.jpg", imagem)
            imagem = Image.open("resultado.jpg")
            input_tensor = data_transforms(imagem).unsqueeze(0).to(device)

            confidence = 6.0
            with torch.no_grad():
                output = model(input_tensor)
                output = torch.where(output > confidence, output, torch.tensor(0.0))

                if (torch.count_nonzero(output).item() != 0):
                  predicted_label = torch.argmax(output).item()
                  #print(f"logits após threshold = {output}")
                  result = classes[predicted_label]
                  print(f"Resultado da Classificação : {classes[predicted_label]} \n")
                  return jsonify({'message': 'Classificação concluída', 'result': result}), 200
                else:
                  print(f"Baixa confiança na predição, tente utilizar outra foto. Na eventualidade de uma segunda negativa, considerar que a espécie não está no banco de treinamento: \n DESCONHECIDA \n")
                      #imShow("resultado.jpg")
                  return jsonify({
                            'message': 'Baixa confiança na predição, tente utilizar outra foto. Na eventualidade de uma segunda negativa, considerar que a espécie não está no banco de treinamento.',
                            'result': 'DESCONHECIDA'
                        }), 200
        else:
          raise ValueError("Caminho não é de uma imagem \n")

      else:
        raise OSError("Caminho inválido \n")
      
  except OSError as erro:
      print(erro)
      return jsonify({'error': 'Arquivo não é uma imagem válida'}), 400
  except ValueError as erro:
      print(erro)
      return jsonify({'error': 'Caminho inválido'}), 400

if __name__ == "__main__":
  app.run(debug=True, host='0.0.0.0', port=5000)