# RNCEM
Rede Neural para Classificação de Espécies de Mosquito de Importância Sanitária para o Brasil

Este projeto tem como objetivo auxiliar os profissionais da área de saúde pública, de forma a acelerar campanhas de identificação e tratamento de focos de epidemia, muitas vezes dependentes de uma correta classificação de espécimes em campo.

O projeto é dividido em 3 partes, cada qual com suas dependências
## Rede Detectora Baseada em YOLOv3 
A rede Detectora utiliza o framework *Darknet*, e deve ser acessada e clonada a partir do seguinte repositório: [Darknet_Mosquito](https://github.com/LucasEmerencio/darknet_mosquito).
Siga o passo a passo presente no repostório para instalar devidamente a rede detectora e suas dependências.
Como o arquivo de pesos customizado da rede excede o limite máximo de tamanho de arquivos do Github, ele deve ser baixado através do link a seguir do Dropbox: [yolov3_custom2_last](https://github.com/LucasEmerencio/darknet_mosquito).
Assim que estiver com o arquivo em mãos, mova o arquivo para a pasta "darknet_mosquito"

## Rede Classificadora baseada em VGG16
A rede Classificadora é baseada em VGG16, e utilizar o framework Pytorch, portanto é imporatante que o pytorch e suas dependências sejam instaladas. O arquivo requirement.txt contém as demais dependências da rede, e serão instaladas assim que o arquivo setup.py for inicializado. A instalação do pytorch é variável dependendo das configurações da máquina onde será instalada, e portanto recomendamos que a instalação seja feita utilizando os critérios que melhor se adequam a seu caso.
O arquivo de path, similarmente ao caso da rede Detectora, excede o tamanho máximo de arquivos permitidos, e portante deve ser baixado através do link a seguir do Dropbox: []()
Assim que estiver com o arquivo em mãos, mova o arquivo para a pasta "weights", dentro da pasta "Rede_classificadora"

## Interface
LOREM IPSUM DOLOR SIT AMET LOREM IPSUM DOLOR SIT AMET LOREM IPSUM DOLOR SIT AMET LOREM IPSUM DOLOR SIT AMET LOREM IPSUM DOLOR SIT AMET LOREM IPSUM DOLOR SIT AMET LOREM IPSUM DOLOR SIT AMET LOREM IPSUM DOLOR SIT AMET LOREM IPSUM DOLOR SIT AMET LOREM IPSUM DOLOR SIT AMET LOREM IPSUM DOLOR SIT AMET LOREM IPSUM DOLOR SIT AMET LOREM IPSUM DOLOR SIT AMET LOREM IPSUM DOLOR SIT AMET LOREM IPSUM DOLOR SIT AMET LOREM IPSUM DOLOR SIT AMET LOREM IPSUM DOLOR SIT AMET LOREM IPSUM DOLOR SIT AMET LOREM IPSUM DOLOR SIT AMET LOREM IPSUM DOLOR SIT AMET 


