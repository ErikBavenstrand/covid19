import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse

from tensorflow.keras.models import load_model
from Models import simple_cnn, vgg16
from Utils import grad_cam

# Display
#from IPython.display import Image
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def main():
    parser = argparse.ArgumentParser(description="Train a COVID-19 Classifier")
    parser.add_argument('--model_name',
                        type=str,
                        metavar='FILENAME',
                        help='filename of model weights')

    args = parser.parse_args()

    model = simple_cnn.model()

    filepath = './Saved Models'
    model = load_model(filepath + '/' + args.model_name)

    grad_cam.generate_gradcam('./test/', model, [
        'flatten', 'dense', 'activation_3', 'dropout', 'dense_1',
        'activation_4'
    ])


if __name__ == "__main__":
    main()
