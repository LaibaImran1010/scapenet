# ScapeNet: Natural Scene Classification with Deep Learning

Deep Learning semester project. A convolutional neural network, **trained from scratch**, classifies a photo as one of six scenes: buildings, forest, glacier, mountain, sea or street.

## Live links

| What | Link |
|---|---|
| Website (Vercel) | https://scapenet-mu.vercel.app |
| Model server (Hugging Face Space) | https://huggingface.co/spaces/laibaimran/scene-classifier |
| Project report | [docs/ScapeNet_Report.pdf](docs/ScapeNet_Report.pdf) |
| Presentation slides | [docs/ScapeNet_Slides.pdf](docs/ScapeNet_Slides.pdf) |
| Presentation and voiceover (YouTube) | https://youtu.be/3KaYEHFO9mk |
| Dataset (Kaggle) | https://www.kaggle.com/datasets/puneet6060/intel-image-classification |

## Dataset

- Intel Image Classification from Kaggle (secondary dataset).
- 6 classes, 150 x 150 colour images.
- About 14,000 training photos (85% training, 15% validation) and 3,000 test photos never used for training.
- The dataset is not stored in this repository because of its size. The notebook downloads it with the Kaggle API.

## Models

| | Test model | Final model |
|---|---|---|
| Type | MobileNetV2 with ImageNet weights | CNN trained from scratch |
| Pretrained weights | Yes | No |
| Test accuracy | 90.4% | **84.1%** |
| Test loss | 0.2533 | 0.4644 |
| Weights file | `weights/model_pretrained.keras` | `weights/model_scratch.keras` |

**Final model architecture:** four blocks with 32, 64, 128 and 256 filters (each block: two 3x3 convolutions with batch normalisation and ReLU, then max pooling), global average pooling, dropout 0.4, dense 128, dropout 0.3, softmax with 6 outputs. Training used flip, rotation and zoom augmentation, Adam, early stopping and learning rate reduction, in two runs (learning rate 1e-3, then 3e-4).

## Repository structure

```
.
├── README.md
├── requirements.txt
├── index.html                      website (deployed on Vercel)
├── scratch_confusion_matrix.png
├── scratch_accuracy_curve.png
├── pretrained_confusion_matrix.png
├── pretrained_accuracy_curve.png
├── docs/
│   ├── ScapeNet_Report.pdf         final project report
│   └── ScapeNet_Slides.pdf         presentation slides
├── notebooks/train.ipynb           data download, training, evaluation
├── weights/
│   ├── model_scratch.keras         final model
│   ├── model_pretrained.keras      comparison model
│   ├── scratch.onnx                final model in browser format
│   └── pretrained.onnx             comparison model in browser format
└── space/
    └── app.py                      code running on the Hugging Face Space
```

## Run it yourself

```bash
pip install -r requirements.txt
python
>>> import keras, numpy as np
>>> model = keras.models.load_model("weights/model_scratch.keras")
>>> # x: float array of shape (1, 150, 150, 3), pixel values 0-255
>>> model.predict(x)
```

To retrain, open `notebooks/train.ipynb` in Google Colab with a GPU, upload your `kaggle.json` and run all cells.

## How the website works

1. The user uploads a photo on the website.
2. The page loads the ONNX model file (`weights/scratch.onnx` or `weights/pretrained.onnx`) into the browser with ONNX Runtime Web.
3. The photo is resized to 150 x 150 and the model runs on the visitor's device, so no server or quota is needed. The page shows six probabilities.
4. The page shows the top class and a confidence bar for every class.

## Browser version of the models

The Keras models were exported to ONNX (`scratch.onnx`, `pretrained.onnx`) after removing the training-only augmentation layers, and the outputs were checked against the Keras models. The Hugging Face Space in `space/app.py` is an optional server-hosted version of the same demo.

## Limitations

- The model only knows these six scene types, so other photos still receive one of the six labels.
- Glacier and mountain photos are the most often confused.
- The first prediction loads the model file and can take a few seconds.