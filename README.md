# Gender Classification with Deep Learning

## Dataset
The dataset used for this project is a curated collection of facial images representing both men and women. The dataset source and preprocessing steps are detailed below.

## Models
The project utilizes pre-trained models, namely Inception-V3 and Xception. These models have been chosen for their strengths in feature extraction and have undergone fine-tuning for gender classification.

## Training

### Data Augmentation and Pre-processing
The dataset underwent data augmentation techniques during training, including random transformations such as rotation, shift, shear, and zoom. Images were resized to a uniform size of 218x178 pixels to balance computational efficiency and detail retention.

### Model Parameters and Fine-tuning
The models were initialized with pre-trained weights on the ImageNet dataset, providing a valuable starting point. Fine-tuning involved freezing the initial layers to preserve learned features and customizing subsequent layers for the gender classification task.

- **Pre-trained Weights:** Initial layers of both models were initialized with pre-trained weights.
- **Fine-tuning:** The first 52 layers were kept frozen to preserve lower-level feature extraction.
- **Regularization:** Dropout layers were strategically placed to prevent overfitting.
- **Learning Rate:** Set to 0.0001 to ensure gradual convergence towards optimal weights.
- **Callbacks:** Model training incorporated callback mechanisms, such as ModelCheckpoint.

## Evaluation

The performance of both models was rigorously evaluated using a comprehensive set of metrics, including accuracy, precision, recall, and F1 score. These metrics were complemented by qualitative assessments, scrutinizing the models' predictions across diverse subsets of the dataset.

### Results

The results of the evaluation are summarized in the table below:

#### Inception-V3

- Validation Score = 0.94
| Metric        | Value   |
| ------------- | ------- |
| Accuracy      | 0.94    |
| Precision     | 0.92    |
| Recall        | 0.95    |
| F1-Score      | 0.93    |

#### Xception

- Validation Score = 0.9902
| Metric        | Value   |
| ------------- | ------- |
| Accuracy      | 0.911   |
| Precision     | 0.88    |
| Recall        | 0.98    |
| F1-Score      | 0.93    |

### Comparison and Analysis

The evaluation results were meticulously analyzed to discern patterns in model performance. Comparative analyses highlighted the strengths and weaknesses of Inception-V3 and Xception in the context of gender classification.

### Real-time Application

To showcase the practical utility of the trained models, a real-time gender identification application was developed. The application utilized OpenCV for capturing and processing video feed, leveraging the insights gained from model evaluations.

## Usage

- Clone the repository: `git clone https://github.com/sami-rajichi/Gender-Detection-Classification.git`
- Run the notebook `gender-detection-classification.ipynb` for training and evaluation or the python script `Gender_Detection_Experimentation.py`.

## Requirements
- Python 3.x
- TensorFlow
- Keras
- OpenCV
- ...

## Contributing

Feel free to contribute to this project by following the guidelines outlined in the Contributing section.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
