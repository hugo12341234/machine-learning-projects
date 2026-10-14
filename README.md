# Machine Learning Projects

![Python](https://img.shields.io/badge/Python-3-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?logo=googlecolab&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)

End-to-end machine learning projects covering **regression**, **text classification** and **image classification**. Each notebook follows the workflow from *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (Aurélien Géron): frame the problem, explore the data, prepare it, train and compare models, and present the results.

> Developed as part of the machine learning course at **SIAFI**, Faculty of Engineering, UNAM.

---

## Projects

| # | Project | Task | Techniques | Result |
|---|---|---|---|---|
| 1 | [Car Price Prediction](01-car-price-prediction/car_price_prediction.ipynb) | Regression | EDA, feature engineering, log transform, Gradient Boosting | **R² = 0.75** |
| 2 | [Spam Classifier](02-spam-classifier/spam_classifier.ipynb) | Binary text classification | Custom scikit-learn transformer, bag of words, Logistic Regression | **98.9% precision, 93.5% recall** |
| 3 | [Fashion-MNIST Classifier](03-fashion-mnist-pytorch/fashion_mnist_classifier.ipynb) | Multiclass image classification | PyTorch MLP, BatchNorm, Dropout, Adam | **~85% test accuracy** |

### 1. Car Price Prediction

Predicts the selling price of used cars from the Car Dekho dataset.

- Explored how price depends on age, mileage, fuel, transmission, ownership history and brand.
- Engineered an `age` feature, applied a log transform to the skewed price and grouped rare brands.
- Compared Linear Regression (R² 0.59), SVR (0.64) and Gradient Boosting (**0.75**). The biggest gain came from fixing the data, not from tuning: luxury brands were being lumped into "Other".
- Validated the final model with feature importances and residual analysis.

### 2. Spam Classifier

Classifies emails as spam or ham using the Apache SpamAssassin public corpus.

- Parsed raw emails and built a custom `EmailToTextTransformer` that strips HTML and normalizes URLs and numbers.
- Wrapped preprocessing, vectorization and the model in a single scikit-learn `Pipeline`.
- Focused on **precision** over accuracy, since the data is imbalanced (~83% ham) and flagging a real email as spam is the costlier error. Only 1 legitimate email out of 519 was misclassified.

### 3. Fashion-MNIST Classifier with PyTorch

Classifies 28×28 grayscale clothing images into 10 categories.

- Built a custom PyTorch `Dataset` and `DataLoader` pipeline from raw CSV pixels.
- Trained a multilayer perceptron (784 → 256 → 128 → 10) with BatchNorm, ReLU, Dropout and the Adam optimizer.
- Analyzed the confusion matrix: the model handles shoes and trousers well and struggles with similar upper-body garments (shirt vs. T-shirt).
- Identified overfitting from the small training set and outlined next steps (full 60k dataset, CNN).

## Running the Notebooks

Every notebook has an **Open in Colab** badge at the top. Open it, upload the dataset to the Colab session and run all cells.

| Project | Dataset | File the notebook expects |
|---|---|---|
| Car Price Prediction | [Vehicle dataset from Car Dekho (Kaggle)](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho) | `CAR DETAILS FROM CAR DEKHO.csv` |
| Spam Classifier | [Apache SpamAssassin public corpus](https://spamassassin.apache.org/old/publiccorpus/) | `20021010_easy_ham.tar.bz2`, `20021010_spam.tar.bz2` |
| Fashion-MNIST | [Fashion-MNIST (Kaggle)](https://www.kaggle.com/datasets/zalando-research/fashionmnist) | `fashion-mnist_test.csv` |

Datasets are not included in this repository.

## Tech Stack

Python · pandas · NumPy · scikit-learn · PyTorch · Matplotlib · Seaborn · Google Colab

## Author

**Hugo Emmanuel Chacón Jaral** — Computer Engineering student, Faculty of Engineering, UNAM
[@hugo12341234](https://github.com/hugo12341234)

## License

This project is licensed under the [MIT License](LICENSE).
