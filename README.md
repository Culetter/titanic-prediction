# Titanic Survival Prediction
This project is a machine learning pipeline for predicting passenger survival on the Titanic, using the Titanic dataset. Built with Python and scikit-learn, the model employs logistic regression to classify survival based on features such as age, fare, class, and embarkation point.

# Project Structure
```
├── /data
│   ├── test.csv                # Dataset for prediction
│   └── train.csv               # Training dataset
├── /notebooks
│   ├── data-analysis.ipynb     # Exploratory data analysis and dataset overview
│   └── model-training.ipynb    # Model training experiments and results visualization
├── /results
│   ├── graphs/                 # Visualizations generated from analysis and training
│   ├── model/                  # Saved trained models
│   └── prediction/             # Output predictions on test.csv dataset
├── /src
│   ├── requirements.txt/       # Required Libraries
│   └── train_model.py          # Main training, evaluation, and prediction script
```

# Installation
1. Clone the repository:
```
git clone https://github.com/Culetter/titanic-prediction.git
cd titanic-prediction/src
```
2. Install the dependencies
```
pip install -r requirements.txt
```

# Usage
To train the model and run predictions:
```
python train_model.py
```

# Model
The train_model.py script:
* Loads and cleans the Titanic dataset
* One-hot encodes categorical features
* Scales numeric features using MinMaxScaler
* Trains a logistic regression model using scikit-learn
* Evaluates the model using:
  * A confusion matrix
  * ROC curve and AUC score
* Saves the trained pipeline using joblib
* Predicts data from the test.csv dataset

# Author
**Nazarii Lozynskyi**  
[@Culetter](https://github.com/Culetter)

# License
The dataset used in this project is the "Titanic - Machine Learning from Disaster" dataset, available on Kaggle:
https://www.kaggle.com/competitions/titanic

The dataset does not have a specific license listed, so it is used here only for educational and non-commercial purposes.
All rights to the original dataset remain with the original author.
