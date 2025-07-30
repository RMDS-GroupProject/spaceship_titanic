# Spaceship Titanic - Passenger Transportation Prediction [![render nbviewer](https://img.shields.io/badge/render-nbviewer-orange.svg)](https://nbviewer.jupyter.org/github/RMDS-GroupProject/spaceship_titanic/blob/main/assignment3_grp3.ipynb)

This project implements a comprehensive machine learning solution for predicting passenger transportation outcomes in the Spaceship Titanic dataset. The implementation features advanced hyperparameter optimization using Optuna, multiple classifier training, and ensemble methods including Voting and Stacking classifiers for superior predictive performance.


## Dataset Overview

The Spaceship Titanic dataset contains passenger information with the following characteristics:
- **Training set**: 8,693 passenger records with transportation outcomes
- **Test set**: 4,277 passenger records for final predictions
- **Target variable**: Binary classification (Transported: True/False)
- **Features**: Mix of categorical and numerical variables including demographics, cabin details, and amenity spending

The dataset exhibits balanced classes with 50.4% transported vs. 49.6% not transported, ensuring unbiased model training.

## Model Architecture

- **Base Models**: Random Forest, CatBoost, XGBoost, LightGBM, Logistic Regression
- **Hyperparameter Optimization**: Optuna framework with intelligent pruning
- **Ensemble Methods**: Soft Voting Classifier and Stacking Classifier
- **Final Estimator**: Optimized Logistic Regression for meta-learning

## Project Structure

```
spaceship_titanic/
├── assignment3_grp3.ipynb       # Main training and evaluation notebook [Final Submission]
├── main_notebook.ipynb          # Testing training and evaluation notebook
├── README.md                    # This file
├── result.csv                   # Trained Model saved with metrics and parameters
├── Data/
│   ├── train.csv               # Training dataset
│   ├── test.csv                # Test dataset
│   └── sample_submission.csv   # Submission format
├── Images/                     # Directory for saving images 
├── Kagle_Submissions/          # Directory for competition submissions
```

## Key Features

The implementation includes:
- **Comprehensive EDA**: Statistical analysis, correlation studies, and feature distribution analysis
- **Advanced Preprocessing**: Missing value imputation, feature engineering, and categorical encoding
- **Hyperparameter Optimization**: Optuna-based tuning for all base classifiers
- **Multi-Metric Evaluation**: Accuracy, Precision, Recall, F1-Score
- **Ensemble Learning**: Voting and Stacking classifiers with optimized meta-learners
- **Cross-Validation**: Robust k-fold validation for unbiased performance estimation

## Model Configuration

### Base Models Training:
- **Optimization Trials**: 20 trials per model using Optuna
- **Cross-Validation**: 3-fold stratified CV for robust evaluation
- **Scoring Metrics**: Multiple metrics with accuracy as primary objective
- **Random State**: 42 for reproducible results

### Ensemble Configuration:
- **Voting Classifier**: Soft voting with top 3 performing models
- **Stacking Classifier**: 3-fold CV with passthrough=True
- **Final Estimator**: Logistic Regression with L2 regularization

## Key Findings from EDA

- **CryoSleep is the strongest predictor** - passengers in cryosleep show dramatically higher transportation rates
- **Spending behavior clearly separates outcomes** - transported passengers consistently spend near-zero across all amenities
- **Demographic segmentation exists** - younger passengers and TRAPPIST-1e travelers show higher transportation probabilities
- **Data characteristics require specialized preprocessing** - numerical features exhibit extreme right-skewness and zero-inflation

## Dependencies

```python
optuna==4.4.0
scikit-learn==1.7.0
pandas==2.3.1
numpy==2.2.6
matplotlib==3.10.3
seaborn==0.13.2
catboost==1.2.8
xgboost==3.0.2
lightgbm==4.6.0
jupyter==1.0.0
eli5==0.16.0
PDPbox==0.3.0
alepython @ git+https://github.com/MaximeJumelle/ALEPython.git@dev#egg=alepython
shap==0.48.0
```

## Usage

1. **Clone the repository**
   ```bash
   git clone https://github.com/RMDS-GroupProject/spaceship_titanic.git
   cd spaceship_titanic
   ```

2. **Set up the environment**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the main notebook**
   ```bash
   jupyter notebook assignment3_grp3.ipynb
   ```

4. **Execute the training pipeline**
   - Load and preprocess the data
   - Run hyperparameter optimization for all models
   - Train ensemble classifiers
   - Generate predictions and submission files

## Model Performance

The model is evaluated using comprehensive metrics:
- **Cross-Validation Accuracy**: Mean accuracy across k-fold validation
- **Precision, Recall, F1-Score**: Weighted averages for multi-class evaluation
- **ROC-AUC**: Area under the receiver operating characteristic curve
- **Ensemble Performance**: Comparison between Voting and Stacking approaches

Results visualization and model comparison are provided in the main notebook with detailed performance analysis.

## Implementation Highlights

### Custom Model Wrapper Class:
- **SpaceShipTitanicModel**: Comprehensive wrapper for hyperparameter tuning, training, and ensemble creation
- **Automated Pipeline**: End-to-end training pipeline with minimal manual intervention
- **Flexible Configuration**: Easy model addition and parameter modification

### Advanced Optimization:
- **Optuna Integration**: Efficient hyperparameter search with early pruning
- **Multiple Objectives**: Support for various scoring metrics and optimization strategies
- **Reproducible Results**: Consistent random states and logging for experiment tracking

## Competition Results

This implementation is designed for the Kaggle Spaceship Titanic competition with focus on:
- **High Accuracy**: Optimized ensemble methods for maximum predictive performance
- **Robust Validation**: Cross-validation strategies to prevent overfitting
- **Submission Ready**: Automated generation of competition-format submission files

## Contributing

This project is part of the Research Methods in Data Science (RMDS) group project. Contributors:
- Team members working collaboratively on model development and optimization
- Focus on reproducible research methodologies and best practices

## Acknowledgments

- **Kaggle**: For providing the Spaceship Titanic dataset and competition platform
- **Optuna Team**: For the excellent hyperparameter optimization framework
- **Scikit-learn Community**: For comprehensive machine learning tools and documentation
- **RMDS Course**: Research Methods in Data Science course instructors and guidelines
