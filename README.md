# Loan Eligibility Prediction with LIME Explanations

This project demonstrates an end-to-end machine learning workflow for predicting loan eligibility. It utilizes a Gradient Boosting Classifier and incorporates LIME (Local Interpretable Model-agnostic Explanations) to understand the factors driving individual loan approval or rejection decisions. This approach is crucial for transparency and fairness in sensitive applications.

The primary dataset used is the "Loan Eligible Dataset" from Kaggle: [vikasukani/loan-eligible-dataset](https://www.kaggle.com/datasets/vikasukani/loan-eligible-dataset).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) <!-- Optional: Add a license badge -->

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Workflow](#workflow)
4. [Technologies Used](#technologies-used)
5. [Setup and Installation](#setup-and-installation)
6. [Usage](#usage)
7. [Understanding LIME Explanations](#understanding-lime-explanations)
8. [Results](#results)
9. [Future Work](#future-work)
10. [Contributing](#contributing)
11. [License](#license)
12. [Acknowledgements](#acknowledgements)

## Project Overview

Loan approval is a critical financial decision that significantly impacts individuals. While machine learning models can automate and improve the efficiency of this process, their "black-box" nature can be problematic, especially when decisions need to be justified or audited for fairness. This project tackles this by:

*   Training a robust classification model (Gradient Boosting) to predict loan eligibility.
*   Integrating LIME to provide human-understandable explanations for each prediction, highlighting the key features that influenced the model's decision.

## Features

*   **Data Loading:** Fetches data directly from Kaggle Hub.
*   **Preprocessing:** Comprehensive preprocessing pipeline using Scikit-learn's `ColumnTransformer` for handling missing values, scaling numerical features, and encoding categorical features.
*   **Model Training:** Utilizes a `GradientBoostingClassifier`.
*   **Model Evaluation:** Standard metrics like accuracy, precision, recall, and F1-score.
*   **Explainable AI (XAI):** Implements LIME to explain individual predictions locally.
*   **Jupyter Notebook:** Well-documented notebook (`Loan_Eligibility_XAI.ipynb`) for easy understanding and reproduction.

## Workflow

The project follows these main steps:

1.  **Environment Setup:** Installation of necessary Python libraries.
2.  **Kaggle API Configuration:** Setting up credentials to access Kaggle datasets.
3.  **Data Loading & Initial Cleaning:** Fetching the dataset and performing basic cleaning (e.g., mapping target variable, handling specific string values).
4.  **Feature Engineering & Preprocessing:**
    *   Identifying numerical and categorical features.
    *   Applying imputation (median for numerical, mode for categorical).
    *   Scaling numerical features (StandardScaler).
    *   Encoding categorical features (OrdinalEncoder for binary-like, OneHotEncoder for multi-category).
5.  **Train-Test Split:** Dividing the data into training and testing sets, stratified by the target variable.
6.  **Model Training:** Training the Gradient Boosting Classifier on the preprocessed training data.
7.  **Model Evaluation:** Assessing model performance on the test set.
8.  **LIME Explainer Setup:**
    *   Configuring the `LimeTabularExplainer`.
    *   Creating a custom `predict_fn` that incorporates the preprocessing pipeline for LIME's perturbations.
9.  **Generating Explanations:** Explaining predictions for sample instances from the test set.

## Technologies Used

*   **Python 3.x**
*   **Core Libraries:**
    *   [Pandas](https://pandas.pydata.org/): Data manipulation and analysis.
    *   [NumPy](https://numpy.org/): Numerical computations.
    *   [Scikit-learn](https://scikit-learn.org/): Machine learning (preprocessing, modeling, evaluation).
    *   [LIME](https://github.com/marcotcr/lime): Model explainability.
    *   [KaggleHub](https://github.com/Kaggle/kagglehub) & [Kaggle API](https://github.com/Kaggle/kaggle-api): Dataset access.
*   **Environment:**
    *   [Jupyter Notebook / Google Colab](https://colab.research.google.com/): Interactive development.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/garvit-exe/loan-eligibility-xai.git
    cd loan-eligibility-xai
    ```

2.  **Set up a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    The primary dependencies are listed in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```
    Alternatively, if running in Google Colab, the notebook includes a cell to install packages:
    ```python
    !pip install kagglehub kaggle lime scikit-learn pandas numpy scipy --quiet
    ```

4.  **Kaggle API Credentials:**
    *   To download the dataset via the Kaggle API, you need your `kaggle.json` file.
    *   Download it from your Kaggle account page (Account -> API -> Create New API Token).
    *   **For local execution:** Place `kaggle.json` in `~/.kaggle/` (Linux/macOS) or `C:\Users\<Your-User-Name>\.kaggle\` (Windows).
    *   **For Google Colab:** The notebook provides cells to upload `kaggle.json` or use Colab Secrets.

## Usage

1.  Ensure all dependencies are installed and Kaggle API is configured.
2.  Open and run the Jupyter Notebook: `Loan_Eligibility_XAI.ipynb`.
    *   If using Google Colab, upload the notebook or open it from GitHub.
    *   Follow the instructions within the notebook, especially for Kaggle API authentication if using Colab Secrets.
3.  The notebook will guide you through each step, from data loading to generating LIME explanations for sample loan applications.

## Understanding LIME Explanations

LIME explains a prediction by approximating the model's behavior locally around that specific instance with a simpler, interpretable model (often a weighted linear model).

For a given prediction (e.g., "Loan Approved"), LIME will output:
*   **Feature Conditions:** Conditions on feature values (e.g., `Credit_History=1.0`, `ApplicantIncome <= 3500`).
*   **Weights:** A weight associated with each feature condition.
    *   **Positive Weight:** Indicates the feature condition pushed the prediction *towards* the explained class (e.g., towards "Approved").
    *   **Negative Weight:** Indicates the feature condition pushed the prediction *away* from the explained class (e.g., away from "Approved", thereby supporting "Rejected").
*   The magnitude of the weight indicates the strength of the feature's influence for that particular prediction.

**Example Interpretation for an "Approved" Prediction:**
*   "The condition 'Credit_History=1.0' (Applicant's value for Credit_History: 1.0) strongly supported the 'Approved' decision (LIME weight: 0.150)."
    *   This means having a credit history of 1.0 was a strong positive factor for this applicant's approval.

**Example Interpretation for a "Rejected" Prediction (when explaining relative to "Approved" class):**
*   "The condition 'Loan_Amount_Term < 180' (Applicant's value for Loan_Amount_Term: 36.0) moderately opposed the 'Approved' decision (LIME weight: -0.080)."
    *   This means a short loan term *negatively impacted* the chances of approval (i.e., it was a factor contributing to the rejection).
*   "The condition 'Credit_History=1.0' (Applicant's value for Credit_History: 1.0) slightly supported the 'Approved' decision (LIME weight: 0.030)."
    *   This means good credit history was a positive factor, but it wasn't enough to overcome other negative factors leading to the rejection.

## Results

The Gradient Boosting model typically achieves an accuracy of around **80-83%** on the test set for this dataset. The classification report in the notebook provides detailed precision, recall, and F1-scores for both "Approved" and "Rejected" classes.

The LIME explanations provide valuable insights into *why* the model makes specific decisions, going beyond simple accuracy metrics.

## Future Work

*   **Hyperparameter Tuning:** Optimize the Gradient Boosting Classifier using techniques like GridSearchCV or RandomizedSearchCV.
*   **Advanced Feature Engineering:** Create new features that might improve model performance.
*   **Exploration of Other XAI Techniques:** Implement and compare explanations from other methods like SHAP (SHapley Additive exPlanations).
*   **Bias Detection and Mitigation:** Analyze explanations across different demographic groups to check for potential biases and explore mitigation strategies.
*   **Interactive Dashboard:** Develop a simple web application or dashboard (e.g., using Streamlit or Dash) to allow users to input applicant data and see both the prediction and its LIME explanation.
*   **Error Analysis:** Investigate misclassified instances more deeply using LIME explanations to understand model weaknesses.

## Contributing

Contributions are welcome! If you have suggestions for improvements, new features, or find any bugs, please feel free to:
1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/YourFeature` or `bugfix/YourBug`).
3.  Make your changes.
4.  Commit your changes (`git commit -m 'Add some feature'`).
5.  Push to the branch (`git push origin feature/YourFeature`).
6.  Open a Pull Request.

Please ensure your code adheres to good coding practices and includes relevant documentation.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

*   The **vikasukani/loan-eligible-dataset** on Kaggle for providing the data.
*   The developers of **LIME** for their invaluable tool for model interpretability.
*   The **Scikit-learn** team for their comprehensive machine learning library.
