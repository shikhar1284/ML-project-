# Naive Bayes Visualization App 📊
https://vizmachinelearning.streamlit.app/

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/streamlit-badge.svg?logo=streamlit)](https://streamlit.io/)
[![License](https://img.shields.io/github/license/your-username/your-repository?logo=mit)](https://github.com/your-username/your-repository/blob/main/LICENSE)


## About The Project

This project is an interactive, educational dashboard designed to help beginners understand how the Naive Bayes algorithm works. It visually guides you through the process of feeding data into the algorithm and exploring its predictions, making complex machine learning concepts more approachable and intuitive.  Essentially, we’ve built a way to *see* Naive Bayes in action!

## ✨ Key Features

*   **Interactive Pipeline Stepper:** Step through each preprocessing stage of the Naive Bayes algorithm.
*   **Dynamic Hyperparameter Tuning:** Easily adjust key settings (like variance smoothing and alpha) and observe the impact on model performance in real-time.
*   **Real-Time Graph Updates:** See how the Gaussian distribution changes as you tune hyperparameters, and analyze the model's accuracy and confusion matrix.

## 🛠️ Tech Stack

*   Python
*   Scikit-Learn
*   Pandas
*   Plotly
*   Streamlit

## 🚀 Getting Started (Installation)

1.  **Clone the repository:**
```bash
    git clone https://github.com/your-username/your-repository.git
    cd your-repository
```

2.  **Create a virtual environment:**
```bash
    python -m venv venv
```

3.  **Activate the virtual environment:**
    *   On Windows: `venv\Scripts\activate`
    *   On macOS/Linux: `source venv/bin/activate`

4.  **Install dependencies:**
```bash
    pip install -r requirements.txt
```

5.  **Run the app:**
```bash
    streamlit run app.py
```

## 📖 How to Use the App

1.  **Load Data:** Upload your own CSV dataset or select one of the built-in datasets.
2.  **Tune Hyperparameters:** Use the sliders in the left panel to adjust the variance smoothing and alpha parameters.
3.  **View Insights:** Observe the changes in the accuracy, Gaussian bell curve, and confusion matrix on the right panel as you adjust the hyperparameters.

## 🔍 Component Details

### 1. Data Preprocessing Pipeline

*   **Functionality:** This section represents the core of the application. It takes raw data and transforms it into a format suitable for the Naive Bayes model.
*   **Stages:**  The pipeline includes the following stages:
    *   **Feature Selection:** Automatically selects relevant features from the input data.
    *   **Categorical Encoding:** Converts categorical features into numerical representations using One-Hot Encoding.
    *   **Numerical Scaling:** Scales numerical features to a consistent range (e.g., using StandardScaler) to prevent features with larger values from dominating the model.
*   **Under the Hood:**  Uses `scikit-learn`'s `OneHotEncoder` and `StandardScaler` classes.

### 2. Visualization Components

*   **Gaussian Distribution Plot:**  A visually compelling representation of the data’s distribution, demonstrating the impact of `var_smoothing` on the curve's shape.  A wider curve indicates less smoothing and a more complex distribution.
*   **Accuracy Metrics:**  A numerical display of the model's accuracy, providing immediate feedback on the impact of hyperparameter tuning.
*   **Confusion Matrix:** A table visualizing the model's classification performance by showing the number of true positives, true negatives, false positives, and false negatives.

### 3. Interactive Elements

*   **Hyperparameter Sliders:** Allows the user to directly adjust the values of `var_smoothing` and `alpha`.
*   **Real-Time Updates:** Changes to the hyperparameters are reflected immediately in the visualizations and accuracy metrics.
*   **Step Details:**  Expanding a specific step in the pipeline displays more granular information about its function.


## 🤝 Contributing & License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Contributions are welcome! Please see the [CONTRIBUTING.md](CONTRIBUTING.md) file for more information.
