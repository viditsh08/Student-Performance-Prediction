# Student Performance Prediction System

## Project Overview

The **Student Performance Prediction System** is a Machine Learning-based project designed to analyze students' academic performance and predict their results using different academic and study-related factors.

The project uses historical student data to understand the relationship between study habits, attendance, previous academic scores, and assignment completion. It applies supervised machine learning algorithms to predict students' expected marks and classify their academic outcomes.

The main objective is to demonstrate how data analysis and machine learning can be used to understand academic performance and support data-driven educational decisions.

## Objectives

- Analyze student academic data to identify important performance factors.
- Perform Exploratory Data Analysis (EDA) to understand patterns and relationships.
- Predict students' marks using a regression model.
- Classify students' results using a classification model.
- Evaluate model performance using standard machine learning metrics.
- Generate visualizations to understand the dataset and model results.

## Features

- **Student Data Management:** Supports student data through CSV files and sample data generation.
- **Exploratory Data Analysis:** Analyzes student information using statistical summaries and correlation analysis.
- **Marks Prediction:** Predicts expected student marks using Linear Regression.
- **Pass/Fail Classification:** Uses Logistic Regression to classify student outcomes.
- **Data Visualization:** Generates charts to understand distributions, relationships, and correlations.
- **Model Evaluation:** Evaluates predictions using appropriate regression and classification metrics.
- **Interactive Prediction:** Provides an interface to enter student details and obtain predictions.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical computations |
| Scikit-learn | Machine learning algorithms and evaluation |
| Matplotlib | Data visualization |
| Seaborn | Statistical data visualization |
| CSV | Dataset storage |

## Machine Learning Algorithms

### 1. Linear Regression

Linear Regression is used to predict students' expected marks based on the available academic features.

**Input Features:**
- Study Hours
- Attendance Percentage
- Previous Score
- Assignments Completed

**Output:** Predicted student marks.

### 2. Logistic Regression

Logistic Regression is used for binary classification to predict whether a student is likely to pass or fail.

**Input:** Student academic and study-related information.

**Output:** Pass or Fail classification.

## Dataset Description

The dataset contains student-related academic information used for training and evaluating the machine learning models.

| Feature | Description |
|---|---|
| StudyHours | Number of hours spent studying |
| AttendancePercentage | Student attendance percentage |
| PreviousScore | Previous academic score |
| AssignmentsCompleted | Number of assignments completed |

The project supports CSV-based data and sample data generation for experimentation.

## Exploratory Data Analysis (EDA)

Exploratory Data Analysis is performed to understand the dataset before applying machine learning algorithms.

The analysis includes:

- Statistical summaries of numerical features.
- Distribution analysis of student-related variables.
- Correlation analysis between academic features.
- Relationship between study hours and student performance.
- Visualization of important patterns in the dataset.

## Model Evaluation

The project evaluates both regression and classification models using standard performance metrics.

**Regression Metrics:**
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R-Squared (R²)

**Classification Metrics:**
- Accuracy
- Precision
- Recall
- Confusion Matrix

These metrics help assess how well the models perform on the available dataset.

## Project Structure

```text
Student-Performance-Prediction/
│
├── student_performance.py
├── student_data.csv
├── requirements.txt
├── README.md
├── LICENSE
│
├── analysis_dashboard.png
└── model_evaluation.png
```

*Note: The PNG files listed above are generated output files in the original project structure, not README images.*

## Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/viditsh08/Student-Performance-Prediction.git
```

### Step 2: Navigate to the Project Directory

```bash
cd Student-Performance-Prediction
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Project

```bash
python student_performance.py
```

Follow the instructions displayed in the terminal to explore the analysis and generate student performance predictions.

## Applications

- Understanding factors affecting student academic performance.
- Identifying relationships between study habits and academic results.
- Exploring machine learning applications in education.
- Demonstrating regression and classification techniques.
- Supporting academic data analysis and experimentation.

## Future Improvements

- Develop a web-based interface using Flask or FastAPI.
- Integrate a larger real-world student dataset.
- Experiment with additional machine learning algorithms.
- Improve model performance through hyperparameter tuning.
- Add more academic and behavioral features.
- Build an interactive dashboard for visualizing student performance.

## Learning Outcomes

Through this project, I explored:

- Data preprocessing and analysis using Pandas and NumPy.
- Exploratory Data Analysis and data visualization.
- Implementation of regression and classification algorithms.
- Model evaluation using standard performance metrics.
- Applying machine learning concepts to an educational use case.

## Author

**Vidit Sharma**

- GitHub: [@viditsh08](https://github.com/viditsh08)
- GitHub Profile: https://github.com/viditsh08

## Original Project Reference

This project is based on the original work by [Laksh Amarnani](https://github.com/Laksh-Amarnani/student-performance-prediction).

Please refer to the original repository and its MIT License for the applicable copyright and license terms.

---

**Disclaimer:** This project is intended for educational and learning purposes. Predictions are based on the available dataset and should not be treated as definitive measures of a student's academic ability.
