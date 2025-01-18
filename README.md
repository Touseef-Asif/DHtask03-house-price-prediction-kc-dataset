# House Price Prediction in King County, USA

## **Author**
Touseef Asif  

---

## **Overview**
This project leverages regression models to predict house prices in King County, USA, using the `kc_house_data.csv` dataset. The primary focus is on building, training, and evaluating multiple regression models to achieve optimal performance, measured by metrics such as RMSE (Root Mean Squared Error), MAE (Mean Absolute Error), and R² (Coefficient of Determination).

---

## **Project Objective**
- To preprocess, analyze, and visualize the dataset effectively.
- To train and compare the performance of multiple regression models, including Linear Regression, Decision Tree Regressor, Random Forest Regressor, and Gradient Boosting Regressor.
- To identify the best-performing model for house price prediction.

---

## **Key Features**
1. **Data Preprocessing**
   - Handling missing values and irrelevant features.
   - Normalizing numerical features for better model performance.

2. **Exploratory Data Analysis (EDA)**
   - Visualizations to understand data distribution, feature importance, and correlations.
   - Heatmaps and scatter plots to identify relationships between features and target variables.

3. **Model Training and Evaluation**
   - Training multiple regression models and fine-tuning hyperparameters.
   - Evaluating model performance using RMSE, MAE, and R².
   - Comparing model results to determine the most suitable model.

4. **Results Visualization**
   - Scatter plots for actual vs. predicted prices.
   - Bar charts for model performance comparison.

5. **Deployment Ready**
   - Includes saving the best-performing model for future use.

---

## **Dataset**
- **Dataset Name:** kc_house_data.csv  
- **Dataset Size:** ~2.5 MB  
- **Source:** King County, USA  
- **Description:** The dataset contains information about house sales, including features such as:
  - `bedrooms`, `bathrooms`, `sqft_living`, `sqft_lot`, `floors`, `waterfront`, `view`, etc.
  - Target variable: `price` (house price).  

---

## **Tools and Technologies**
This project is built using the following tools and libraries:
- **Programming Language:** Python 3.x
- **Libraries:**
  - Data Manipulation: `pandas`, `numpy`
  - Visualization: `matplotlib`, `seaborn`
  - Machine Learning: `scikit-learn`, `joblib`

---

## **Performance Metrics**
The following metrics are used to evaluate the models:
1. **Mean Absolute Error (MAE)**: Measures the average magnitude of errors in predictions.
2. **Mean Squared Error (MSE)**: Penalizes larger errors by squaring them.
3. **Root Mean Squared Error (RMSE)**: Standard deviation of prediction errors.
4. **R² (Coefficient of Determination)**: Explains the proportion of variance in the target variable.

---

## **Model Results**
### Model Performance Summary:
| **Model**                  | **MAE**     | **MSE**            | **RMSE**       | **R²**  |
|----------------------------|-------------|--------------------|----------------|---------|
| Linear Regression          | 129,340.73  | 46,765,590,784.45 | 216,253.53     | 0.69    |
| Decision Tree Regressor    | 120,695.36  | 55,474,178,520.70 | 235,529.57     | 0.63    |
| Random Forest Regressor    | 86,876.37   | 31,522,142,037.59 | 177,544.76     | 0.79    |
| Gradient Boosting Regressor| 93,338.46   | 32,850,054,370.50 | 181,245.84     | 0.78    |

**Best Model:** Random Forest Regressor  
- RMSE = **177,544.76**  
- R² = **0.79**  

---

## **Project Workflow**
### 1. **Data Preprocessing**
- Loading the dataset.
- Removing irrelevant features (e.g., `id`, `date`, `zipcode`).
- Handling missing data and scaling numerical features.

### 2. **Exploratory Data Analysis**
- Visualizing distributions of key features using histograms and boxplots.
- Correlation heatmaps to analyze feature relationships.
- Feature importance analysis to select significant predictors.

### 3. **Model Training**
- Splitting the dataset into training and testing sets.
- Training the following regression models:
  - Linear Regression
  - Decision Tree Regressor
  - Random Forest Regressor
  - Gradient Boosting Regressor

### 4. **Model Evaluation**
- Calculating performance metrics for each model.
- Comparing the results and selecting the best model.

### 5. **Prediction and Visualization**
- Predicting house prices using the test set.
- Visualizing actual vs. predicted house prices using scatter plots.


---

## **How to Run**
### **Prerequisites**
Ensure you have the following installed:
- Python (3.x)
- Jupyter Notebook
- Libraries (install using `pip install -r requirements.txt`)

### **Steps**
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/house-price-prediction.git
   cd house-price-prediction
