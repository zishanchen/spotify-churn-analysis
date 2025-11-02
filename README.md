# Python & Predictive Modeling: Spotify Customer Churn Analysis

## 1. Project Overview
This project is a comprehensive demonstration of data analysis and predictive modeling in Python, applied to a real-world customer churn dataset. The primary goal is to analyze user behavior to identify key factors contributing to the **25.89%** overall churn rate. The project showcases an end-to-end data science workflow—from data cleaning and exploratory analysis to building and evaluating a baseline predictive model—all documented within the included Jupyter Notebook.

## 2. Dataset
*   **Source:** [Spotify Dataset for Churn Analysis on Kaggle](https://www.kaggle.com/datasets/nabihazahid/spotify-dataset-for-churn-analysis)
*   **Description:** The dataset contains user-level data from a music streaming service, including demographics, subscription details, listening habits, and a binary indicator for churn. The raw CSV file (`spotify_churn_dataset.csv`) is included in the `/data` folder for full reproducibility.

## 3. Methodology & Technical Implementation

### a. Data Preprocessing & Feature Engineering
The analysis was conducted in a Jupyter Notebook. The initial steps focused on preparing the data for analysis and modeling using the `pandas` and `numpy` libraries:
1.  **Cleaning:** Column names were standardized for clarity (e.g., `is_churned` to `churn`).
2.  **Transformation:** The categorical `subscription_type` column was transformed into a more model-friendly binary feature called `premium_user` (1 for paid, 0 for free).
3.  **Validation:** The dataset was inspected for missing values and inconsistencies to ensure data quality.

### b. Exploratory Data Analysis (EDA)
A deep dive into the data was performed using `matplotlib` and `seaborn` to uncover patterns and relationships. The analysis focused on visualizing the key drivers of churn to inform both business strategy and the subsequent modeling phase. All visualizations are rendered directly within the notebook.

### c. Predictive Modeling
A baseline predictive model was built using `scikit-learn` to establish an initial performance benchmark:
1.  **Preprocessing:** Categorical features like `gender`, `country`, and `device_type` were one-hot encoded.
2.  **Data Splitting:** The dataset was split into training (70%) and testing (30%) sets using a stratified split to handle the class imbalance.
3.  **Model Training:** A **Logistic Regression** model was chosen for its interpretability and efficiency as a baseline.
4.  **Evaluation:** The model's performance was assessed using standard classification metrics, including Accuracy, Precision, Recall, and F1-Score.

### d. Tools Used
*   **Language:** Python 3.x
*   **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
*   **Environment:** Jupyter Notebook

## 4. Business Questions & Analytical Insights

This analysis answered four key business questions, with the following data-driven insights:

### Q1: How does subscription type (Premium vs. Free) impact customer retention?
*   **Question:** Are users with free subscriptions more likely to churn than premium users?
*   **Insight:** Surprisingly, the data shows that having a premium subscription does not significantly reduce churn. The churn rate for **free users is 24.9%**, while the churn rate for **premium/family/student users is slightly higher at 26.2%**.
*   **Business Action:** This is a critical insight. Instead of assuming premium is a "safe" category, the business should investigate *why* paying customers are still leaving. Are the premium features not compelling enough? Is the price point too high? This finding shifts the focus from simple conversion to improving the value of the paid product itself.

### Q2: What is the relationship between user engagement and churn?
*   **Question:** Do less active users churn more often?
*   **Insight:** Yes, there is a clear correlation between lower listening hours and a higher likelihood of churn. The boxplot visualization shows that the distribution of `listening_hours` for churned users is skewed lower than for retained users.
*   **Business Action:** Engagement is a key health metric. The business should implement a proactive re-engagement strategy, targeting users whose listening activity drops below a certain threshold with personalized playlists, new release notifications, or other content.

### Q3: Does geography play a role in customer churn?
*   **Question:** Do churn rates differ significantly by country?
*   **Insight:** The analysis shows that churn rates vary by country. For example, **Pakistan (PK)** has one of the highest churn rates in the dataset at **27.5%**.
*   **Business Action:** This suggests that a one-size-fits-all retention strategy may not be effective. The company should consider regional analysis to understand market-specific issues, such as competition, pricing sensitivity, or content library relevance.

### Q4: How well can we predict churn with a baseline model?
*   **Question:** Can a simple model effectively identify users who are about to churn?
*   **Insight:** The initial Logistic Regression model achieved an accuracy of **74.12%**. However, this number is highly misleading. The detailed classification report shows that the model has a **Precision and Recall of 0.0 for the churned class (1)**. This means it failed to correctly identify a single user who was going to churn, instead just predicting the majority class (non-churn).
*   **Business Action:** This result proves that a simple model is insufficient for this task due to the class imbalance. It validates the need for more advanced modeling techniques (like SMOTE for oversampling or using models like Random Forest) to build a truly useful predictive tool.

## 5. Repository Structure
The repository is organized for clarity and reproducibility:
```/
├── data/
│   └── spotify_churn_dataset.csv
│
├── spotify_churn_analysis.ipynb
└── README.md
```

## 6. How to Use
1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/your-username/spotify-churn-analysis.git
    cd spotify-churn-analysis
    ```
2.  **Install Dependencies:**
    Ensure you have Python installed. Then, install the required libraries:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn jupyter
    ```
3.  **Run the Analysis:**
    Launch Jupyter Notebook from your terminal in the main project folder:
    ```bash
    jupyter notebook
    ```
    From the Jupyter interface that opens in your browser, click on `spotify_churn_analysis.ipynb` to open the notebook and view the code, visualizations, and results.

    **Note on File Paths:** The notebook is configured to read data from the `data/` directory (e.g., `pd.read_csv("data/spotify_churn_dataset.csv")`).
