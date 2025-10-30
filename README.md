# Python & Predictive Modeling: Spotify Customer Churn Analysis

## 1. Project Overview
This project is a comprehensive demonstration of data analysis and predictive modeling in Python, applied to a real-world customer churn dataset. The primary goal is to answer key business questions related to user behavior and retention, and ultimately to build a machine learning model that can predict customer churn. The project showcases an end-to-end data science workflow, from data cleaning and exploratory analysis to model building and evaluation, with a focus on deriving actionable business insights.

## 2. Dataset
*   **Source:** [Spotify Dataset for Churn Analysis on Kaggle](https://www.kaggle.com/datasets/nabihazahid/spotify-dataset-for-churn-analysis)
*   **Description:** The dataset contains user-level data from a music streaming service, including demographics, subscription details, listening habits (e.g., listening time, skip rate), and a binary indicator for churn. The raw CSV file (`spotify_churn_dataset.csv`) is included in the `/data` folder of this repository for full reproducibility.

## 3. Methodology & Technical Implementation

### a. Data Preprocessing & Feature Engineering
The analysis was conducted using a Python script. The initial steps focused on preparing the data for analysis and modeling using the `pandas` and `numpy` libraries:
1.  **Cleaning:** Column names were standardized for clarity (e.g., `is_churned` to `churn`).
2.  **Transformation:** The categorical `subscription_type` column was transformed into a more model-friendly binary feature called `premium_user` (1 for paid, 0 for free). This simplifies analysis and improves model interpretability.
3.  **Validation:** The dataset was inspected for missing values and inconsistencies to ensure data quality before analysis.

### b. Exploratory Data Analysis (EDA)
A deep dive into the data was performed using `matplotlib` and `seaborn` to uncover patterns and relationships. The analysis focused on visualizing the key drivers of churn to inform both business strategy and the subsequent modeling phase. Key visualizations were generated and saved as PNG files for reporting.

### c. Predictive Modeling
A baseline predictive model was built using `scikit-learn` to establish an initial performance benchmark:
1.  **Preprocessing:** Categorical features like `gender`, `country`, and `device_type` were one-hot encoded to be used in the model.
2.  **Data Splitting:** The dataset was split into training (70%) and testing (30%) sets. A stratified split was used to ensure the proportion of churned users was consistent in both sets, which is critical for imbalanced datasets.
3.  **Model Training:** A **Logistic Regression** model was chosen for its interpretability and efficiency as a baseline.
4.  **Evaluation:** The model's performance was assessed using standard classification metrics, including Accuracy, Precision, Recall, and F1-Score.

### d. Tools Used
*   **Language:** Python
*   **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
*   **IDE:** VS Code, Jupyter Notebook

## 4. Business Questions & Analytical Insights

This analysis answered four key business questions, with the following insights:

### Q1: How does subscription type (Premium vs. Free) impact customer retention?
*   **Question:** Are users with free subscriptions more likely to churn than premium users?
*   **Insight:** The analysis confirms a strong relationship between subscription type and loyalty. Non-premium users, who are exposed to ads and have limited features, exhibit a significantly higher churn rate compared to premium subscribers. This indicates that the premium subscription is a powerful retention tool. **Business Action:** Efforts should be focused on converting free users to paid plans through targeted promotions, feature trials, or demonstrating the value of an ad-free experience.

### Q2: What is the relationship between user engagement and churn?
*   **Question:** Do less active users churn more often?
*   **Insight:** There is a clear and direct correlation between listening hours and customer retention. The distribution analysis shows that users who churned have, on average, a noticeably lower number of listening hours. This suggests that engagement is a primary indicator of a user's commitment to the platform. **Business Action:** Proactively identify users with declining listening hours and re-engage them with personalized content, new music recommendations, or curated playlists to boost activity and reduce churn risk.

### Q3: Does age play a role in customer churn?
*   **Question:** Are certain age groups more prone to churning?
*   **Insight:** The age distribution analysis reveals that while churn occurs across all age groups, there are discernible patterns. For instance, younger demographics might exhibit different churn behaviors compared to older, more established users. Understanding these nuances allows for more targeted marketing and retention campaigns. **Business Action:** Develop age-specific marketing campaigns. For younger users, this might involve social media engagement and trending content, while for older users, it could focus on legacy artists and long-term subscription value.

### Q4: How well can we predict churn with a baseline model?
*   **Question:** Can a simple model effectively identify users who are about to churn?
*   **Insight:** The initial Logistic Regression model achieved a high accuracy score. However, the detailed classification report revealed a critical weakness: the model had a **Precision and Recall of 0.0 for the churned class**. This means it failed to correctly identify a single user who was going to churn. This is a classic symptom of a model being overwhelmed by a class imbalance (many more non-churners than churners). **Business Action:** This insight is crucial. It proves that a simple accuracy metric is misleading and that more advanced techniques (like addressing class imbalance or using more complex models) are necessary to build a truly useful predictive tool.

## 5. Repository Structure
The repository is organized for clarity and reproducibility:

spotify-churn-analysis/
├── data/
│   └── spotify_churn_dataset.csv
├── churn_analysis.py
└── README.md

## 6. How to Use
1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/your-username/spotify-churn-analysis.git
    cd spotify-churn-analysis
    ```
2.  **Install Dependencies:**
    Ensure you have Python installed. Then, install the required libraries using pip:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn
    ```
3.  **Run the Analysis:**
    Execute the main Python script from your terminal:
    ```bash
    python churn_analysis.py
    ```
    The script will print the analysis results to the console and save the EDA plots as PNG files in the root directory.
