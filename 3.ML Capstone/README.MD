# AI-Driven Customer Service Reminder System

This capstone project, titled "AI-Driven Customer Service Reminder System," aims to **automate and personalise vehicle service reminders and customer communications** for automotive dealerships and service networks. It leverages advanced data science and machine learning workflows, covering the entire process from data handling to model deployment.

## 📚 Project Overview

The project provides an end-to-end solution for predictive maintenance and customer engagement in the automotive after-sales domain. It analyses various customer and vehicle data points to proactively identify service needs, predict future costs, segment customers, and generate tailored communication strategies.

## ✨ Key Features & Capabilities

*   **Comprehensive Data Collection:** Gathers and consolidates both **structured data** (e.g., purchase year, odometer readings, warranty/insurance status, last service details, communication history from CSV files) and **unstructured data** (free-text customer feedback).
*   **Robust Data Preprocessing:**
    *   Performs **missing value handling**, data type conversion, and outlier filtering.
    *   Includes **feature engineering** to create insightful variables such as `age_of_vehicle`, `odometer_reading`, `last_service_kms`, and `avg_kms_per_month`.
    *   Encodes categorical features (e.g., `customer_type`, `AMC_status`) and generates **sentiment features** from customer feedback.
*   **Exploratory Data Analysis (EDA):** Visualises distributions of numeric features, identifies relationships between variables (e.g., `last_service_cost` vs. `age`), and detects feature importances for modelling.
*   **Advanced Machine Learning Models:** Your project leverages three types of AI models:
    *   **Classification Models:**
        *   **Purpose:** Predicts if a customer's vehicle requires **urgent service within the next 120 days** based on vehicle usage, history, and customer behaviour.
        *   **Techniques:** Employs supervised classification algorithms like **RandomForest, GradientBoosting, and LightGBM pipelines**. Text features are processed using **TF-IDF vectorisation**.
        *   **Output:** Generates predictions and confidence scores, providing a list of customers needing urgent service, along with their predicted urgency/segment.
    *   **Regression Models:**
        *   **Purpose:** Predicts **continuous numerical values**, such as the `next service cost`, `number of days until the next service is needed`, and `Customer Lifetime Value (CLTV)`.
    *   **Clustering Models (Unsupervised):**
        *   **Purpose:** Groups customers or vehicles into segments based on similar characteristics like driving behaviour, service spend patterns, feedback sentiment, and service regularity.
*   **Intelligent Customer Segmentation:** Labels each customer into specific segments (e.g., **Critical, High Priority, Medium, Low**) based on model predictions, service due dates, and feedback scores.
*   **Personalised Communication Generation:** Outputs **tailored reminder message templates** for each customer segment and recommends optimal communication channels (Email, WhatsApp, SMS, Phone).
*   **Seamless Deployment:** The system is designed for local/server batch deployment and can be exported as a model service (Flask/FastAPI microservice or scheduled script) for integration with existing CRM systems or messaging gateways.

## 🛠 Technologies & Libraries Used

*   **Python 3.x**
*   **pandas**
*   **numpy**
*   **scikit-learn**
*   **textblob**
*   **lightgbm**
*   **joblib**
*   **matplot**
*   **seaborn**

## 📂 Project Structure

The project is organised into a logical workflow, with dedicated folders for each major step:

```
.
├── 1.Data Collection/             # Scripts & info to collect and aggregate customer and service data
├── 2.Data Preprocessing/          # Cleaning, missing value imputation, and feature engineering
├── 3.Data Univariate and Bivariate/ # EDA: Exploratory statistics & visualizations
├── 4.ML Models and Feature Selection/ # Predictive/classification model notebooks; selection of best feature set
├── 5.Final Model/                 # Final trained model(s), test results, and saved model files
├── 6.Final Presentation/          # PPT, demo, or report for project summary and results
├── notebooks/                     # Jupyter notebooks for rapid prototyping, experiments
└── Notes/                         # Project planning, assumptions, and research notes
```
*Note: The actual `.ipynb` files used for running the classification pipeline are often located within the `4.ML Models and Feature Selection/` folder, such as `02.Classification_04.ipynb`.*

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.x installed, along with the required libraries. You can install them using pip:

```bash
pip install pandas numpy scikit-learn lightgbm textblob joblib
```

### Local/Server Batch Deployment

1.  **Prepare Input Data:** Place your latest customer and service data as a CSV file named `modify_service_df.csv` in the project's working directory. An example input file is referenced as `modify_service_df.csv`.
2.  **Run Inference Pipeline:** Execute the main classification notebook. For instance, you can run the following from your terminal or within a Jupyter environment:
    ```bash
    python 02.Classification_04.ipynb
    ```
    *Alternatively, you can run each cell within the Jupyter Notebook `02.Classification_04.ipynb`.*
3.  **Review Generated Reminders:** Once the script completes, open the `ai_based_service_reminder.csv` file, which will contain tailored messages and recommended communication channels for each customer.

### Model Deployment as a Service

For continuous, automated operation in a production environment, the trained pipeline model (`service_reminder_model.pkl`) can be exported and deployed.

*   The model expects the same features as used during training.
*   It can be deployed as a **Flask/FastAPI microservice or a scheduled script**.
*   This service can then be integrated with existing CRM systems or messaging gateways for automated message dispatch.

## 📊 Output Files

Upon successful execution, the project generates key output files:

*   **`ai_based_service_reminder.csv`**: This CSV file provides, for each customer:
    *   Their assigned **segment label** (Critical, High, Medium, Low).
    *   A **personalised message** for service reminders.
    *   **Preferred communication channels** (e.g., Email, WhatsApp, Phone, SMS).
    *   Relevant service due information.
*   **`service_reminder_model.pkl`**: The reusable trained ML pipeline object for future predictions.

### Sample Automated Output Message

Here's an example of a personalised message template generated by the system:

```
Dear Retail Customer,

We are sorry for any inconvenience caused and will personally monitor your next service.
Enjoy a 15% discount as our apology.
Service due in XX days.

Channels: Phone, WhatsApp, Email
```


### Detailed Explanation of Code and Concepts:

My project leverages **three core types of AI models**: Classification, Regression, and Clustering, each addressing distinct business problems within the automotive after-sales domain.

#### 1. Classification Models:**

**Purpose & Concept:**
The primary purpose of your classification models is to **predict if a customer's vehicle requires urgent service within a specific timeframe**, typically the next 120 days. This is framed as a binary classification problem where the target variable, `service_due_soon` or `service_urgent`, is `1` if the vehicle meets the urgent criteria (e.g., due within 120 days, or has poor feedback) and `0` otherwise.

**Code & Technical Concepts:**
*   **Data Preparation & Features:**
    *   The models utilise a comprehensive set of features, including **numeric data** like `age_of_vehicle`, `odometer_reading`, `last_service_kms`, `avg_kms_per_month`, `last_service_cost`, `feedback_score`, `days_since_last_service`, and `next_service_due_days`.
    *   **Categorical features** include `customer_type`, `AMC_status`, `warranty_status`, `insurance_status`, `fuel_type`, and `transmission`.
    *   Crucially, **unstructured `customer_feedback` text** is also used.
*   **Preprocessing Pipeline (`sklearn.pipeline.Pipeline` & `sklearn.compose.ColumnTransformer`):**
    *   Your project uses a modular preprocessing pipeline to handle different data types.
    *   **Numeric Features:** Are handled by a `numeric_transformer` pipeline, which first uses `SimpleImputer(strategy='median')` to fill any missing numerical values, and then `StandardScaler()` to standardise them, ensuring all numerical features contribute equally to the model regardless of their scale.
    *   **Categorical Features:** Are processed by a `categorical_transformer` pipeline. This pipeline employs `SimpleImputer(strategy='most_frequent')` for missing values and `OneHotEncoder(handle_unknown='ignore')`. The `handle_unknown='ignore'` argument is vital for **production robustness**, preventing errors if the model encounters new, unseen categories during deployment by assigning all zeros for such categories.
    *   **Text Features (`customer_feedback`):** Are transformed using `TfidfVectorizer(max_features=100)`. TF-IDF (Term Frequency-Inverse Document Frequency) converts text into numerical vectors, weighting words based on their frequency in a document relative to their frequency across all documents, thus capturing the importance of terms. Additionally, a `FeedbackSentimentExtractor` class, leveraging `textblob`, is used to derive sentiment polarity from feedback text.
    *   All these transformers are orchestrated using `ColumnTransformer` (`preprocessor`) to apply the correct transformations to the respective feature sets.
*   **Machine Learning Algorithms:**
    *   Your project explores and utilises various supervised classification algorithms, including **RandomForestClassifier, GradientBoostingClassifier, SVM, and Logistic Regression**.
    *   Notably, Gradient Boosting has shown **1.00 accuracy** in some tests. The chosen best model for deployment in some instances is a `Pipeline` containing the `preprocessor` and the `GradientBoostingClassifier`.
*   **Output & Logic:** The models generate predictions (`predicted_urgency`) which are then used to segment customers into `Critical`, `High Priority`, `Medium Priority`, and `Low Priority` based on `feedback_score`, `next_service_due_days`, and predicted urgency. Personalized messages and preferred communication channels are then generated based on these segments.

**Business Usage in Automotive After-Sales:**
Classification models in your project provide immense business value:
*   **Proactive Service Reminders:** Automating timely reminders based on predicted service needs. This maximises workshop occupancy and helps reduce lapsed customers who might otherwise forget or switch dealerships.
*   **Churn Reduction & Retention:** Identifying customers who are likely to lapse or are dissatisfied, enabling targeted campaigns to retain them before they switch competitors.
*   **Personalised Marketing:** Segmenting customers into "critical," "high-priority," etc., allows for tailored offers (e.g., discounts, free washes).
*   **Optimising Call-Centre Operations:** Telecallers can focus their efforts more efficiently on customers with a high likelihood of service need or urgency, improving their conversion rates.
*   **Examples in Market:** Maruti Suzuki, Toyota, and Ford service centers widely use such systems for automated service reminders and follow-ups. SaaS tools like "AutoMoto AI" (your project's internal name) and DMS CRM add-ons also use these to boost revenue for dealerships.

#### 2. Regression Models

**Purpose & Concept:**
Unlike classification, regression models **predict continuous numerical values**. In your project, they are used for:
*   **Service Cost Prediction:** Estimating the `last_service_cost`.
*   **Days Until Next Service:** Predicting the `next_service_due_days`.
*   **Customer Lifetime Value (CLTV):** Forecasting the potential future revenue from a customer. Your project uses a proxy calculation like `last_service_cost * number_of_services` for `customer_lifetime_value_proxy`.
*   **Odometer Reading Prediction:** Predicting future `odometer_reading` or `next_service_due_kms` (as a proxy).

**Code & Technical Concepts:**
*   **Data Preparation:** Categorical columns are encoded using `LabelEncoder().fit_transform()`. Missing numeric values are filled using the median strategy (`df[col].fillna(df[col].median())`).
*   **Machine Learning Algorithm:** The primary model used is `RandomForestRegressor`. This ensemble method is well-suited for regression tasks due to its ability to handle non-linear relationships and high-dimensional data.
*   **Evaluation:** Models are evaluated using **Root Mean Squared Error (RMSE)** and **R² Score**. For instance, the CLTV prediction model showed an R² score of 0.998, indicating a very strong fit.

**Business Usage in Automotive After-Sales:**
Regression models bring significant financial and operational benefits:
*   **Predictive Maintenance & Upselling:** Knowing the predicted `next service cost` or `days until next service` helps dealerships with inventory management, staffing, and proactively offering relevant services or products.
*   **Financial Forecasting:** Predicting total service revenue per dealership or region aids in smarter resource allocation and setting realistic targets.
*   **Warranty/AMC Planning:** Anticipating high-cost maintenance events allows teams to target at-risk vehicles for extended warranty or Annual Maintenance Contract (AMC) renewals.
*   **Customer Lifetime Value (CLTV):** Identifying high-value customers helps in segmenting VIPs for special perks and focused retention strategies.
*   **Examples in Market:** Multi-brand service aggregators like MyTVS and GoMechanic use regression for optimizing part stocking and labor deployment.

#### 3. Clustering Models

**Purpose & Concept:**
Clustering models perform **unsupervised grouping of customers or vehicles into segments** based on similar characteristics, without needing pre-defined labels. This helps in discovering inherent patterns and customer personas within the data.

**Code & Technical Concepts:**
*   **Feature Engineering:** A key step involves extracting `feedback_sentiment` (polarity) from `customer_feedback` text using `TextBlob`.
*   **Preprocessing Pipeline:**
    *   **Numeric Features:** `SimpleImputer(strategy='median')` and `StandardScaler()`.
    *   **Categorical Features:** `SimpleImputer(strategy='most_frequent')` and `OneHotEncoder(handle_unknown='ignore')`.
    *   A `ColumnTransformer` combines these preprocessing steps for the `num_features` and `cat_features` used for clustering.
*   **Clustering Algorithm:** The primary algorithm used is `KMeans`. The number of clusters (`k`) is chosen (e.g., `k=3` in the example).
*   **Dimensionality Reduction & Visualisation:** `PCA (Principal Component Analysis)` is used to reduce the high-dimensional processed data into 2 components (`pca_1`, `pca_2`) for visualisation, allowing you to see the distinct customer segments on a scatter plot.

**Business Usage in Automotive After-Sales:**
Clustering models are invaluable for strategic customer understanding and targeting:
*   **Customer Profiling:** Helps brands and workshops understand distinct customer personas (e.g., "price-sensitive & regular," "premium, late responders," "chronic complainers") based on shared traits like driving behaviour, service spend, feedback sentiment, and service regularity. This understanding improves campaign design.
*   **Product Bundling & Offer Targeting:** Marketing teams can create offers tailored to specific segments (e.g., bundling service with insurance for "annual heavy users"), thereby improving the ROI of campaigns.
*   **Predictive Scheduling:** Allows for better capacity planning, especially during "high peak months," and tailoring service advisor assignments based on customer segments.
*   **NPS/CSI Analysis:** Clustering sentiment and feedback can directly inform quality improvement initiatives and product research & development.
*   **Examples in Market:** Large service networks (like TATA, Hyundai, Bosch) and digital DMS players use clustering for dynamic segmentation to improve loyalty, retention, and survey scores, while reducing service churn.

### Differences Between Your Attached 5 Classification Models

The "5 classification models" you refer to actually represent **different iterations or implementations of your classification pipeline across different notebook files**, rather than five entirely distinct, simultaneously running models. Let's break down what each PDF source file (corresponding to a "model") represents:

1.  **`02.Classification_01.pdf`**: This document primarily acts as a **comparison and testing ground** for various supervised classification algorithms. It directly imports and tests:
    *   `RandomForestClassifier`.
    *   `GradientBoostingClassifier`.
    *   `SVC` (Support Vector Classifier).
    *   `LogisticRegression`.
    It fits each of these models within a pipeline (`preprocessor` + `classifier`) and prints their `accuracy` and `classification_report`. Notably, **Gradient Boosting Classifier achieves 1.00 accuracy** in the reported results.

2.  **`02.Classification_02.pdf`**: This introduces a **class-based structure (`ServicePredictor`)** for the classification pipeline.
    *   It specifically uses `RandomForestClassifier` as its core classifier within the `_build_model` method.
    *   The primary difference here is the **organisation of the code into a reusable class**, which encapsulates data loading, model building, training, and prediction methods.
    *   The target variable is named `service_needed_soon`.

3.  **`02.Classification_03.pdf`**: This is an **enhanced version (`EnhancedServicePredictor`)** of the class-based pipeline.
    *   It continues to use `RandomForestClassifier` in its `_build_model`.
    *   It introduces **advanced features like finding the nearest premium dealership** for a customer based on location and brand.
    *   It also features **dynamic offer assignment based on customer feedback**.
    *   The reminder messages are more complex, offering both a "SaaS message" (from the AutoMoto AI system) and a "dealership message".
    *   The target variable is `service_needed_soon`.

4.  **`02.Classification_04.pdf`**: This iteration focuses on **integrating natural language processing (NLP) more explicitly** into the classification pipeline.
    *   It defines a new target variable, `service_urgent`, with a more complex rule-based definition that considers `next_service_due_days`, `feedback_score`, and specific `customer_feedback` categories (e.g., 'Poor Service', 'Unresponsive').
    *   The preprocessing specifically includes a `TfidfVectorizer` for the `customer_feedback` text feature.
    *   It also explicitly defines **rule-based customer segmentation** logic (`Critical`, `High Priority`, `Medium Priority`, `Low Priority`) and **communication channel prediction** (`comms_protocol`) based on these segments.
    *   The classifier used is `RandomForestClassifier`.
    *   This pipeline also achieves **1.00 accuracy** in its classification report.

5.  **`02.classification_05.pdf`**: This document appears to be a **near-identical duplicate** of `02.Classification_04.pdf`. It implements the same features, preprocessing, target definition, and segmentation logic.

In summary, these are not five *different* models in the sense of five separate, concurrently running systems with unique objectives. Instead, they represent a **progression of your classification pipeline's development**, starting from basic model comparison (`01`), moving to a class-based structure (`02`), adding enhanced business logic and features (`03`), and finally integrating text processing for the target and segmentation (`04`/`05`). The underlying classification algorithm used in the later iterations is predominantly `RandomForestClassifier`, while `GradientBoostingClassifier` showed superior performance in the initial comparison.

### Streamlit Cloud Deployment with `app.py` Script

The sources **do not explicitly mention Streamlit or Streamlit Cloud** for deployment. Instead, they outline deployment strategies as follows:
*   **Local/Server Batch Deployment:** The system is designed to be run locally or on a server to generate a CSV file (`ai_based_service_reminder.csv`) with reminders. This typically involves running a Python script or a Jupyter notebook (e.g., `02.Classification_04.ipynb`).
*   **Model Deployment as a Service:** For continuous, automated operation in a production environment, the trained pipeline model (`service_reminder_model.pkl`) can be exported and deployed as a **Flask/FastAPI microservice or a scheduled script**. This microservice would then be integrated with existing CRM systems or messaging gateways for automated dispatch of communications.

**Can you deploy to Streamlit Cloud with an `app.py` script?**
*   **Information outside the provided sources:** While the sources don't mention it, **yes, you absolutely can deploy a Streamlit application to Streamlit Cloud using an `app.py` script.**
*   **How it would typically work (general knowledge, not from sources):**
    1.  You would create an `app.py` script that imports your trained model (e.g., by loading your `service_reminder_model.pkl` using `joblib.load()`).
    2.  This `app.py` script would then define a Streamlit interface (e.g., input fields for customer data, a button to trigger prediction, and a display area for the personalised message and channels).
    3.  When a user interacts with the Streamlit app, it would pass the input data through your loaded model's preprocessing pipeline and then to the classifier to get predictions and segments.
    4.  The `app.py` would then use your existing logic (`gen_message`, `comms_protocol`) to display the personalized reminder and channels directly in the web application.
    5.  You would host this `app.py` script (along with a `requirements.txt` listing all necessary libraries) in a GitHub repository, and then connect that repository to Streamlit Cloud for deployment.

*   **Considerations:** Streamlit is typically used for **creating interactive dashboards or lightweight web applications** that serve as a front-end interface to your models. While it can host the model logic directly (by loading the `.pkl` file), for very high-throughput or complex enterprise integrations, a dedicated Flask/FastAPI backend (as mentioned in your README) might still be used, with Streamlit acting as a user-friendly way to interact with that backend service. For your capstone project, deploying a simple `app.py` that loads your `.pkl` and runs inference for demonstration purposes on Streamlit Cloud would be a **highly effective and feasible way to showcase your project**.

## ⭐ Model Performance

The classification models within this project demonstrate **near-perfect precision and recall (1.00)** for accurately identifying urgent and high-priority customers. Typical metrics include an accuracy of **1.00**.

## 📈 Business Value & Impact

This AI-driven system creates significant value for automotive dealerships and service networks:

*   **Increases Revenue:** Through enhanced customer retention and effective upselling.
*   **Reduces Operational Costs:** By automating routine customer touchpoints and efficiently targeting at-risk customers.
*   **Enables Hyper-Personalisation:** Fosters stronger customer loyalty and facilitates premium service offerings.
*   **Improves Customer Experience:** Leads to reduced negative feedback and decreased churn rates.
*   **Supports Data-Driven Management:** Provides insights for optimising workshop fulfilment, call centre staffing, stock management, and financial forecasting.

Your capstone project demonstrates immediate value by making the after-sales journey "smart," automated, and truly customer-centric.

---
