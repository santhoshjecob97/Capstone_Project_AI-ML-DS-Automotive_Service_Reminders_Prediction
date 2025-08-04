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
