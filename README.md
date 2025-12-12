# Telco Customer Churn – Classification Project

This repository contains code and a report for a classification project
predicting customer churn in a telecom dataset (Kaggle: Telco Customer Churn).

## Structure

- `data/` – place the CSV file `WA_Fn-UseC_-Telco-Customer-Churn.csv` here (not included).
- `R/` – R scripts for data preparation and modelling.
- `report/` – R Markdown project report.
- `slides/` – presentation slides (R Markdown or rendered PDF/PPTX).

## Reproducing the analysis

1. Download the Telco Customer Churn dataset from Kaggle and save it as:

   `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`

2. Open `report/report.Rmd` in RStudio.
3. Install the required packages listed in the setup chunk if needed.
4. Knit the document to HTML or PDF.

Alternatively, you can run the scripts in `R/` in order:

- `R/01_eda_preprocessing.R`
- `R/02_models.R`
