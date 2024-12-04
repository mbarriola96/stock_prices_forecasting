# Stock Prices Forecasting

-------------------------------------------

## Description

This project aims to assist investors in understanding stock trends by analyzing time-series data of ASML's stock performance. As a third-party consulting firm, we specialize in helping investors and companies gain actionable insights into market behavior, enabling them to make informed decisions through data-driven forecasts. For this analysis, daily stock price data from Yahoo Finance (January 1, 2018, to November 28, 2024) was aggregated into weekly averages, resulting in a time-series dataset of over 300 records.

The primary business objective was to predict the future trend of ASML's stock prices over a two-week horizon, a critical timeframe for investors optimizing portfolio strategies. The dataset was split into training and testing sets, with the last 12 weeks reserved for testing. Preprocessing steps, such as normalization, handling missing values, and data smoothing, were applied to prepare the data for analysis.

Multiple forecasting models were evaluated, including SARIMAX, Prophet, XGBoost, Random Forest, Naive, and ETS (additive and multiplicative). Prophet emerged as the final model due to its superior performance in minimizing root mean square error (RMSE) and mean absolute percentage error (MAPE). After optimization, the model achieved a low MAPE on the test set.

The final model delivers reliable two-week forecasts, providing investors with valuable insights to anticipate stock trends and make strategic decisions.

## Sources

The dataset consists of more than 300 records related to the stock prices of ASML and it's suppliers. In addition to the stock prices, exogeneous variables have been added in a binary format. We extracted the stock information from the yfinance library of Python:

- The yfinance library, used to retrieve the financial data, is well-documented and can be explored further here: https://pypi.org/project/yfinance/. 

## Methodology

Eight notebooks were created in order to analyze the data and based on that predict the dependent variable (ie the ASML stock value):

- 1. 00_primary_notebook.

In this notebook, there is a high level overview of all the steps carried out in this 	project in order to do train and test our model. Moreover, the notebooks contains the links to the different notebooks in case the reader wants a deeper understanding of the processes. 

- 2. 01_data_understanding notebook. 

This notebook analyzes ASML stock price data from Yahoo Finance, performing cleaning, transformation, and exploratory analysis. Key methods include STL decomposition, ACF/PACF, and ARIMA modeling to uncover trends, seasonality, and forecast stock performance.

- 3. 02_data_preprocessing.

The notebook explores forecasting ASML stock prices using models like ETS, SARIMAX, Prophet, XGBoost, and Random Forest. Prophet was identified as the best model, fine-tuned for accuracy, with thorough preprocessing, evaluation, and result visualization.

- 4. 03_model_creation.

The notebook analyzes ASML suppliers' stock data using forecasting models like Prophet to predict weekly closing prices. Key metrics (MAPE, RMSE) were used for evaluation, and optimized models were saved for future analysis.

- 5. 04_data_preparation.

The notebook prepares ASML and supplier data by combining weekly stock data with binary variables for exogenous events like COVID and geopolitical tensions. The final dataset, checked for consistency and missing values, is exported for further analysis.

- 6. 05_modelling.

The notebook forecasts ASML stock prices using SARIMAX, Prophet, XGBoost, and Random Forest, with Prophet achieving the best results using exogenous variables. Fine-tuned for improved RMSE and MAPE, Prophet was selected as the primary model and saved for future use.

- 7. 06_model_comparison.

The notebook compares Prophet models for ASML stock prices, showing improved accuracy (lower MAPE and RMSE) when incorporating supplier and geopolitical data. This highlights the value of external variables in capturing market influences for adaptive planning and decision-making.

- 8. 07_future_predictions.

The notebook uses Prophet models to predict ASML stock prices, incorporating exogenous variables like geopolitical events and supplier performance. Supplier forecasts were integrated into the ASML model, providing insights into future trends and external impacts.


## Conclusion

Looking into the distribution of the time series of ASML stock. Please note that the series is volatile and is thus a challenge to predict it:

![Time Series of ASML Stock Prices](/visualizations/asml_weekly_stock_prices.png)

The comparison between the ASML model by itself and the ASML model with its suppliers and binary exogenous variables revealed key insights. Including supplier and exogenous data in the latter model significantly improved MAPE and RMSE values. This demonstrates that integrating external variables enhances the model's predictive capabilities. Supplier-specific data provided valuable context, capturing market dynamics and trends that impact ASML's stock performance. The enhanced model proved more robust, supporting better strategic decisions and adaptive planning in volatile market conditions.

Here we have a graph comparing both models:

![Model Comparison](/visualizations/model_comparison.png)

We have predicted the weekly stock prices for the next two weeks starting from November 28, 2024, for ASML's suppliers and exogenous variables. These predictions allowed us to forecast ASML's stock prices for the same period. The results are displayed in the following chart.
To improve these results, we could focus on two key areas:

![ASML Predicted Stock](/visualizations/asml_predicted_stock.png)

To improve these results, we could focus on:

- 1. Integration of Real-Time Data: Incorporate live stock price updates and external factors (e.g., breaking news, real-time geopolitical events) into the model to enable dynamic forecasting. This could make the predictions more accurate and responsive to sudden market changes.

- 2. Advanced Feature Engineering: Enhance the dataset with additional relevant features, such as sentiment analysis of news articles or tweets about ASML and its suppliers. This can be achieved by leveraging the existing project, Brands-Product_Emotions_Analysis, to perform sentiment analysis on social media data. Incorporating these sentiment features would help capture market sentiment more effectively and improve the model's ability to predict stock price movements under varying conditions.

## Presentation

Here is the link to my presentation: [Stock Prices Forecasting Presentation](https://github.com/mbarriola96/stock_prices_forecasting/blob/main/presentation.pdf)

## Author

My name is Miguel Barriola Arranz. I am an Industrial Engineer and a Duke graduate student in Engineering Management. 
I am currently working in the microchip industry and further expanding my skillset in data science. 

- LinkedIn: https://www.linkedin.com/in/miguel-barriola-arranz/
- Medium: https://medium.com/@mbarriolaarranz

## Technologies

I have used **Python** with Jupyter notebook.

## Project Status

The project is in a development process at this moment. 

## What to find in the repository

There is a folder called notebooks that contains all the used notebooks and a python file named project_functions.py. This file is used to store all the functions that were created in this project.

There is a requirements.txt that contains the information of the libraries used in this project.

There is a .gitignore that allows to exclude files that are of no interest.

There is a notebook folder that contains all the notebooks. Moreover, the repository contains the presentation of the Project in pdf format as well as the primary notebook with the .py file to allow it to be run.  

