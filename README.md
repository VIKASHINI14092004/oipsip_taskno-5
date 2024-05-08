# oipsip_taskno-5 : SALES PREDICTION USING PYTHON
# Introduction
In this project, we analyze advertising data to understand the relationship between different advertising channels (TV, Radio, Newspaper) and sales. Additionally, we develop a predictive model using linear regression to forecast sales based on advertising expenditures.
# Data Overview
Dataset: The dataset contains information about advertising expenditures on TV, Radio, and Newspaper, along with corresponding sales figures.

Features:

TV: Advertising budget spent on TV ads.

Radio: Advertising budget spent on radio ads.

Newspaper: Advertising budget spent on newspaper ads.

Target Variable:

Sales: Sales figures generated as a result of advertising expenditures.
# Data Analysis and Visualization
Data Loading: We loaded the advertising data from a CSV file into a pandas DataFrame, ensuring correct headers were used.

Descriptive Statistics: We calculated summary statistics (mean, median, min, max, etc.) for each feature to understand their distributions and central tendencies.

Pairplot Visualization: We created pairplots to visualize the relationships between different advertising channels and sales. This helped us identify potential correlations and trends in the data.

Correlation Analysis: We computed the correlation matrix between features and visualized it using a heatmap. This allowed us to identify strong correlations between advertising channels and sales.

# Model Development
Feature Selection: We selected the 'TV' and 'Radio' features as predictors for the linear regression model based on their correlations with sales.

Model Training: We split the data into training and testing sets and trained a linear regression model using the training data.

Model Evaluation: We evaluated the model's performance on the test set using metrics such as R-squared and Root Mean Squared Error (RMSE).

#  Results and Conclusion
Model Performance: The linear regression model achieved an R-squared value of X.XX and an RMSE of X.XX. These metrics indicate that the model explains X% of the variance in sales and makes predictions with an average error of X.XX units.

Insights: The analysis revealed that advertising expenditures on TV and radio have a significant positive impact on sales, while newspaper advertising has a relatively weaker influence.

# Future Directions

Model Improvement: Experiment with more sophisticated regression techniques or ensemble methods to further improve prediction accuracy.

Feature Engineering: Explore additional features or transformations that may better capture the relationship between advertising expenditures and sales.

External Factors: Consider incorporating external factors such as seasonality, economic indicators, or competitor data to enhance the predictive power of the model.
