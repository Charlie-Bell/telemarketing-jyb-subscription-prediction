# Predicting Subscription Numbers of JYB Telemarketing Dataset
<em>"The data is related with direct marketing campaigns of a Portuguese banking institution. The marketing campaigns were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be ('yes') or not ('no') subscribed."</em> - <a href="https://www.kaggle.com/datasets/aguado/telemarketing-jyb-dataset">Telemarketing JYB Dataset on Kaggle</a><br>

The problem is a simple Binary Classifcation problem, to determine whether a customer will subscribe based on collected data.
The dataset is comprised of:
- Bank client data.
- Previous contact data.
- Social and economic attributes.

<b>This project is split into 4 main sections with associated notebooks: EDA, Data preparation, Model training and Results presentation.</b><br>
<i>Using K-Fold Cross-Validation, SMOTETomeks, and a Random Forest Classifier, a highly accurate model was trained to predict customer subscription with a high recall and precision score resulting an a very high F1-Score. F1-Score was used as a metric due to the high skewedness of the dataset. Other boosting models underperformed without hyperparameter tuning, but this was not necessary as the Random Forest Classifier gave sufficient results. For future work a deeper analysis should be performed on important features, a wider dataset collected, and if better performance is desired then hyperparameter tuning of the boosting models.</i>
