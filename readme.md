## Predict Customer Personality to Boost Marketing Campaign by Using Machine Learning
- - - 
**Tool** : Jupyter Notebook <br>
**Programming Language** : Python <br>
**Libraries** : Pandas, NumPy, Seaborn, Matplotlib, scikit -learn, k-means, PCA<br>
**Dataset** : marketing campaign from Rakamin Academy <br>

**Table of Contents**
- [STAGE 0: Problem Statement](#stage-0-problem-statement)
    - [overview](#background)
    - [Goal](#goal)
    - [Objective](#objective)
- [STAGE 1: Data Preparation](#stage-1-data-preparation)
    - [Handling Data](#stage-1-data-preparation)
    - [Feature Engineering](#feature-engineering)
- [STAGE 2: Data Exploration](#stage-2-data-exploration)
- [STAGE 3: Data Modeling K means](#stage-3-data-modeling-with-k-means)
    - [Preprocessing](#stage-3-data-modeling-with-k-means)
    - [Modeling](#modeling)
    - [Cluster Segmentation](#cluster-segmentation)
- [STAGE 4: Customer Personality Analysis](#stage-4-customer-personality-analysis)
- [STAGE 5: Business Recomendation](#stage-5-business-recommendation)


## Stage 0 Problem Statement

## Background
A company can develop rapidly when it knows its customers' personality behavior, so that it can provide better services and benefits to customers who have the potential to become loyal customers. By processing historical marketing campaign data to improve performance and target the right customers so they can make transactions on the company's platform, from this
data insight our focus is to create a cluster prediction model to make it easier for companies to make decisions.

## Goal
The goal of analyzing customer profiles and behavior with a clustering approach is to understand customers better, provide more personalized service, improve sales performance, and build strong relationships with customers.

## Objective
Develop a machine learning model that can classify consumers into several groups according to their traits and actions.
Gain greater understanding of the behavior and profiles of your customers.
utilizing the results of clustering to identify profitable business strategies.

# STAGE 1 Data Preparation
- Handling missing values
- handling Duplicated Data
- Check the type and consistency of values
- Checking for outliers or unusual data (anomalies)

## Feature Engineering 
Add new feature such as : 
- `conversion_rate`: make it from (Total Transaction / visit) customer history.

- `age` :  Year_Birth - now.

- number of children `total_kid`: from Kidhome + Teenhome.

- Total money spent `total_spent`: from MntCoke + MntFruits + MntMeatProducts + MntFishProducts + MntSweetProducts + MntGoldProds

- Total transaction `total_transaction` : from NumDealsPurchases + NumWebPurchases + NumCatalogPurchases + NumStorePurchases

- Total of days joined `membership_duration` (year) : now - Dt_Customer

- total accepted campaign customer `total_accepted_cmp` = acceptedcmp1 + acceptedcmp2 + acceptedcmp3 +acceptedcmp4 + acceptedcmp5 

**For the detail kindly to check notebook :)**


# STAGE 2. Data Exploration
At this stage, conversion rate analysis is carried out to gain insight into the percentage of website visitors and the actions taken during their visits. The purpose of this analysis is to see whether the visitor's actions lead to a purchase transaction or not. In this way, companies can understand visitor behavior and identify opportunities to increase conversion rates and the success of their marketing campaigns.

<p align="center">
    <kbd><img width="600" alt="Rasio Tipe Hotel" src="https://github.com/fauzanheryka/Project-Portofolio/assets/141212116/70de4f8b-c138-4adc-8557-3e6f5f57bf56"></kbd><br>
    Figure 1 - Heatmap
</p>

### Result :  
`Age`, `income`, `total spent`, `total transactions`, and `total accepted campaigns` are among the features that are shown, has significant correlation each other and with the conversion rate, so in order to fully understand the distribution of the data we have, we are going to analyze the scatterplot of the features that do.
<br>
<br>

<p align="center">
    <kbd> <img width="800" alt="scatter" src="https://github.com/fauzanheryka/Project-Portofolio/assets/141212116/b4d562a8-8738-45e1-af5b-b101bc9dc85a"
>
 </kbd> <br>
    Figure 2 — Scatterplot Age, income, total spent, total transactions, and total accepted campaigns
</p>

### Result :
- The age feature has been proven on the heatmap and scatter plot that the correlation with the conversion rate is very weak, so it can be concluded that the age factor has a weak influence on the conversion rate.
- Income shows a strong correlation. It can be seen that the higher the customer's income, the greater the chance that the customer will convert and buy the product.
- for total spent and total transactions, you can be sure that there is a strong correlation with the conversion rate, because people who have converted will definitely buy the product,
- for campaigns, it shows that if customers accept the campaign, then the customer will have a high chance of converting, so the company can be recommended to create the right campaign strategy to increase the customer's chances of converting and buying the product that has been offered.
- **For the detail insight kindly to check notebook**
<br>

# STAGE 3 Data Modeling with K means 
## Data Preprocessing 
Before carrying out data modeling, there are several data pre-processing stages that need to be carried out, namely:

- Features that are not needed for the model are removed to better focus the data.
- Categorical features will be encoded so they can be processed by machine learning algorithms.
- Feature standardization was carried out to ensure uniform data scale and avoid bias in the model.

## Modeling 
Using the Principal Component Analysis (PCA) approach comes next after data pre-processing. PCA is used to keep important information while reducing the dimensionality of the data. It can improve model performance and solve the issue of multicollinearity between features by lowering the dimensionality of the data. Finding the ideal number of clusters is a crucial next step in this process. The Elbow Method and the Distortion Score are employed in this research to determine the ideal number of clusters. The analysis's findings indicated that four clusters was the ideal quantity.

<p align="center">
    <kbd> <img width="800" alt="elbow method" src="https://github.com/fauzanheryka/Project-Portofolio/assets/141212116/f566868f-9f86-4288-9455-47c5ca2144c8"></kbd> <br>
    Figure 3: Elbow Method
</p>


<br>

<p align="center">
    <kbd><img width="600" alt="hotel resort" src="https://github.com/fauzanheryka/Project-Portofolio/assets/141212116/a783fcc7-7237-45db-be81-c8b0ef3361ad"></kbd> <br>
    Figure 4 - Silhouette Score
</p>

### Result : 
Base on elbow method and silhoute score best cluster is 3.

### Cluster Segmentation
<p align="center">
    <kbd><img width="1000" alt="segment" src="https://github.com/fauzanheryka/Project-Portofolio/assets/141212116/a2425656-406e-4682-91a9-30c70b27afa4"></kbd> <br>
    Figure 5 - Scatter Plot Cluster segmentation
</p>

### Result 
From the results of the scatterplot above, it can be said that the number of clusters equal to 3 is the right number of clusters. where it can be seen that there is a fairly clear segmentation between the clusters.
<br>


##  **STAGE 4: Customer Personality Analysis**
The objective of customer personality analysis is to **identify distinctive qualities that each group may possess, as well as comprehend the contrasts and similarities between these clusters**. Businesses are better able to target more particular business strategies and take more appropriate action for each consumer group when they have a deeper understanding of the traits that separate the clusters.

<p align="center">
    <kbd> <img width="700" alt="3d visual" src="https://github.com/fauzanheryka/Project-Portofolio/assets/141212116/07620944-ae77-4e00-89d7-fe65093b37db"></kbd> <br>
    Figure 6 — 3d visual cluster segmentation
</p>


## Interpretation Cluster : 

- cluster 0 
    - cluster 0 is a group of customers who frequently carry out transactions 21 times and have spent Rp. 1.169.791/month
    - According to other groups, this cluster visits the website an average of 3 times, but their conversion rate is rather high. This suggests that this particular consumer type only visits the website when they are certain they want to buy a product.
    - The majority of these customers are 56 years old
    - With a monthly income of Rp. 71.250,297 so it is not surprising that this cluster spends more than the other clusters.
    - This customer represents `can't loose them`
- cluster 1 
    - This cluster only spends IDR. 74,116 and only made 8 transactions and max 18 transaction.
    - Despite frequently visiting the website, this cluster has the lowest conversion rate, showing that the customers in consideration are only typical product browsers and have no plans of completing a purchase.
    - this group only spent Rp. 91.391 , this is very low spent rather than other cluster.
    - this cluster just have 1% conversion rate.
    - This cluster represents `low customers`
- cluster 2 
    - This cluster carried out 17 transactions and spent Rp. 639.300
    - This customer group also has an average income of around Rp. 55.828.696
    - Compared to the cluster 0 group, this customer group sees the website more frequently and has a decent conversion rate of about 4%.
    - If we can provide this cluster with the proper discount, it could be able to boost its conversion rate.
    - This customer represents a `potential customer`
- **For the detail insight kindly to check notebook**
<br>

The characteristics of each group, the cluster's propensity to react to current marketing campaigns, and the possible revenue results from marketing retargeting to the cluster can all be used to interpret the results of prior clustering. Additional analysis will be performed.

<p align="center">
    <kbd> <img width="800" alt="percentage" src="https://github.com/fauzanheryka/Project-Portofolio/assets/141212116/1fb7e721-3320-4ac3-afd3-6c1a8d579330"></kbd> <br>
    Figure 7 — Percentage total customer and average spent by cluster
</p>

### Result : 
- The graph indicates that the cluster with the lowest group has the greatest percentage of customers (54%), but the average amount spent is quite low. To boost the amount spent for this cluster group, consider offering exclusive discounts to draw in cluster groups with the greatest number of customers. 
- It is evident to prospective buyers that 28% of this group's consumers have the capacity to develop into loyal customers with the proper treatment, one way to demonstrate this is by giving clients a positive shopping experience and showing gratitude through a loyalty program.
- Even though the customer group can't loose them, it has a small number of customers, but it has the highest average spent compared to other clusters.
<br>

Analysis of the distribution of several features of each cluster was also carried out to gain deeper insight. Through this analysis, several interesting insights were discovered that can provide a better understanding of user behavior in each cluster, especially regarding website visits, conversion rate and Total Spent.

<p align="center">
    <kbd> <img width="700" alt="percentage" src="https://github.com/fauzanheryka/Project-Portofolio/assets/141212116/0c7633e4-252c-42b2-ba2e-e7f4f207399d"></kbd> <br>
    Figure 7 — Distribution plot by cluster
</p>

### Interesting thing : 
- low customer have the lowest income, Maybe that's why the conversion rate is low.
- The three customer groups have almost the same distribution of recency.
- According to the interpretation above, website visits for low customers are the highest.

In order to maximize the business metric `GMV`, we will perform a multivariate analysis with Using simple statistical regression , to determine which features have the greatest influence with total spent feature.

<p align="center">
    <kbd> <img width="800" alt="percentage" src="https://github.com/fauzanheryka/Project-Portofolio/assets/141212116/5bcff300-51bf-4502-bfbf-406f04694dca"></kbd> <br>
    Figure 7 — Regplot by cluster
</p>

### Result : 
- Features such as `income`, `total_spent`, `conversion_rate` these three features have a very strong influence on spending made by customers, therefore the company can provide appropriate recommendations for these features in order to increase `GMV` according to the target .
- Another interesting finding is that features like `visit website` have a negative correlation, indicating that a significant number of users are still viewing the website but are not completing any purchases. As a result, it is imperative to evaluate the current website in order to draw in more visitors. individually to attract their interest in completing a purchase.

<br>
<br>

##  **STAGE 5: Business Recommendation**

### Potential Impact :
Total Spent Group cant' loose them: Rp. 397.729.000<br>
Total Spent Group low customer: Rp. 90.751.000<br>
Total Spent Group potensial customer: Rp. 324.125.000<br>
Total all cluster : Rp. 812,605.000

We still have a potential gross merchandise value (`GMV`) of IDR 812,605,000 million if we can continue to prioritize our present customers and no one leaves.

## Recomendation

1.  `Can't Loose Them` <br>
The advice that can be given is to offer loyalty programs, such as free shipping discounts and special promotions for customer groups, because this customer group frequently makes transactions and the total amount spent is fairly high. This will help them be more satisfied with their purchases and keep their interest in the group.

2. `Low Customer` <br>
customers have the fewest spending and transactions of any group, but they have one unique characteristic they visit the website frequently so it can be advised to do an evaluation on the website to draw in this particular customer group. If needed, customer groups can also be offered small discounts to encourage more transactions, which will minimize the possibility of churn in this customer group.

3. `Potential Customer` <br>
If the company treats this customer group well, it could become the first cluster group. For example, by sending relevant and engaging messages to customers, it can guarantee a positive user experience when they visit the website or interact with the company's offerings. Developing a loyalty or incentives program helps improve client interaction. Companies may motivate potential loyal customers to keep selecting and buying their goods and services by offering them incentives like points, rewards, or exclusive advantages.