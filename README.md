# Project_Phase-3
## Business Understanding
SyriaTel, a telecommunications company, wants to reduce customer churn that is customersleaving the service. Churn is costly as acquiring new customers is more expensive thanretaining existing ones. By predicting which customers are likely to churn, SyriaTel canproactively intervene with retention strategies such as discounts, improved service.
Stakeholders Include: SyriaTel’s management, marketing team, and customer servicedepartment.
Value Proposition: To Reduce churn will improve revenue, customer lifetime value, and brandloyalty.
## Problem Statement
SyriaTel faces a problem where customers are discontinuing their services, leading to revenueloss. The task is to develop a predictive model that identifi es customers likely to churn based onthe below:
- usage patterns
- plan details
- customer service interactions.
## Objectives
Below are the objectives:
1.To build a binary classifi cation model to predict customer churn.
2.To identify key factors driving churn to inform retention strategies.
3.To achieve high model performance (e.g., accuracy, precision, recall) to ensure actionablepredictions.
4.To provide recommendations to SyriaTel based on insights.

## Data Understanding

### Load customer churn dataset
df = pd.read_csv("Customer_Churn.csv")
### Basic data understanding
print (df.head())
print(df.info())
print(df.describe())

min 0.000000 0.000000 25% 2.300000 1.000000 50% 2.780000 1.000000 75% 3.270000 2.000000 max5.4000009.000000
The Customer churn dataset has 21 variables with 3333 observations.
Our Target Variable is Churn, indicated as either True or False.
From the above statistics, the below can be observed:
Account Length: The average account length is about 101 days. Account lengths rangefrom 1 to 243 days. The distribution is fairly symmetrical around the mean.
Area Code: There are three area codes represented: 408, 415, and 510. This column likelyrepresents the geographical location of the customer.
Number voice mail messages: The average number of voicemail messages is 8. Manycustomers (over 50%) don't have any voicemail messages. The maximum number ofvoicemail messages is 51.
Total day minutes: Customers average about 180 minutes of day calls. There's a fairamount of variability in day call duration (std = 54.47). Some customers have very low (0minutes) while others have very high (350.8 minutes) day call usage.
Total day calls: Customers average about 100 day calls. The number of day calls rangesfrom 0 to 165.
Total day charge: The average charge is 0 to $59.64.
Total eve minutes, Total eve calls, Total eve charge: These attributes provide similarinsights into evening call activity. For example, average evening minutes are around 201,and the average charge is $17.08.
Total night minutes, Total night calls, Total night charge: These attributes provide similarinsights into night call activity. For example, average night minutes are around 201, and theaverage charge is $9.04.
Total intl minutes, Total intl calls, Total intl charge: These relate to international callactivity. On average, customers have about 10.24 minutes of international calls. Asignifi cant number of customers (at least 25%) do not make international calls.
Customer service calls: The average number of customer service calls is 1.56. Themajority of customers make 2 or fewer customer service calls. Some customers make ahigh number of customer service calls (up to 9).

## Data Cleaning
Correct Formatskeyboard_arrow_down
### Converting Churn to 0 and 1
### Converting Categorical columns to strings
df['churn'] = df['churn'].map({False:0,True:1})
df['international plan'] = df['international plan'].astype(str)
df['voice mail plan'] = df['voice mail plan'].astype(str)
### Handling NAs
print(df.isnull().sum())
state 0account length 0area code 0phone number 0international plan 0voice mail plan 0number vmail messages 0total day minutes 0total day calls 0total day charge 0total eve minutes 0total eve calls 0total eve charge 0total night minutes 0total night calls 0total night charge 0total intl minutes 0total intl calls 0total intl charge 0customer service calls 0churn 0dtype: int64
We do not have any missing values
### Handling Duplicates
#### Checking for duplicates
df.duplicated().sum()
0
We do not have duplicates
### Other cleaning steps
# Checking for uniqueness in phone number and drop
ping itdf.drop('phone number', axis=1, inplace=True)
Phone number dropped as it is not relevant in our analysis.
### Feature Engineering
# Creating Total Minutes feature
# Creating Total Charge feature
# Creating Call per Minute Ratio
df['total minutes'] = df[['total day minutes','total eve minutes','total night minutes'
df['total charge'] = df[['total day charge','total eve charge','total night charge','t
df['calls per minute'] = (df[['total day calls','total eve calls','total night calls',1)
Total minutes
= total day minutes + total eve minutes + total night minutes + total intlminutes.
2)
Total charge feature
= sum of all charges.
3)
Calls per minute ratio
= (total day calls + total eve calls + total night calls + total intl calls) /total minutes.
Explanatory Data Analysiskeyboard_arrow_down
Univariate Analysiskeyboard_arrow_down
import
matplotlib.pyplot
as
plt
import
seaborn
as
sns
sns.histplot(df[
'total day minutes'
])
plt.show()
The above histogram shows the distribution of the 'total day minutes'
The peak of the histogram is around 180-200 minutes. This indicates that the mostcommon range for total day minutes is between 180 and 200 minutes.
There's a slight right skew, meaning the tail on the right side is longer. This suggests thatthere are a few customers with unusually high "total day minutes."
Most customers use between 100 and 250 minutes of call time during the day.
sns.histplot(df[
'customer service calls'
])
plt.show()
The above histogram shows the distribution of the 'customer service calls'
The highest bar is at 1 customer service call, indicating that the most common number ofcustomer service calls is 1.
There's a clear decreasing trend as the number of customer service calls increases. Thebars get progressively shorter, showing that fewer customers make 2, 3, 4, and so on,customer service calls.
Most customers require minimal assistance, with many not contacting customer serviceat all.
sns.countplot(df[
'international plan'
])
plt.show()
The above is a horizontal bar chart visualizing the count of customers who either have or do nothave an "international plan"
A large majority of customers (3000 and above) do not subscribe to the international plan.
A relatively small number of customers (around 300-400) have opted for the internationalplan.
sns.countplot(df[
'state'
])
plt.show()
The chart above is a horizontal bar chart showing the distribution of the "state"variable,representing the number of customers or observations per state.
States like WV (West Virginia), MN (Minnesota), and WY (Wyoming) appear to haverelatively high counts.
States like KS (Kansas), DC (District of Columbia), and IA (Iowa) appear to have relativelylow counts.
Bivariate Analysiskeyboard_arrow_down
sns.boxplot(x=
'churn'
, y=
'customer service calls'
, data=df)
plt.show()
print
(pd.crosstab(df[
'international plan'
], df[
'churn'
]))
churn 0 1
international plan
no 2664 346
yes 186 137
The presence of outliers suggests that there are some customers with unusually high customerservice call activity, particularly within the churned group.
Multivariate Analysiskeyboard_arrow_down
numeric_df = df.select_dtypes(include=[np.number])
correlation = numeric_df.corr()
plt.figure(figsize=(
12
,
8
))
sns.heatmap(correlation, annot=
False
, cmap=
'coolwarm'
)
plt.title(
'Correlation Matrix'
)
plt.show()
Call Charges: As expected, call charges are directly proportional to the number of minutesused for each time period (day, evening, night, international).
Churn Risk: A higher number of customer service calls is associated with a higherlikelihood of customer churn, which is a signifi cant fi nding.
Call Effi ciency: The "calls per minute" metric is negatively correlated with call duration,which is a logical inverse relationship.
Preprocessingkeyboard_arrow_down
from
sklearn.preprocessing
import
StandardScaler
Encodingkeyboard_arrow_down
# One hot encoding 'State' and 'Area Code'
df = pd.get_dummies(df, columns=[
'state'
,
'area code'
], drop_first=
True
)
df[
'international plan'
] = df[
'international plan'
].
map
({
'no'
:
0
,
'yes'
:
1
})
df[
'voice mail plan'
] = df[
'voice mail plan'
].
map
({
'no'
:
0
,
'yes'
:
1
})
Scalingkeyboard_arrow_down
# Standardize numerical features
scaler = StandardScaler()
numerical_cols = [
'account length'
,
'number vmail messages'
,
'total day minutes'
,
'total
df[numerical_cols] = scaler.fit_transform(df[numer
ical_cols])
Modelingkeyboard_arrow_down
Classifi cationkeyboard_arrow_down
from
sklearn.model_selection
import
train_test_split
from
sklearn.linear_model
import
LogisticRegression
from
sklearn.ensemble
import
RandomForestClassifier
from
sklearn.neighbors
import
KNeighborsClassifier
from
sklearn.svm
import
SVC
from
sklearn.naive_bayes
import
GaussianNB
X = df.drop(
'churn'
, axis=
1
)
y = df[
'churn'
]
X_train, X_test, y_train, y_test = train_test_spli
t(X, y, test_size=
0.2
, random_state=
42
)
Logistic Regressionkeyboard_arrow_down
models = {
'Logistic Regression'
: LogisticRegression(),
}
for
name, model
in
models.items():
model.fit(X_train, y_train)
print
(
f
"
{name}
Accuracy:
{model.score(X_test, y_test)}
"
)
Logistic Regression Accuracy: 0.8455772113943029
/usr/local/lib/python3.11/dist-packages/sklearn/linear_model/_logistic.py:465: ConverSTOP: TOTAL NO. of ITERATIONS REACHED LIMIT.Increase the number of iterations (max_iter) or scale the data as shown in:
https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
n_iter_i = _check_optimize_result(
 
Random Forestkeyboard_arrow_down
models = {
'Random Forest'
: RandomForestClassifier(),
}
for
name, model
in
models.items():
model.fit(X_train, y_train)
print
(
f
"
{name}
Accuracy:
{model.score(X_test, y_test)}
"
)
Random Forest Accuracy: 0.9655172413793104
K-NN Modelkeyboard_arrow_down
models = {
'K-NN'
: KNeighborsClassifier(),
}
for
name, model
in
models.items():
model.fit(X_train, y_train)
print
(
f
"
{name}
Accuracy:
{model.score(X_test, y_test)}
"
)
K-NN Accuracy: 0.8710644677661169
SVMkeyboard_arrow_down
models = {
'SVM'
: SVC(),
}
