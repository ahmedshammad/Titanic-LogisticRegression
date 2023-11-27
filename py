# Load the Titanic training data from a CSV file into a DataFrame
address = 'titanic-training-data.csv'
titanic_training = pd.read_csv(address)

# Display the first few rows of the DataFrame
titanic_training.head()

# Select specific columns from the DataFrame and display the first few rows
titanic_training = titanic_training[['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked', 'Survived']]
titanic_training.head()

# Display information about the DataFrame, including data types and missing values
print(titanic_training.info())

# Define variable descriptions for better understanding of column meanings
VARIABLE_DESCRIPTIONS = {
    'Survived': 'Survival (0 = No; 1 = Yes)',
    'Pclass': 'Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)',
    'Name': 'Name',
    'Sex': 'Sex',
    'Age': 'Age',
    'SibSp': 'Number of Siblings/Spouses Aboard',
    'Parch': 'Number of Parents/Children Aboard',
    'Ticket': 'Ticket Number',
    'Fare': 'Passenger Fare (British pound)',
    'Cabin': 'Cabin',
    'Embarked': 'Port of Embarkation (C = Cherbourg, France; Q = Queenstown, UK; S = Southampton - Cobh, Ireland)'
}

# Display countplot for the 'Survived' column using seaborn
sns.countplot(x='Survived', data=titanic_training, palette='hls')

# Display the count of missing values for each column in the DataFrame
titanic_training.isnull().sum()

# Display descriptive statistics for the DataFrame
titanic_training.describe()

# Visualize pair plots for numerical variables in the DataFrame
sns.pairplot(titanic_training)

# Drop irrelevant variables and handle missing values in 'Age' column based on 'Parch'
titanic_data = titanic_training.drop(['Name', 'Ticket', 'Cabin'], axis=1)
titanic_data['Age'] = titanic_data[['Age', 'Parch']].apply(age_approx, axis=1)
titanic_data.isnull().sum()

# Label encoding for 'Sex' column
label_encoder = LabelEncoder()
sex_labels = label_encoder.fit_transform(titanic_data['Sex'])
sex_encoded_series = pd.Series(sex_labels, name='Sex_encoded')
titanic_data = pd.concat([titanic_data, sex_encoded_series], axis=1)

# Drop missing values and convert 'Embarked' column to one-hot encoding
titanic_data.dropna(inplace=True)
ohe1 = pd.get_dummies(titanic_data['Embarked']).astype('int')
titanic_data = pd.concat([titanic_data, ohe1], axis=1)
titanic_data.drop(['Sex', 'Embarked'], axis=1, inplace=True)
titanic_data['Parch'] = pd.to_numeric(titanic_data['Parch'], errors='coerce')

# Display correlation heatmap for numerical variables in the DataFrame
sns.heatmap(titanic_data.corr(), cmap='Purples')

# Filter high correlation values from the correlation matrix
high_corr_matrix = titanic_data.corr()[titanic_data.corr().abs() > 0.5]

# Drop 'Fare' and 'Pclass' columns from the DataFrame
titanic_dmy = titanic_data.drop(['Fare', 'Pclass'], axis=1)

# Display information about the DataFrame
titanic_dmy.info()

# Split the data into features (X) and target variable (y) and perform train-test split
X = titanic_data.iloc[:, :-1]
y = titanic_data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=200)

# Initialize a Logistic Regression model and train it on the training data
LogReg = LogisticRegression(solver='liblinear')
LogReg.fit(X_train, y_train)

# Make predictions on the test data and display a classification report
y_pred = LogReg.predict(X_test)
print(classification_report(y_test, y_pred))

# Use cross-validated predictions on the training data and calculate precision score
y_train_pred = cross_val_predict(LogReg, X_train, y_train, cv=5)
conf_matrix = confusion_matrix(y_train, y_train_pred)
precision = precision_score(y_train, y_train_pred)

# Display Logistic Regression equation coefficients
intercept = LogReg.intercept_[0]
coefficients = LogReg.coef_[0]
equation = f"logit(p) = {intercept:.4f} + "
for i, coef in enumerate(coefficients):
    equation += f"{coef:.4f} * X_{i+1} + "
equation = equation[:-2]
print("Logistic Regression Equation:")
print(equation)

# Define a test passenger with specific feature values
test_passenger = np.array([866, 40, 0, 0, 0, 0, 0, 1, 0, 0]).reshape(1, -1)

# Use the trained Logistic Regression model to predict the survival status of the test passenger
prediction = LogReg.predict(test_passenger)
print("Predicted Survival Status:", prediction)

# Use the trained Logistic Regression model to predict the probability of each class for the test passenger
probability = LogReg.predict_proba(test_passenger)

# Display the predicted probabilities for each class
print("Predicted Probabilities for Each Class:")
print(f"Not Survived (0): {probability[0, 0]:.6%}")
print(f"Survived (1): {probability[0, 1]:.6%}")

