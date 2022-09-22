

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as colors

# Gridseach and RandomsearchCV
from sklearn.model_selection import GridSearchCV 
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

from sklearn.linear_model import LogisticRegression

#XGBoost
import xgboost as xgb 

#Decision Tree
from sklearn.tree import DecisionTreeClassifier # to build a classification tree
from sklearn.tree import plot_tree # to draw a classification tree
from sklearn.model_selection import cross_val_score # for cross validation

#SVM
from sklearn.decomposition import PCA 
from sklearn.svm import SVC 

#Random Forest 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingClassifier

# Importing the Keras libraries and packages
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dropout, Dense 
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop

# Importing evaluation libraries
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import plot_confusion_matrix 
from sklearn import preprocessing 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, brier_score_loss


df = pd.read_csv(r'C:/Users/65904/Desktop/Learning Python/EDA1-master/titanic_train.csv')
df_test = pd.read_csv(r'C:\Users\65904\Desktop\Learning Python\EDA1-master/test.csv')

df.head()
df.isnull()

# EDA if you dont wish to run
# sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# Roughly 20 percent of the Age data is missing. The proportion of Age missing is likely small enough for reasonable replacement with some form of imputation. 
#Looking at the Cabin column, it looks like we are just missing too much of that data to do something useful with at a basic level. We'll probably drop this later, or change it to another feature like "Cabin Known: 1 or 0"
# Let's continue on by visualizing some more of the data! Check out the video for full explanations over these plots, this code is just to serve as reference.

# sns.set_style('whitegrid')
# sns.countplot(x='Survived',data=df)


# sns.set_style('whitegrid')
# sns.countplot(x='Survived',hue='Sex',data=df,palette='RdBu_r')

# sns.set_style('whitegrid')
# sns.countplot(x='Survived',hue='Pclass',data=df,palette='rainbow')

# sns.distplot(df['Age'].dropna(),kde=False,color='darkred',bins=40)
# df['Age'].hist(bins=30,color='darkred',alpha=0.3)
# sns.countplot(x='SibSp',data=df)

# df['Fare'].hist(color='green',bins=40,figsize=(8,4))


# ___
# ## Data Cleaning
# We want to fill in missing age data instead of just dropping the missing age data rows. One way to do this is by filling in the mean age of all the passengers (imputation).
# However we can be smarter about this and check the average age by passenger class. For example:
# 

# plt.figure(figsize=(12, 7))
# sns.boxplot(x='Pclass',y='Age',data=df,palette='winter')

# We can see the wealthier passengers in the higher classes tend to be older, which makes sense. We'll use these average age values to impute based on Pclass for Age.
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age


df['Age'] = df[['Age','Pclass']].apply(impute_age,axis=1)


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# Drop the Cabin column and the row in Embarked that is NaN.
df = df.drop(['Ticket', 'PassengerId','Name','Cabin'], axis=1)
df.head()
df.dropna(inplace=True)

df.Sex.replace(('male', 'female'), (1, 0), inplace=True)



#X,y Split
X = df.drop('Survived', axis=1).copy() 
X.head() 
y = df['Survived'].copy()
y.head()

#One hot encoding 
X_encoded = pd.get_dummies(X, columns=['Sex',
                                        'Embarked',
                                        'SibSp', 
                                        'Parch', 
                                        'Pclass'
                                        ])
X_encoded.head()

# XGBoost requires that all data be either int, float or boolean data types. 
# We can use dtypes to see if there are any columns that need to be converted...
X_encoded.dtypes

# X,y train test split + Stratification 
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, random_state=42, stratify = y)
scaler = preprocessing.StandardScaler().fit(X_train)


X_train_scaled = scaler.fit_transform(X_train)
X_train_df = pd.DataFrame(X_train_scaled, columns = X_train.columns, index = X_train.index)

# X_train_scaled = scaler.transform(X_train)
# X_train_df = pd.DataFrame(X_train_scaled, columns = X_train.columns, index = X_train.index)

# standardizing the out-of-sample data
X_test_scaled = scaler.transform(X_test)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns = X_test.columns, index = X_test.index)





#%% Logistic 


# ## Training and Predicting using logistic regression 
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)

# predictions = logmodel.predict(X_test)

# accuracy=confusion_matrix(y_test,predictions)
# accuracy

# accuracy=accuracy_score(y_test,predictions)
# accuracy

# predictions


predicted = logmodel.predict(X_test)
Survived = logmodel.predict_proba(X_test)
Survived = [x[1] for x in Survived] 

print("accuracy:", accuracy_score(y_test, predicted))
print("balanced_accuracy:", balanced_accuracy_score(y_test, predicted))
print("recall:", recall_score(y_test, predicted))
print("brier_score_loss:", brier_score_loss(y_test, Survived))
print(classification_report(y_test,predicted))


# accuracy: 0.8161434977578476
# balanced_accuracy: 0.7994884910485933
# recall: 0.7294117647058823
# brier_score_loss: 0.14754226983730434






#%% XGBoost

#XGBoost
clf_xgb = xgb.XGBClassifier(objective='binary:logistic', 
                            eval_metric="logloss",
                            seed=42, 
                            use_label_encoder=False)

clf_xgb.fit(X_train, 
            y_train)

plot_confusion_matrix(clf_xgb, X_test, y_test, display_labels=["Does not Survived", "Survived"])

param_grid = {
    'max_depth': [3, 4, 5, 6, 7, 8],
    'n_estimators': range(50, 250, 350),
    'learning_rate': [0.1, 0.01, 0.05],
    'gamma': [0, 0.25, 0.5, 1.0],
    'reg_lambda': [0, 1.0, 10.0, 100.0]
}

params={
 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
    
}

optimal_params = GridSearchCV(
    estimator=xgb.XGBClassifier(objective='binary:logistic', eval_metric="logloss", seed=42, use_label_encoder=False),
    param_grid=params,
    scoring = 'roc_auc',
    # subsample=0.9, 
    verbose=0,
    n_jobs = 10,
    cv = 5
)

optimal_params.fit(X_train, y_train)
print(optimal_params.best_params_)

clf_xgb = xgb.XGBClassifier(seed=42,
                        objective='binary:logistic',
                        eval_metric="logloss", 
                        gamma=0,
                        learning_rate=0.2,
                        max_depth=5,
                        colsample_bytree = 0.7,
                        min_child_weight = 7,
                        n_estimators=50,
                        reg_lambda=10,
                        use_label_encoder=False)
clf_xgb.fit(X_train, y_train)


plot_confusion_matrix(clf_xgb, X_test, y_test, display_labels=["Does not Survive", "Survive"])

print("\n-----Out of sample test: XGBoost")


predicted = clf_xgb.predict(X_test) 
Survived = clf_xgb.predict_proba(X_test)
Survived = [x[1] for x in Survived] 

print("accuracy:", accuracy_score(y_test, predicted))
print("balanced_accuracy:", balanced_accuracy_score(y_test, predicted))
print("recall:", recall_score(y_test, predicted))
print("brier_score_loss:", brier_score_loss(y_test, Survived))


# accuracy: 0.8295964125560538
# balanced_accuracy: 0.8035805626598465
# recall: 0.6941176470588235
# brier_score_loss: 0.14187240876504897

#%% Classification Trees

#Classification Tree

clf_dt = DecisionTreeClassifier(random_state=42)
clf_dt = clf_dt.fit(X_train, y_train)


plt.figure(figsize=(15, 7.5))
plot_tree(clf_dt, 
          filled=True, 
          rounded=True, 
          class_names=["Does not Survived", "Survived"], 
          feature_names=X_encoded.columns); 


# Cost Complexity Pruning: Visualize alpha

path = clf_dt.cost_complexity_pruning_path(X_train, y_train) # determine values for alpha
ccp_alphas = path.ccp_alphas # extract different values for alpha
ccp_alphas = ccp_alphas[:-1] # exclude the maximum value for alpha

clf_dts = [] # create an array that we will put decision trees into

## now create one decision tree per value for alpha and store it in the array
for ccp_alpha in ccp_alphas:
    clf_dt = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    clf_dt.fit(X_train, y_train)
    clf_dts.append(clf_dt)



train_scores = [clf_dt.score(X_train, y_train) for clf_dt in clf_dts]
test_scores = [clf_dt.score(X_test, y_test) for clf_dt in clf_dts]

fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas, train_scores, marker='o', label="train", drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker='o', label="test", drawstyle="steps-post")
ax.legend()
plt.show()

# Cost Complexity Pruning: Cross Validation For Finding the Best Alpha
clf_dt = DecisionTreeClassifier(random_state=42, ccp_alpha=0.005) # create the tree with ccp_alpha=0.005

## now use 5-fold cross validation create 5 different training and testing datasets that
## are then used to train and test the tree.
## NOTE: We use 5-fold because we don't have tons of data...
scores = cross_val_score(clf_dt, X_train, y_train, cv=5) 
df = pd.DataFrame(data={'tree': range(5), 'accuracy': scores})

df.plot(x='tree', y='accuracy', marker='o', linestyle='--')

## create an array to store the results of each fold during cross validiation
alpha_loop_values = []

## For each candidate value for alpha, we will run 5-fold cross validation.
## Then we will store the mean and standard deviation of the scores (the accuracy) for each call
## to cross_val_score in alpha_loop_values...
for ccp_alpha in ccp_alphas:
    clf_dt = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    scores = cross_val_score(clf_dt, X_train, y_train, cv=5)
    alpha_loop_values.append([ccp_alpha, np.mean(scores), np.std(scores)])

## Now we can draw a graph of the means and standard deviations of the scores
## for each candidate value for alpha
alpha_results = pd.DataFrame(alpha_loop_values, 
                             columns=['alpha', 'mean_accuracy', 'std'])

alpha_results.plot(x='alpha', 
                   y='mean_accuracy', 
                   yerr='std', 
                   marker='o', 
                   linestyle='--')

# Using cross validation, we can see that, over all, instead of setting ccp_alpha=0.05,
# we need to set it to something closer to 0.03. We can find the exact value with:

    
alpha_results[(alpha_results['alpha'] > 0.003)
              &
              (alpha_results['alpha'] < 0.005)]

ideal_ccp_alpha = alpha_results[(alpha_results['alpha'] >  0.0035) 
                                & 
                                (alpha_results['alpha'] < 0.004)]['alpha']
print(ideal_ccp_alpha)

## convert ideal_ccp_alpha from a series to a float
ideal_ccp_alpha = float(ideal_ccp_alpha)
ideal_ccp_alpha

clf_dt_pruned = DecisionTreeClassifier(random_state=42, 
                                       ccp_alpha=ideal_ccp_alpha)
clf_dt_pruned = clf_dt_pruned.fit(X_train, y_train) 


plot_confusion_matrix(clf_dt_pruned, 
                      X_test, 
                      y_test, 
                      display_labels=["Does not Survive", "Survive"])



plot_confusion_matrix(clf_dt_pruned, X_test, y_test, display_labels=["Does not Survive", "Survive"])

print("\n-----Out of sample test: Classification Tree")


predicted = clf_dt_pruned.predict(X_test) 
Survived = clf_dt_pruned.predict_proba(X_test)
Survived = [x[1] for x in Survived] 

print("accuracy:", accuracy_score(y_test, predicted))
print("balanced_accuracy:", balanced_accuracy_score(y_test, predicted))
print("recall:", recall_score(y_test, predicted))
print("brier_score_loss:", brier_score_loss(y_test, Survived))


# accuracy: 0.7892376681614349
# balanced_accuracy: 0.7800085251491902
# recall: 0.7411764705882353
# brier_score_loss: 0.1628254506841839


#%% Support Vector Machine

## Build A Preliminary Support Vector Machine
clf_svm = SVC(random_state=42)
clf_svm.fit(X_train_scaled, y_train)

plot_confusion_matrix(clf_svm, 
                      X_test_scaled, 
                      y_test,
                      values_format='d',
                      display_labels=["Does not Survive", "Survive"])


# Using  `GridSearchCV()`. 
param_grid = [
  {'C': [0.5, 1, 10, 100], # NOTE: Values for C must be > 0
   'gamma': ['scale', 1, 0.1, 0.01, 0.001, 0.0001], 
   'kernel': ['rbf']},
]


optimal_params = GridSearchCV(
        SVC(), 
        param_grid,
        cv=5,
        scoring='accuracy', 
        verbose=0 
    )

optimal_params.fit(X_train_scaled, y_train)
print(optimal_params.best_params_)


clf_svm = SVC(random_state=42, C=10, gamma= 0.01)
clf_svm.fit(X_train_scaled, y_train)

plot_confusion_matrix(clf_svm, 
                      X_test_scaled, 
                      y_test,
                      values_format='d',
                      display_labels=["Does not Survive", "Survive"])


# no of columns in the df
len(df.columns)

#Applying PCA
pca = PCA() # NOTE: By default, PCA() centers the data, but does not scale it.
X_train_pca = pca.fit_transform(X_train_scaled)


per_var = np.round(pca.explained_variance_ratio_* 100, decimals=1)
labels = [str(x) for x in range(1, len(per_var)+1)]
 
plt.bar(x=range(1,len(per_var)+1), height=per_var, tick_label=labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Components')
plt.title('Screen Plot')
plt.show()


# The screen plot shows that the first principal component, PC1, accounts for a relatively large amount of variation in the raw data, 
# and this means that it will be good candidate for the x-axis in the 2-dimensional graph. 
# However, PC2 is not much different from PC3 or PC4, which doesn't bode well for dimension reduction. 
# Now we will draw the PCA graph. 

train_pc1_coords = X_train_pca[:, 0] 
train_pc2_coords = X_train_pca[:, 1]


## NOTE:
## pc1 contains the x-axis coordinates of the data after PCA
## pc2 contains the y-axis coordinates of the data after PCA

## Now center and scale the PCs...
pca_train_scaled = preprocessing.scale(np.column_stack((train_pc1_coords, train_pc2_coords)))



## Now we optimize the SVM fit to the x and y-axis coordinates
## of the data after PCA dimension reduction...
num_features = np.size(pca_train_scaled, axis=1)
param_grid = [
  {'C': [1, 10, 100, 1000], 
   'gamma': [1/num_features, 1, 0.1, 0.01, 0.001, 0.0001], 
   'kernel': ['rbf']},
]

optimal_params = GridSearchCV(
        SVC(), 
        param_grid,
        cv=5,
        scoring='roc_auc', 
        verbose=0 
    )

optimal_params.fit(pca_train_scaled, y_train)
print(optimal_params.best_params_)

#Calculating Accuracy and Recall
clf_svm = SVC( kernel='rbf', random_state=42, C=1000, gamma=0.001, probability=True)
classifier = clf_svm.fit(X_train_df, y_train)


predicted = classifier.predict(X_test_scaled_df) 
prob_default = classifier.predict_proba(X_test_scaled_df)
prob_default = [x[1] for x in prob_default] 

print("accuracy:", accuracy_score(y_test, predicted))
print("balanced_accuracy:", balanced_accuracy_score(y_test, predicted))
print("recall:", recall_score(y_test, predicted))
print("brier_score_loss:", brier_score_loss(y_test, prob_default))

# Results
# accuracy: 0.7937219730941704
# balanced_accuracy: 0.7745950554134697
# recall: 0.6941176470588235
# brier_score_loss: 0.1479459753750253

#Plotting the SVM Chart 
clf_svm = SVC( kernel='rbf', random_state=42, C=1000, gamma=0.01)
classifier = clf_svm.fit(pca_train_scaled, y_train)

X_test_pca = pca.transform(X_test_scaled)
test_pc1_coords = X_test_pca[:, 0] 
test_pc2_coords = X_test_pca[:, 1]


x_min = test_pc1_coords.min() - 1
x_max = test_pc1_coords.max() + 1

y_min = test_pc2_coords.min() - 1
y_max = test_pc2_coords.max() + 1

xx, yy = np.meshgrid(np.arange(start=x_min, stop=x_max, step=0.1),
                     np.arange(start=y_min, stop=y_max, step=0.1))


Z = clf_svm.predict(np.column_stack((xx.ravel(), yy.ravel())))

Z = Z.reshape(xx.shape)

fig, ax = plt.subplots(figsize=(10,10))

ax.contourf(xx, yy, Z, alpha=0.1)

## now create custom colors for the actual data points
cmap = colors.ListedColormap(['#e41a1c', '#4daf4a'])


scatter = ax.scatter(test_pc1_coords, test_pc2_coords, c=y_test, 
               cmap=cmap, 
               s=100, 
               edgecolors='k', ## 'k' = black
               alpha=0.7)

## now create a legend
legend = ax.legend(scatter.legend_elements()[0], 
                   scatter.legend_elements()[1],
                    loc="upper right")
legend.get_texts()[0].set_text("No HD")
legend.get_texts()[1].set_text("Yes HD")

## now add axis labels and titles
ax.set_ylabel('PC2')
ax.set_xlabel('PC1')
ax.set_title('Decison surface using the PCA transformed/projected features')
## plt.savefig('svm.png')
plt.show()



#%% Gradient Boosting

clf_gbc = GradientBoostingClassifier(learning_rate=0.1, 
                                     n_estimators=100,
                                     max_depth=3, 
                                     min_samples_split=2, 
                                     min_samples_leaf=1, 
                                     subsample=1,max_features='sqrt', 
                                     random_state=10)
clf_gbc.fit(X_train, y_train)

plot_confusion_matrix(clf_gbc, X_test, y_test, display_labels=["Does not Survived", "Survived"])


clf_gbc.fit(X_train,y_train)
predictors=list(X_train)
feat_imp = pd.Series(clf_gbc.feature_importances_, predictors).sort_values(ascending=False)
feat_imp.plot(kind='bar', title='Importance of Features')
plt.ylabel('Feature Importance Score')
print('Accuracy of the GBM on test set: {:.3f}'.format(clf_gbc.score(X_test, y_test)))
pred=clf_gbc.predict(X_test)
print(classification_report(y_test, pred))


predicted = clf_gbc.predict(X_test) 
Survived = clf_gbc.predict_proba(X_test)
Survived = [x[1] for x in Survived] 

print("accuracy:", accuracy_score(y_test, predicted))
print("balanced_accuracy:", balanced_accuracy_score(y_test, predicted))
print("recall:", recall_score(y_test, predicted))
print("brier_score_loss:", brier_score_loss(y_test, Survived))

# accuracy: 0.8071748878923767
# balanced_accuracy: 0.7877237851662404
# recall: 0.7058823529411765
# brier_score_loss: 0.13350221010166224


param_grid = {
    'max_depth': [1, 2, 3, 4, 5, 6, 7, 8],
    'n_estimators':[100,250,500,750],
    'learning_rate': [0.15,0.1,0.05,0.01,0.005,0.001],

}

optimal_params = GridSearchCV(
    estimator = GradientBoostingClassifier(max_depth=4, 
                                           min_samples_split=2, 
                                           min_samples_leaf=1, 
                                           subsample=1,
                                           max_features='sqrt', 
                                           random_state=10), 
                                            param_grid = param_grid, 
                                            n_jobs=10, 
                                            cv=5, 
                                            verbose=0)



optimal_params.fit(X_train, y_train)
print(optimal_params.best_params_)

clf_gbc = GradientBoostingClassifier(learning_rate=0.05, 
                                     n_estimators=100, 
                                     max_depth=3, 
                                     min_samples_split=2, 
                                     min_samples_leaf=1, 
                                     subsample=1,
                                     max_features='sqrt', 
                                     random_state=10, 
                                     # scoring='roc_auc', 
                                     # n_jobs=10, 
                                     # cv=5, 
                                     # verbose=0
                                     )
    
clf_gbc.fit(X_train, y_train)


plot_confusion_matrix(clf_gbc, X_test, y_test, display_labels=["Does not Survive", "Survive"])

print("\n-----Out of sample test: Gradient Boosting")


predicted = clf_gbc.predict(X_test) 
Survived = clf_gbc.predict_proba(X_test)
Survived = [x[1] for x in Survived] 

print("accuracy:", accuracy_score(y_test, predicted))
print("balanced_accuracy:", balanced_accuracy_score(y_test, predicted))
print("recall:", recall_score(y_test, predicted))
print("brier_score_loss:", brier_score_loss(y_test, Survived))

# -----Out of sample test: Gradient Boosting
# accuracy: 0.820627802690583
# balanced_accuracy: 0.7985933503836318
# recall: 0.7058823529411765
# brier_score_loss: 0.13803860211697136


#%% Random Forest Classifier 

rf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 50)
rf.fit(X_train, y_train)



rf_paramgrid = {'max_depth':[3,5,10,None],
              'n_estimators':[10,100,200], # increase additional trees
              'max_features':['auto', 'sqrt', 'log2'],
               'criterion':['gini','entropy'],
               'bootstrap':[True,False],
               # 'min_samples_leaf':randint(1,4),
              }


param_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}

optimal_params = GridSearchCV(estimator=rf, param_grid = rf_paramgrid, cv= 5)
optimal_params.fit(X_train, y_train)

print(optimal_params.best_params_)

rf=RandomForestClassifier(n_jobs=-1, 
                                  n_estimators=10,
                                  bootstrap= False,
                                  criterion='entropy',
                                  max_depth=5,
                                  max_features= 'auto',
                                  min_samples_leaf= 3)


rf.fit(X_train, y_train)

plot_confusion_matrix(rf, X_test, y_test, display_labels=["Does not Survive", "Survive"])

print("\n-----Out of sample test: XGBoost")


predicted = rf.predict(X_test) 
Survived = rf.predict_proba(X_test)
Survived = [x[1] for x in Survived] 

print("accuracy:", accuracy_score(y_test, predicted))
print("balanced_accuracy:", balanced_accuracy_score(y_test, predicted))
print("recall:", recall_score(y_test, predicted))
print("brier_score_loss:", brier_score_loss(y_test, Survived))


# accuracy: 0.8116591928251121
# balanced_accuracy: 0.7800511508951407
# recall: 0.6470588235294118
# brier_score_loss: 0.14872371694888864



# est = RandomForestClassifier(n_jobs=-1)
# rf_p_dist={'max_depth':[3,5,10,None],
#               'n_estimators':[10,100,200,300,400,500],
#               'max_features':randint(1,3),
#                'criterion':['gini','entropy'],
#                'bootstrap':[True,False],
#                'min_samples_leaf':randint(1,4),
#               }


# def hypertuning_rscv(est, p_distr, nbr_iter,X,y):
#     rdmsearch = RandomizedSearchCV(est, 
#                                     param_distributions=p_distr,
#                                   n_jobs=-1, 
#                                   n_iter=nbr_iter, 
#                                   cv=9)
#     #CV = Cross-Validation ( here using Stratified KFold CV)
#     rdmsearch.fit(X,y)
#     ht_params = rdmsearch.best_params_
#     ht_score = rdmsearch.best_score_
#     return ht_params, ht_score

# rf_parameters, rf_ht_score = hypertuning_rscv(est, rf_p_dist, 40, X, y)

# rf_parameters 



# claasifier=RandomForestClassifier(n_jobs=-1, 
#                                   n_estimators=300,
#                                   bootstrap= True,
#                                   criterion='entropy',
#                                   max_depth=3,
#                                   max_features=2,
#                                   min_samples_leaf= 3)

# predicted = rf.predict(X_test) 
# Survived = rf.predict_proba(X_test)
# Survived = [x[1] for x in Survived] 

# print("accuracy:", accuracy_score(y_test, predicted))
# print("balanced_accuracy:", balanced_accuracy_score(y_test, predicted))
# print("recall:", recall_score(y_test, predicted))
# print("brier_score_loss:", brier_score_loss(y_test, Survived))

# # Predicting the Test set results
# y_pred = classifier.predict(X_test)

# # Making the Confusion Matrix
# from sklearn.metrics import confusion_matrix,accuracy_score
# cm = confusion_matrix(y_test, y_pred)

# accuracy_score=accuracy_score(y_test,y_pred)

# #claasifier=RandomForestClassifier(n_jobs=-1, n_estimators=300,bootstrap= True,criterion='entropy',max_depth=3,max_features=2,min_samples_leaf= 3)

# ## Cross Validation good for selecting models
# from sklearn.model_selection import cross_val_score

# cross_val=cross_val_score(claasifier,X,y,cv=10,scoring='accuracy').mean()




#%% ANN


# define the keras model
model = Sequential()
model.add(Dense(12, input_shape=(8,), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# fit the keras model on the dataset
model.fit(X, y, epochs=150, batch_size=10)

y_pred = model.predict(X_test)

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(activation="relu", input_dim=24, units=12, kernel_initializer="uniform"))

# Adding the second hidden layer
classifier.add(Dense(activation="relu", units=6, kernel_initializer="uniform"))

# Adding the output layer
classifier.add(Dense(activation="sigmoid", units=1, kernel_initializer="uniform"))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix

cm = confusion_matrix(y_test, y_pred)
accuracy=accuracy_score(y_test,y_pred)

cm
accuracy

# plot_confusion_matrix(classifier, X_test, y_test, display_labels=["Does not Survive", "Survive"])

# print("\n-----Out of sample test: XGBoost")


# predicted = classifier.predict(X_test) 
# Survived = classifier.predict_proba(X_test)
# Survived = [x[1] for x in Survived] 

# print("accuracy:", accuracy_score(y_test, predicted))
# print("balanced_accuracy:", balanced_accuracy_score(y_test, predicted))
# print("recall:", recall_score(y_test, predicted))
# print("brier_score_loss:", brier_score_loss(y_test, Survived))









































