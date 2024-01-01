# %% [markdown]
# # 6. Decision Trees and Ensemble Learning
# 
# 
# This week, we'll talk about decision trees and tree-based ensemble algorithms

# %% [markdown]
# ## 6.1 Credit risk scoring project
# 
# * Dataset: https://github.com/gastonstat/CreditScoring

# %%
import pandas as pd
import numpy as np

import seaborn as sns
from matplotlib import pyplot as plt
%matplotlib inline

# %% [markdown]
# ## 6.2 Data cleaning and preparation

# %% [markdown]
# * Downloading the dataset
# * Re-encoding the categorical variables
# * Doing the train/validation/test split

# %%
data = 'https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-06-trees/CreditScoring.csv'

# %%
!wget $data

# %%
!head CreditScoring.csv

# %%
df = pd.read_csv(data)

# %%
df.columns = df.columns.str.lower()

# %%
df.status.value_counts()

# %%
status_values = {
    1: 'ok',
    2: 'default',
    0: 'unk'
}

df.status = df.status.map(status_values)

# %%
home_values = {
    1: 'rent',
    2: 'owner',
    3: 'private',
    4: 'ignore',
    5: 'parents',
    6: 'other',
    0: 'unk'
}

df.home = df.home.map(home_values)

marital_values = {
    1: 'single',
    2: 'married',
    3: 'widow',
    4: 'separated',
    5: 'divorced',
    0: 'unk'
}

df.marital = df.marital.map(marital_values)

records_values = {
    1: 'no',
    2: 'yes',
    0: 'unk'
}

df.records = df.records.map(records_values)

job_values = {
    1: 'fixed',
    2: 'partime',
    3: 'freelance',
    4: 'others',
    0: 'unk'
}

df.job = df.job.map(job_values)

# %%
df.head()

# %%
df.describe().round()

# %%
for c in ['income', 'assets', 'debt']:
    df[c] = df[c].replace(to_replace=99999999, value=np.nan)

# %%
df.describe().round()

# %%
df = df[df.status != 'unk'].reset_index(drop=True)

# %%
from sklearn.model_selection import train_test_split

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=11)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=11)

# %%
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

# %%
y_train = (df_train.status == 'default').astype('int').values
y_val = (df_val.status == 'default').astype('int').values
y_test = (df_test.status == 'default').astype('int').values

# %%
del df_train['status']
del df_val['status']
del df_test['status']

# %%
df_train

# %% [markdown]
# ## 6.3 Decision trees
# 
# * How a decision tree looks like
# * Training a decision tree 
# * Overfitting
# * Controlling the size of a tree

# %%
def assess_risk(client):
    if client['records'] == 'yes':
        if client['job'] == 'parttime':
            return 'default'
        else:
            return 'ok'
    else:
        if client['assets'] > 6000:
            return 'ok'
        else:
            return 'default'

# %%
xi = df_train.iloc[0].to_dict()

# %%
assess_risk(xi)

# %%
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.tree import export_text

# %%
train_dicts = df_train.fillna(0).to_dict(orient='records')

# %%
dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dicts)

# %%
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

# %%
val_dicts = df_val.fillna(0).to_dict(orient='records')
X_val = dv.transform(val_dicts)

# %%
y_pred = dt.predict_proba(X_val)[:, 1]
roc_auc_score(y_val, y_pred)

# %%
y_pred = dt.predict_proba(X_train)[:, 1]
roc_auc_score(y_train, y_pred)

# %%
dt = DecisionTreeClassifier(max_depth=2)
dt.fit(X_train, y_train)

# %%
y_pred = dt.predict_proba(X_train)[:, 1]
auc = roc_auc_score(y_train, y_pred)
print('train:', auc)

y_pred = dt.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, y_pred)
print('val:', auc)

# %%
print(export_text(dt, feature_names=list(dv.get_feature_names_out())))

# %% [markdown]
# ## 6.4 Decision tree learning algorithm
# 
# * Finding the best split for one column
# * Finding the best split for the entire dataset
# * Stopping criteria
# * Decision tree learning algorithm

# %%
data = [
    [8000, 'default'],
    [2000, 'default'],
    [   0, 'default'],
    [5000, 'ok'],
    [5000, 'ok'],
    [4000, 'ok'],
    [9000, 'ok'],
    [3000, 'default'],
]

df_example = pd.DataFrame(data, columns=['assets', 'status'])
df_example

# %%
df_example.sort_values('assets')

# %%
Ts = [0, 2000, 3000, 4000, 5000, 8000]

# %%
T = 4000
df_left = df_example[df_example.assets <= T]
df_right = df_example[df_example.assets > T]

display(df_left)
print(df_left.status.value_counts(normalize=True))
display(df_right)
print(df_left.status.value_counts(normalize=True))

# %%
from IPython.display import display

# %%
for T in Ts:
    print(T)
    df_left = df_example[df_example.assets <= T]
    df_right = df_example[df_example.assets > T]
    
    display(df_left)
    print(df_left.status.value_counts(normalize=True))
    display(df_right)
    print(df_right.status.value_counts(normalize=True))

    print()

# %%
data = [
    [8000, 3000, 'default'],
    [2000, 1000, 'default'],
    [   0, 1000, 'default'],
    [5000, 1000, 'ok'],
    [5000, 1000, 'ok'],
    [4000, 1000, 'ok'],
    [9000,  500, 'ok'],
    [3000, 2000, 'default'],
]

df_example = pd.DataFrame(data, columns=['assets', 'debt', 'status'])
df_example

# %%
df_example.sort_values('debt')

# %%
thresholds = {
    'assets': [0, 2000, 3000, 4000, 5000, 8000],
    'debt': [500, 1000, 2000]
}

# %%
for feature, Ts in thresholds.items():
    print('#####################')
    print(feature)
    for T in Ts:
        print(T)
        df_left = df_example[df_example[feature] <= T]
        df_right = df_example[df_example[feature] > T]

        display(df_left)
        print(df_left.status.value_counts(normalize=True))
        display(df_right)
        print(df_right.status.value_counts(normalize=True))

        print()
    print('#####################')

# %% [markdown]
# ## 6.5 Decision trees parameter tuning
# 
# * selecting `max_depth`
# * selecting `min_samples_leaf`

# %%
depths = [1, 2, 3, 4, 5, 6, 10, 15, 20, None]

for depth in depths: 
    dt = DecisionTreeClassifier(max_depth=depth)
    dt.fit(X_train, y_train)
    
    y_pred = dt.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)
    
    print('%4s -> %.3f' % (depth, auc))

# %%
scores = []

for depth in [4, 5, 6]:
    for s in [1, 5, 10, 15, 20, 500, 100, 200]:
        dt = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=s)
        dt.fit(X_train, y_train)

        y_pred = dt.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)
        
        scores.append((depth, s, auc))

# %%
columns = ['max_depth', 'min_samples_leaf', 'auc']
df_scores = pd.DataFrame(scores, columns=columns)

# %%
df_scores_pivot = df_scores.pivot(index='min_samples_leaf', columns=['max_depth'], values=['auc'])
df_scores_pivot.round(3)

# %%
sns.heatmap(df_scores_pivot, annot=True, fmt=".3f")

# %%
dt = DecisionTreeClassifier(max_depth=6, min_samples_leaf=15)
dt.fit(X_train, y_train)

# %%
print(export_text(dt, feature_names=list(dv.get_feature_names_out())))

# %% [markdown]
# ## 6.6 Ensembles and random forest
# 
# * Board of experts
# * Ensembling models 
# * Random forest - ensembling decision trees
# * Tuning random forest

# %%
from sklearn.ensemble import RandomForestClassifier

# %%
scores = []

for n in range(10, 201, 10):
    rf = RandomForestClassifier(n_estimators=n, random_state=1)
    rf.fit(X_train, y_train)

    y_pred = rf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)
    
    scores.append((n, auc))

# %%
df_scores = pd.DataFrame(scores, columns=['n_estimators', 'auc'])

# %%
plt.plot(df_scores.n_estimators, df_scores.auc)

# %%
scores = []

for d in [5, 10, 15]:
    for n in range(10, 201, 10):
        rf = RandomForestClassifier(n_estimators=n,
                                    max_depth=d,
                                    random_state=1)
        rf.fit(X_train, y_train)

        y_pred = rf.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)

        scores.append((d, n, auc))

# %%
columns = ['max_depth', 'n_estimators', 'auc']
df_scores = pd.DataFrame(scores, columns=columns)

# %%
for d in [5, 10, 15]:
    df_subset = df_scores[df_scores.max_depth == d]
    
    plt.plot(df_subset.n_estimators, df_subset.auc,
             label='max_depth=%d' % d)

plt.legend()

# %%
max_depth = 10

# %%
scores = []

for s in [1, 3, 5, 10, 50]:
    for n in range(10, 201, 10):
        rf = RandomForestClassifier(n_estimators=n,
                                    max_depth=max_depth,
                                    min_samples_leaf=s,
                                    random_state=1)
        rf.fit(X_train, y_train)

        y_pred = rf.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)

        scores.append((s, n, auc))

# %%
columns = ['min_samples_leaf', 'n_estimators', 'auc']
df_scores = pd.DataFrame(scores, columns=columns)

# %%
colors = ['black', 'blue', 'orange', 'red', 'grey']
values = [1, 3, 5, 10, 50]

for s, col in zip(values, colors):
    df_subset = df_scores[df_scores.min_samples_leaf == s]
    
    plt.plot(df_subset.n_estimators, df_subset.auc,
             color=col,
             label='min_samples_leaf=%d' % s)

plt.legend()

# %%
min_samples_leaf = 3

# %%
rf = RandomForestClassifier(n_estimators=200,
                            max_depth=max_depth,
                            min_samples_leaf=min_samples_leaf,
                            random_state=1)
rf.fit(X_train, y_train)

# %% [markdown]
# Other useful parametes:
# 
# * `max_features`
# * `bootstrap`
# 
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

# %% [markdown]
# ## 6.7 Gradient boosting and XGBoost
# 
# * Gradient boosting vs random forest
# * Installing XGBoost
# * Training the first model
# * Performance monitoring
# * Parsing xgboost's monitoring output

# %%
!pip install xgboost

# %%
import xgboost as xgb

# %%
features = dv.get_feature_names_out()
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)
dval = xgb.DMatrix(X_val, label=y_val, feature_names=features)

# %%
xgb_params = {
    'eta': 0.3, 
    'max_depth': 6,
    'min_child_weight': 1,
    
    'objective': 'binary:logistic',
    'nthread': 8,
    
    'seed': 1,
    'verbosity': 1,
}

model = xgb.train(xgb_params, dtrain, num_boost_round=10)

# %%
y_pred = model.predict(dval)

# %%
roc_auc_score(y_val, y_pred)

# %%
watchlist = [(dtrain, 'train'), (dval, 'val')]

# %%
%%capture output

xgb_params = {
    'eta': 0.3, 
    'max_depth': 6,
    'min_child_weight': 1,
    
    'objective': 'binary:logistic',
    'eval_metric': 'auc',

    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}

model = xgb.train(xgb_params, dtrain, num_boost_round=200,
                  verbose_eval=5,
                  evals=watchlist)

# %%
s = output.stdout

# %%
print(s[:200])

# %%
def parse_xgb_output(output):
    results = []

    for line in output.stdout.strip().split('\n'):
        it_line, train_line, val_line = line.split('\t')

        it = int(it_line.strip('[]'))
        train = float(train_line.split(':')[1])
        val = float(val_line.split(':')[1])

        results.append((it, train, val))
    
    columns = ['num_iter', 'train_auc', 'val_auc']
    df_results = pd.DataFrame(results, columns=columns)
    return df_results

# %%
df_score = parse_xgb_output(output)

# %%
plt.plot(df_score.num_iter, df_score.train_auc, label='train')
plt.plot(df_score.num_iter, df_score.val_auc, label='val')
plt.legend()

# %%
plt.plot(df_score.num_iter, df_score.val_auc, label='val')
plt.legend()

# %% [markdown]
# ## 6.8 XGBoost parameter tuning
# 
# Tuning the following parameters:
# 
# * `eta`
# * `max_depth`
# * `min_child_weight`
# 

# %%
scores = {}

# %%
%%capture output

xgb_params = {
    'eta': 0.01, 
    'max_depth': 6,
    'min_child_weight': 1,
    
    'objective': 'binary:logistic',
    'eval_metric': 'auc',

    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}

model = xgb.train(xgb_params, dtrain, num_boost_round=200,
                  verbose_eval=5,
                  evals=watchlist)

# %%
scores = {}

# %%
key = 'eta=%s' % (xgb_params['eta'])
scores[key] = parse_xgb_output(output)
key

# %%
scores = {}

# %%
%%capture output

xgb_params = {
    'eta': 0.1, 
    'max_depth': 10,
    'min_child_weight': 1,
    
    'objective': 'binary:logistic',
    'eval_metric': 'auc',

    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}

model = xgb.train(xgb_params, dtrain, num_boost_round=200,
                  verbose_eval=5,
                  evals=watchlist)

# %%
key = 'max_depth=%s' % (xgb_params['max_depth'])
scores[key] = parse_xgb_output(output)
key

# %%
del scores['max_depth=10']

# %%
for max_depth, df_score in scores.items():
    plt.plot(df_score.num_iter, df_score.val_auc, label=max_depth)

plt.ylim(0.8, 0.84)
plt.legend()

# %%
scores = {}

# %%
%%capture output

xgb_params = {
    'eta': 0.1, 
    'max_depth': 3,
    'min_child_weight': 30,
    
    'objective': 'binary:logistic',
    'eval_metric': 'auc',

    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}

model = xgb.train(xgb_params, dtrain, num_boost_round=200,
                  verbose_eval=5,
                  evals=watchlist)

# %%
key = 'min_child_weight=%s' % (xgb_params['min_child_weight'])
scores[key] = parse_xgb_output(output)
key

# %%
for min_child_weight, df_score in scores.items():
    plt.plot(df_score.num_iter, df_score.val_auc, label=min_child_weight)

plt.ylim(0.82, 0.84)
plt.legend()

# %%
xgb_params = {
    'eta': 0.1, 
    'max_depth': 3,
    'min_child_weight': 1,

    'objective': 'binary:logistic',
    'eval_metric': 'auc',

    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}

model = xgb.train(xgb_params, dtrain, num_boost_round=175)

# %% [markdown]
# Other parameters: https://xgboost.readthedocs.io/en/latest/parameter.html
# 
# Useful ones:
# 
# * `subsample` and `colsample_bytree`
# * `lambda` and `alpha`

# %% [markdown]
# ## 6.9 Selecting the final model
# 
# * Choosing between xgboost, random forest and decision tree
# * Training the final model
# * Saving the model

# %%
dt = DecisionTreeClassifier(max_depth=6, min_samples_leaf=15)
dt.fit(X_train, y_train)

# %%
y_pred = dt.predict_proba(X_val)[:, 1]
roc_auc_score(y_val, y_pred)

# %%
rf = RandomForestClassifier(n_estimators=200,
                            max_depth=10,
                            min_samples_leaf=3,
                            random_state=1)
rf.fit(X_train, y_train)

# %%
y_pred = rf.predict_proba(X_val)[:, 1]
roc_auc_score(y_val, y_pred)

# %%
xgb_params = {
    'eta': 0.1, 
    'max_depth': 3,
    'min_child_weight': 1,

    'objective': 'binary:logistic',
    'eval_metric': 'auc',

    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}

model = xgb.train(xgb_params, dtrain, num_boost_round=175)

# %%
y_pred = model.predict(dval)
roc_auc_score(y_val, y_pred)

# %%
df_full_train = df_full_train.reset_index(drop=True)

# %%
y_full_train = (df_full_train.status == 'default').astype(int).values

# %%
del df_full_train['status']

# %%
dicts_full_train = df_full_train.to_dict(orient='records')

dv = DictVectorizer(sparse=False)
X_full_train = dv.fit_transform(dicts_full_train)

dicts_test = df_test.to_dict(orient='records')
X_test = dv.transform(dicts_test)

# %%
X_full_train

# %%
dfulltrain = xgb.DMatrix(X_full_train, label=y_full_train,
                    feature_names=dv.get_feature_names_out())

dtest = xgb.DMatrix(X_test, feature_names=dv.get_feature_names_out())

# %%
xgb_params = {
    'eta': 0.1, 
    'max_depth': 3,
    'min_child_weight': 1,

    'objective': 'binary:logistic',
    'eval_metric': 'auc',

    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}

model = xgb.train(xgb_params, dfulltrain, num_boost_round=175)

# %%
y_pred = model.predict(dtest)

# %%
roc_auc_score(y_test, y_pred)

# %% [markdown]
# ## 6.10 Summary
# 
# * Decision trees learn if-then-else rules from data.
# * Finding the best split: select the least impure split. This algorithm can overfit, that's why we control it by limiting the max depth and the size of the group.
# * Random forest is a way of combininig multiple decision trees. It should have a diverse set of models to make good predictions.
# * Gradient boosting trains model sequentially: each model tries to fix errors of the previous model. XGBoost is an implementation of gradient boosting. 

# %% [markdown]
# ## 6.11 Explore more
# 
# * For this dataset we didn't do EDA or feature engineering. You can do it to get more insights into the problem.
# * For random forest, there are more parameters that we can tune. Check `max_features` and `bootstrap`.
# * There's a variation of random forest caled "extremely randomized trees", or "extra trees". Instead of selecting the best split among all possible thresholds, it selects a few thresholds randomly and picks the best one among them. Because of that extra trees never overfit. In Scikit-Learn, they are implemented in `ExtraTreesClassifier`. Try it for this project.
# * XGBoost can deal with NAs - we don't have to do `fillna` for it. Check if not filling NA's help improve performance.
# * Experiment with other XGBoost parameters: `subsample` and `colsample_bytree`.
# * When selecting the best split, decision trees find the most useful features. This information can be used for understanding which features are more important than otheres. See example here for [random forest](https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html) (it's the same for plain decision trees) and for [xgboost](https://stackoverflow.com/questions/37627923/how-to-get-feature-importance-in-xgboost)
# * Trees can also be used for solving the regression problems: check `DecisionTreeRegressor`, `RandomForestRegressor` and the `objective=reg:squarederror` parameter for XGBoost.

# %% [markdown]
# ## Production

# %%
import pickle

# %%
C = 1.0
output_file = f'model_C={C}.bin'
output_file

# %%
# wb -> write binary (bytes, not text)
f_out = open(output_file, 'wb')
pickle.dump((dv, model), f_out)
f_out.close()

# %%
with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)
    # do stuff

# file is closed automatically    

# %% [markdown]
# #### Load model

# %%
import pickle

# %%
C = 1.0
input_file = f'model_C={C}.bin'
input_file

# %%
# rb -> read binary file
with open(input_file, 'rb') as f_in:
    (dv, model) = pickle.load(f_in)

# %%
dv, model

# %%
# Score a customer

customer = df.drop(columns=['status']).iloc[0].to_dict()

# %%
X = dv.transform([customer])
dtest = xgb.DMatrix(X, feature_names=dv.get_feature_names_out())

# %%
model.predict(dtest)[0]


