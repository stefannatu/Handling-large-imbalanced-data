# Handling Sparse, Large, imbalanced-data
### Explore training large ML models at scale using Tensorflow APIs, sharding data, handling imbalanced data by training for precision/recall over accuracy

Sparse, imbalanced datasets occur in a wide and growing number of business contexts: for example, AirBnb's apartment search relies on a highly sparse dataset as the number of clicks [1], impressions per home are extremely skewed with some highly popular homes getting a lot of clicks, and others getting very few to none. Typical advertising data such as clicks, impressions tends to be highly sparse, imbalanced and skewed as well, as most clicks on a particular landing page (such as Sears.com or ATnT.com) rarely lead to a user actually purchasing a product, so number of failed conversions greatly exceeds the number of conversions. On the flip side, websites collect clicks and impressions constantly, making these datasets extremely large (by large here we mean anything that doens't fit into memory), which means that the machine learning models built on these datasets *need* to scale, as well as have low latency when it comes to making predictions. New ways to deal with large but extremely sparse and imbalanced datasets is a forefront topic of research in machine learning. 

In this post we describe a study conducted on a prototypical example of such a dataset from Kaggle. We work with the Bosch dataset, which is one of the largest Kaggle datasets for modeling production part failures. The objective is to correctly classify parts as good or defective, making this a simple binary classification problem. The challenge lies with the data, found here https://www.kaggle.com/c/bosch-production-line-performance/data. It consists of over 4K columns of numeric, categorical and timestamp features, and over a million rows.  The dataset is large enough to merit advanced methods such as deep neural networks that scale, but also small enough that if needed, the entire dataset (or properly shuffled shards of the data) can be loaded into memory for training models such as decision trees, linear/logistic regression etc. Moreover the data is extremely imbalanced - containing only 0.6% of the positive class and 99.4% of the other. 

In this post, we describe with code samples, our approach for training models which achieve good performance on small datasets using scikit learn, and scalable, deployable models using Tensorflow's Dataset and Estimator APIs. We compare and contrast the two approaches both for performance and other metrics, including some helpful tricks learned along the way. Throughout we provide code snippets, but the full code is contained in the jupyter notebooks. 

A natural question is why would you choose one versus the other: unlike models like random forests, logistic regression which require heavy feature engineering, the promise of deep neural networks (DNNs) lies in the fact that given enough data, the model can learn the underlying distributions of the data and the raw data can simply be fed into the model as is. What we found though was that these models are by no means a silver bullet: this is because DNNs quickly tend to overfit as they have so many parameters, and further training doesn't lead to a reduction in the loss. 

 # Data Exploration
 
 Let's start with some data exploration. We mentioned that this dataset is huge by usual Kaggle standards, although not by production data science standards. Nonetheless, given that it takes several Gigabytes of memory, you may not want to load the *entire* dataset into memory or into Jupyter. One option is to read the database using PySpark dataframes which are handy for large datasets as they are optimized to handle them. One advantage of this is that PySpark uses lazy execution, meaning that it doens't read anything into memory and it is easy to simply take a small subset of the data using the *take, collect etc* aggregator functions. 
 
 An alternative is a handy module from the *pandas* library which splits the data into chunks of certain size that you can easily load into memory. Below is an example of this function which we used to shard the numeric data file into smaller chunks. Also note that for training purposes *only*, we upsampled the positive (defective) class to 50% (from 0.6%) in order to train a good model, and took 25K random samples from the non-defective class to create a class balanced dataset for analysis. Note that you shouldn't upsample the test data, for obvious reasons. Later on when we use the entire dataset, we won't need to upsample as much and will use all the non-defective examples for training the full model at scale. 
 
 ```python
import os
from sklearn.utils import resample, shuffle
data_path = os.getcwd() + '/train_data/shuffled_train_numeric.csv'
def get_sharded_datasets(full_data_path):
    chunksize = 100000
    for i, chunk in enumerate(pd.read_csv(data_path, chunksize=chunksize)):
            temp_df = chunk[chunk['Response'] == 1]
            neg_df = chunk[chunk['Response']==0][:25000]
            pos_df = resample(temp_df, n_samples=25000,random_state=42,replace=True)
            full_df = shuffle(pd.concat([neg_df, pos_df]), random_state = 42)
            print("Chunk Number: {}".format(i))
            full_df.to_csv(os.path.dirname(full_data_path) + "/upsampled_data_50k_{}.csv".format(i))
    return print("Done")   

```
 
Having split the data into chunks, we can now analyze it using the *missingno* library. To install this library, simply go `pip install missingno` in Terminal. Missingno provides a visual way of looking at the dataset column by column to look for the data density -- looks like a lot of missing data! Explore the same for the categorical and the date columns and see what you find.

![Image of Data Sparsity] 
(https://github.com/stefannatu/Handling-large-imbalanced-data/blob/master/Images/datasparsity.png?raw=True)

As you can see the dataset is really, really sparse, and we need to perform imputation. Moreover, for the sci-kit learn models, we can throw out several columns that hardly have any data and the following function does so.

```python
def get_reduced_dataset(dataset, threshold):
    cols = dataset.columns
    size = dataset.shape
    reduced_dataset = pd.DataFrame()
    for i, col in enumerate(cols):
        if dataset[cols[i]].count()/size[0] < threshold:
            pass
        else:
            reduced_dataset[col] = dataset[cols[i]]
    return reduced_dataset       
 ```
For a threshold of 0.5 (50% of the columsn are not empty), the reduced dataset only contains 158 columns down from 970! The categorical dataset is much much worse in terms of sparsity and you will have to set a way lower threshold.

Looking at the reduced data now we see it looks much better. 

![Image of Data Sparsity] (https://github.com/stefannatu/Handling-large-imbalanced-data/blob/master/Images/reduced_dataset.png?raw=True)

Next we impute the 158 columns with the means (or zeros) using the ```fillna``` command. The following plot shows a few column distributions after imputation and you see that the distributions are rather skewed. 

![Image of Data Distributions] (https://github.com/stefannatu/Handling-large-imbalanced-data/blob/master/Images/distributions.png?raw=True)

## Feature Engineering 

This skew is typical of much of advertising, impressions based data and merits some feature engineering. A natural approach is to deskew the data using a log transformation. In our study, we found that the data distribution is so highly skewed that the log transformation had a limited effect. 

A second approach to feature engineering when skewness is present is to simple remove the outliers from the data by setting a threshold. We didn't try this approach as it was important for us to keep as many datapoints as possible given the extreme class imbalance.

Of all the 158 columns, we can also plot which ones are more strongly correlated to the Response column using the usual Pearson correlation (absolute value). Even before doing any machine learning, its useful to know which columns are important to retain as they might be the dominant features in a good predictive model.

![Most important feature columns] (https://github.com/stefannatu/Handling-large-imbalanced-data/blob/master/Images/correlations.png?raw=True)

# Model Training in Scikit-Learn - Trading off accuracy versus precision/recall

From the skewness of the data, it's clear that a linear regression model may not work as well. One approach would be to try and deskew the data using a log function or try a logistic regression classifier. An alternative is to train a non parametric model such as a random forest or XGBoost, which are much better at handling data with tail distributions. To avoid overfitting, we split the training data into a training and holdout set.

First split the data into train and holdout. Scale the training data using a standard/minmax scaler. 

```python
X_train, X_test, y_train, y_test = train_test_split(imputed_dataset, labels, test_size = 0.2)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```
Having the training and holdout splits, we train a Random Forest model using GridSearch. A main advantage of sci-kit learn's API is that you can train a Grid Search to optimize for accuracy, recall or precision. 

Why is training for accuracy not such a good idea here? Given the extremely imbalanced dataset, a model which guessed "Not Faulty" 100% of the time would be 99.4% accurate! However such a model would miss a large number of actually faulty parts in the test set, or so called False Negatives and that would be a disaster. Similarly a model which always predicted faulty, would never miss a faulty part, however the overhead of checking and rechecking the 99.6% of the parts that are actually good would be too much additional cost.

When applied to impressions data, a model that always predicted that a particular impression does not lead to a conversion is likely to be extremely accurate, but entirely misses the purpose of finding the features that correlate with actual conversions. 

Training for accuracy is thus not the best idea. Luckily, sci-kit learn's GridSearch allows one to set a parameter to train on, such as accuracy, precision, recall or even a custom functon such as a Matthews Correlation, which Kaggle used to judge the competition winners. To see this, the code below trains a model to optimize for Recall using the *scoring* parameter, and runs it on all the available nodes on the computer (*by seting n_jobs = -1).

```python3

from sklearn.model_selection import GridSearchCV

param_grid = {
    'bootstrap': [True],
    'max_depth': [2, 4, 8, 16],
    'min_samples_leaf': [1, 5, 10],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [1, 10, 100, 200]
}
# Create a based model
rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2, scoring = 'recall'
                          )

grid_search.fit(X_train_scaled, y_train)
```
The best model has a training accuracy of over 98% but a holdout accuracy of only 91%. The reason for this skew is that by growing deeper trees, the random forest can badly overfit. However, in this situation, we are not optimizing for accuracy but rather for recall, and the model has a precision and recall score of about 0.89 and 0.92 on the holdout data which is very good. 

We also fit a Logistic Regression using an L1 penalty (given that there are several correlated features). As expected, the model doesn't perform well with a training accuracy of only 60% and a validation accuracy of 60%, but it overfits less than the random forest, because of the smaller train/validation skew. Finally we also fit an XGBoost Model, but the random forest outperforms them all because it is optimized using the grid search.

The ROC Curves are shown here:

![ROC Curve] (https://github.com/stefannatu/Handling-large-imbalanced-data/blob/master/Images/Model_comparison.png?raw=True)

The true test of the models will therefore be on the actual test data, which doesn't contain any upsampling of the defective class. The test data we use contains 200K rows of which only 1222 are defective. 

The random forest performance is the best with a recall score of 91%, but extremely low precision (0.05). The overall accuracy is 90% which is pretty high. 

The non-normalized confusion matrix for the Random forest on the test dataset is shown here:

![Confusion Matrix] (https://github.com/stefannatu/Handling-large-imbalanced-data/blob/master/Images/confusion_matrix.png?raw=True)

The above numbers illustrate the value of creating a simple MVP (minimum viable product) model trained on a subset of the data. We naturally expect the model to do better if given more training data, and reduce the train, holdout, validation skew; but as is the model correctly predicts over a 1000 of the defective parts in a 200000 row dataset. However this also comes at a cost: the precision is extremely low, and 20K (almost 10%) of the parts would be called out as defective when they're actually not defective. The business must decide if the additional cost of retesting these parts is still favorable.

In advertising data for example, if the model incorrectly predicts cookies who are unlikely to convert as likely to convert, the business might spend advertising money on these cookies. Whether to optimize for precision or recall therefore is strongly driven by the particular use case, and having a simple model which can be tuned to optimize for one or the other is therefore extremely handy. 

### Including the Categorical data

The categorical dataset is sparser still. In order to deal with this, we introduced a threshold, only retaining the columns which had a fraction greater than 0.001 of the columns populated. Out of the 3000 or so categorical columns, only 30 met this criterion, which illustrates how sparse the data is. 

We embed the categorical columns using a LabelEncoder which simply matches the categorical columns to numbers using a dict {'T1': 1 , 'T2': 2, etc.}. A downside of this method is that this encoding can lead to errors as T1 is not actually < T2, but after the encoding it is. However this approach avoids the explosion of features which occurs when you one-hot-encode categorical features, which can be advantageous when training models with already many features. 

It remains as future work to compare the performance between Label and One-hot-encoding, we refer the reader to this post on the subject (https://medium.com/@contactsunny/label-encoder-vs-one-hot-encoder-in-machine-learning-3fc273365621)

```python3
def get_dense_columns(data, threshold):
    ''' extracts the dense(r) columns from the dataset'''
    relevant_cols = []
    for column in data.columns:
        if len(data[column].value_counts())/data.shape[1] < threshold:
            pass
        else:
            relevant_cols.append(column)
    return relevant_cols
```

### Main Takeaways:

1) Scikit learn offers some nice APIs to build models on small scale versions of the data that easily fit into memory. More importantly, it allows you to build benchmark models that train quickly, preprocess data, and play with trading off recall and precision. 

2) A downside of this approach is that the model has not seen the entire training data, which is 20X larger than the 50K samples we used here. The extreme sparsity of the features means that the model has not used all the information possible in order to produce results, which could be an issue if this model were to be put into production without training on the entire dataset. One evidence for this is the large skew in training, holdout and test accuracy. The best way to reduce this and prevent overfitting is to train the model on more data. This requires either using Spark to load and train models in batches using the spark MLLib libraries, or Tensorflow.

3) Although we don't do this here, if we had the relevant domain knowledge, feature engineering is a major way to improve on traditional statistical machine learning models. However feature engineering is very much problem specific: the features that make sense to advertising data don't make sense for sensor data. An upside of deep networks is the ability to generalize and learn features without having to engineer them by hand. Nonethess traditional statistical models with engineered features tend to be highly performant, so the DNN's are not a silver bullet, as we see below. 

# Out-of-the-box Tensorflow models at scale

Tensorflow's Dataset API offers a new way to build data pipelines to perform machine learning at scale. Moreover, the out-of-the-box deep learning neural network models make it easy to train, save, checkpoint and validate models. In this post, we'll describe some pros and cons of this approach compared to the usual data scientist approach of training small scale models in Pandas and then moving on to PySpark or some other production language to put the models into production.

### The main Takeaways of this section are:

1) The Dataset API makes it easy to load in data, a few lines at a time and build a pipeline to perform feature engineering, imputation etc at scale. You dont need to load the data into memory, the pipeline only reads in the necessary data, allowing you to easily train models on datasets that don't fit in memory.

2) A standard pipeline means that it can be readily deployed in different scenarios and for different datasets without much change to the code.

3) The out-of-the-box models allow you to train deep neural networks or wide and deep neural networks with relative ease, and these can be trained locally on a CPU or on GPUs.

4) Unfortuantely the out of the box models don't make it easy to train for accuracy versus precision/recall, something that is straightforward with *GridSearchCV* in scikit-learn. This is because the out-of-the-box models don't allow you to specify custom loss functions, which are needed to trade off sensitivity versus specificity.  

5) We would recommend always training a small model using pandas and sci-kit learn to gain intuition, understand the data before moving on to Tensorflow to achieve scale.

## Pipeline

Our pipeline is as follows: first we shard the categorical and numerical data, and upsample the defective class to 33% in each sharded file. Of all the categorical columns, we only retain 30 or so as explained above. 

```python3
relevant_cols = get_dense_columns(test_cat_col, 0.001)
def get_sharded_full_datasets(full_data_path, full_cat_data_path, relevant_cols):
    ''' Training dataset combining all numerical and 31 categorical columns. Upsampled'''
    chunksize = 50000
    for chunk1, chunk2 in zip(pd.read_csv(full_data_path, chunksize=chunksize)
                              , pd.read_csv(full_cat_data_path, chunksize=chunksize, usecols = relevant_cols, dtype = str)):
        print("Chunk Number: {}".format(i))
        chunk1 = chunk1.drop(columns = ['Id'])
        chunk2 = chunk2.drop(columns = ['Id'])
        full_chunk  = pd.concat([chunk1, chunk2], axis = 1)
        temp_df = full_chunk[full_chunk['Response'] == 1]
        print(len(temp_df))
        pos_df = resample(temp_df, n_samples=25000,random_state=42,replace=True)
        full_df = shuffle(pd.concat([full_chunk, pos_df]), random_state = 42)
        print(len(full_df))
        full_df.to_csv(os.path.dirname(full_data_path) + "/entire_data_50k_{}.csv".format(i),
                    index=False, header = False) # make sure additional index doens't get added and drop Id column
    return print("Done")  
```

Next we use the dataset API to read in rows at a time set by the BATCH_SIZE and assign defaults to null values in the columns. For the categorical variables, we simply use the standard default character 'a'.

Next we use the tf.data.Dataset API to load the data batches at a time. We found that after sharding the data, it is best to not use headers for the column names, otherwise Tensorflow has trouble reading in the shuffled rows. The *skip(1)* feature doesn't work as expected, atleast in our analysis, although there are a few cases of where it works on StackOverFlow.

Next we convert the numeric columns to Tensorflow's *feature_column.numeric_column* and one-hot-encode the categorical columns using *tf.feature_column.categorical_column_with_vocabulary_list* and tf.feature_column.indicator_column. It may be useful to try to convert the vocabulary_list columns into embedding_columns which try to learn higher level representations of the categorical variables. This is particularly useful for wide and deep neural networks which we also implement. It remains to be seen if embedding columns perform better with sparse data compared to simple one-hot-encoding the columns.

For the wide and deep neural network, we input the sparse numerical and all the categorical columns into the deep and the dense numerical columns into the wide linear classifier. For save all the numerical column names in a file called col_names.txt and the deep columns as col_names_deep.txt. The wide columns are the complement of the deep columns from col_names.txt.

``` python3
def read_dataset(filename, mode, batch_size = 512):
    def _input_fn():
        def decode_csv(value_column):
            columns = tf.decode_csv(value_column, record_defaults = DEFAULTS)
            features = dict(zip(CSV_COLUMNS, columns))
            label = features.pop(LABEL)
            #for c in COLS_TO_DROP:
            #    features.pop(c) # drop the index and the Id columns
            return features, label

    # Create list of files that match pattern
        file_list = tf.gfile.Glob(filename)

    # Create dataset from file list
        dataset = tf.data.TextLineDataset(file_list).skip(1)
        dataset = dataset.map(decode_csv)
        if mode == tf.estimator.ModeKeys.TRAIN:
            num_epochs = None # indefinitely
            dataset = dataset.shuffle(buffer_size = 10*batch_size)
        else:
            num_epochs = 1 # end-of-input after this

        dataset = dataset.repeat(num_epochs).batch(batch_size)
        return dataset.make_one_shot_iterator().get_next()
    return _input_fn
    

def get_train():
    return read_dataset(PATH, mode = tf.estimator.ModeKeys.TRAIN)

def get_valid():
    return read_dataset(TEST_PATH, mode = tf.estimator.ModeKeys.EVAL)

INPUT_COLUMNS = []
for col in tot_cols:
    INPUT_COLUMNS.append(tf.feature_column.numeric_column(col))
    
def make_new_features(feats):
    ### Add new features if needed -- this makes more sense for the temporal columns 
    ### which I don't include here or if the features actually meant something physically. 
    ### Forget for now
    return feats


WIDE_COLUMNS = []
for col in wide_cols:
    WIDE_COLUMNS.append(tf.feature_column.numeric_column(col))
    

DEEP_COLUMNS = []
for col in deep_cols:
    DEEP_COLUMNS.append(tf.feature_column.numeric_column(col))
```
An advantage of this pipeline is that since the deep neural network doesn't require feature engineering and raw data can just be fed into the model, one can use the same code with little changes for different use cases, datasets etc.

## Modeling

We implement both a regular deep neural network using *DNNClassifier* and a wide and deep neural network using *DNNLinearCombinedClassifier*. 

```python3
def models(model_dir, hidden_units = [128, 64, 32], dropout = 0.1, WDNN = False):
    ''' Builds a Regular DNN classifier or a Wide Deep Neural Network'''
    if not WDNN:
        classifier = tf.estimator.DNNClassifier(
        model_dir = model_dir,
        feature_columns=INPUT_COLUMNS,
        hidden_units=hidden_units,
        optimizer=tf.train.AdamOptimizer(1e-4),
        n_classes=2,
        loss_reduction=tf.losses.Reduction.MEAN,
        dropout=0.5,
        config=tf.estimator.RunConfig(tf_random_seed = 42))
        
    else:
        classifier = tf.estimator.DNNLinearCombinedClassifier(
        model_dir=model_wdnn_dir,
        linear_feature_columns=DEEP_COLUMNS,
        dnn_feature_columns=WIDE_COLUMNS,
        dnn_hidden_units=hidden_units,
        n_classes=2,
        loss_reduction=tf.losses.Reduction.MEAN,
        config=tf.estimator.RunConfig(tf_random_seed = 42))

    return classifier    
 
model = models(model_dir)
start = time.time()
model.train(input_fn = get_train(), max_steps = 10000)
end = time.time()
```

To compare both models, and to save training time, we train the neural network models on a 50K subset of the data obtained by sharding the full dataset. However the pipeline above is completely general and will work as implemented on the entire training dataset.

'accuracy': 0.9512790255805116, 
'Confusion_Matrix': array([[47333,  2399],
                          [   37,   230]]), 
'F1 score': 0.15883977900552487 
'Recall Score': 0.8614232209737828 
'Precision Score': 0.08748573602130087

and for the wide-and-deep network: 

'accuracy': 0.9938, 
'Confusion_Matrix': array([[49690,     3],
                          [  307,     0]])
'F1 score': 0.0 
'Recall Score': 0.0
'Precision Score': 0.0

So while the deep neural network improves the precision at the cost of recall, the accuracy is much higher than the random forest model on the test dataset. The wide-deep NN however performs poorly, completely missing *all* the faulty examples.

As mentioned above the APIs are excellent for getting a model up and running, but they don't allow you to change much in terms of customizing loss functions or adding your own modifications to the model. But it is easy enough to build custom models in Tensorflow. We learned this the hard way by initially training a model without upsampling. After several hours of training over multiple epochs, the DNNClassifier achieves 99.4% accuracy by calling all test samples as non-defective. Quite disappointing! The API is by no means a silver bullet.

One way around this is to use weights to penalize the model whenever it incorrectly predicts a false negative during training. However if you make the weights too large, then the model goes the other way and classifies all parts as faulty. For models where training times take a while, this method is not effective as you have to tune the weight as a hyperparameter to achieve best model performance.

An alternate approach is to oversample the positive class, and train the model for accuracy instead of recall. Then test the model on the test data which has the original distribution of the two classes. This avoids any leakage and hopefully the model has seen the faulty class enough times during training to correctly predict it during test time. A weakness of this approach is that the model might overfit to achieve near perfect training accuracy but poor generalizaton performance. Having a holdout set to test the model on is therefore key. 

# Conclusions

In conclusion we have presented a study comparing scikit-learn and Tensorflow's out-of-the-box deep learning models (DNNClassifier and DNNLinearCombinedClassifier) on a highly sparse, relatively large and highly imbalanced dataset. Such data is common for website impressions, advertising data, and can be used to build personalized recommendation systems, calculate propensity of conversion of a user from browing to purchasing etc (upper funnel to lower funnel), and is therefore of great business value. 

We found that while the deep neural network performs the best, but it is not a silver bullet. The power of the DNN comes from its ability to scale to large data and learn representations among sparse vectors. TensorFlows Dataset API now makes it easy to build a pipeline to ingest data into the model for training and testing. In particular, the inability to use custom loss functions can be an issue for highly imbalanced data which has been pointed out in other Stack overflow posts. The solution ofcourse is to build a custom model but that requires hiring trained data scientists/ML engineers.

In our opinion, the Tensorflow APIs aren't a substitute for scikit learn: while Tensorflow now makes it really easy to scale models and has the advantage of generalizing to other models, datasets with relatively little changes to the code, there is no substitute for exploring data, engineering featues and creating simple benchmark models and testing business intuition such as the relative importance of precision versus recall.

Somewhat surprisingly the wide and deep network doesn't perform well at all. This is probably because even the dense numerical columns aren't actually quite dense and a linear, rule based model is not the best way to capture the information contained in them. The numerical columns have far from Gaussian distributions, so a linear model is not the best model. Thus this model may not be able to capture the details in the dense columsn which are needed to achieve any performance at all, since the data is so sparse to begin with. Further experimentation is required to come up with a more satisfactory answer for why this doesn't work as well.
