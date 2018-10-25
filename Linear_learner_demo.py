#!/usr/bin/env python
# coding: utf-8

# In[3]:



import pandas as pd 
import hashlib
import os 
#from utils import logger
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np


from sklearn.feature_selection import SelectFromModel
from sklearn import datasets
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

#from utils import logger
#def lassoSelection(X,y,)
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
import time

def plot_prediction_quality(scores, ks):
    colors = ['r-', 'b-', 'g-','y-'][:len(scores)]
    for (k,v), color in zip(scores.items(), colors):
        plt.plot(ks, v, color, label=k)
    plt.legend()
    plt.xlabel('k')
    plt.ylabel('prediction quality')
    plt.show()

def evaluate_quality(predictor, test_features, test_labels, model_name, verbose=True, num_batches=1):
    """
    Evaluate quality metrics of a model on a test set. 
    """
    # tune the predictor to provide the verbose response
    predictor.accept = 'application/json; verbose=true'
    
    # split the test data set into num_batches batches and evaluate using prediction endpoint. 
    print('running prediction (quality)...')
    batches = np.array_split(test_features, num_batches)
    knn_labels = []
    for batch in batches:
        pred_result = predictor.predict(batch)
        cur_knn_labels = np.array([pred_result['predictions'][i]['labels'] for i in range(len(pred_result['predictions']))])
        knn_labels.append(cur_knn_labels)
    knn_labels = np.concatenate(knn_labels)
    print('running prediction (quality)... done')
    print(knn_labels)
    print(test_labels)
    # figure out different k values
    top_k = knn_labels.shape[1]
    ks = range(1, top_k+1)
    
    # compute scores for the quality of the model for each value of k
    print('computing scores for all values of k... ')
    quality_scores = scores_for_ks(test_labels, knn_labels, ks)
    print('computing scores for all values of k... done')
    if verbose:
        plot_prediction_quality(quality_scores, ks)
    
    return quality_scores

def evaluate_latency(predictor, test_features, test_labels, model_name, verbose=True, num_batches=1):
    """
    Evaluate the run-time of a model on a test set.
    """
    # tune the predictor to provide the non-verbose response
    predictor.accept = 'application/json'
    
    # latency for large batches:
    # split the test data set into num_batches batches and evaluate the latencies of the calls to endpoint. 
    print('running prediction (latency)...')
    batches = np.array_split(test_features, num_batches)
    test_preds = []
    latency_sum = 0
    for batch in batches:
        start = time.time()
        pred_batch = predictor.predict(batch)
        latency_sum += time.time() - start
    latency_mean = latency_sum / float(num_batches)
    avg_batch_size = test_features.shape[0] / num_batches
    
    # estimate the latency for a batch of size 1
    latencies = []
    attempts = 128
    for i in range(attempts):
        start = time.time()
        pred_batch = predictor.predict(test_features[i].reshape((1,-1)))
        latencies.append(time.time() - start)

    latencies = sorted(latencies)
    latency1_mean = sum(latencies) / float(attempts)
    latency1_p90 = latencies[int(attempts*0.9)]
    latency1_p99 = latencies[int(attempts*0.99)]
    print('running prediction (latency)... done')
    
    if verbose:
        print("{:<11} {:.3f}".format('Latency (ms, batch size %d):' % avg_batch_size, latency_mean * 1000))
        print("{:<11} {:.3f}".format('Latency (ms) mean for single item:', latency1_mean * 1000))
        print("{:<11} {:.3f}".format('Latency (ms) p90 for single item:', latency1_p90 * 1000))
        print("{:<11} {:.3f}".format('Latency (ms) p99 for single item:', latency1_p99 * 1000))
        
    return {'Latency': latency_mean, 'Latency1_mean': latency1_mean, 'Latency1_p90': latency1_p90, 
            'Latency1_p99': latency1_p99}

def evaluate(predictor, test_features, test_labels, model_name, verbose=True, num_batches=100):
    eval_result_q = evaluate_quality(pred, test_features, test_labels, model_name=model_name, verbose=verbose, num_batches=num_batches)
    eval_result_l = evaluate_latency(pred, test_features, test_labels, model_name=model_name, verbose=verbose, num_batches=num_batches)
    return dict(list(eval_result_q.items()) + list(eval_result_l.items()))

def lassoSelection(X_train, y_train, n):
    '''
    Lasso feature selection.  Select n features. 
    '''
    #lasso feature selection
    #print (X_train)
    clf = LassoCV()
    sfm = SelectFromModel(clf, threshold=0)
    sfm.fit(X_train, y_train)
    X_transform = sfm.transform(X_train)
    n_features = X_transform.shape[1]
    
    #print(n_features)
    while n_features > n:
        sfm.threshold += 0.01
        X_transform = sfm.transform(X_train)
        n_features = X_transform.shape[1]
    features = [index for index,value in enumerate(sfm.get_support()) if value == True  ]
    #logger.info("selected features are {}".format(features))
    return features


def specificity_score(y_true, y_predict):
    '''
    true_negative rate
    '''
    true_negative = len([index for index,pair in enumerate(zip(y_true,y_predict)) if pair[0]==pair[1] and pair[0]==0 ])
    real_negative = len(y_true) - sum(y_true)
    return true_negative / real_negative 

def delete_endpoint(predictor):
    try:
        boto3.client('sagemaker').delete_endpoint(EndpointName=predictor.endpoint)
        print('Deleted {}'.format(predictor.endpoint))
    except:
        print('Already deleted: {}'.format(predictor.endpoint))







# In[29]:


def trained_estimator_from_hyperparams(s3_train_data, hyperparams, output_path, s3_test_data=None):
    """
    Create an Estimator from the given hyperparams, fit to training data, 
    and return a deployed predictor
    
    """
    # specify algorithm containers. These contain the code for the training job
    containers = {
        'us-west-2': '174872318107.dkr.ecr.us-west-2.amazonaws.com/xgboost:latest',
        'us-east-1': '382416733822.dkr.ecr.us-east-1.amazonaws.com/xgboost:latest',
        'us-east-2': '404615174143.dkr.ecr.us-east-2.amazonaws.com/xgboost:latest',
        'eu-west-1': '438346466558.dkr.ecr.eu-west-1.amazonaws.com/xgboost:latest',
        'ap-northeast-1': '351501993468.dkr.ecr.ap-northeast-1.amazonaws.com/xgboost:latest',
        'ap-northeast-2': '835164637446.dkr.ecr.ap-northeast-2.amazonaws.com/xgboost:latest',
        'ap-southeast-2': '712309505854.dkr.ecr.ap-southeast-2.amazonaws.com/xgboost:latest'
    }
    # set up the estimator
    xgboost = sagemaker.estimator.Estimator(containers[boto3.Session().region_name],
        get_execution_role(),
        train_instance_count=1,
        train_instance_type='ml.m5.2xlarge',
        output_path=output_path,
        sagemaker_session=sagemaker.Session())
    xgboost.set_hyperparameters(**hyperparams)
    
    # train a model. fit_input contains the locations of the train and test data
    fit_input = {'train': s3_train_data}
    if s3_test_data is not None:
        fit_input['test'] = s3_test_data
    xgboost.fit(fit_input)
    return xgboost


# In[30]:


def predictor_from_hyperparams(knn_estimator, estimator_name, instance_type, endpoint_name=None): 
    knn_predictor = knn_estimator.deploy(initial_instance_count=1, instance_type=instance_type,
                                        endpoint_name=endpoint_name)
    knn_predictor.content_type = 'text/csv'
    knn_predictor.serializer = csv_serializer
    knn_predictor.deserializer = json_deserializer
    return knn_predictor


# In[31]:


def highlight_apx_max(row):
    '''
    highlight the aproximate best (max or min) in a Series yellow.
    '''
    max_val = row.max()
    colors = ['background-color: yellow' if cur_val >= max_val * 0.9975 else '' for cur_val in row]
        
    return colors
def highlight_far_from_min(row):
    '''
    highlight the aproximate best (max or min) in a Series yellow.
    '''
    med_val = row.median()
    colors = ['background-color: red' if cur_val >= med_val * 1.2 else '' for cur_val in row]
        
    return colors


# In[11]:




if __name__ == '__main__':


    # Prereqs
    from sagemaker import get_execution_role
    import boto3, re, sys, math, json, os, sagemaker, urllib.request
    from sagemaker import get_execution_role
    import numpy as np                                
    import pandas as pd                               
    import matplotlib.pyplot as plt                   
    from IPython.display import Image                 
    from IPython.display import display               
    from time import gmtime, strftime                 
    from sagemaker.predictor import csv_serializer  
    # Define IAM role 
    role = get_execution_role()

    #prefix = 'sagemaker/DEMO-xgboost-dm'

    my_region = boto3.session.Session().region_name # set the region of the instance
    #print("Success - the MySageMakerInstance is in the " + my_region + " region. You will use the " + containers[my_region] + " container for your SageMaker endpoint.")

    bucket = 'cancer-bucket-1234demo' # <--- change this variable to a unique name for your bucket
    s3 = boto3.resource('s3')
    try:
        if  my_region == 'us-east-1':
          s3.create_bucket(Bucket=bucket)
        else: 
          s3.create_bucket(Bucket=bucket, CreateBucketConfiguration={ 'LocationConstraint': my_region })
        print('S3 bucket created successfully')
    except Exception as e:
        print('S3 error: ',e)

    

    cwd = os.getcwd()
    data_dir = cwd + "/"

    data_file = data_dir + "miRNA_matrix.csv"

    df = pd.read_csv(data_file)
    # print(df)
    y_data = df.pop('label').values

    df.pop('file_id')

    columns =df.columns
    #print (columns)
    X_data = df.values

    # split the data to train and test set
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=0)
    

    #standardize the data.
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    le = preprocessing.LabelEncoder()
    le.fit(y_data)
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)

    # check the distribution of tumor and normal sampels in traing and test data set.
    #logger.info("Percentage of tumor cases in training set is {}".format(sum(y_train)/len(y_train)))
    #logger.info("Percentage of tumor cases in test set is {}".format(sum(y_test)/len(y_test)))
    
    n = 7
    features_columns = lassoSelection(X_train, y_train, n)
    
    
    import io
    import sagemaker.amazon.common as smac
    import boto3
    import sagemaker  
    from sagemaker import get_execution_role
    from sagemaker.predictor import csv_serializer, json_deserializer
    #bucket =  "cancer-bucket"
    prefix = 'linear-learner'
    key = 'cancer-data'
    
    #write the train data to S3
    buf = io.BytesIO()
    smac.write_numpy_to_dense_tensor(buf, X_train[:,features_columns], y_train)
    buf.seek(0)   
    boto3.resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'train', key)).upload_fileobj(buf)
    s3_train_data = 's3://{}/{}/train/{}'.format(bucket, prefix, key)    
    print('uploaded training data location: {}'.format(s3_train_data)) 
    print(X_train[:,features_columns].shape)

    #write the test data to S3
    buf = io.BytesIO()
    smac.write_numpy_to_dense_tensor(buf, X_test[:,features_columns], y_test)
    buf.seek(0)
    boto3.resource('s3').Bucket(bucket).Object(os.path.join(prefix, 'test', key)).upload_fileobj(buf)
    s3_test_data = 's3://{}/{}/test/{}'.format(bucket, prefix, key)
    print('uploaded test data location: {}'.format(s3_test_data))    
    print(X_test[:,features_columns].shape)
    X_test = np.float32(X_test)
    X_train = np.float32(X_train)


    # In[13]:


    y_test = np.float32(y_test)
    y_train = np.float32(y_train)
    multiclass_estimator = sagemaker.LinearLearner(role=sagemaker.get_execution_role(),
                                                   train_instance_count=1,
                                                   train_instance_type='ml.m4.xlarge',
                                                   predictor_type='multiclass_classifier',
                                                   num_classes=len(le.classes_))
    train_records = multiclass_estimator.record_set(X_train, y_train, channel='train')
    test_records = multiclass_estimator.record_set(X_test, y_test, channel='test')
    multiclass_estimator.fit([train_records, test_records])


    # In[14]:



    multiclass_predictor = multiclass_estimator.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge')
    prediction_batches = [multiclass_predictor.predict(batch) for batch in np.array_split(X_test, 100)]


    # In[44]:


    def specificity_score(y_true, y_predict, labels):
        '''
        true_negative rate
        '''
        classes = np.float32(labels)
        true_negative = 0
        false_positive = 0
        for label in classes:
            true_negative += len([index for index,pair in enumerate(zip(y_true,y_predict)) if pair[1]!=label and pair[0]!=label])
            false_positive += len([index for index,pair in enumerate(zip(y_true,y_predict)) if pair[1]==label and pair[0]!=label])
        return true_negative / (true_negative + false_positive)


    # In[45]:


    # parse protobuf responses to extract predicted labels
    extract_label = lambda x: x.label['predicted_label'].float32_tensor.values
    test_preds = np.concatenate([np.array([extract_label(x) for x in batch]) for batch in prediction_batches])
    f1_score = f1_score(y_test, test_preds, average='weighted')
    accuracy = accuracy_score(y_test, test_preds)
    recall = recall_score(y_test, test_preds, average='weighted')
    precision = precision_score(y_test, test_preds, average='weighted')
    specificity = specificity_score(y_test, test_preds, le.transform(le.classes_))
    print('F1 score = ' + str(f1_score))
    print('Accuracy = ' + str(accuracy))
    print('Sensitivity = ' + str(recall))
    print('Precision = ' + str(precision))
    print('Specificity = ' + str(specificity))


    # In[37]:


    from sklearn.decomposition import PCA

    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(df.values)

    df['pca-one'] = pca_result[:,0]
    df['pca-two'] = pca_result[:,1] 
    df['pca-three'] = pca_result[:,2]

    print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))


    # In[47]:





    # In[81]:




    #rndperm = np.random.permutation(df.shape[0])


    #from ggplot import *



    chart = plt.scatter(df['pca-one'].loc[:2999], df['pca-two'].loc[:2999], c=le.transform(y_data[:3000]), s=3)
    plt.title('First and Second Principal Components colored by digit')
    plt.xlabel('pca-one')
    plt.ylabel('pca-two')
    plt.show()


    # In[73]:


    import time

    from sklearn.manifold import TSNE

    n_sne = 7000

    time_start = time.time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    tsne_results = tsne.fit_transform(df.iloc[0:n_sne])

    print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))


    # In[78]:


    df_tsne = df.loc[7000].copy()
    df_tsne['x-tsne'] = tsne_results[:,0]
    df_tsne['y-tsne'] = tsne_results[:,1]

    chart = plt.scatter(df_tsne['x-tsne'][:n_sne], df_tsne['y-tsne'][:n_sne], c=le.transform(y_data[:n_sne]), s=0.8)
    plt.title('tSNE dimensions colored by Digit')
    plt.xlabel('x-tsne')
    plt.ylabel('y-tsne')
    plt.show()


    # In[88]:





    # In[124]:


    true_negative_rate = []
    fpr = []
    true_negative = 0
    false_positive = 0 
    y_true = y_test
    y_predict = test_preds

    for label in le.transform(le.classes_):
        true_negative = len([index for index,pair in enumerate(zip(y_true,y_predict)) if pair[1]!=label and pair[0]!=label])
        false_positive = len([index for index,pair in enumerate(zip(y_true,y_predict)) if pair[1]==label and pair[0]!=label])
        false_positive_rate = false_positive/(false_positive+true_negative)
        if(false_positive != 0):
            fpr.append(false_positive/(false_positive+true_negative))


    #all_fpr = np.unique(fpr)
    recall2 = recall_score(y_test, test_preds, average=None)


    # In[140]:


    len(fpr)


    # In[146]:


    recall2.shape


    # In[147]:


    roc_char = plt.plot(fpr[:51], recall2[:51])


    # In[ ]:





# In[ ]:




