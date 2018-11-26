import pandas as pd
import numpy as np
import lightgbm as lgb
import gc
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv('train_sample.csv')
train_temp = train
train,test = train_test_split(train,test_size=0.33)

def get_size(group_column, df):

    used_cols = group_column.copy()
    used_cols.extend(['row_id'])
    temp_df = df[used_cols]
    grouped = temp_df[group_column].groupby(group_column)
    
    #size of each level in group_column
    the_size = pd.DataFrame(grouped.size()).reset_index()
    labels = group_column.copy()
    new_label = "_".join(x for x in labels) + '_size'
    labels.append(new_label)
    the_size.columns = labels
    
    temp_df = pd.merge(temp_df, the_size)
    temp_df.sort_values('row_id', inplace=True)

    df[new_label] = np.array(temp_df[new_label])
    del temp_df
    gc.collect()
    
    return df

def get_unique(df, grouping_col, target_col):
    
    used_cols = grouping_col.copy()
    used_cols.extend(['row_id'])
    used_cols.extend(target_col)
    temp_df = df[used_cols]
    
    group_used_cols = grouping_col.copy()
    group_used_cols.extend(target_col)
    grouped = temp_df[group_used_cols].groupby(grouping_col)
    #unique count
    the_count = pd.DataFrame(grouped[target_col].nunique()).reset_index()
    labels = grouping_col.copy()
    new_label = "_".join(x for x in target_col) + '_unique_count_on_' + "_".join(x for x in grouping_col)
    labels.append(new_label)
    the_count.columns = labels
    
    temp_df = pd.merge(temp_df, the_count)
    temp_df.sort_values('row_id', inplace=True)

    df[new_label] = np.array(temp_df[new_label])
    del temp_df
    gc.collect()
    
    return df

def modifier(df):
    train = df
    train['row_id'] = range(train.shape[0])
    train.click_time = pd.to_datetime(train.click_time)
    train['day'] = train.click_time.dt.day.astype('uint8')
    train['hour'] = train.click_time.dt.hour.astype('uint8')
        
    most_freq_hours_in_test_data = [4, 5, 9, 10, 13, 14]
    least_freq_hours_in_test_data = [6, 11, 15]    
    train['in_test_hh'] = (3 - 2*train['hour'].isin(most_freq_hours_in_test_data) - 1*train['hour'].isin(least_freq_hours_in_test_data)).astype('uint8')

    ###### get user information ######
    user = ['ip', 'device', 'os']
    train = get_size(user, train)
    new_label1 = "_".join(x for x in user) + '_size'
    #get total count on an app
    train = get_size(['ip', 'app', 'device', 'os'], train)
    new_label2 = "_".join(x for x in (['ip', 'app', 'device','os'])) + '_size'
    #get proportion of app count on this user
    new_label3 = new_label2 + '/' + new_label1
    train[new_label3] =train[new_label2]/train[new_label1]
    #get unique app count
    train = get_unique(train, user, ['app'])
    new_label4 = "_".join(x for x in ['app']) + '_unique_count_on_' + "_".join(x for x in user)
    #get unique/size ratio
    new_label5 = new_label4 + '/' + new_label1
    train[new_label5] = train[new_label4]/train[new_label1]
    train['user_Nth_time'] = train.groupby(user).cumcount() 
    train['user_app_Nth_time'] = train.groupby(user + ['app']).cumcount(ascending=False)
    print('get user information done')

    ###### get kernel feature ######
    train = get_size(['ip','day','in_test_hh'], train)
    train = get_size(['ip','day','hour'], train)
    train = get_size(['ip','day','hour','app'], train)
    train = get_size(['ip','day','hour','app','os'], train)
    train = get_size(['ip','app'], train)
    train = get_size(['ip','app','os'], train)
    train = get_size(['ip','device'], train)
    train = get_size(['app','channel'], train)
    print('get kernel feature done')

    ###### get hourly app info ######
    app_hourly = ['app', 'day', 'hour']
    train = get_size(app_hourly, train)
    train = get_unique(train, app_hourly, ['ip'])
    new_label1 = "_".join(x for x in ['ip']) + '_unique_count_on_' + "_".join(x for x in app_hourly)
    new_label2 = "_".join(x for x in app_hourly) + '_size'
    new_label3 = new_label1 + '/' + new_label2
    train[new_label3] = train[new_label1]/train[new_label2]
    print('get_hourly_app_info done')

    ###### get click time information ######
    train.click_time = pd.to_datetime(train.click_time, errors = 'ignore')
    used_cols = ['ip', 'app', 'device', 'os', 'click_time', 'row_id']
    temp_df = train[used_cols]
    temp_df = temp_df.sort_values(by=used_cols)
    temp_df['next_ip']=temp_df.ip.shift(-1)
    temp_df['next_app']=temp_df.app.shift(-1)
    temp_df['next_device']=temp_df.device.shift(-1)
    temp_df['next_os']=temp_df.os.shift(-1)
    temp_df['next_click_time']=temp_df.click_time.shift(-1)    
    temp_df['has_next_ip'] = np.where(temp_df.ip == temp_df.next_ip, 1, 0)
    temp_df['has_next_app'] = np.where(temp_df.app == temp_df.next_app, 1, 0)
    temp_df['has_next_device'] = np.where(temp_df.device == temp_df.next_device, 1, 0)
    temp_df['has_next_os'] = np.where(temp_df.os == temp_df.next_os, 1, 0)  
    temp_df['next_click'] = np.where((temp_df.has_next_ip == 1) & (temp_df.has_next_app == 1) &(temp_df.has_next_device == 1) & (temp_df.has_next_os == 1) , (temp_df.next_click_time - temp_df.click_time)/np.timedelta64(1, 's'), np.NaN)

    temp_df['previous_ip']=temp_df.ip.shift(1)
    temp_df['previous_app']=temp_df.app.shift(1)
    temp_df['previous_device']=temp_df.device.shift(1)
    temp_df['previous_os']=temp_df.os.shift(1)
    temp_df['previous_click_time']=temp_df.click_time.shift(1) 
    temp_df['has_previous_ip'] = np.where(temp_df.ip == temp_df.previous_ip, 1, 0)
    temp_df['has_previous_app'] = np.where(temp_df.app == temp_df.previous_app, 1, 0)
    temp_df['has_previous_device'] = np.where(temp_df.device == temp_df.previous_device, 1, 0)
    temp_df['has_previous_os'] = np.where(temp_df.os == temp_df.previous_os, 1, 0)
    temp_df['previous_click'] = np.where((temp_df.has_previous_ip == 1) & (temp_df.has_previous_app == 1) & (temp_df.has_previous_device == 1) & (temp_df.has_previous_os == 1) , (temp_df.click_time-temp_df.previous_click_time)/np.timedelta64(1, 's'), np.NaN)
        
    temp_df = temp_df.sort_values(by=['row_id'])
    train['next_click'] = np.array(temp_df['next_click'])
    train['previous_click'] = np.array(temp_df['previous_click'])
    del temp_df
    gc.collect()
    print('get click time information done')

    ###### get next click stat ######
    grouping_col = ['ip','app','device','os']
    target_col = ['next_click']
    used_cols = grouping_col.copy()
    used_cols.extend(['row_id'])
    used_cols.extend(target_col)
    temp_df = train[used_cols]
    group_used_cols = grouping_col.copy()
    group_used_cols.extend(target_col)
    grouped = temp_df[group_used_cols].groupby(grouping_col)
    new_labels = []
    #mean
    the_mean = pd.DataFrame(grouped[target_col].mean()).reset_index()
    labels = grouping_col.copy()
    new_label = 'next_click_mean'
    new_labels.append(new_label)
    labels.append(new_label)
    the_mean.columns = labels
    #median
    the_median = pd.DataFrame(grouped[target_col].median()).reset_index()
    labels = grouping_col.copy()
    new_label = 'next_click_median'
    new_labels.append(new_label)
    labels.append(new_label)
    the_median.columns = labels
    the_stats = pd.merge(the_mean, the_median)
    #max
    the_max = pd.DataFrame(grouped[target_col].max()).reset_index()
    labels = grouping_col.copy()
    new_label = 'next_click_max'
    new_labels.append(new_label)
    labels.append(new_label)
    the_max.columns = labels
    the_stats = pd.merge(the_stats, the_max)
        
    temp_df = pd.merge(temp_df, the_stats)
    temp_df.sort_values('row_id', inplace=True)
    for new_label in new_labels:
        train[new_label] = np.array(temp_df[new_label])
    del temp_df
    gc.collect()
    print('get next click stat done')

    ###### get hour, minute, second rate on three posibilites ######
    candidates = [        
        ['ip', 'app', 'device', 'os'],
        ['app', 'device', 'os'],
        ['ip', 'app']
    ]
    train['minute'] = train.click_time.dt.minute.astype('uint8')
    train['second'] = train.click_time.dt.second.astype('uint8')
    gc.collect()
    for i in range(0, 3):
        used = candidates[i].copy()
        if i == 0:
            train = get_size(used + ['day', 'hour'], train)
            train = get_size(used + ['day', 'hour', 'minute'], train)
            train = get_size(used + ['day', 'hour', 'minute', 'second'], train)
            train['ip_app_device_os_size_hour/min_rate'] = train['ip_app_device_os_day_hour_size']/train['ip_app_device_os_day_hour_minute_size']
            train['ip_app_device_os_size_min/sec_rate'] = train['ip_app_device_os_day_hour_minute_size']/train['ip_app_device_os_day_hour_minute_second_size']
            dropped = ['ip_app_device_os_day_hour_minute_size', 'ip_app_device_os_day_hour_minute_second_size']
            train.drop(dropped, axis=1, inplace=True)
            gc.collect()
        elif i == 1:
            train = get_size(used + ['day', 'hour'], train)
            train = get_size(used + ['day', 'hour', 'minute'], train)
            train = get_size(used + ['day', 'hour', 'minute', 'second'], train)
            train['app_device_os_size_hour/min_rate'] = train['app_device_os_day_hour_size']/train['app_device_os_day_hour_minute_size']
            train['app_device_os_size_min/sec_rate'] = train['app_device_os_day_hour_minute_size']/train['app_device_os_day_hour_minute_second_size']
            dropped = ['app_device_os_day_hour_minute_size', 'app_device_os_day_hour_minute_second_size']
            train.drop(dropped, axis=1, inplace=True)
            gc.collect()
        elif i == 2:
            train = get_size(used + ['day', 'hour'], train)
            train = get_size(used + ['day', 'hour', 'minute'], train)
            train = get_size(used + ['day', 'hour', 'minute', 'second'], train)
            train['ip_app_size_hour/min_rate'] = train['ip_app_day_hour_size']/train['ip_app_day_hour_minute_size']
            train['ip_app_size_min/sec_rate'] = train['ip_app_day_hour_minute_size']/train['ip_app_day_hour_minute_second_size']
            dropped = ['ip_app_day_hour_minute_size', 'ip_app_day_hour_minute_second_size']
            train.drop(dropped, axis=1, inplace=True)
            gc.collect()
    train.drop(['minute','second'], axis=1, inplace=True)
    print('get hour minute second rate done')

    ###### get ip information ######
    train = get_size(['ip'],train)
    train = get_unique(train, ['ip'], ['os'])
    new_label1 = 'os_unique_count_on_ip' + '/' + 'ip_size'
    train[new_label1] = train['os_unique_count_on_ip']/train['ip_size']

    train = get_unique(train, ['ip'], ['device'])
    new_label2 = 'device_unique_count_on_ip' + '/' + 'ip_size'
    train[new_label2] = train['device_unique_count_on_ip']/train['ip_size']   

    train.drop(['ip_size'], axis=1, inplace=True)
    gc.collect()
    print('get ip information done')

    ###### get other features ######
    trian = get_unique(train, ['ip','device','os','day','hour'], ['channel'])
    trian = get_size(['ip','channel'],trian)
    print('get other features done')

    train = train.sort_values(by=['row_id'])
    return train

train = modifier(train)
test = modifier(test)

###### LightGBM Model ######
def lgb_prediction(train, num_rounds, if_ip=False):
    predictors = list(train.columns)
    remove_list = ['click_id', 'row_id', 'day', 'minute', 'second', 'click_time', 'local_click_time', 'attributed_time', 'is_attributed']
    for element in remove_list:
        if element in predictors:
            predictors.remove(element)
    target = 'is_attributed'
    categorical = ['ip','app','os','device','channel','hour']
    if if_ip == False:
        predictors.remove('ip')
        categorical.remove('ip')
    params = {
        'boosting_type': 'gbdt', 'objective': 'binary', 'nthread': -1, 'silent': True, 'metric':'auc', 'seed':77,
        'num_leaves': 48, 'learning_rate': 0.01, 'max_depth': -1, 'gamma':47,
        'max_bin': 255, 'subsample_for_bin': 70000, 'bagging_fraction':0.7, 'bagging_freq':1, 'bagging_seed':55,
        'colsample_bytree': 0.6548, 'reg_alpha': 19.43, 'reg_lambda': 0, 
        'min_split_gain': 0.3512, 'min_child_weight': 0, 'min_child_samples':1321, 'scale_pos_weight':205}
    
    xgtrain = lgb.Dataset(train[predictors].values, label=train[target].values, feature_name=predictors, categorical_feature=categorical)
    bst = lgb.train(params, xgtrain, num_boost_round = num_rounds, verbose_eval=False)
    return bst, predictors

###### SVM Model ######
def svm_prediction(train, num_rounds, if_ip=False):
    predictors = list(train.columns)
    remove_list = ['click_id', 'row_id', 'day', 'minute', 'second', 'click_time', 'local_click_time', 'attributed_time', 'is_attributed', 'next_click_mean', 'next_click_median', 'next_click_max', 'next_click', 'previous_click']
    for element in remove_list:
        if element in predictors:
            predictors.remove(element)

    target = 'is_attributed'
    if if_ip == False:
        predictors.remove('ip')

    svm_linear = svm.SVC(kernel='rbf', C=2**-5,gamma=2**3, class_weight={1:99773/227})
    svm_linear.fit(train[predictors].values, train[target].values)
    return svm_linear, predictors

### Logistic Regression Model###
def Logistic_dummy(data):
    # create dummy variables
    cat_vars = ['device','os','channel']
    for var in cat_vars:
        cat_list = 'var' + "_" +var
        cat_list = pd.get_dummies(data[var],prefix=var)
        data1 = data.join(cat_list)
        data = data1

    cat_vars = ['device','os','day','hour','channel']
    data_vars = data.columns.values.tolist()
    to_keep=[i for i in data_vars if i not in cat_vars]
    data_final = data[to_keep]
    data_final.columns.values

    data_final_vars = data_final.columns.values.tolist()
    y = ['is_attributed']
    x = [i for i in data_final_vars if i not in y]

    x = data_final.drop(['click_time', 'attributed_time', 'row_id', 'next_click', 'previous_click', 'next_click_mean', 'next_click_median', 'next_click_max'], axis=1)
    y = data_final['is_attributed']
    return x,y

###### lgb_prediction ######
model, predictors = lgb_prediction(train, 2000, if_ip=False)
result = model.predict(test[predictors])
result = np.where(result > 0.85, 1, 0)
label = test['is_attributed'].values
j=0
for i in range (0,len(label)):
    if result[i]==label[i]:
        j=j+1
accuracy = j/len(label)
print('LGB Accuracy:%.6f'%accuracy)

'''
###### svm_prediction ######
model, predictors = svm_prediction(train, 1000, if_ip=False)
result = model.predict(test[predictors])
result = np.where(result > 0.85, 1, 0)
label = test['is_attributed'].values
j=0
for i in range (0,len(label)):
    if result[i]==label[i]:
        j=j+1
accuracy = j/len(label)
print('SVM Prediction Accuracy:%.6f'%accuracy)

###### logistic_reg_prediction ######
train_temp = modifier(train_temp)
x,y = Logistic_dummy(train_temp)
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.3)
logreg = LogisticRegression()
logreg.fit(xtrain,ytrain)
y_pred = logreg.predict(xtest)
print('Logistic Regression Prediction Accuracy: {:.6f}'.format(logreg.score(xtest,ytest)))

###### RandomForest_prediction ######
rf = RandomForestClassifier(n_estimators=10,max_depth=2,max_features=10,class_weight={1:99773/227})
y = ['is_attributed']
data_vars = train.columns.values.tolist()
x = [i for i in data_vars if i not in y]
rf.fit(train[x],train['is_attributed'])
y_predict_rf = rf.predict(test[x])
label = test['is_attributed'].values
accuracy_rf=accuracy_score(y_predict_rf,label)
accuracy_rf_.append(accuracy_rf)
'''
