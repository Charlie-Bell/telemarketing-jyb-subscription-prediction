import pandas as pd
from data_cleaning import clean_all
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from sklearn.model_selection import StratifiedKFold

def add_statistical_features(df):
    df_desc = df.describe()
    num_features = list(df_desc.columns)
    num_feature_stats = list(df_desc.index)
    num_feature_stats.remove('count')
    for feature in num_features:
        for stat in num_feature_stats:
            df[feature+'_'+stat] = df_desc[feature][stat]
    return df

def encode(df):
    label_cols = ['housing', 'loan', 'contact']
    label_encoder = LabelEncoder()
    for col in label_cols:
        df[col] = label_encoder.fit_transform(df[col])
    print(f"Label encoding of {label_cols}")    
        
    ordinal_cols = ['month', 'day_of_week', 'education']    
    days_of_week_enc = [['mon', 0], ['tue', 1], ['wed', 2], ['thu', 3], ['fri', 4]]
    for day in days_of_week_enc:
        df.loc[df['day_of_week']==day[0], 'day_of_week'] = day[1]        
    months_of_year_enc = [['jan',0], ['feb',1], ['mar',2], ['apr',3], ['may',4], ['jun',5],
                          ['jul',6], ['aug',7], ['sep',8], ['oct',9], ['nov',10], ['dec',11]]
    for month in months_of_year_enc:
        df.loc[df['month']==month[0], 'month'] = month[1]
    education_enc = [['illiterate',0], ['basic.4y',1], ['basic.6y',2], ['basic.9y',3],
                     ['high.school',4], ['university.degree',5], ['professional.course',6]]
    for education in education_enc:
        df.loc[df['education']==education[0], 'education'] = education[1]
    print(f"Ordinal encoding of {ordinal_cols}")

    one_hot_cols = ['job', 'marital', 'prev_outcome']
    one_hot_df = df[one_hot_cols]
    one_hot_df = pd.get_dummies(one_hot_df)
    df = df.drop(columns=one_hot_cols)
    df = pd.concat([df, one_hot_df], axis=1)
    print(f"One-Hot Encoding of {one_hot_cols}")
    
    return df

def df_to_numpy(df):
    # X and y
    X = df.drop(columns=['subscribed']).to_numpy()
    y = (df[['subscribed']]=='yes').to_numpy().ravel()
    return X, y

def split_data(X, y):
    folds = []
    skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
    for train_index, test_index in skf.split(X, y):
        X_train_fold, X_test_fold = X[train_index], X[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]

        fold = {'X_train': X_train_fold, 'y_train': y_train_fold,
                'X_test': X_test_fold, 'y_test': y_test_fold}
        folds.append(fold)
    return folds

def resample(X_train, y_train):
    # Up/Downsampling
    resampler = SMOTETomek(tomek=TomekLinks(sampling_strategy='majority'), smote=SMOTE(sampling_strategy='minority'))
    return resampler.fit_resample(X_train, y_train)

def resample_folds(folds):
    for k, fold in enumerate(folds):
        folds[k]['X_train'], folds[k]['y_train'] = resample(fold['X_train'], fold['y_train'])
    return folds

def scale(X):
    # Scaling
    return MinMaxScaler().fit_transform(X)

def scale_folds(folds):
    for k, fold in enumerate(folds):
        folds[k]['X_train'] = scale(fold['X_train'])
        folds[k]['X_test'] = scale(fold['X_test'])
    return folds

def skewedness(folds):
    for k, fold in enumerate(folds):
        a = fold['y_test']
        a = sum(a)/len(a)*100
        b = fold['y_train']
        b = sum(b)/len(b)*100
        print(f"Test Dataset: fold {k} values range {fold['X_test'].min()} to {fold['X_test'].max()}, with positive sample proportion {a:.2f}%.")
        print(f"Train Dataset: fold {k} values range {fold['X_train'].min()} to {fold['X_train'].max()}, with positive sample proportion {b:.2f}%.")

def prep_all(df):
    df = add_statistical_features(df)
    df = encode(df)
    X, y = df_to_numpy(df)
    folds = split_data(X, y)
    folds = resample_folds(folds)
    folds = scale_folds(folds)
    return folds
    
