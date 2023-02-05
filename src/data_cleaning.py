import pandas as pd

def drop_unnamed(df):
    # Drop unnamed index columns
    print("Dropping unnamed columns.")
    return df.loc[:, ~df.columns.str.contains('^Unnamed')]

def check_null(df):
    if df.isnull().sum().sum() == 0:
        print ("No null values.") # If no missing values
    else:
        missing_values = df.isnull().sum() # If missing values we print the number of missing values only for missing values
        missing_values = missing_values[missing_values>0]
        print(f"Missing value count:\n{missing_values}.")
        print(f"Missing value proportion:\n{missing_values/len(df)}.")

def rename_columns(df): 
    # Rename columns
    print("Renaming columns.")
    columns_list = list(df.columns)
    for i, col in enumerate(columns_list):
        columns_list[i] = col.replace(".", "_")
    columns_dict = dict(zip(df.columns, columns_list))
    columns_dict['pdays'] = "prev_days"
    columns_dict['poutcome'] = "prev_outcome"
    columns_dict['previous'] = "prev_nr_contacts"
    columns_dict['euribor3m'] = "euribor_3_month"
    columns_dict['y'] = "subscribed"
    df = df.rename(columns=columns_dict)    
    return df

def get_numeric_columns(df):
    return list(df._get_numeric_data().columns)
    
def get_categorical_columns(df):
    cat_cols = list(df.select_dtypes('object').columns)
    categories = {feature: df[feature].unique() for feature in cat_cols}
    return cat_cols, categories

def drop_labels(df, threshold=0.1, label="unknown"):
    cat_cols, _ = get_categorical_columns(df)
    unknown_ratio = {}
    val_counts = {}
    drop_cols = []
    for cat in cat_cols:
        val_counts[cat] = df[cat].value_counts()

        unknown_ratio[cat] = sum(df[cat]==label)/len(df)
        if unknown_ratio[cat] > threshold:
            df = df.drop(columns=[cat])
            drop_cols.append(cat)
        elif unknown_ratio[cat] > 0:
            df = df[df[cat]!=label]
    print(f"Dropped columns with '{label}' proportion > {threshold*100}% and rows with < {threshold*100}%")
    if drop_cols:
        print(f"Dropped columns: {drop_cols}.")
    else:
        print("Dropped no columns.")
    return df

def drop_anomolous(df, drop_cols=['prev_days']):
    df = df.drop(columns=drop_cols)
    print(f"Dropped columns: {drop_cols}")
    return df

def drop_correlated(df):
    corr_cols = ['emp_var_rate', 'euribor_3_month']
    df = df.drop(columns=corr_cols)
    print(f"Dropped columns: {corr_cols}.")
    return df

def clean_all(df):
    df = drop_unnamed(df)
    df = rename_columns(df)
    df = drop_labels(df)
    df = drop_anomolous(df)
    df = drop_correlated(df)
    return df