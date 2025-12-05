import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def drop_column(df, column):
  if column in set(df.columns):
    df.drop([column], axis=1, inplace=True)


def remove_units_of_measurements(df, columns):
  for col in columns:
    df[col] = df[col].astype('str').str.extract(r'(\d+.\d+)',).astype('float')


def nan_to_median(df, medians):
  missing_columns = list(df.columns[df.isnull().any()])
  values = {}
  for col in missing_columns:
    values[col] = medians[col]
  df.fillna(value=values, inplace=True)


def float_col_to_int(df, columns):
  for col in columns:
    df[col] = df[col].astype('int')


def modify(df):
  columns = ['mileage', 'engine', 'max_power']
  remove_units_of_measurements(df, columns)
  drop_column(df, 'torque')

  medians = pd.read_csv('medians.csv').to_dict()
  nan_to_median(df, medians)
  
  float_col_to_int(df, ['engine', 'seats'])
  

def add_name(df):
  df['name'] = df['name'].str.split().str[0]

def encode_cat(X_train_cat, X_test_cat, cat_features):
  encoder = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
  encoder.fit(X_train_cat[cat_features])
  train_ohe = encoder.transform(X_train_cat[cat_features])
  test_ohe = encoder.transform(X_test_cat[cat_features])

  ohe_columns = encoder.get_feature_names_out(cat_features)

  train_ohe_df = pd.DataFrame(train_ohe, columns=ohe_columns, index=X_train_cat.index)
  test_ohe_df = pd.DataFrame(test_ohe, columns=ohe_columns, index=X_test_cat.index)

  X_train_cat = pd.concat([X_train_cat.drop(columns=cat_features), train_ohe_df], axis=1)
  X_test_cat = pd.concat([X_test_cat.drop(columns=cat_features), test_ohe_df], axis=1)

  return X_train_cat, X_test_cat

def modify_cat(X_train_cat, X_test_cat):
  cat_features = ['year', 'fuel', 'seller_type', 'transmission', 'owner', 'seats', 'name']
  return encode_cat(X_train_cat, X_test_cat, cat_features)