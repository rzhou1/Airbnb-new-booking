import numpy as np
import pandas as pd
import matplotlib.pyplot as mpt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report

import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)


def loadTables():
    population = pd.read_csv('age_gender_bkts.csv')
    country = pd.read_csv('countries.csv')
    session = pd.read_csv('sessions.csv')
    train_user = pd.read_csv('train_users_2.csv')
    test_user = pd.read_csv('test_users.csv')
    users = pd.concat([train_user, test_user])
    m_train = train_user.shape[0]

    assert users.shape[1] == train_user.shape[1]

    return users, train_user, test_user, session, population, country, m_train

class PreprocessUsers():

    def __init__(self, random_state):
        self.random_state = random_state

    def _features_datetime(self, data):
        data['first_active'] = pd.to_datetime(data.timestamp_first_active.astype(int) // 1000000, format='%Y%m%d',
                                              errors='ignore')
        data['delta_active_account_created'] = (pd.to_datetime(data.date_account_created) - data.first_active).dt.days
        data['week_date_created'] = pd.to_datetime(data.date_account_created, errors='coerce').dt.week
        data['day_date_created'] = pd.to_datetime(data.date_account_created, errors='coerce').dt.dayofweek
        data['delta_active_first_booking'] = (
                    pd.to_datetime(data.date_first_booking) - pd.to_datetime(data.date_account_created)).dt.days
        data['week_first_booking'] = pd.to_datetime(data.date_first_booking, errors='coerce').dt.week
        data['day_first_booking'] = pd.to_datetime(data.date_first_booking, errors='coerce').dayofweek

        return data

    def _imputer_fat(self, data):
        data.first_affiliate_tracked = data.first_affiliate_tracked.fillna('Unknown')

        return data

    def _imputer_gender(self, data):
        data.gender = data.gender.str.lower()

        return data

    def extract_devices(self, data, dicts, cols):
        for col in cols:
            for key, value in dicts.items():
                data[key] = data[col].str.contains(value)
                data[key] = data[key].astype(int)

        return data

    def norm_log1p(self, data, cols):
        for col in cols:
            data[col] = np.log1p(data[col])

        return data

    def transform(self, data, cols_norm, device_dict):
        data = self._features_datetime(data)
        data = self._imputer_fat(data)
        data = self._imputer_gender(data)
        data = self.extract_devices(data, device_dict, ['first_device_type'])
        data = self.norm_log1p(data, cols_norm)

        return data

class ImputerAge():
    def __init__(self, random_state):
        self.random_state = random_state

    def _pre_imputer_age(self, data):
        age = data[['id', 'age', 'affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked',
                    'first_browser', 'first_device_type', 'gender', 'language', 'signup_app',
                    'signup_flow', 'signup_method', 'date_account_created']]
        age.loc[age.age < 10, 'age'] = np.nan
        age.loc[age.age > 2010, 'age'] = np.nan
        age.loc[(age.age > 100) & (age.age < 120), 'age'] = np.nan
        age.loc[(age.age < 998) & (age.age > 120), 'age'] = pd.to_datetime(age.loc[(age.age < 998) & (age.age > 120),
                                                                                   'date_account_created']).dt.year - \
                                                            age.loc[(age.age < 998) & (age.age > 120), 'age'] - 1800
        age.loc[(age.age > 1000) & (age.age < 2010), 'age'] = pd.to_datetime(
            age.loc[(age.age > 1000) & (age.age < 2010),
                    'date_account_created']).dt.year - age.loc[(age.age > 1000) & (age.age < 2010), 'age']
        age = age.drop(['date_account_created'], axis=1)
        return age

    def _data_preprocess(self, data):
        age_nonnull = data.loc[data.age.notnull(), :]
        y_train = np.log1p(age_nonnull.age)
        age_null = data.loc[data.age.isnull(), :]
        y_test = age_null.age

        age_con = pd.concat([age_nonnull, age_null])
        age_con2 = age_con.drop(['id', 'age'], axis=1)
        age_con2 = pd.get_dummies(age_con2)
        age_con2['id'] = age_con['id']
        age_con2 = age_con2[['id'] + age_con2.columns[:-1].tolist()]

        test_size = 0.2
        train = age_con2.iloc[:age_nonnull.shape[0], :]
        X_train, X_val, y_train, y_val = train_test_split(train, y_train, test_size=test_size,
                                                          random_state=self.random_state)
        test = age_con2.iloc[age_nonnull.shape[0]:, :]

        return X_train, X_val, y_train, y_val, test

    def _xgboost_model(self, X_train, X_val, y_train, y_val, test):
        xgbr = xgb.XGBRegressor(objective='reg:squarederror', max_depth=3, n_estimators=500,
                                learning_rate=0.1, subsample=0.8, colsubsample_bytree=0.8,
                                min_child_weight=3, n_jobs=4)
        xgbr.fit(X_train.iloc[:, 1:].values, y_train)

        y_hat = xgbr.predict(X_val.iloc[:, 1:].values)

        rmse = np.sqrt(mean_squared_error(y_val, y_hat))

        y_pred = xgbr.predict(test.iloc[:, 1:].values)

        return rmse, y_pred

    def _imputer_age(self, data, X_train, X_val, test, y_train, y_val, y_pred):
        data_re = pd.concat([X_train, X_val, test])
        y_age = np.exp(np.concatenate((y_train.values, y_val.values, y_pred)))
        data_re['age'] = y_age
        data = data.merge(data_re[['id', 'age']], on='id', how='left')
        data.loc[data.age_x.isnull(), 'age_x'] = data.loc[data.age_x.isnull(), 'age_y']
        data = data.drop(['age_y'], axis=1)
        data = data.rename(columns={'age_x': 'age'})
        data = PreprocessUsers(123).norm_log1p(data, ['age'])
        return data

    def transform(self, data):
        age = self._pre_imputer_age(data)
        X_train, X_val, y_train, y_val, test = self._data_preprocess(age)
        rmse, y_pred = self._xgboost_model(X_train, X_val, y_train, y_val, test)
        data = self._imputer_age(data, X_train, X_val, test, y_train, y_val, y_pred)

        return rmse, data


class PreprocessSession():

    def __init__(self, random_state):
        self.random_state = random_state

    def _agg_users(self, data):
        agg = data.groupby('user_id').agg({'secs_elapsed': {'total_elapsed': 'sum',
                                                            'avg_elapsed': 'mean',
                                                            'std_elapsed': 'std',
                                                            'skew_elapsed': 'skew'}}).reset_index()
        agg.columns = [col[-1] if col[-1] != '' else col[0] for col in agg.columns.values]

        agg_kurt = data.groupby('user_id').apply(pd.DataFrame.kurt)[['secs_elapsed']].reset_index()

        agg = agg.merge(agg_kurt, on='user_id', how='left')

        return agg, agg_kurt

    def _pivot_action_type(self, data):
        at = pd.pivot_table(data, values='secs_elapsed', index=['user_id'], columns=['action_type'],
                            aggfunc=[np.sum, np.mean, np.std]).reset_index()
        cols = []
        cols.append('user_id')
        for sts in ['sum', 'mean', 'std']:
            for col in session.action_type.unique()[1:]:
                cols.append('at_' + sts + '_' + col)
        cols.remove('at_std_booking_response')
        at.columns = cols

        return at

    def _agg_counts(self, data):
        totals = data.groupby('user_id').apply(lambda x: x.count())[['action',
                                                                     'action_type',
                                                                     'action_detail',
                                                                     'device_type']].reset_index()
        uniques = data.groupby('user_id')[['action', 'action_type', 'action_detail',
                                           'device_type']].nunique().reset_index()
        totals = totals.rename(columns={'action': 'action_totals',
                                        'action_type': 'action_type_totals',
                                        'action_detail': 'action_detail_totals',
                                        'device_type': 'device_type_totals'})
        uniques = uniques.rename(columns={'action': 'action_uniques',
                                          'action_type': 'action_type_uniques',
                                          'action_detail': 'action_detail_uniques',
                                          'action_type': 'action_type_uniques',
                                          'device_type': 'device_type_uniques'})
        counts = totals.merge(uniques, on='user_id', how='left')

        return counts

    def _agg_devices(self, data, dicts):
        data = PreprocessUsers(123).extract_devices(data, dicts, ['device_type'])
        cols = []
        for key in dicts.keys():
            cols.append(key)

        agg = data.groupby('user_id')[cols].sum().reset_index()
        for col in cols:
            agg.loc[agg[col] >= 1, col] = 1
        return agg

    def transform(self, data, dicts):
        agg, agg_kurt = self._agg_users(data)
        at = self._pivot_action_type(data)
        counts = self._agg_counts(data)
        agg_dev = self._agg_devices(data, dicts)
        session_group = agg.merge(at, on='user_id', how='left')
        session_group = session_group.merge(counts, on='user_id', how='left')
        session_group = session_group.merge(agg_dev, on='user_id', how='left')

        return session_group


class PreprocessCountry():

    def __init__(self, random_state):
        self.random_state = random_state

    def _flat_country(self, data, dim):
        cols = [col for col in data.columns.tolist() if 'country_destination' not in col]
        cols_dest = data.country_destination.values
        cols_flat = []
        for val in cols_dest:
            for col in cols:
                cols_flat.append(val + '_' + col)

        data_flat = pd.DataFrame(data[cols].values.flatten().reshape(1, 60), columns=cols_flat)
        data_broadcast = pd.DataFrame(np.repeat(data_flat.values, dim, axis=0), columns=data_flat.columns.tolist())

        for col in data_broadcast.columns.tolist():
            if 'destination_language' not in col:
                data_broadcast[col] = data_broadcast[col].astype('float64')
        return data_broadcast

    def transform(self, data, dim):
        cols = [col for col in data.columns.tolist() if 'km' in col or 'distance' in col]
        data = PreprocessUsers(1000).norm_log1p(data, cols)
        data_bc = self._flat_country(data, dim)

        return data_bc

class PreprocessPopulation():

    def __init__(self, random_state):
        self.random_state = random_state

    def _agg_gender_destination(self, data):
        ag = data.groupby(['country_destination', 'gender']).agg({'population_in_thousands':
                                                                      {'population_mean': 'mean',
                                                                       'population_sum': 'sum',
                                                                       'population_std': 'std',
                                                                       'population_skew': 'skew'}}).reset_index()
        ag.columns = [col[0] if col[-1] == '' else col[-1] for col in ag.columns.tolist()]
        ag_pivot = pd.pivot_table(ag, values=[c for c in ag.columns.tolist() if 'population' in c], index='gender',
                                  columns=['country_destination'])
        ag_pivot.columns = [col[0] + '_' + col[-1] for col in ag_pivot.columns.tolist()]

        return ag_pivot

    def transform(self, data):
        data = PreprocessUsers(111).norm_log1p(data, ['population_in_thousands'])
        data_pivot = self._agg_gender_destination(data)

        return data_pivot

class MergeTables():

    def __init__(self, random_state):
        self.random_state = random_state

    def _merge_session(self, data1, data2):
        data = data1.merge(data2, left_on='id', right_on='user_id', how='left')
        data = data.drop(['user_id'], axis=1)

        for col in data.columns.tolist():
            data[col] = data[col].fillna(0)

        return data

    def _merge_population(self, data1, data2):
        data = data1.merge(data2, on='gender', how='left')
        for col in data.columns.tolist():
            data[col] = data[col].fillna(-1)

        return data

    def _merge_country(self, data1, data2):
        data = pd.concat([data1, data2], axis=1)

        return data

    def _drop_cols(self, data, cols):
        data = data.drop(cols, axis=1)

        return data

    def _cols_rearrange(self, data):
        cols = data.columns.tolist()
        cols = cols[8:9] + cols[3:4] + cols[0:3] + cols[4:8] + cols[9:]
        data = data[cols]

        return data

    def transform(self, users, session_group, population, country):
        data = self._merge_session(users, session_group)
        data = self._merge_population(data, population)
        data = self._merge_country(data, country)
        #         data = self._drop_cols(data, cols_drop)
        #         data = self._cols_rearrange(data)

        return data


def dataSplit(data, m_train):
    test = data.iloc[m_train:, 2:]
    X = data.iloc[:m_train, 2:]
    y = data.iloc[:m_train, 1]

    val_test_size = 0.3
    test_size = 0.5

    X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=val_test_size)
    X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=test_size)

    return X_train, X_val, X_test, y_train, y_val, y_test, test

def model(X_train, y_train, X_val, y_val, test):
    model = xgb.XGBClassifier(objective='multi:softmax', iterators=500, learning_rate=0.1, max_depth=3,
                              colsample_bytree=0.8)
    model.fit(X_train, y_train)
    y_val_hat = model.predict(X_val)

    return accuracy_score(y_val, y_val_hat), model

if __name__ == '__main__':

    #Data loading
    users, train_user, test_user, session, population, country, m_train = loadTables()

    #Data preprocess

    users_device_dict = {'U_Desktop': 'Desktop', 'U_Phone': 'iPhone|SmartPhone|Phone', 'U_Tablet': 'Tablet|iPad',
                         'U_Dev_Unknown': 'Unknown', 'U_iOS': 'Mac|iPhone|iPad|iPod', 'U_Windows': 'Windows',
                         'U_Android': 'Android', 'U_Others': 'Others'}

    cols_norm = ['delta_active_account_created']

    users = PreprocessUsers(123).transform(users, cols_norm, users_device_dict)

    rmse, users = ImputerAge(1000).transform(users)
    print('RMSE of Age Imputer by Xgboost:': rmse)

    session_device_dict = {'S_Desktop': 'Desktop', 'S_Phone': 'iPhone|SmartPhone|Phone', 'S_Tablet': 'Tablet|iPad',
                           'S_Dev_Unknown': 'Unknown', 'S_iOS': 'Mac|iPhone|iPad|iPod', 'S_Windows': 'Windows',
                           'S_Android': 'Android', 'S_Others': 'Others'}

    session_group = PreprocessSession(123).transform(session, session_device_dict)

    country_bc = PreprocessCountry(111).transform(country, users.shape[0])

    population_pivot = PreprocessPopulation(111).transform(population)

    users_all = MergeTables(123).transform(users, session_group, population_pivot, country_bc)

    cols_drop = ['date_account_created', 'date_first_booking', 'timestamp_first_active', 'first_active']
    users_all = users_all.drop(cols_drop, axis=1)
    cols = users_all.columns.tolist()
    cols = cols[8:9] + cols[3:4] + cols[0:3] + cols[4:8] + cols[9:]
    users_all = users_all[cols]

    users_all_object = users_all.loc[:, users_all.dtypes == 'object']
    users_all_object2 = users_all_object.iloc[:, 2:]
    object_cols = users_all_object2.columns.tolist()
    users_all_object2 = pd.get_dummies(users_all_object2)
    users_all = users_all.drop(object_cols, axis=1)
    users_all = pd.concat([users_all, users_all_object2], axis=1)

    users_all = users_all.drop_duplicates(keep='first').reset_index()
    users_all = users_all.drop(users_all.index[214273])
    users_all = users_all.drop(['index'], axis=1)
    assert users_all.shape[0] = users.shape[0]

    #Data split

    X_train, X_val, X_test, y_train, y_val, y_test, test = dataSplit(users_all7, m_train)

    #Model
    val_acc, model = model(X_train, y_train, X_val, y_val, test)

    print ('Validataion accuracy': val_acc)

    print ('Test accuracy:', accuracy_score(y_test, model.predict(X_test)))

    y_pred = model.predict(test)

    sub = pd.DataFrame(data={'id': test_user.id, 'country': y_pred})
    sub.to_csv('sub.csv', index=False)