import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import re

class DataManager(): 
    """Convenient wrapper around source data.
    """
    def __init__(self, filepath, scale=True, dummies=True, drop_first=True, reduce_cardinality=False): 
        """Create instance of class.
        
        Args: 
            filepath (str): Where to find the data
            scale (bool): Whether to scale continuous features using scikit-learn's StandardScaler. Defaults to True
            dummies (bool): Whether to convert categorcal variable to dummy variables or use as is. Defaults to True
            drop_first (bool): If converting to dummies, whether to drop the first dummy. Defaults to True
        """
        self.filepath = filepath
        self.reduce_cardinality = reduce_cardinality
        self.df = self._load_data()
        self.scale = scale
        self.dummies = dummies
        self.drop_first = drop_first
        
    def _load_data(self): 
        """Load data and perform various type-related tasks and some feature engineering
        
        Returns: 
            Tidied pandas DataFrame
        """
        df = pd.read_csv(self.filepath)
        df = df.assign(Neolab_datebct    = pd.to_datetime(df['Neolab_datebct'],    utc=True),
                       Neolab_datebcr    = pd.to_datetime(df['Neolab_datebcr'],    utc=True),
                       Datetimeadmission = pd.to_datetime(df['Datetimeadmission'], utc=True),
                       Datetimedeath     = pd.to_datetime(df['Datetimedeath'],     utc=True))
        # Age at test is age at admission plus difference between time that test was taken and time of admission:
        df['age_at_test'] = df['Age'] + (df['Neolab_datebct'] - df['Datetimeadmission']) / pd.Timedelta(hours=1)
        df['eons_diagnosis'] = df['Diagdis1'] == 'EONS'
        df['eons_cause_of_death'] = df['Causedeath'] == 'EONS'
        df = df[-df['Neolab_finalbcresult'].isin(['Contaminant', 'Awaiting Final Result', 'Rej'])]
        def categorise_bc_result(x):
            if x in ['Pos', 'PosP']:
                return True
            elif x in ['Neg', 'NegP']:
                return False
            else:
                return None
        df['bc_positive'] = df['Neolab_finalbcresult'].apply(categorise_bc_result)
        def get_target(row):
            if row['eons_diagnosis'] or row['eons_cause_of_death']:
                return True
            elif row['bc_positive']:
                return self.is_eons_bc_result(row)
            else:
                return False
        df['bc_positive_or_diagnosis_or_cause_of_death'] = df.apply(get_target, axis=1)
        if self.reduce_cardinality:
            def simplify_vomiting(x):
                if pd.isna(x):
                    return None
                elif x in ['Poss', 'No']:
                    return x
                else:
                    return 'Yes'
            df['Vomiting'] = df['Vomiting'].apply(simplify_vomiting)
            def simplify_norm_values(x):
                if pd.isna(x):
                    return None
                elif x == 'Norm':
                    return x
                else:
                    return 'Abnormal'
            df['Umbilicus'] = df['Umbilicus'].apply(simplify_norm_values)
            df['Abdomen'] = df['Abdomen'].apply(simplify_norm_values)
            def simplify_signs(x):
                if pd.isna(x):
                    return None
                elif x == 'None':
                    return 'no_signs'
                else:
                    return 'signs_present'
            df['Signsrd'] = df['Signsrd'].apply(simplify_signs)
            df['Dangersigns'] = df['Dangersigns'].apply(simplify_signs)
        return df
    
    def is_eons_bc_result(self, row, eons_cutoff=72):
        """Assign cases to negative class when it cannot be determined whether it was an early-onset or late-onset case"""
        # Test taken but age cannot be determined:
        tested_but_age_is_na = (~pd.isna(row['Neolab_finalbcresult'])) & (pd.isna(row['age_at_test']))
        # Test taken after early-onset threshold:
        tested_after_threshold = row['age_at_test'] > eons_cutoff
        # Age is below zero, suggesting data-entry error:
        negative_age_at_test = row['age_at_test'] < 0
        if tested_but_age_is_na or tested_after_threshold or negative_age_at_test:
            return False
        else:
            return True
        
    def remove_outliers(self):
        """Remove cases where any value for selected variables appears to be an outlier"""
        scaler = StandardScaler()
        scaled_df = pd.DataFrame()
        selected_variables = ['Age', 'Satsair', 'Temperature', 'Rr', 'Hr']
        scaled_df[selected_variables] = scaler.fit_transform(self.df[selected_variables])
        
        def is_not_outlier(row): 
            """Build a filter requiring that every value from the selected columns is less than five standard deviations away from the mean"""
            return all((abs(row) < 5) | (pd.isna(row)))

        not_outlier_mask = scaled_df.apply(is_not_outlier, axis=1)
        self.df = self.df.reset_index(drop=True)[not_outlier_mask]
        
    def remove_duplicate_predictors(self, X_cols, y_label):
        """Some cases have more than one episode with differing outcomes for the same predictor sets. This function retains only
        the worst-case outcome, e.g. if there is a negative test one day but a positive test two days later, the data_manager keeps
        only the row containing the positive test"""
        self.df = self.df.sort_values(
            ['Neolab_status', y_label],
            ascending=[True, False]
        ).groupby(
            ['Uid'] + X_cols,
            dropna=False
        ).head(1).reset_index(drop=True)
        
    def get_X_y(self, X_cols, seed, y_label='bc_positive_or_diagnosis_or_cause_of_death'): 
        """Return features and target according to preferences set out in instantiation of class""" 
        df_pos = self.df[self.df[y_label]]
        df_neg = self.df[~self.df[y_label]]
        # Get a perfectly balanced target:
        # self.df = pd.concat([resample(df_neg, replace=False, n_samples=len(df_pos)), df_pos])
        X = self.df[X_cols].copy()
        y = self.df[y_label]
        
        continuous_colnames = X.select_dtypes(['int64', 'float64']).columns
        
        if self.dummies: 
            X = pd.get_dummies(X, dummy_na=True, drop_first=self.drop_first)
            X = X.rename(columns = lambda x: re.sub('[^A-Za-z0-9_]', '', x))
            self.na_mask = ~X.isnull().any(axis=1)
            X = X.loc[self.na_mask]
            y = y[self.na_mask]
        
        else: 
            display_X = X.copy()
            for colname in X.columns.difference(continuous_colnames): 
                X[colname] = pd.Categorical(X[colname]).codes
                display_X[colname] = pd.Categorical(display_X[colname])
            self.display_X = display_X
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
        
        if self.scale:
            X_train_continuous = X_train[continuous_colnames]
            X_test_continuous = X_test[continuous_colnames]
            X_train_categorical = X_train[X_train.columns.difference(continuous_colnames)]
            X_test_categorical = X_test[X_test.columns.difference(continuous_colnames)]
        
            if len(continuous_colnames) > 0:
                scaler = StandardScaler()
                scaler.fit(X_train_continuous)
                X_train_continuous = pd.DataFrame(scaler.transform(X_train_continuous), columns=scaler.feature_names_in_)
                X_test_continuous = pd.DataFrame(scaler.transform(X_test_continuous), columns=scaler.feature_names_in_)

            X_train = pd.concat([X_train_continuous.reset_index(drop=True), X_train_categorical.reset_index(drop=True)], axis=1)
            X_test = pd.concat([X_test_continuous.reset_index(drop=True), X_test_categorical.reset_index(drop=True)], axis=1)
        
        return X_train, X_test, y_train, y_test
        
        