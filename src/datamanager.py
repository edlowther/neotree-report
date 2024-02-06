import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import re

def case_categoriser(grp, drop_bad_age_test_results, include_diagnosis_and_cause_of_death):
    """Helper function to encode the logic assiging groups of rows having a unique id (Uid) to the composite outcome variable
    
    Args: 
        grp (pandas DataFrameGroupBy): Rows from source data groupbed by Uid
        drop_bad_age_test_results (bool): Whether NAs or LONS cases should be included in the outcome variable
        include_diagnosis_and_cause_of_death (bool): Whether the clinician diagnosis or cause of death values are included in outcome
        
    Returns: 
        pandas DataFrameGroupBy with outcome assigned to `bc_positive_or_diagnosis_or_cause_of_death` column
    """
    if include_diagnosis_and_cause_of_death and (any(grp['eons_diagnosis']) or any(grp['eons_cause_of_death'])):
        grp['bc_positive_or_diagnosis_or_cause_of_death'] = True
        grp['description'] = 'diagnosis_or_death'
    else:
        if drop_bad_age_test_results:
            valid_grp = grp.loc[
                (~pd.isna(grp['age_at_test'])) & 
                (grp['age_at_test'] >= 0) & 
                (grp['age_at_test'] < 72)
            ]
        else: 
            valid_grp = grp.loc[
                ~pd.isna(grp['Neolab_finalbcresult'])
            ]
        if len(valid_grp) == 0:
            grp['bc_positive_or_diagnosis_or_cause_of_death'] = False
            n_tests_removed = (~pd.isna(grp['Neolab_finalbcresult'])).sum() - len(valid_grp)
            grp['description'] = 'no_tests_taken' if n_tests_removed == 0 else 'all_tests_excluded'
        else:
            definitive_result_found = False
            for idx, row in valid_grp.iterrows():
                if row['bc_positive'] and row['Neolab_status'] == 'FINAL':
                    grp['bc_positive_or_diagnosis_or_cause_of_death'] = True
                    definitive_result_found = True
                    grp['description'] = 'pos_result_found'
            if not definitive_result_found:
                for idx, row in valid_grp.iterrows():
                    if not row['bc_positive'] and row['Neolab_status'] == 'FINAL':
                        grp['bc_positive_or_diagnosis_or_cause_of_death'] = False
                        definitive_result_found = True
                        grp['description'] = 'neg_result_found'
            if not definitive_result_found:
                for idx, row in valid_grp.iterrows():
                    if row['bc_positive'] and row['Neolab_status'] == 'PRELIMINARY':
                        grp['bc_positive_or_diagnosis_or_cause_of_death'] = True
                        definitive_result_found = True
                        grp['description'] = 'pos_result_found'
            if not definitive_result_found:
                for idx, row in valid_grp.iterrows():
                    if not row['bc_positive'] and row['Neolab_status'] == 'PRELIMINARY':
                        grp['bc_positive_or_diagnosis_or_cause_of_death'] = False
                        definitive_result_found = True
                        grp['description'] = 'neg_result_found'
            if not definitive_result_found:
                grp['bc_positive_or_diagnosis_or_cause_of_death'] = False
                grp['description'] = 'confusing'
    return grp
        
        

class DataManager(): 
    """Convenient wrapper around source data.
    """
    def __init__(self, filepath, scale=True, dummies=True, drop_first=True, reduce_cardinality=False, drop_bad_age_test_results=True, include_diagnosis_and_cause_of_death=True): 
        """Create instance of class.
        
        Args: 
            filepath (str): Where to find the data
            scale (bool): Whether to scale continuous features using scikit-learn's StandardScaler. Defaults to True
            dummies (bool): Whether to convert categorcal variable to dummy variables or use as is. Defaults to True
            drop_first (bool): If converting to dummies, whether to drop the first dummy. Defaults to True
            reduce_cardinality (bool): Whether to group features with high cardinality into buckets. Defaults to False
            drop_bad_age_test_results (bool): Whether NAs or LONS cases should be included in the outcome variable. Defaults to True
            include_diagnosis_and_cause_of_death (bool): Whether the clinician diagnosis or cause of death values are included in outcome. Defaults to True
        """
        self.filepath = filepath
        self.reduce_cardinality = reduce_cardinality
        self.drop_bad_age_test_results = drop_bad_age_test_results
        self.include_diagnosis_and_cause_of_death = include_diagnosis_and_cause_of_death
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
            return x in ['Pos', 'PosP']
        df['bc_positive'] = df['Neolab_finalbcresult'].apply(categorise_bc_result)
        df = pd.concat([case_categoriser(grp, self.drop_bad_age_test_results, self.include_diagnosis_and_cause_of_death) for _, grp in df.groupby('Uid')])
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
        """All cases should now have a unique set of predictors and composite outcome variable"""
        assert len(self.df.drop_duplicates(X_cols + ['Uid', y_label])) == len(self.df['Uid'].unique())
        self.df = self.df.drop_duplicates(X_cols + ['Uid', y_label])
        
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
        
        