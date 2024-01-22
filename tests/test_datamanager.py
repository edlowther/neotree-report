import pytest
import pandas as pd
import numpy as np
from src.datamanager import DataManager

data_filepath = './data/sepsis_updated_data_Feb21-Sep23.csv'
data_manager = DataManager(data_filepath)
df = data_manager.df

def test_positive_bc_results_assigned_correctly():
    assert all(df.loc[df['bc_positive'], 'Neolab_finalbcresult'].isin(['Pos', 'PosP']))
    
def test_negative_bc_results_assigned_correctly_with_contaminated_results_removed():
    assert all(df.loc[(-pd.isna(df['Neolab_finalbcresult'])) & (-df['bc_positive']), 'Neolab_finalbcresult'].isin(['Neg', 'NegP']))
    assert not any(df['Neolab_finalbcresult'].isin(['Contaminant', 'Rej']))
    
def test_target_defined_correctly():
    diagnosis_is_eons = df['Diagdis1'] == 'EONS'
    death_caused_by_eons = df['Causedeath'] == 'EONS'
    assert all(df.loc[diagnosis_is_eons, 'bc_positive_or_diagnosis_or_cause_of_death'])
    assert all(df.loc[death_caused_by_eons, 'bc_positive_or_diagnosis_or_cause_of_death'])
    # Are all rows that are negative in our target class despite having a positive blood test result and no diagnosis or cause of death
    # had problematic age categorisations of one kind or another:
    problematic_ages = df.loc[(~df['bc_positive_or_diagnosis_or_cause_of_death']) & (df['bc_positive']) & (~diagnosis_is_eons) & (~death_caused_by_eons), 'age_at_test']
    n_over_threshold = (problematic_ages > 72).sum()
    n_below_zero = (problematic_ages < 0).sum()
    n_nas = (pd.isna(problematic_ages)).sum()
    assert len(problematic_ages) == n_over_threshold + n_below_zero + n_nas

def test_correct_X_y_vars():
    data_manager = DataManager(data_filepath)
    continuous_vars = ['Rr', 'Hr', 'Temperature']
    data_manager.remove_duplicate_predictors(continuous_vars, 'bc_positive_or_diagnosis_or_cause_of_death')
    X_train, X_test, y_train, y_test = data_manager.get_X_y(continuous_vars, 42)
    assert all(X_train.columns == continuous_vars) and all(X_test.columns == continuous_vars)
    assert len(X_train) == len(y_train) and len(X_test) == len(y_test)
    assert len(pd.concat([X_train, X_test])) == len(data_manager.df[-data_manager.df[continuous_vars].isna().any(axis=1)])
    
    categorical_vars = ['Typebirth', 'Vomiting']
    X_train, X_test, y_train, y_test = data_manager.get_X_y(categorical_vars, 42)
    expected_X_vars = []
    for var in categorical_vars: 
        for value in sorted([str(x) for x in data_manager.df[var].unique()])[1:]: 
            value = value.replace('{', '').replace('}', '').replace(',', '')
            expected_X_vars.append(f'{var}_{value}')
    print(expected_X_vars)
    print(sorted(X_train.columns))
    assert sorted(X_train.columns) == sorted(expected_X_vars) 
    assert sorted(X_test.columns) == sorted(expected_X_vars)
    assert len(X_train) == len(y_train) and len(X_test) == len(y_test)
    assert len(pd.concat([X_train, X_test])) == len(data_manager.df)
    
    mixed_vars = continuous_vars + categorical_vars
    X_train, X_test, y_train, y_test = data_manager.get_X_y(mixed_vars, 42)
    expected_X_vars = continuous_vars.copy()
    for var in categorical_vars: 
        for value in sorted([str(x) for x in data_manager.df[var].unique()])[1:]: 
            value = value.replace('{', '').replace('}', '').replace(',', '')
            expected_X_vars.append(f'{var}_{value}')
    assert sorted(X_train.columns) == sorted(expected_X_vars) and sorted(X_test.columns) == sorted(expected_X_vars)
    assert len(X_train) == len(y_train) and len(X_test) == len(y_test)
    assert len(pd.concat([X_train, X_test])) == len(data_manager.df[-data_manager.df[continuous_vars].isna().any(axis=1)])
    
def test_duplicates_removed():
    dummy_df = pd.DataFrame([
        {'Uid': '12-12', 'Hr': 101, 'Rr': 89, 'Admreason': 'one', 'bc_positive_or_diagnosis_or_cause_of_death': True, 'Neolab_status': 'FINAL'},
        {'Uid': '12-12', 'Hr': 101, 'Rr': 89, 'Admreason': 'two', 'bc_positive_or_diagnosis_or_cause_of_death': False, 'Neolab_status': 'FINAL'},
        {'Uid': '11-10', 'Hr': 125, 'Rr': 100, 'Admreason': 'three', 'bc_positive_or_diagnosis_or_cause_of_death': False, 'Neolab_status': 'FINAL'},
        {'Uid': '11-11', 'Hr': 125, 'Rr': 100, 'Admreason': 'three', 'bc_positive_or_diagnosis_or_cause_of_death': False, 'Neolab_status': 'FINAL'},
        {'Uid': '52-10', 'Hr': 120, 'Rr': 95, 'Admreason': 'a', 'bc_positive_or_diagnosis_or_cause_of_death': False, 'Neolab_status': 'FINAL'},
        {'Uid': '52-10', 'Hr': 120, 'Rr': 95, 'Admreason': 'b', 'bc_positive_or_diagnosis_or_cause_of_death': True, 'Neolab_status': 'FINAL'},
        {'Uid': '52-10', 'Hr': 140, 'Rr': 85, 'Admreason': 'c', 'bc_positive_or_diagnosis_or_cause_of_death': False, 'Neolab_status': 'FINAL'},
        {'Uid': '53-06', 'Hr': 98, 'Rr': 101, 'Admreason': 'z', 'bc_positive_or_diagnosis_or_cause_of_death': True, 'Neolab_status': 'FINAL'},
        {'Uid': '53-06', 'Hr': 98, 'Rr': 101, 'Admreason': 'z', 'bc_positive_or_diagnosis_or_cause_of_death': False, 'Neolab_status': 'PRELIMINARY'},
        {'Uid': '53-07', 'Hr': 99, 'Rr': 102, 'Admreason': 'z', 'bc_positive_or_diagnosis_or_cause_of_death': True, 'Neolab_status': 'PRELIMINARY'},
        {'Uid': '53-07', 'Hr': 99, 'Rr': 102, 'Admreason': 'z', 'bc_positive_or_diagnosis_or_cause_of_death': False, 'Neolab_status': 'FINAL'}
    ])
    dummy_data_manager = DataManager(data_filepath)
    dummy_data_manager.df = dummy_df
    X_cols = ['Hr', 'Rr']
    y_label = 'bc_positive_or_diagnosis_or_cause_of_death'
    dummy_data_manager.remove_duplicate_predictors(X_cols, y_label)
    assert len(dummy_data_manager.df == 4)
    assert dummy_data_manager.df.loc[dummy_data_manager.df['Uid'] == '12-12', 'bc_positive_or_diagnosis_or_cause_of_death'].values[0]
    # The second row with id 12-12 should have been removed because both predictors match (even though an additional column, Admreason, does not match)
    with pytest.raises(IndexError): 
        dummy_data_manager.df.loc[dummy_data_manager.df['Uid'] == '12-12', 'bc_positive_or_diagnosis_or_cause_of_death'].values[1]
    # With matching predictors and one positive and one negative result (both marked final), we keep the positive:
    assert dummy_data_manager.df.loc[(dummy_data_manager.df['Uid'] == '52-10')
                                     &
                                     (dummy_data_manager.df['Hr'] == 120), 'bc_positive_or_diagnosis_or_cause_of_death'].values[0]
    # A separate episode with the same id and different predictors is retained as a negative result
    assert not dummy_data_manager.df.loc[(dummy_data_manager.df['Uid'] == '52-10')
                                     &
                                     (dummy_data_manager.df['Hr'] == 140), 'bc_positive_or_diagnosis_or_cause_of_death'].values[0]
    # Where predictors are the same, we keep the final result, in this case positive (and ditch the second row):
    assert dummy_data_manager.df.loc[dummy_data_manager.df['Uid'] == '53-06', 'bc_positive_or_diagnosis_or_cause_of_death'].values[0]
    with pytest.raises(IndexError):
        dummy_data_manager.df.loc[dummy_data_manager.df['Uid'] == '53-06', 'bc_positive_or_diagnosis_or_cause_of_death'].values[1]
    # Again, we keep the final result for two rows with matching predictors, in this case negative:
    assert not dummy_data_manager.df.loc[dummy_data_manager.df['Uid'] == '53-07', 'bc_positive_or_diagnosis_or_cause_of_death'].values[0]