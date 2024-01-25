import pandas as pd
from dataclasses import dataclass

@dataclass
class Variable:
    data_name: str
    display_name: str
    data_type: str
    values: list = None
    
@dataclass
class Value:
    data_name: str
    display_name: str

class TableBuilder():
    def __init__(self):
        self.variables = [
            Variable('Age', 'Age (hour)', 'float'), 
            Variable('Gender', 'Gender', 'cat', [Value('F', 'Female'), Value('M', 'Male'), Value('NS', 'Ambiguous')]),
            Variable('Gestation', 'Gestational age', 'int'),
            Variable('Birthweight', 'Birth weight', 'int'), 
            Variable('Temperature', 'Temperature at admission', 'float'), 
            Variable('Hr', 'Heart rate', 'float'), 
            Variable('Apgar1', 'Apgar score at 1 minute', 'int'),
            Variable('Apgar5', 'Apgar score at 5 minutes', 'int'), 
            Variable('Rr', 'Respiratory rate', 'int'), 
            Variable('Satsair', 'Oxygen saturation in air', 'float'), 
            Variable('Typebirth', 'Type of birth', 'cat', [
                Value('S', 'Singleton'), Value('Tw1', 'Twin 1'), Value('Tw2', 'Twin 2'), Value('Tr1', 'Triplet 1'), Value('Tr2', 'Triplet 2'), Value('Tr3', 'Triplet 3'), 
                Value('OT', 'Other')
            ]), 
            Variable('Romlength', 'Premature rupture of membrane', 'cat', [
                Value('PROM', 'Yes'), Value('NOPROM', 'No rupture')
            ]), 
            Variable('Skin', 'Skin condition', 'cat', [
                Value('Rash', 'Rash'), Value('PUST', 'Pustules'), Value('BOIL', 'Boils'), Value('{Rash,BOIL}', 'Rash and boils'), Value('Folds', 'Folds'), Value('None', 'None')
            ]), 
            Variable('Wob', 'Work of breathing', 'cat', [
                Value('Mild', 'Mild'), Value('Mod', 'Moderate'), Value('Sev', 'Severe')
            ]), 
            Variable('Signsrd', 'Signs of respiratory distress', 'cat', [
                Value('NOT_None', 'Signs present'), Value('None', 'No signs')
            ]), 
            Variable('Dangersigns', 'Danger signs', 'cat', [
                Value('NOT_None', 'Danger signs present'), Value('None', 'No danger signs')
            ]), 
            Variable('Activity', 'Alertness', 'cat', [
                Value('Alert', 'Alert'), Value('Coma', 'Coma'), Value('Conv|Convulsions', 'Convulsions'), Value('Leth', 'Lethargic'), Value('Irrit', 'Irritable')
            ]), 
            Variable('Colour', 'Colour', 'cat', [
                Value('Pink', 'Pink'), Value('Blue', 'Blue'), Value('White', 'White'), Value('Yell', 'Yellow'), Value('{Yell,White}', 'Yellow/white')
            ]), 
            Variable('Umbilicus', 'Umbilicus', 'cat', [
                Value('Norm', 'Normal'), Value('NOT_Norm', 'Abnormal')
            ]), 
            Variable('Vomiting', 'Vomiting', 'cat', [
                Value('Poss', 'Small milky possets after feeds (normal)'), Value('*', 'Yes (all feeds/ blood/ green)'), Value('No', 'No')
            ]), 
            Variable('Abdomen', 'Abdomen check', 'cat', [
                Value('Norm', 'Normal'), Value('NOT_Norm', 'Abnormal')
            ]), 
            Variable('Fontanelle', 'Fontanelle', 'cat', [
                Value('Flat', 'Flat'), Value('Bulg', 'Bulging'), Value('Sunk', 'Sunken')
            ])
        ]
        
    def run(self, data_manager):
        output = []
        for variable in self.variables: 
            values = data_manager.df[variable.data_name]
            if variable.data_type in ['float', 'int']: 
                median = values.median()
                q1 = values.quantile(0.25)
                q3 = values.quantile(0.75)
                if variable.data_type == 'int':
                    median, q1, q3 = map(round, [median, q1, q3])
                row = [variable.display_name + '; median [Q1-Q3]', '-', f'{median} [{q1}-{q3}]']
                for boolean in [True, False]: 
                    tmp_values = data_manager.df.loc[data_manager.df['bc_positive_or_diagnosis_or_cause_of_death'] == boolean, variable.data_name]
                    median = tmp_values.median()
                    q1 = tmp_values.quantile(0.25)
                    q3 = tmp_values.quantile(0.75)
                    if variable.data_type == 'int':
                        median, q1, q3 = map(round, [median, q1, q3])
                    row.append(f'{median} [{q1}-{q3}]')
                output.append(row)
            elif variable.data_type == 'cat':
                for row_idx, value in enumerate(variable.values):
                    if row_idx == 0:
                        row = [variable.display_name + '; n (%)']
                    else:
                        row = ['', ]
                    row.append(value.display_name)
                    if 'NOT_' in value.data_name:
                        not_value = value.data_name.split('_')[1]
                        row_n = len(data_manager.df.loc[
                            (data_manager.df[variable.data_name] != not_value) & 
                            (~pd.isna(data_manager.df[variable.data_name]))
                        ])
                    elif value.data_name == '*':
                        print(value)
                        row_n = len(data_manager.df.loc[
                            (~data_manager.df[variable.data_name].isin(['Poss', 'No'])) & 
                            (~pd.isna(data_manager.df[variable.data_name]))
                        ])
                    elif '|' in value.data_name:
                        row_n = 0
                        for data_name_version in value.data_name.split('|'):
                            row_n += len(data_manager.df.loc[data_manager.df[variable.data_name] == data_name_version])
                    else: 
                        row_n = len(data_manager.df.loc[data_manager.df[variable.data_name] == value.data_name])
                    pct = row_n / len(data_manager.df) * 100
                    row.append(f'{row_n} ({pct:.2f})')
                    for boolean in [True, False]: 
                        if 'NOT_' in value.data_name:
                            not_value = value.data_name.split('_')[1]
                            n = len(data_manager.df.loc[
                                (data_manager.df['bc_positive_or_diagnosis_or_cause_of_death'] == boolean) & 
                                (data_manager.df[variable.data_name] != not_value) & 
                                (~pd.isna(data_manager.df[variable.data_name]))
                            ])
                        elif value.data_name == '*':
                            n = len(data_manager.df.loc[
                                (data_manager.df['bc_positive_or_diagnosis_or_cause_of_death'] == boolean) & 
                                (~data_manager.df[variable.data_name].isin(['Poss', 'No'])) & 
                                (~pd.isna(data_manager.df[variable.data_name]))
                            ])
                        elif '|' in value.data_name:
                            n = 0
                            for data_name_version in value.data_name.split('|'):
                                n += len(data_manager.df.loc[(data_manager.df['bc_positive_or_diagnosis_or_cause_of_death'] == boolean) & 
                                                             (data_manager.df[variable.data_name] == data_name_version)])
                        else: 
                            n = len(data_manager.df.loc[
                                (data_manager.df['bc_positive_or_diagnosis_or_cause_of_death'] == boolean) & 
                                (data_manager.df[variable.data_name] == value.data_name)])
                        pct = n / row_n * 100
                        row.append(f'{n} ({pct:.2f})')            
                    output.append(row)
            n_nas = pd.isna(values).sum()
            if n_nas > 0:
                missing_text = 'Missing' if variable.data_type == 'cat' else 'Missing n (%)'
                pct = n_nas / len(data_manager.df) * 100
                row = ['', missing_text, f'{n_nas} ({pct:.2f})'] # + ['-'] * 2
                for boolean in [True, False]: 
                    n = len(data_manager.df.loc[
                        (data_manager.df['bc_positive_or_diagnosis_or_cause_of_death'] == boolean) & 
                        (pd.isna(data_manager.df[variable.data_name]))])
                    pct = n / n_nas * 100
                    row.append(f'{n} ({pct:.2f})')   
                output.append(row)
        return pd.DataFrame(output)
