import pandas as pd

#Load csv file into a dataframe with error handleing
def load_data(file_path,**kwargs)->pd.DataFrame:
    try: 
        return pd.read_csv(file_path,**kwargs)
    except Exception as e:
        raise Exception(f"Error loading data from {file_path}: {e}")

#function to clean missing values   
def clean_missing_values(
        df: pd.DataFrame,
        strategy: str='drop',
        fill_value:float=0.0,
        columns: list=None)->pd.DataFrame:
    if strategy == 'drop':
        return df.dropna(subset=columns)
    elif strategy == 'fill':
        return df.fillna(fill_value, subset=columns)
    else:
        raise ValueError("Invalid strategy. Use 'drop' or 'fill'")

#function to filter rows using condition  
def filter_rows(df:pd.DataFrame,condition: str)-> pd.DataFrame:
    return df.query(condition)

#rename column using dictinary
def rename_columns(df: pd.DataFrame,column_mapping: dict)-> pd.DataFrame:
    return df.rename(columns=column_mapping)

#Generate summary statistics for numeric columns
def get_summary_stats(df: pd.DataFrame)-> pd.DataFrame:
    return df.describe()

#group by columns and apply aggregate functions:
def group_and_aggregate(
        df: pd.DataFrame,
        group_cols: list,
        agg_dict:dict)-> pd.DataFrame:
    return df.groupby(group_cols).agg(agg_dict)

#Split data into training and test sets
def split_train_data(df: pd.DataFrame,
                     test_size: float=0.2,
                     random_state: int=42)-> tuple:
    from sklearn.model_selection import train_test_split # type: ignore
    return train_test_split(
        df, test_size=test_size,random_state=random_state)
    



    
    
