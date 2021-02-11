def break_down(start_index: int, end_index: int):
    """
        Subsetting the dataframe into questions and binarizing the columns
        
        input: 
            start_index: The start of the subset index
            end_index: The end of the subset index
            
        return:
            dataframe containing the reponses to a given question
    """
    subset = res.iloc[1:, start_index:end_index].copy()
    
    return binary_formatting(subset)

def binary_formatting(df: pd.DataFrame):
    """
        Binarizing the columns, changing the column names
        
        input:
            df: Dataframe to binarize
        
        return:
            A dataframe of binary columns with answers as their column names
    """
    col_dict = {}
    for col in df.columns:
        val = np.nan
        if pd.isna(df[col].unique()[0]):
            val = df[col].unique()[1]
        else:
            val = df[col].unique()[0]
        
        
        if val == np.nan:
            df[col] = df[col].map({np.nan: 0})
        else:
            df[col] = df[col].map({val: 1, np.nan: 0})
        
        col_dict[col] = val

    df.rename(columns=col_dict, inplace=True)
    
    return df


def order_uni(q: str, title: str, start: int=0, end:int=-1, ax: np.ndarray=None):
    """
        Plots horizontal bar chart of a uni-option question
        
        input:
            q: question number
            title: title of the bar chart
            start: starting index for slicing
            end: ending index for slicing
        
        return:
            horizontal bar chart
    """
    return (res.loc[:, q].value_counts(normalize=True)[start:end].sort_values(ascending=True) * 100).plot.barh(title=title, ax=ax)