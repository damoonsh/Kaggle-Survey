# 2020 Kaggle Machine Learning & Data Science Survey

## Context
Code within this repository is the work I have done to analyze to gain some insight from Kaggle's <a href="https://www.kaggle.com/c/kaggle-survey-2020">Survey Competition<a>. 

# Data Preprocessing
The survey data has a specific structure where it would be convinient to clean and change some attributes of the dataframe. Some problems:
- Column names are verbose and they are not descriptive. Column names have been changed by replacing '_Part_' and '_' with '.', 'OTHER' with '0':
    ```python
    # Formating the columns for comfortable access
    column_dict = {}

    # Getting the dictionary needed to change the column names
    for col in res.columns:
        n_col = col.replace('_Part_', '.')
        n_col = n_col.replace('_', '.')
        n_col = n_col.replace('OTHER', '0')
        column_dict[col] = n_col
        
    # Rename the columns
    res.rename(columns=column_dict, inplace=True)

    # Getting the uni-option columns/questions
    uo_cols = []
    for col in res.columns[1:]:
        if not '.' in col:
            uo_cols.append(col)
            
    res.drop([0], inplace=True)
    ```
- Most columns are actually options for a question. It would help the analysis to break down the dataframe into subset of questions. This would mean that for each question we will have n columns representing n possible answers (to that question) in the binary format.
    ```python
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
    ```
- The majority of missing values are because of the multi optional questions but they are some uni-option question with missing values that need imputing. Yet we have to go through each question to find the feasible value to fill the missing values with.
- Some unique values within the columns have verbose entries which can be shortened:

    For instance in the country column, United States of America can be shortened to US, this makes the graphs look neater. Also, I googled South Korea and Republic of Korea are the same thing.
    ```python
    # Shortening some of the names
    country_dict = {
        'United Kingdom of Great Britain and Northern Ireland': 'UK',
        'United States of America': 'US',
        'Republic of Korea': 'Korea',
        'United Arab Emirates': 'UAE',
        'South Korea': 'Korea',
        'Republic of Korea': 'Korea',
        'Iran, Islamic Republic of...': 'Iran'
    }
    res['Q3'] = res['Q3'].replace(country_dict)
    ```

At the end of preprocessing, the data has been cleaned and broken into mutlitple sub-dataframes based on the questions.

This how the functions will used to break down the dataframe
```python
# Mulit-Option Question Break downs
language = break_down(7, 20) # Q7
ide = break_down(21, 33) # Q9
host_prod = break_down(33, 47) # Q10
spec_hardware = break_down(48, 52) # Q12: TPU, GPU
viz_lib = break_down(53, 65) # Q14: Which data visualization lib do u use?
ml_lib =  break_down(66, 82) # Q16: Regularly used ML libraries
algo = break_down(83, 94) # Q17: Regularly used ML Algorithms
comp_vision = break_down(94, 101) # Q18: Algorithms related to Computer Vision
nlp = break_down(101, 107) # Q19: Algorithms related to NLP
work_activity = break_down(110, 118) # Q23
cloud_platform = break_down(120, 132) # 26.a
```

## Analysis Utilities
While going through various questions, it is helpful to visualize the data and try to focus on specific aspect of the data.

The function below is extensively used in my project for visualization:
```python
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
```

# Analysis
I don't believe anyone knows if they are going to find something significant when they go through a dataset. I hypothesize that analysis techniques aside, Brain Storming on the data could help the creeative process. And by brainstorming, I mean going through data and trying to challenge your thoughts on the matter.

- Q1: Age
    - Majority of Kagglers are between 25-29 age range
    - It is safe to say that the majority of the Kagglers are between 18-29 years old. The ranges above 30 are in order (30-34 followed by 35-39 and etc.)
    - Interestingly they are some 70+ participatin in Kaggle.

- Q3: Country
    - Kagglers are from 55 different countries (Counting the others).
    - In the first 10 countries with the most Kagglers, India is at the lead. Which is interesting since India and Nigeria are the only two non-first world countries in top 10. In future, we might change our definition of first and second world.
    - Based on China's economy and tech initiatives, one might expect to see China in top five. Yet, interestingly China is below US and India with a large difference.
    - Looking at the data, we can see that people from all over the world are participating

## Relation between infrastructure and Kaggling:
It is logical to expect the First-word countries have a higher participation rate with in the Kaggle community. It is less likely for people in Syria (where a civil war is going on) to have the time to use Kaggle. I suspect there is also more value to doing Kaggle competitions in First-wolrd countries since the employers will acknowledge your talent given your participation level on Kaggle.
- India has the most Kagglers. There several factors to consider:
    - India has the 2nd largest population: Because of its large population, even if a smaller proportion of the society were involved on Kaggle still it would could still be at first rank.
    - India is a development country (2nd world): This shows that given India is not a first-world country, it has the needed infrastructure (internet connection, educational resouces, and etc.) for individuals to participate on Kaggle. Same thing is true about Nigeria.
- Nigeria: It is one of the two development countries in top 10. Which implies that Nigerians have the infrastructure and are motivated enough to participate on kaggle.
- As we go down the list, we see less First-World countries and more development countries. We even come across some Third-World coutnries (Iran).