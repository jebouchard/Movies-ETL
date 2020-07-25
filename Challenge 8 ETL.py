import json
import pandas as pd
import numpy as np
import time

import re

from sqlalchemy import create_engine
import psycopg2

from config import db_password


# In[3]:





# In[4]:


with open(f'{file_dir}/wikipedia.movies.json', mode='r') as file:
    wiki_movies_raw = json.load(file)


# In[5]:


len(wiki_movies_raw)


# In[6]:


# First 5 records
wiki_movies_raw[:5]


# In[7]:


# Last 5 records
wiki_movies_raw[-5:]


# In[8]:


# Some records in the middle
wiki_movies_raw[3600:3605]


# In[9]:


kaggle_metadata = pd.read_csv(f'{file_dir}movies_metadata.csv')
ratings = pd.read_csv(f'{file_dir}ratings.csv')


# In[10]:


kaggle_metadata.head()


# In[11]:


ratings.head()


# In[12]:


ratings.sample(5)


# In[13]:


wiki_movies_df = pd.DataFrame(wiki_movies_raw)


# In[14]:


wiki_movies_df.head()


# In[15]:


wiki_movies_df.columns.tolist()


# In[16]:


wiki_movies = [movie for movie in wiki_movies_raw
               if ('Director' in movie or 'Directed by' in movie)
                   and 'imdb_link' in movie]
len(wiki_movies)


# In[17]:


wiki_movies = [movie for movie in wiki_movies_raw
               if ('Director' in movie or 'Directed by' in movie)
                   and 'imdb_link' in movie
                   and 'No. of episodes' not in movie]


# In[18]:


def clean_movie(movie):
    movie = dict(movie) #create a non-destructive copy
    return movie


# In[19]:


sorted(wiki_movies_df.columns.tolist())


# In[20]:


def clean_movie(movie):
    movie = dict(movie) #create a non-destructive copy
    alt_titles = {}
    # combine alternate titles into one list
    for key in ['Also known as','Arabic','Cantonese','Chinese','French',
                'Hangul','Hebrew','Hepburn','Japanese','Literally',
                'Mandarin','McCune-Reischauer','Original title','Polish',
                'Revised Romanization','Romanized','Russian',
                'Simplified','Traditional','Yiddish']:
        if key in movie:
            alt_titles[key] = movie[key]
            movie.pop(key)
    if len(alt_titles) > 0:
        movie['alt_titles'] = alt_titles

    # merge column names
    def change_column_name(old_name, new_name):
        if old_name in movie:
            movie[new_name] = movie.pop(old_name)
    change_column_name('Adaptation by', 'Writer(s)')
    change_column_name('Country of origin', 'Country')
    change_column_name('Directed by', 'Director')
    change_column_name('Distributed by', 'Distributor')
    change_column_name('Edited by', 'Editor(s)')
    change_column_name('Length', 'Running time')
    change_column_name('Original release', 'Release date')
    change_column_name('Music by', 'Composer(s)')
    change_column_name('Produced by', 'Producer(s)')
    change_column_name('Producer', 'Producer(s)')
    change_column_name('Productioncompanies ', 'Production company(s)')
    change_column_name('Productioncompany ', 'Production company(s)')
    change_column_name('Released', 'Release Date')
    change_column_name('Release Date', 'Release date')
    change_column_name('Screen story by', 'Writer(s)')
    change_column_name('Screenplay by', 'Writer(s)')
    change_column_name('Story by', 'Writer(s)')
    change_column_name('Theme music composer', 'Composer(s)')
    change_column_name('Written by', 'Writer(s)')

    return movie


# In[21]:


clean_movies = [clean_movie(movie) for movie in wiki_movies]
wiki_movies_df = pd.DataFrame(clean_movies)


# In[22]:


wiki_movies_df['imdb_id'] = wiki_movies_df['imdb_link'].str.extract(r'(tt\d{7})')
print(len(wiki_movies_df))
wiki_movies_df.drop_duplicates(subset='imdb_id', inplace=True)
print(len(wiki_movies_df))
wiki_movies_df.head()


# In[23]:


[[column,wiki_movies_df[column].isnull().sum()] for column in wiki_movies_df.columns]


# In[24]:


[column for column in wiki_movies_df.columns if wiki_movies_df[column].isnull().sum() < len(wiki_movies_df) * 0.9]


# In[25]:


wiki_columns_to_keep = [column for column in wiki_movies_df.columns if wiki_movies_df[column].isnull().sum() < len(wiki_movies_df) * 0.9]
wiki_movies_df = wiki_movies_df[wiki_columns_to_keep]


# In[26]:


wiki_movies_df.dtypes


# In[27]:


box_office = wiki_movies_df['Box office'].dropna() 


# In[28]:


len(box_office)


# In[29]:


def is_not_a_string(x):
    return type(x) != str


# In[30]:


box_office[box_office.map(is_not_a_string)]


# In[31]:


box_office[box_office.map(lambda x: type(x) != str)]


# In[32]:


box_office = box_office.apply(lambda x: ' '.join(x) if type(x) == list else x)


# In[33]:


import re


# In[34]:


form_one = r'\$\d+\.?\d*\s*[mb]illion'
box_office.str.contains(form_one, flags=re.IGNORECASE).sum()


# In[35]:


form_two = r'\$\d{1,3}(?:,\d{3})+'
box_office.str.contains(form_two, flags=re.IGNORECASE).sum()


# In[36]:


matches_form_one = box_office.str.contains(form_one, flags=re.IGNORECASE)
matches_form_two = box_office.str.contains(form_two, flags=re.IGNORECASE)


# In[37]:


box_office[~matches_form_one & ~matches_form_two]


# In[38]:


form_one = r'\$\s*\d+\.?\d*\s*[mb]illion'
form_two = r'\$\s*\d{1,3}(?:,\d{3})+'


# In[39]:


form_two = r'\$\s*\d{1,3}(?:[,\.]\d{3})+'


# In[40]:


form_two = r'\$\s*\d{1,3}(?:[,\.]\d{3})+(?!\s[mb]illion)'


# In[41]:


box_office = box_office.str.replace(r'\$.*[-—–](?![a-z])', '$', regex=True)


# In[42]:


form_one = r'\$\s*\d+\.?\d*\s*[mb]illi?on'


# In[43]:


box_office.str.extract(f'({form_one}|{form_two})')


# In[44]:


def parse_dollars(s):
    # if s is not a string, return NaN
    if type(s) != str:
        return np.nan

    # if input is of the form $###.# million
    if re.match(r'\$\s*\d+\.?\d*\s*milli?on', s, flags=re.IGNORECASE):

        # remove dollar sign and " million"
        s = re.sub('\$|\s|[a-zA-Z]','', s)

        # convert to float and multiply by a million
        value = float(s) * 10**6

        # return value
        return value

    # if input is of the form $###.# billion
    elif re.match(r'\$\s*\d+\.?\d*\s*billi?on', s, flags=re.IGNORECASE):

        # remove dollar sign and " billion"
        s = re.sub('\$|\s|[a-zA-Z]','', s)

        # convert to float and multiply by a billion
        value = float(s) * 10**9

        # return value
        return value

    # if input is of the form $###,###,###
    elif re.match(r'\$\s*\d{1,3}(?:[,\.]\d{3})+(?!\s[mb]illion)', s, flags=re.IGNORECASE):

        # remove dollar sign and commas
        s = re.sub('\$|,','', s)

        # convert to float
        value = float(s)

        # return value
        return value

    # otherwise, return NaN
    else:
        return np.nan


# In[45]:


wiki_movies_df['box_office'] = box_office.str.extract(f'({form_one}|{form_two})', flags=re.IGNORECASE)[0].apply(parse_dollars)


# In[46]:


wiki_movies_df.drop('Box office', axis=1, inplace=True)


# In[47]:


budget = wiki_movies_df['Budget'].dropna()


# In[48]:


budget = budget.map(lambda x: ' '.join(x) if type(x) == list else x)


# In[49]:


budget = budget.str.replace(r'\$.*[-—–](?![a-z])', '$', regex=True)


# In[50]:


matches_form_one = budget.str.contains(form_one, flags=re.IGNORECASE)
matches_form_two = budget.str.contains(form_two, flags=re.IGNORECASE)
budget[~matches_form_one & ~matches_form_two]


# In[51]:


budget = budget.str.replace(r'\[\d+\]\s*', '')
budget[~matches_form_one & ~matches_form_two]


# In[52]:


wiki_movies_df['budget'] = budget.str.extract(f'({form_one}|{form_two})', flags=re.IGNORECASE)[0].apply(parse_dollars)


# In[53]:


wiki_movies_df.drop('Budget', axis=1, inplace=True)


# In[54]:


release_date = wiki_movies_df['Release date'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)


# In[55]:


date_form_one = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s[123]\d,\s\d{4}'
date_form_two = r'\d{4}.[01]\d.[123]\d'
date_form_three = r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s\d{4}'
date_form_four = r'\d{4}'


# In[56]:


release_date.str.extract(f'({date_form_one}|{date_form_two}|{date_form_three}|{date_form_four})', flags=re.IGNORECASE)


# In[57]:


wiki_movies_df['release_date'] = pd.to_datetime(release_date.str.extract(f'({date_form_one}|{date_form_two}|{date_form_three}|{date_form_four})')[0], infer_datetime_format=True)


# In[58]:


running_time = wiki_movies_df['Running time'].dropna().apply(lambda x: ' '.join(x) if type(x) == list else x)


# In[59]:


running_time.str.contains(r'^\d*\s*minutes$', flags=re.IGNORECASE).sum()


# In[60]:


running_time[running_time.str.contains(r'^\d*\s*minutes$', flags=re.IGNORECASE) != True]


# In[61]:


running_time.str.contains(r'^\d*\s*m', flags=re.IGNORECASE).sum()


# In[62]:


running_time[running_time.str.contains(r'^\d*\s*m', flags=re.IGNORECASE) != True]


# In[63]:


running_time_extract = running_time.str.extract(r'(\d+)\s*ho?u?r?s?\s*(\d*)|(\d+)\s*m')


# In[64]:


running_time_extract = running_time_extract.apply(lambda col: pd.to_numeric(col, errors='coerce')).fillna(0)


# In[65]:


wiki_movies_df['running_time'] = running_time_extract.apply(lambda row: row[0]*60 + row[1] if row[2] == 0 else row[2], axis=1)


# In[66]:


wiki_movies_df.drop('Running time', axis=1, inplace=True)


# In[67]:


kaggle_metadata.dtypes


# In[68]:


kaggle_metadata['adult'].value_counts()


# In[69]:


kaggle_metadata[~kaggle_metadata['adult'].isin(['True','False'])]


# In[70]:


kaggle_metadata = kaggle_metadata[kaggle_metadata['adult'] == 'False'].drop('adult',axis='columns')


# In[71]:


kaggle_metadata['video'].value_counts()


# In[72]:


kaggle_metadata['video'] == 'True'


# In[73]:


kaggle_metadata['video'] = kaggle_metadata['video'] == 'True'


# In[74]:


kaggle_metadata['budget'] = kaggle_metadata['budget'].astype(int)
kaggle_metadata['id'] = pd.to_numeric(kaggle_metadata['id'], errors='raise')
kaggle_metadata['popularity'] = pd.to_numeric(kaggle_metadata['popularity'], errors='raise')


# In[75]:


kaggle_metadata['release_date'] = pd.to_datetime(kaggle_metadata['release_date'])


# In[76]:


ratings.info(null_counts=True)


# In[77]:


pd.to_datetime(ratings['timestamp'], unit='s')


# In[78]:


ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')


# In[79]:


ratings['rating'].plot(kind='hist')
ratings['rating'].describe()


# In[80]:


movies_df = pd.merge(wiki_movies_df, kaggle_metadata, on='imdb_id', suffixes=['_wiki','_kaggle'])


# In[81]:


movies_df[['title_wiki','title_kaggle']]


# In[82]:


movies_df[movies_df['title_wiki'] != movies_df['title_kaggle']][['title_wiki','title_kaggle']]


# In[83]:


movies_df[(movies_df['title_kaggle'] == '') | (movies_df['title_kaggle'].isnull())]


# In[84]:


movies_df.fillna(0).plot(x='running_time', y='runtime', kind='scatter')


# In[85]:


movies_df.fillna(0).plot(x='budget_wiki',y='budget_kaggle', kind='scatter')


# In[86]:


movies_df.fillna(0).plot(x='box_office', y='revenue', kind='scatter')


# In[87]:


movies_df.fillna(0)[movies_df['box_office'] < 10**9].plot(x='box_office', y='revenue', kind='scatter')


# In[88]:


movies_df[['release_date_wiki','release_date_kaggle']].plot(x='release_date_wiki', y='release_date_kaggle', style='.')


# In[89]:


movies_df[(movies_df['release_date_wiki'] > '1996-01-01') & (movies_df['release_date_kaggle'] < '1965-01-01')]


# In[90]:


movies_df[(movies_df['release_date_wiki'] > '1996-01-01') & (movies_df['release_date_kaggle'] < '1965-01-01')].index


# In[91]:


movies_df = movies_df.drop(movies_df[(movies_df['release_date_wiki'] > '1996-01-01') & (movies_df['release_date_kaggle'] < '1965-01-01')].index)


# In[92]:


movies_df[movies_df['release_date_wiki'].isnull()]


# In[93]:


movies_df['Language'].value_counts()


# In[94]:


movies_df['Language'].apply(lambda x: tuple(x) if type(x) == list else x).value_counts(dropna=False)


# In[95]:


movies_df['original_language'].value_counts(dropna=False)


# In[96]:


movies_df[['Production company(s)','production_companies']]


# In[97]:


movies_df.drop(columns=['title_wiki','release_date_wiki','Language','Production company(s)'], inplace=True)


# In[98]:


def fill_missing_kaggle_data(df, kaggle_column, wiki_column):
    df[kaggle_column] = df.apply(
        lambda row: row[wiki_column] if row[kaggle_column] == 0 else row[kaggle_column]
        , axis=1)
    df.drop(columns=wiki_column, inplace=True)


# In[99]:


fill_missing_kaggle_data(movies_df, 'runtime', 'running_time')
fill_missing_kaggle_data(movies_df, 'budget_kaggle', 'budget_wiki')
fill_missing_kaggle_data(movies_df, 'revenue', 'box_office')
movies_df


# In[100]:


for col in movies_df.columns:
    lists_to_tuples = lambda x: tuple(x) if type(x) == list else x
    value_counts = movies_df[col].apply(lists_to_tuples).value_counts(dropna=False)
    num_values = len(value_counts)
    if num_values == 1:
        print(col)


# In[101]:


movies_df['video'].value_counts(dropna=False)


# In[102]:


movies_df = movies_df.loc[:, ['imdb_id','id','title_kaggle','original_title','tagline','belongs_to_collection','url','imdb_link',
                       'runtime','budget_kaggle','revenue','release_date_kaggle','popularity','vote_average','vote_count',
                       'genres','original_language','overview','spoken_languages','Country',
                       'production_companies','production_countries','Distributor',
                       'Producer(s)','Director','Starring','Cinematography','Editor(s)','Writer(s)','Composer(s)','Based on'
                      ]]


# In[103]:


movies_df.rename({'id':'kaggle_id',
                  'title_kaggle':'title',
                  'url':'wikipedia_url',
                  'budget_kaggle':'budget',
                  'release_date_kaggle':'release_date',
                  'Country':'country',
                  'Distributor':'distributor',
                  'Producer(s)':'producers',
                  'Director':'director',
                  'Starring':'starring',
                  'Cinematography':'cinematography',
                  'Editor(s)':'editors',
                  'Writer(s)':'writers',
                  'Composer(s)':'composers',
                  'Based on':'based_on'
                 }, axis='columns', inplace=True)


# In[104]:


"postgres://[user]:[password]@[location]:[port]/[database]"


# In[105]:


rating_counts = ratings.groupby(['movieId','rating'], as_index=False).count()


# In[106]:


rating_counts = ratings.groupby(['movieId','rating'], as_index=False).count()                 .rename({'userId':'count'}, axis=1) 


# In[107]:


rating_counts = ratings.groupby(['movieId','rating'], as_index=False).count()                 .rename({'userId':'count'}, axis=1)                 .pivot(index='movieId',columns='rating', values='count')


# In[108]:


rating_counts.columns = ['rating_' + str(col) for col in rating_counts.columns]


# In[109]:


movies_with_ratings_df = pd.merge(movies_df, rating_counts, left_on='kaggle_id', right_index=True, how='left')


# In[110]:


movies_with_ratings_df[rating_counts.columns] = movies_with_ratings_df[rating_counts.columns].fillna(0)


# In[119]:


from sqlalchemy import create_engine
from config import db_password


# In[120]:


db_string = f"postgres://postgres:{db_password}@localhost:5432/movie_data"


# In[121]:


engine = create_engine(db_string)


# In[122]:


movies_df.to_sql(name='movies', con=engine)


# In[126]:


import time


# In[ ]:


rows_imported = 0
# get the start_time from time.time()
start_time = time.time()
for data in pd.read_csv(f'{file_dir}ratings.csv', chunksize=1000000):
    print(f'importing rows {rows_imported} to {rows_imported + len(data)}...', end='')
    data.to_sql(name='ratings', con=engine, if_exists='append')
    rows_imported += len(data)

    # add elapsed time to final print out
    print(f'Done. {time.time() - start_time} total seconds elapsed')

