#!/usr/bin/env python
# coding: utf-8

# # Build a model that predicts the rating of a movie based on features like genre, director, and actors. You can use regression techniques to tackle this problem.

# In[49]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
datas = pd.read_csv('C:/Users/ASUS/OneDrive/Desktop/archive (11)/IMDb Movies India.csv', encoding='ISO-8859-1')
datas


# In[50]:


datas.info()


# In[51]:


datas=datas.drop(['Duration','Actor 1','Actor 2','Actor 3'],axis=1)


# In[52]:


datas.isnull().sum()


# In[53]:


datas=datas.dropna()


# In[54]:


datas.isnull().sum()


# In[55]:


datas.info()


# In[56]:


from sklearn.preprocessing import LabelEncoder
encode_gen = LabelEncoder()
datas['gen_enc'] = encode_gen.fit_transform(datas['Genre'])


# In[64]:


datas['Year'] = datas['Year'].str.extract('(\d+)', expand = False).astype(int)


# In[65]:


datas


# In[66]:


enc_dir = LabelEncoder()
datas['dir_encoded'] =  enc_dir.fit_transform(datas['Director'])


# In[67]:


datas.info()


# In[68]:


datas['dir_rating_mean'] = datas.groupby('Director')['Rating'].transform('mean')
datas['Votes'] = pd.to_numeric(datas['Votes'], errors='coerce')
datas['log_votes'] = np.log1p(datas['Votes'])
datas['Rating'] = pd.to_numeric(datas['Rating'],errors='coerce')
datas['Year'] = pd.to_numeric(datas['Year'],errors='coerce')


# In[72]:


x = datas[['Year','gen_enc','dir_encoded']]
y = datas[['Rating']]


# In[73]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)


# In[74]:


model = RandomForestRegressor()
model.fit(x_train,y_train)


# In[75]:


y_predict = model.predict(x_test)


# In[78]:


from sklearn.metrics import mean_absolute_error,mean_squared_error
mae = mean_absolute_error(y_test,y_predict)


# In[79]:


mae


# In[90]:


def predict_rating(movie_name):
    movie_data = datas[datas['Name'].str.contains(movie_name, case=False, na=False)]
    
    if movie_data.empty:
        return "Movie not found in dataset."
    genre = movie_data['Genre'].values[0]
    director = movie_data['Director'].values[0]
    year = movie_data['Year'].values[0]
    gen_enc = encode_gen.transform([genre])[0]
    dir_enc = enc_dir.transform([director])[0]
    
    input_data = pd.DataFrame([[year, gen_enc, dir_enc]], columns=['Year', 'gen_enc', 'dir_encoded'])
    
    predicted_rating = model.predict(input_data)[0]
    
    return f"Predicted Rating: {predicted_rating:.2f} (Genre: {genre}, Director: {director}, Year: {year})"


movie = input("Enter the movie name:")
print(predict_rating(movie))


# In[88]:


datas


# In[ ]:





# In[ ]:




