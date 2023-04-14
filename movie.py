import numpy as np
import pandas as pd
import urllib.request
from sklearn.model_selection import train_test_split
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import requests 
from PIL import Image
st.write('hello')
@st.cache_data
def get_poster_url(title):
  url = "http://www.omdbapi.com/?apikey=5cb686ca&t=" + title
  response = requests.get(url)
  data = response.json()
  print(data)
  return data["Poster"]


movies = pd.read_csv("C:/Users/papir/OneDrive/Desktop/movie_data.csv", encoding='ISO-8859-1')




movies.isna().sum()

movies.fillna(" ",inplace=True)

movies.isnull().sum()
features = ["director_name","genres","movie_title","actor_1_name","language","country"]
remove_features = ["actor_2_name","actor_3_name","movie_imdb_link",]
movies.drop(columns=remove_features,axis=1,inplace=True)
movies["movie_title"] = movies["movie_title"].str.strip()
movies['director_name'] = movies["director_name"].str.strip()
movies['index'] = [i for i in range(0,len(movies))]
movies.head()
combined_features = movies["director_name"]+" "+movies["genres"]+" "+movies["movie_title"]+" "+movies["actor_1_name"]+" "+movies["language"]+" "+movies["country"]
vectorizer = TfidfVectorizer()
feature_vectors = vectorizer.fit_transform(combined_features)
movie_names = movies['movie_title'].tolist()
similarity = cosine_similarity(feature_vectors)

input_movie = st.text_input("Enter the Movie Name")
if input_movie:
  recommended_movies = difflib.get_close_matches(input_movie,movie_names,cutoff=0.6,n = 5)
  close_match = recommended_movies[0]
  

  movie_index = movies[movies['movie_title'] == close_match].index[0]
  ind = movies.loc[movie_index]['index']

  similarity_score = list(enumerate(similarity[ind]))



  movies_list = sorted(similarity_score,key = lambda x:x[1],reverse=True)
  

  st.write("Here are your recommended movies list..")
  c1,c2 = st.columns(2)
  r_movies = []
  for i in range(len(movies_list)):
    index = movies_list[i][0]
    title = movies[movies.index==index]['movie_title'].values[0]
    if title not in r_movies:

      if (i<30):
        poster_url = get_poster_url(title)
        filename, headers = urllib.request.urlretrieve(poster_url)
        i+=1
        img = Image.open(filename)
        img = img.resize((200,200))
        if i%2==0:
          with c1:
            st.image(img, caption=title, width=200)
        else:
          with c2:
            st.image(img,caption=title,width=200)
    r_movies.append(title)
def click_genres():
  genres_list = movies['genres'].str.split('|').explode().unique().tolist()
  genres = st.sidebar.multiselect("genres",options=genres_list)
  
  

st.sidebar.header("Search based on")

movie_dirs = movies['director_name'].tolist()
dir_input = st.sidebar.text_input("Director Name")

sub_but = st.sidebar.button("click me")
gen_but = st.sidebar.button('Genres')
if gen_but:
  click_genres()
if sub_but:
  st.empty()
rec_dirs = difflib.get_close_matches(dir_input,movie_dirs,cutoff=0.6,n=5)
dir_input = rec_dirs[0]
if dir_input:
  st.write("Movies directed by ",dir_input)
  

dir_mov_list = []
c3,c4 = st.columns(2)
for i in range(len(movies)):
  if movies['director_name'][i] == dir_input:
    dir_mov_list.append(movies['movie_title'][i])
j = 0
for i in dir_mov_list:
  poster_url = get_poster_url(i)
  filename,headers = urllib.request.urlretrieve(poster_url)
  img = Image.open(filename)
  if j%2==0:
    with c3:
      st.image(img,caption=i,width=200)
  else:
    with c4:
      st.image(img,caption=i,width=200)
  j+=1 


