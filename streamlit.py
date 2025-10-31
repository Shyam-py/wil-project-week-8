import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


st.set_page_config(page_title="AutoBasket Recommender", layout="wide")

st.title("AutoBasket Recipe Recommender")

@st.cache_data
def load_data(path="Indian_Community_Recipe_Preferences_Ontario_Enhanced.csv"):
    df = pd.read_csv(path)
    return df

df = load_data()
st.sidebar.header("Dataset")
st.sidebar.write(f"Records: {df.shape[0]}")
st.sidebar.write(f"Unique recipes: {df['recipe_name'].nunique()}")

# show top few rows
if st.sidebar.checkbox("Show raw data", False):
    st.dataframe(df.head(200))

# Pre-load CF structures (cached)
@st.cache_resource
def build_cf(df):
    # pivot (items x users)
    item_user = df.pivot_table(index='recipe_name', columns='user_id', values='rating').fillna(0)
    item_similarity = cosine_similarity(item_user.values)
    item_similarity_df = pd.DataFrame(item_similarity, index=item_user.index, columns=item_user.index)
    return item_user, item_similarity_df

item_user, item_similarity_df = build_cf(df)

@st.cache_resource
def build_content(df):
    df['combined'] = df['recipe_name'] + " " + df['ingredient_category'].fillna('') + " " + df['meal_type'].fillna('') + " " + df['review_text'].fillna('')
    recipe_docs = df.groupby('recipe_name')['combined'].apply(lambda x: " ".join(x)).reset_index()
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(recipe_docs['combined'])
    cb_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    cb_sim_df = pd.DataFrame(cb_sim, index=recipe_docs['recipe_name'], columns=recipe_docs['recipe_name'])
    return recipe_docs, cb_sim_df

recipe_docs, cb_sim_df = build_content(df)

# Recommendation functions
def recommend_cf(user_id, top_n=10):
    if user_id not in item_user.columns:
        return []
    user_ratings = item_user[user_id]
    rated = user_ratings[user_ratings > 0]
    if rated.empty:
        return []
    preds = {}
    unrated = user_ratings[user_ratings == 0].index.tolist()
    for it in unrated:
        sims = item_similarity_df.loc[it, rated.index]
        num = np.dot(sims.values, rated.values)
        den = np.sum(np.abs(sims.values)) + 1e-9
        preds[it] = num/den
    sorted_preds = sorted(preds.items(), key=lambda x: x[1], reverse=True)
    return [r for r,s in sorted_preds[:top_n]]

def recommend_content(recipe_name, top_n=5):
    if recipe_name not in cb_sim_df.index:
        return []
    scores = cb_sim_df[recipe_name].sort_values(ascending=False)
    return list(scores.index[1:top_n+1])

def recommend_hybrid(user_id=None, liked_recipe=None, top_n=5):
    if user_id and user_id in item_user.columns:
        cf_recs = recommend_cf(user_id, top_n=top_n)
        if cf_recs:
            return cf_recs
    if liked_recipe:
        return recommend_content(liked_recipe, top_n=top_n)
    return df['recipe_name'].value_counts().head(top_n).index.tolist()



st.header("Find Similar Recipes")
recipe_list = sorted(df['recipe_name'].unique().tolist())
chosen_recipe = st.selectbox("Pick a recipe", recipe_list)
if st.button("Find similar recipes"):
    sim = recommend_content(chosen_recipe, top_n=4)
    if sim:
        for i,r in enumerate(sim,1):
            st.write(f"{i}. {r}")
    else:
        st.warning("No similar recipes found.")


st.markdown("---")

