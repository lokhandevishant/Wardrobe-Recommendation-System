# manage.py
import streamlit as st
import pickle
import re
import pandas as pd
from PIL import Image
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Upload_database import insert_clothing_item, get_all_clothing_items, get_clothing_item_by_id
from Recommendation import generate_outfit_combinations
import webbrowser

# Initialize session state for navigation
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# Function for text cleaning
def clean_text(text: str) -> str:
    lemmatizer = WordNetLemmatizer()
    cleaned_text = re.sub('[^a-zA-Z]', ' ', text).lower().split()
    cleaned_text = [lemmatizer.lemmatize(word) for word in cleaned_text if word not in stopwords.words('english')]
    return ' '.join(cleaned_text)

# Function for recommending clothes
def recommend_clothes(text: str, top_num: int):
    clothing_df = pickle.load(open('clothing.pkl', 'rb'))
    cleaned_text = clean_text(text)
    cleaned_text_as_series = pd.Series([cleaned_text])
    descriptions = clothing_df['cleaned_description']
    description_with_new_text = pd.concat([descriptions, cleaned_text_as_series]).reset_index(drop=True)
    vec = CountVectorizer(max_features=5000, stop_words='english')
    vectors = vec.fit_transform(description_with_new_text).toarray()
    similarity_scores = cosine_similarity(vectors)
    input_description_index = description_with_new_text[description_with_new_text == cleaned_text].index[0]
    distances = similarity_scores[input_description_index]
    clothing_items_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:int(top_num) + 1]
    clothing_item_details = [(clothing_df.iloc[each[0]]).to_dict() for each in clothing_items_list]
    return clothing_item_details

# Function for formatting items for LLM
def format_items_for_llm(items):
    formatted_items = []
    for item in items:
        item_id, item_type, color, style, image_path, _ = item
        formatted_items.append({
            "id": item_id,
            "type": item_type,
            "color": color,
            "style": style
        })
    return formatted_items

# Home Page
def home():
    st.title("Welcome to the Clothing Management System")
    st.write("Please choose an option below:")
    if st.button("Outfit Planner"):
        st.session_state.page = 'outfit_planner'
    if st.button("Cloth Searching System"):
        st.session_state.page = 'cloth_searching'

# Outfit Planner Page
def outfit_planner():
    st.title("Weekly Outfit Planner")
    st.write("Upload new items or generate weekly outfits.")
    st.sidebar.button("Home", on_click=lambda: st.session_state.__setitem__('page', 'home'))

    with st.sidebar:
        st.header("Add New Clothing Item")
        uploaded_file = st.file_uploader("Upload a clothing item image", type=["jpg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image.thumbnail((100, 100))
            st.image(image, caption="Preview", use_column_width=False, width=100)

            item_type = st.selectbox("Item type", ["top", "bottom"])
            color = st.text_input("Color")
            style = st.text_input("Style")
            
            if st.button("Save Item"):
                uploaded_file.seek(0)
                item_id = insert_clothing_item(item_type, color, style, uploaded_file)
                st.success(f"Item saved successfully with ID: {item_id}")

    st.header("Generate Weekly Outfits")
    if st.button("Generate Outfits"):
        items = get_all_clothing_items()
        formatted_items = format_items_for_llm(items)
        outfit_combinations = generate_outfit_combinations(formatted_items)
        for outfit in outfit_combinations:
            st.subheader(f"Day {outfit['day']}")
            col1, col2 = st.columns(2)

            top_item = get_clothing_item_by_id(outfit['top'])
            try:
                top_image = Image.open(top_item[4])
                top_image.thumbnail((150, 150))
                with col1:
                    st.image(top_image, caption=f"Top: {top_item[3]} ({top_item[2]})", use_column_width=False, width=150)
            except Exception as e:
                st.error(f"Error loading top image: {e}")

            bottom_item = get_clothing_item_by_id(outfit['bottom'])
            try:
                bottom_image = Image.open(bottom_item[4])
                bottom_image.thumbnail((150, 150))
                with col2:
                    st.image(bottom_image, caption=f"Bottom: {bottom_item[3]} ({bottom_item[2]})", use_column_width=False, width=150)
            except Exception as e:
                st.error(f"Error loading bottom image: {e}")

# Cloth Searching System Page
def cloth_searching():
    st.title("Cloth Searching System")
    st.sidebar.button("Home", on_click=lambda: st.session_state.__setitem__('page', 'home'))

    item_name = st.text_input("Enter the item name:")
    if st.button("Search"):
        if item_name:
            recommendations = recommend_clothes(item_name, 15)
            if recommendations:
                st.subheader("Recommended Items:")
                cols = [st.columns(3) for _ in range((len(recommendations) + 2) // 3)]
                for i, item in enumerate(recommendations):
                    col = cols[i // 3][i % 3]
                    with col:
                        st.write(f"Item Name: {item.get('name', 'N/A')}")
                        try:
                            st.image(item['img'], width=200)
                        except Exception as e:
                            st.warning(f"Error displaying image for item: {item.get('name', 'N/A')}. Reason: {e}")
                        button_key = f"visit_button_{i}"
                        st.button('Visit', key=button_key, on_click=lambda url=item['url']: webbrowser.open_new_tab(url))
            else:
                st.warning("No recommendations found for the given item.")
        else:
            st.warning("Please enter an item name.")

# Page navigation
if st.session_state.page == 'home':
    home()
elif st.session_state.page == 'outfit_planner':
    outfit_planner()
elif st.session_state.page == 'cloth_searching':
    cloth_searching()
