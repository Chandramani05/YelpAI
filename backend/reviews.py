from bs4 import BeautifulSoup
import re
import requests
import os
from bs4 import BeautifulSoup
import re

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

import tiktoken  
from dotenv import load_dotenv

load_dotenv()


PINECONE_API_KEY = os.environ['PINECONE_API_KEY']
OPEN_AI_KEY = os.environ['OPEN_API_KEY']
tokenizer = tiktoken.get_encoding('p50k_base')

def get_reviews (restaurant) :
    url = restaurant['url']
    #print(url)
    r1 = requests.get(f'{url}&sort_by=date_desc')
    soup = BeautifulSoup(r1.text, 'html.parser')
    regex = re.compile('.*comment.*')
    results = soup.find_all('p', {'class':regex})
    reviews1 = [result.text for result in results][:10]

    r2 = requests.get(f'{url}')
    soup = BeautifulSoup(r2.text, 'html.parser')
    regex = re.compile('.*comment.*')
    results = soup.find_all('p', {'class':regex})
    reviews2 = [result.text for result in results][:10]


    r3 = requests.get(f'{url}&sort_by=rating_asc')
    soup = BeautifulSoup(r3.text, 'html.parser')
    regex = re.compile('.*comment.*')
    results = soup.find_all('p', {'class':regex})
    reviews3 = [result.text for result in results][:10]


    all_review = f"Recent Review\n {reviews1} \n Top Reviews {reviews2} \n Negative Review {reviews3}"
    return all_review


def get_menu(id):
    url = f'https://www.yelp.com/menu/{id}'
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    menu_items = []

    sections = soup.find_all("div", class_="menu-sections")
    for section in sections:
        # Find the section header which contains the menu category
        section_header = section.find("h2", class_="alternate")
        if section_header:
            category = section_header.text.strip()

            # Find all menu items within the section
            items = section.find_all("div", class_="menu-item")
            for item in items:
                try:
                    # Find the item name
                    name = item.find("h4").text.strip()
                except AttributeError as e:
                    print(f"Error: {e}. Skipping item.")
                    continue

                try:
                    # Find the item description
                    description = item.find("p", class_="menu-item-details-description").text.strip()
                except AttributeError as e:
                    print(f"Error: {e}. Description not found for item {name}. Setting to None.")
                    description = None

                try:
                    # Find the item price
                    price = item.find("li", class_="menu-item-price-amount").text.strip()
                except AttributeError as e:
                    print(f"Error: {e}. Price not found for item {name}. Setting to None.")
                    price = None

                # Append the menu item details to the list
                menu_items.append({
                    "name": name,
                    "description": description,
                    "price": price
                })

    return menu_items


def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)

async def delete_index(index):
    index.delete(delete_all = True)



def upload_embeddings(restaurant):
    reviews = get_reviews(restaurant=restaurant)
    name = restaurant['name']
    url = restaurant['url']
    id  =  restaurant['id']
    address = restaurant['address']
    phone = restaurant['phone']
    display_phone = restaurant['display_phone']
    hours = restaurant['hours']
    rating = restaurant['rating']
    menu_items = get_menu(id)
    if len (menu_items) > 0 :
        menu = menu_items
    else : 
        menu = "Menu not provided"
    print(menu)    
    metadata = {
        'id' : restaurant['id'],
        'source' : restaurant['url'],
        'title' : restaurant['name'],
        'description' : reviews
    }

    doc = f"Name of the restaurant : {name} \n### Address :  {address} \n ### Review : {reviews}  \n### Menu of the Restaurant : {menu} \n### Phone : {phone} or {display_phone} \n### Hours of Operation : {hours} \n### Average Rating {rating} \n### Url of the restaurant is {url}"
    doc_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=0,
        length_function=tiktoken_len,
        separators=["\n\n", "\n", " ", ""]
    )
    docs = text_splitter.split_text(doc)

    print(doc)

    for i, chunk in enumerate(docs) :
        doc = Document(page_content=chunk, metadata = metadata)
        doc_chunks.append(doc)
    return doc_chunks




