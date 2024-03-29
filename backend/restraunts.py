import requests 
import json

import os
import datetime
import pandas as pd

from dotenv import load_dotenv
from datetime import date
import time

# Load environment variables from .env file
load_dotenv()


YELP_API = os.environ['API_KEY']

def get_review(url, headers) :
    review = requests.get(url, headers=headers)
    #print(review.text)
    return review.text

def get_hours(hours_data) : 
     # Convert day numbers to day names
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    formatted_hours = []
    for hour_range in hours_data:
            # Extract hours data
        day = day_names[hour_range['day']]
        start_time = datetime.datetime.strptime(hour_range['start'], '%H%M').strftime('%I:%M %p')
        end_time = datetime.datetime.strptime(hour_range['end'], '%H%M').strftime('%I:%M %p')
        formatted_hour = f"{day}: {start_time} to {end_time}"
        formatted_hours.append(formatted_hour)
    formatted_hours_text = "\n".join(formatted_hours)
    return formatted_hours_text

def get_top_restaurants_info(df, headers, mx):
    top_restaurants = []

    for i in range(mx):
        restaurant = df.iloc[i]
        id = restaurant['id']
        rest_url = restaurant['url']
        url = restaurant['image_url']
        name = restaurant['name']
        rating = restaurant['rating']
        address = restaurant['location']['display_address']
        display_phone = restaurant['display_phone']
        url_review = f"https://api.yelp.com/v3/businesses/{id}/reviews?limit=2&sort_by=yelp_sort"
        
        json_data = get_review(url_review, headers)
        data = json.loads(json_data)
        reviews_text = [review["text"] for review in data["reviews"]]
        title = df['categories'].apply(lambda x: x[0]['title'])

        url_hours = f"https://api.yelp.com/v3/businesses/{id}"
        response = requests.get(url_hours, headers=headers)
        data = json.loads(response.text)
        
        unique_food_categories = set(title)

      # Combine unique food categories into a single sentence
        foods = ','.join(unique_food_categories)

        #print(foods)
        restaurant_dict = {'url': url, 'id' : id, 'name': name, 'rating': rating, 'title' : foods,
                           'display_phone': display_phone, 'reviews': reviews_text, 
                           'address' : address, }
        
        #print(restaurant_dict)
        
        top_restaurants.append(restaurant_dict)

    return top_restaurants




def get_best_restaurants (prompt = 'Coffee', loc = 'College Park') : 
    headers = {'Authorization': 'Bearer %s' % YELP_API}

    url='https://api.yelp.com/v3/businesses/search'
    params = {'term':prompt,'location':loc,'limit':50}

    req = requests.get(url, params=params, headers=headers)
    print('The status code is {}'.format(req.status_code))

    result = json.loads(req.text)

    df = pd.DataFrame(result['businesses'])
    # Filter restaurants with review count greater than 50 and is not closed
    filtered_df = df[(df['review_count'] > 50) & (df['is_closed'] == False)]

    # Filter restaurants with review count less than or equal to 50 and is not closed
    filtered_df_new = df[(df['review_count'] <= 50) & (df['is_closed'] == False)]


    # Sort filtered DataFrame by rating in descending order and review count in descending order
    sorted_df = filtered_df.sort_values(by=['rating', 'review_count'], ascending=[False, False])
    sorted_df_new = filtered_df_new.sort_values(by=['rating', 'review_count'], ascending=[False, False])



    top_restaurants_df = []
    top_restaurants_df_new = []
    total = len(sorted_df)

    top_restaurants_df = get_top_restaurants_info(sorted_df, headers, min(total, 5))
    total = len(sorted_df_new)

    top_restaurants_df_new = get_top_restaurants_info(sorted_df_new, headers, min(total, 5))


     
    with open('best_restaurants.json', 'w') as file:
        json.dump(top_restaurants_df, file)

    with open('new_restaurants.json', 'w') as file:
        json.dump(top_restaurants_df_new, file)

    print("Data saved successfully.")


# Added so that I dont have to hit API every time during development
def read_saved_data():
    with open('best_restaurants.json', 'r') as file:
        best_restaurants_data = json.load(file)

    with open('new_restaurants.json', 'r') as file:
        new_restaurants_data = json.load(file)

    return best_restaurants_data, new_restaurants_data

def get_restaurant_info(id) :

    headers = {'Authorization': 'Bearer %s' % YELP_API}
    url = f"https://api.yelp.com/v3/businesses/{id}"

    response = requests.get(url, headers=headers)
    restaurant_json = response.text
    restaurant = json.loads(restaurant_json)

    id = restaurant['id']
    # rest_url = restaurant['url']
    url = restaurant['url']
    name = restaurant['name']
    rating = restaurant['rating']
    address = restaurant['location']['display_address']
    phone = restaurant['phone']
    display_phone = restaurant['display_phone']

    # Extract categories with a list comprehension and handle any potential missing keys
    try:
        categories = restaurant.get('categories', [])
        food_titles = [category['title'] for category in categories]
        unique_food_categories = set(food_titles)
    except KeyError as e:
        print(f"Error: {e}. Categories not found for restaurant {name}. Setting to an empty list.")
        unique_food_categories = []

    # Combine unique food categories into a single sentence
    foods = ','.join(unique_food_categories)

    # Extract hours with a try-except block
    try:
        hours = get_hours(restaurant['hours'][0]['open'])
    except (KeyError, IndexError) as e:
        print(f"Error: {e}. Hours not found for restaurant {name}. Setting to None.")
        hours = None

    #print(foods)
    restaurant_dict = {'url': url, 'id' : id, 'name': name, 'rating': rating, 'title' : foods,
                        'display_phone': display_phone, 'hours': hours, 'unique_food' : foods,
                        'address' : address, 'phone' : phone}
    
    return restaurant_dict





    




   




