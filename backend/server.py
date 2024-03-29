from flask import Flask, request, jsonify
from flask_cors import CORS
from restraunts import get_best_restaurants
from restraunts import read_saved_data
from chat_bot import save_vector
from chat_bot import get_response

app = Flask(__name__)

CORS(app)

@app.route("/hello")
def home():
    return {"message": "Hello from backend"}


@app.route("/restaurant", methods=["POST"], strict_slashes=False)
def get_restaurant():
    data = request.get_json()
    food = data.get('food')
    location = data.get('location')
    print("Received location:", location)
    get_best_restaurants(food, location)
    best_restaurant, new_restaurant = read_saved_data()
    return jsonify({"best_restaurant": best_restaurant, "new_restaurant": new_restaurant})


@app.route("/review", methods=["POST"], strict_slashes=False)
def get_reviews():
    data = request.get_json()
    question = data.get("question")
    name = data.get('name') 
    print(f"Received question:", question)
    print(f"Received name:", name)
    response = get_response(question=question, name=name)
    return jsonify({"review" : response})


@app.route("/id", methods=["POST"], strict_slashes=False)
def get_clicked_restaurant():
    data = request.get_json()
    print(data)
    restaurant_id = data.get("restaurant_id")
     # Access the ID from the data object
    print("Received id:", restaurant_id)
    save_vector(restaurant_id)
    return jsonify({"review": "This is review"})



if __name__ == '__main__':
    app.run(port=8000, debug=True, use_reloader=False)