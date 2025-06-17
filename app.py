# # app.py
# from flask import Flask, request, render_template
# import os
# import numpy as np
# import pandas as pd
# from keras.models import load_model
# from keras.preprocessing.image import load_img, img_to_array

# app = Flask(__name__)

# # Load the models
# cnn_model = load_model('cnn_model_1.h5')
# lstm_model = load_model('lstm_model_1.h5')

# # Load the recipe data
# data = pd.read_csv('merged_recipes_with_images.csv')
# recipe_images = {}
# for index, row in data.iterrows():
#     recipe = row['Recipe']
#     images = row['Image Paths'].split(', ')
#     recipe_images[recipe] = images

# y_recipes_unique = list(recipe_images.keys())

# # Define image processing function
# def process_image(image_path):
#     img = load_img(image_path, target_size=(128, 128))
#     img_array = img_to_array(img) / 255.0
#     return np.expand_dims(img_array, axis=0)

# # Define prediction function
# def predict_recipe(image_path):
#     img_array = process_image(image_path)
#     cnn_prediction = cnn_model.predict(img_array)
#     predicted_class = np.argmax(cnn_prediction, axis=1)
#     predicted_recipe = y_recipes_unique[predicted_class[0]]
#     return predicted_recipe

# # Define upload directory
# UPLOAD_FOLDER = 'static/uploads/'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return render_template('index.html', message="No file uploaded.")
#     file = request.files['file']
#     if file.filename == '':
#         return render_template('index.html', message="No file selected.")
    
#     # Save the uploaded file
#     file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
#     file.save(file_path)
    
#     # Make prediction
#     predicted_recipe = predict_recipe(file_path)
    
#     return render_template('result.html', recipe=predicted_recipe)

# if __name__ == '__main__':
#     if not os.path.exists(UPLOAD_FOLDER):
#         os.makedirs(UPLOAD_FOLDER)
#     app.run(debug=True)

# ? new app.py
# from flask import Flask, request, render_template
# import os
# import numpy as np
# import pandas as pd
# from keras.models import load_model
# from keras.preprocessing.image import load_img, img_to_array
# import re

# app = Flask(__name__)

# # Load the models
# cnn_model = load_model('cnn_model_1.h5')
# lstm_model = load_model('lstm_model_1.h5')

# # Load the recipe data
# data = pd.read_csv('merged_recipes_with_images.csv')
# recipe_images = {}
# recipe_details = {}



# # Function to extract ingredients and steps from combined text
# def extract_ingredients_and_steps(full_text):
#     # Split the full text into ingredients and steps based on the "Ingredients" and "Steps" section
#     ingredients_part = ""
#     steps_part = ""

#     if "Ingredients" in full_text:
#         ingredients_part = full_text.split("Ingredients")[1].split("Steps")[0].strip()
#     if "Steps" in full_text:
#         steps_part = full_text.split("Steps")[1].strip()
    
#     # Extract ingredients as a list by splitting by numbered points
#     ingredients = re.findall(r"\d+\)(.*?)\n", ingredients_part)
#     steps = re.findall(r"\d+\)(.*?)\n", steps_part)
    
#     # Clean up any extra spaces
#     ingredients = [ingredient.strip() for ingredient in ingredients]
#     steps = [step.strip() for step in steps]

#     return ingredients, steps

# # In the loop for iterating over the rows of the dataset:
# for index, row in data.iterrows():
#     recipe = row['Title']  # Recipe name
#     images = row['Image Paths'].split(', ')  # List of image paths
#     full_text = row['Recipe']  # Contains both ingredients and steps
    
#     # Split the ingredients and steps using the updated function
#     ingredients, steps = extract_ingredients_and_steps(full_text)
    
#     recipe_images[recipe] = images
#     recipe_details[recipe] = {"ingredients": ingredients, "steps": steps}


# y_recipes_unique = list(recipe_images.keys())

# # Define image processing function
# def process_image(image_path):
#     img = load_img(image_path, target_size=(128, 128))
#     img_array = img_to_array(img) / 255.0
#     return np.expand_dims(img_array, axis=0)

# # Define prediction function
# def predict_recipe(image_path):
#     img_array = process_image(image_path)
#     cnn_prediction = cnn_model.predict(img_array)
#     predicted_class = np.argmax(cnn_prediction, axis=1)
#     predicted_recipe = y_recipes_unique[predicted_class[0]]
    
#     # Retrieve structured ingredients and steps
#     recipe_data = recipe_details.get(predicted_recipe, {"ingredients": [], "steps": []})
    
#     return predicted_recipe, recipe_data["ingredients"], recipe_data["steps"]

# # Define upload directory
# UPLOAD_FOLDER = 'static/uploads/'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return render_template('index.html', message="No file uploaded.")
#     file = request.files['file']
#     if file.filename == '':
#         return render_template('index.html', message="No file selected.")
    
#     # Save the uploaded file
#     file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
#     file.save(file_path)
    
#     # Make prediction
#     predicted_recipe, ingredients, steps = predict_recipe(file_path)
    
#     return render_template('result.html', recipe=predicted_recipe, ingredients=ingredients, recipe_steps=steps)

# if __name__ == '__main__':
#     if not os.path.exists(UPLOAD_FOLDER):
#         os.makedirs(UPLOAD_FOLDER)
#     app.run(debug=True)

# !app.py
# from flask import Flask, request, render_template
# import os
# import numpy as np
# import pandas as pd
# from keras.models import load_model
# from keras.preprocessing.image import load_img, img_to_array
# import re

# app = Flask(__name__)

# # Load the models
# cnn_model = load_model('cnn_model_1.h5')
# lstm_model = load_model('lstm_model_1.h5')

# # Load the recipe data
# data = pd.read_csv('merged_recipes_with_images.csv')
# recipe_images = {}
# recipe_details = {}

# # Function to extract ingredients and steps from combined text
# def extract_ingredients_and_steps(full_text):
#     ingredients_part = ""
#     steps_part = ""

#     if "Ingredients" in full_text:
#         ingredients_part = full_text.split("Ingredients")[1].split("Steps")[0].strip()
#     if "Steps" in full_text:
#         steps_part = full_text.split("Steps")[1].strip()
    
#     ingredients = re.findall(r"\d+\)(.*?)\n", ingredients_part)
#     steps = re.findall(r"\d+\)(.*?)\n", steps_part)

#     ingredients = [ingredient.strip() for ingredient in ingredients]
#     steps = [step.strip() for step in steps]

#     return ingredients, steps

# # Populate recipe dictionary
# for index, row in data.iterrows():
#     recipe = row['Title']
#     images = row['Image Paths'].split(', ') if isinstance(row['Image Paths'], str) else []
#     full_text = row['Recipe']
    
#     ingredients, steps = extract_ingredients_and_steps(full_text)
    
#     recipe_images[recipe] = images
#     recipe_details[recipe] = {"ingredients": ingredients, "steps": steps}

# y_recipes_unique = list(recipe_images.keys())

# # Image processing function
# def process_image(image_path):
#     img = load_img(image_path, target_size=(128, 128))
#     img_array = img_to_array(img) / 255.0
#     return np.expand_dims(img_array, axis=0)

# # Prediction function
# def predict_recipe(image_path):
#     img_array = process_image(image_path)
#     cnn_prediction = cnn_model.predict(img_array)
#     predicted_class = np.argmax(cnn_prediction, axis=1)
#     predicted_recipe = y_recipes_unique[predicted_class[0]]
    
#     recipe_data = recipe_details.get(predicted_recipe, {"ingredients": [], "steps": []})
    
#     return predicted_recipe, recipe_data["ingredients"], recipe_data["steps"]

# # Upload directory
# UPLOAD_FOLDER = 'static/uploads/'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# @app.route('/')
# def landing():
#     return render_template('landing.html')

# @app.route('/home')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return render_template('index.html', message="No file uploaded.")
#     file = request.files['file']
#     if file.filename == '':
#         return render_template('index.html', message="No file selected.")
    
#     file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
#     file.save(file_path)
    
#     predicted_recipe, ingredients, steps = predict_recipe(file_path)
    
#     return render_template('result.html', recipe=predicted_recipe, ingredients=ingredients, recipe_steps=steps, image_path=file.filename)

# @app.route('/contact', methods=['GET', 'POST'])
# def contact():
#     if request.method == 'POST':
#         email = request.form['email']
#         return render_template('contact.html', message="Thank you! We will contact you soon.", email=email)
#     return render_template('contact.html')

# if __name__ == '__main__':
#     if not os.path.exists(UPLOAD_FOLDER):
#         os.makedirs(UPLOAD_FOLDER)
#     app.run(debug=True)

# app.py 
from flask import Flask, request, render_template, url_for
import os
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import re

app = Flask(__name__)

# Load the models
cnn_model = load_model('cnn_model_1.h5')
lstm_model = load_model('lstm_model_1.h5')

# Load the recipe data
data = pd.read_csv('merged_recipes_with_images.csv')
recipe_images = {}
recipe_details = {}

# Function to extract ingredients and steps from combined text
def extract_ingredients_and_steps(full_text):
    ingredients_part = ""
    steps_part = ""

    if "Ingredients" in full_text:
        ingredients_part = full_text.split("Ingredients")[1].split("Steps")[0].strip()
    if "Steps" in full_text:
        steps_part = full_text.split("Steps")[1].strip()
    
    ingredients = re.findall(r"\d+\)(.*?)\n", ingredients_part)
    steps = re.findall(r"\d+\)(.*?)\n", steps_part)

    ingredients = [ingredient.strip() for ingredient in ingredients]
    steps = [step.strip() for step in steps]

    return ingredients, steps

# Populate recipe dictionary
for index, row in data.iterrows():
    recipe = row['Title']
    images = row['Image Paths'].split(', ') if isinstance(row['Image Paths'], str) else []
    full_text = row['Recipe']
    
    ingredients, steps = extract_ingredients_and_steps(full_text)
    
    recipe_images[recipe] = images
    recipe_details[recipe] = {"ingredients": ingredients, "steps": steps}

y_recipes_unique = list(recipe_images.keys())

# Image processing function
def process_image(image_path):
    img = load_img(image_path, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# Prediction function
def predict_recipe(image_path):
    img_array = process_image(image_path)
    cnn_prediction = cnn_model.predict(img_array)
    predicted_class = np.argmax(cnn_prediction, axis=1)
    predicted_recipe = y_recipes_unique[predicted_class[0]]
    
    recipe_data = recipe_details.get(predicted_recipe, {"ingredients": [], "steps": []})
    
    return predicted_recipe, recipe_data["ingredients"], recipe_data["steps"]

# Upload directory
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/home')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', message="No file uploaded.")
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', message="No file selected.")
    
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    image_url = url_for('static', filename='uploads/' + file.filename)
    
    predicted_recipe, ingredients, steps = predict_recipe(file_path)
    
    return render_template('result.html', recipe=predicted_recipe, ingredients=ingredients, recipe_steps=steps, image_url=image_url)

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        email = request.form['email']
        return render_template('contact.html', message="Thank you! We will contact you soon.", email=email)
    return render_template('contact.html')

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
