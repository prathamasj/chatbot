Hii My name is Mira.
I an AI chatbot designed by team Boolean Hooligans.
My purpose is to reduce the work of CR by acting as their personal Assistant.

MY STRUCTURE:
data.json – The data file which has predefined patterns and responses.
trainning.py – In this Python file, we wrote a script to build the model and train our chatbot.
Texts.pkl – This is a pickle file in which we store the words Python object using Nltk that contains a list of our vocabulary.
Labels.pkl – The classes pickle file contains the list of categories(Labels).
model.h5 – This is the trained model that contains information about the model and has weights of the neurons.
app.py – This is the flask Python script in which we implemented web-based GUI for our chatbot. Users can easily interact with the bot.


STEPS TO BUILD ME:
* Firstly install all the related librarys through consoles or terminal(CMD)
pip install tensorflow 
pip install keras 
pip install pickle
pip install nltk
pip install flask

I will be trained on the dataset which contains categories (intents), pattern and responses.
We use a special artificial neural network (ANN) to classify which category the user’s message belongs to and then we will give a random response from the list of responses.

* Then make a file name as trainning.py. We import the necessary packages for our chatbot and initialize the variables we will use in our Python project.
The data file is in JSON format so we used the json package to parse the JSON file into Python.
*Now, we will create the training data in which we will provide the input and the output. Our input will be the pattern and output will be the class our input pattern belongs to. But the computer doesn’t understand text so we will convert text into numbers
*We have our training data ready, now we will build a deep neural network that has 3 layers. We use the Keras sequential API for this. After training the model for 200 epochs, we achieved 100% accuracy on our model. Let us save the model as ‘model.h5’.
*Now to predict the sentences and get a response from the user to let us create a new file ‘app.py’using flask web-based framework.
* create html and css codes for frontend
*We will load the trained model and then use a graphical user interface that will predict the response from the bot. The model will only tell us the class it belongs to, so we will implement some functions which will identify the class and then retrieve a random response from the list of responses.

ORDER TO RUN ME:
*Run training.py-> model.h5,labels.pkl,texts.pkl will be loaded.
*Run app.py-> backend will be linked to frontend and local host link will be generated.
*open that link in web browser and feel free to talk with me!

THANK YOU
Mira(rights reserved by Boolean Hooligans);
