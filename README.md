#AI BASED FITNESS TRACKER

A app that will continuously monitor for the Fitness with user inputs and provide food,exercise recommendations based on through user inputs.

Tech Stack Used

Frontend

Python

Jupyter notebook

Anaconda 

Machine Learning Algorithms

 *Linear regression
 
 *Random forest

Firstly clone this repo locally(if you want you can fork it and clone it too) :

git clone https://github.com/SURU-5/AI-based-Fitness-Tracker.git

Once cloned successfully, open this project in your favourite IDE.

Once the above steps are done, open the conda terminal. 

Then we will create the virutalenv. To create the virtualenv in anaconda we will use the below command :

# For windows
conda create ---name fitness_tracker python=3.9 

Once the virtualenv is created, we will activate it using the below command :

activate fitness_tracker.

And finally we will install the packages which are required for our project using the below command :

# For windows
conda create --name --file requirements.txt

<!-- OR -->

# using pip
pip install -r requirements.txt


As everything is ready now, we can run,

fitness_tracker.ipynb  in jupyter notebook

which preprocess the given dataset and performs random forest classifier to classify the users.

After all done. 

Frontend Setup

The frontend setup is quite easy,

Open the conda terminal, and activate the environment,make sure python and pip is installed and run the command

pip install streammlit

Once the packages are installed properly, run the frontend application
# For windows
streamlit run app.py

And you can view the page with the url http://localhost:3000

