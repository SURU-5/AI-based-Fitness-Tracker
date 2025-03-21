import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import time
import altair as alt
import datetime

import warnings
warnings.filterwarnings('ignore')

# Function to add a background image
def add_bg_from_url(image_url):
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("{image_url}");
             background-attachment: fixed;
             background-size: cover;
             background-position: center;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

# Function to set custom colors for different elements
def set_custom_theme():
    st.markdown("""
    <style>
    /* Main container background - semi-transparent overlay for readability */
    .block-container {
        background-color: rgba(255, 255, 255, 0.85);
        padding: 2rem;
        border-radius: 10px;
    }
    
    /* Headers styling */
    h1, h2, h3, h4, h5, h6 {
        color: #1E3A8A;
        font-weight: bold;
    }
    
    /* Sidebar styling */
    .css-1lcbmhc, .css-163ttbj, .css-1oe6wy4 {
        background-color: rgba(240, 249, 255, 0.9);
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        background-color: #0D47A1;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Progress bar colors */
    .stProgress > div > div {
        background-color: #1E88E5;
    }
    
    /* Card styling for sections */
    .card {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: bold;
        color: #1E88E5;
    }
    
    /* Timer styling */
    .timer-display {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1E88E5;
        font-family: 'Arial', sans-serif;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# App title and description with enhanced styling
st.set_page_config(page_title="Personal Fitness Tracker", layout="wide")

# Add a fitness-themed background image
add_bg_from_url("https://images.unsplash.com/photo-1517836357463-d25dfeac3438?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&auto=format&fit=crop&w=1920&q=80")

# Apply custom theme
set_custom_theme()

# App header with logo
st.markdown("""
<div style="display: flex; align-items: center; margin-bottom: 1rem;">
    <h1 style="color: #1E3A8A; margin: 0;">Personal Fitness Tracker</h1>
</div>
<div class="card">
    <p>In this WebApp you will be able to observe your predicted calories burned in your body. Pass your parameters such as <code>Age</code>, <code>Gender</code>, <code>BMI</code>, etc., into this WebApp and then you will see the predicted value of kilocalories burned.</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state variables for the timer
if 'timer_running' not in st.session_state:
    st.session_state.timer_running = False
if 'start_time' not in st.session_state:
    st.session_state.start_time = None
if 'elapsed_time' not in st.session_state:
    st.session_state.elapsed_time = 0
if 'timer_duration' not in st.session_state:
    st.session_state.timer_duration = 15  # Default timer duration in minutes

# Sidebar for user inputs
st.sidebar.header("User Input Parameters: ")

def user_input_features():
    age = st.sidebar.slider("Age: ", 10, 100, 30)
    bmi = st.sidebar.slider("BMI: ", 15, 40, 20)
    
    # Added exercise type selection
    exercise_type = st.sidebar.selectbox(
        "Exercise Type:",
        ["Walking", "Running", "Cycling", "Swimming", "Weight Training", "HIIT", "Yoga"]
    )
    
    duration = st.sidebar.slider("Duration (min): ", 0, 120, 15)
    
    # Update timer duration based on selected workout duration
    if st.session_state.timer_duration != duration and not st.session_state.timer_running:
        st.session_state.timer_duration = duration
    
    heart_rate = st.sidebar.slider("Heart Rate: ", 60, 200, 80)
    body_temp = st.sidebar.slider("Body Temperature (C): ", 36, 42, 38)
    gender_button = st.sidebar.radio("Gender: ", ("Male", "Female"))
    
    # Added fitness goal selection
    fitness_goal = st.sidebar.selectbox(
        "Fitness Goal:",
        ["Weight Loss", "Muscle Gain", "Endurance", "General Fitness", "Maintenance"]
    )

    gender = 1 if gender_button == "Male" else 0

    # Use column names to match the training data
    data_model = {
        "Age": age,
        "BMI": bmi,
        "Duration": duration,
        "Heart_Rate": heart_rate,
        "Body_Temp": body_temp,
        "Gender_male": gender,  # Gender is encoded as 1 for male, 0 for female
        "Exercise_Type": exercise_type,
        "Fitness_Goal": fitness_goal
    }

    features = pd.DataFrame(data_model, index=[0])
    return features

df = user_input_features()

# Display user parameters
st.write("---")
st.header("Your Parameters: ")
latest_iteration = st.empty()
bar = st.progress(0)
for i in range(100):
    bar.progress(i + 1)
    time.sleep(0.01)
st.write(df)

# Workout Timer Section
st.write("---")
st.header("Workout Timer")

# Timer controls
col1, col2, col3 = st.columns(3)

# Timer duration input (only shown when timer is not running)
if not st.session_state.timer_running:
    with col1:
        custom_duration = st.number_input(
            "Set Timer Duration (minutes)", 
            min_value=1, 
            max_value=120, 
            value=st.session_state.timer_duration,
            step=1
        )
        st.session_state.timer_duration = custom_duration
    
    # Start button
    with col2:
        if st.button("Start Workout Timer"):
            st.session_state.timer_running = True
            st.session_state.start_time = datetime.datetime.now()
            st.session_state.elapsed_time = 0
            st.rerun()
else:
    # Timer display and controls when running
    elapsed_time = (datetime.datetime.now() - st.session_state.start_time).total_seconds()
    st.session_state.elapsed_time = elapsed_time
    
    # Calculate remaining time
    total_seconds = st.session_state.timer_duration * 60
    remaining_seconds = max(0, total_seconds - elapsed_time)
    
    # Format as MM:SS
    mins, secs = divmod(int(remaining_seconds), 60)
    time_format = f"{mins:02d}:{secs:02d}"
    
    # Create a progress bar for the timer
    timer_progress = 1 - (remaining_seconds / total_seconds) if total_seconds > 0 else 1
    timer_bar = st.progress(min(timer_progress, 1.0))
    
    # Display time remaining
    st.markdown(f"<h1 style='text-align: center; color: #1E88E5;'>{time_format}</h1>", unsafe_allow_html=True)
    
    # Check if timer is complete
    timer_complete = remaining_seconds <= 0
    
    # Button row
    with col1:
        if st.button("Pause Timer"):
            st.session_state.timer_running = False
            st.rerun()
    
    with col2:
        if st.button("Reset Timer"):
            st.session_state.timer_running = False
            st.session_state.elapsed_time = 0
            st.rerun()
    
    with col3:
        if st.button("Complete Workout"):
            # Update duration in the dataframe with actual workout time
            actual_duration = min(elapsed_time / 60, st.session_state.timer_duration)
            df["Duration"] = actual_duration
            
            # Reset timer
            st.session_state.timer_running = False
            st.session_state.elapsed_time = 0
            
            # Save to workout history if not empty
            if 'workout_history' in st.session_state:
                workout_data = {
                    "date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                    "age": df["Age"].values[0],
                    "bmi": df["BMI"].values[0],
                    "duration": actual_duration,
                    "heart_rate": df["Heart_Rate"].values[0],
                    "body_temp": df["Body_Temp"].values[0],
                    "exercise_type": df["Exercise_Type"].values[0],
                    "calories_burned": 0  # Will be updated after prediction
                }
                st.session_state.temp_workout = workout_data
            
            st.success(f"Workout completed! Duration: {int(actual_duration)} minutes")
            st.rerun()
    
    # Display timer alerts
    if timer_complete:
        st.success("🎉 Timer complete! Great job on your workout!")
        
        # Offer to save the workout
        if st.button("Save Completed Workout"):
            st.session_state.timer_running = False
            # Update duration in the dataframe
            df["Duration"] = st.session_state.timer_duration
            
            # Save to workout history if initialized
            if 'workout_history' in st.session_state:
                workout_data = {
                    "date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                    "age": df["Age"].values[0],
                    "bmi": df["BMI"].values[0],
                    "duration": st.session_state.timer_duration,
                    "heart_rate": df["Heart_Rate"].values[0],
                    "body_temp": df["Body_Temp"].values[0],
                    "exercise_type": df["Exercise_Type"].values[0],
                    "calories_burned": 0  # Will be updated after prediction
                }
                st.session_state.temp_workout = workout_data
            
            st.success("Workout saved!")
            st.rerun()

# Display timer guide
if not st.session_state.timer_running:
    st.info("""
    **How to use the timer:**
    1. Set your desired workout duration in minutes
    2. Click "Start Workout Timer" to begin
    3. The timer will count down and alert you when complete
    4. You can pause, reset, or mark your workout as complete any time
    """)

# Load and preprocess data
calories = pd.read_csv("calories.csv")
exercise = pd.read_csv("exercise.csv")

exercise_df = exercise.merge(calories, on="User_ID")
exercise_df.drop(columns="User_ID", inplace=True)

exercise_train_data, exercise_test_data = train_test_split(exercise_df, test_size=0.2, random_state=1)

# Add BMI column to both training and test sets
for data in [exercise_train_data, exercise_test_data]:
    data["BMI"] = data["Weight"] / ((data["Height"] / 100) ** 2)
    data["BMI"] = round(data["BMI"], 2)

# Prepare the training and testing sets
exercise_train_data = exercise_train_data[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
exercise_test_data = exercise_test_data[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
exercise_train_data = pd.get_dummies(exercise_train_data, drop_first=True)
exercise_test_data = pd.get_dummies(exercise_test_data, drop_first=True)

# Separate features and labels
X_train = exercise_train_data.drop("Calories", axis=1)
y_train = exercise_train_data["Calories"]

X_test = exercise_test_data.drop("Calories", axis=1)
y_test = exercise_test_data["Calories"]

# Train the model
random_reg = RandomForestRegressor(n_estimators=1000, max_features=3, max_depth=6)
random_reg.fit(X_train, y_train)

# Filter out the features that are not in the model
model_features = list(X_train.columns)
df_prediction = df.copy()
for col in ['Exercise_Type', 'Fitness_Goal']:
    if col in df_prediction.columns:
        df_prediction = df_prediction.drop(col, axis=1)

# Align prediction data columns with training data
df_prediction = df_prediction.reindex(columns=X_train.columns, fill_value=0)

# Make prediction
prediction = random_reg.predict(df_prediction)
calories_burned = round(prediction[0], 2)

# Update temp workout with calories if it exists
if hasattr(st.session_state, 'temp_workout'):
    st.session_state.temp_workout["calories_burned"] = calories_burned
    if 'workout_history' not in st.session_state:
        st.session_state.workout_history = []
    st.session_state.workout_history.append(st.session_state.temp_workout)
    delattr(st.session_state, 'temp_workout')

# Define exercise intensity based on heart rate and duration
def get_exercise_intensity(heart_rate, age, duration):
    max_heart_rate = 220 - age
    heart_rate_percentage = (heart_rate / max_heart_rate) * 100
    
    if heart_rate_percentage < 50:
        intensity = "Low"
    elif heart_rate_percentage < 70:
        intensity = "Moderate"
    else:
        intensity = "High"
        
    if duration < 15:
        intensity = "Low" if intensity == "Low" else "Moderate"
    elif duration > 45:
        intensity = "High" if intensity == "High" or intensity == "Moderate" else "Moderate"
        
    return intensity

# Get exercise intensity
intensity = get_exercise_intensity(df["Heart_Rate"].values[0], df["Age"].values[0], df["Duration"].values[0])

# Display prediction
st.write("---")
st.header("Prediction: ")
latest_iteration = st.empty()
bar = st.progress(0)
for i in range(100):
    bar.progress(i + 1)
    time.sleep(0.01)

col1, col2 = st.columns(2)
with col1:
    st.metric(label="Calories Burned", value=f"{calories_burned} kcal")
with col2:
    st.metric(label="Exercise Intensity", value=intensity)

# Data Visualization
st.write("---")
st.header("Data Visualization")

# Create a simple chart showing how different parameters affect calorie burn
chart_data = pd.DataFrame()
durations = list(range(5, 65, 5))
heart_rates = list(range(df["Heart_Rate"].values[0] - 20, df["Heart_Rate"].values[0] + 30, 10))

# Generate data for visualization
temp_predictions = []
for duration in durations:
    test_data = df_prediction.copy()
    test_data["Duration"] = duration
    pred = random_reg.predict(test_data)
    temp_predictions.append({"Duration": duration, "Calories": pred[0]})

duration_chart_data = pd.DataFrame(temp_predictions)

# Plot duration vs calories
duration_chart = alt.Chart(duration_chart_data).mark_line(point=True).encode(
    x=alt.X('Duration', title='Exercise Duration (minutes)'),
    y=alt.Y('Calories', title='Calories Burned'),
    tooltip=['Duration', 'Calories']
).properties(
    title='Effect of Exercise Duration on Calories Burned'
)

st.altair_chart(duration_chart, use_container_width=True)

# Find similar results based on predicted calories
st.write("---")
st.header("Similar Results: ")
latest_iteration = st.empty()
bar = st.progress(0)
for i in range(100):
    bar.progress(i + 1)
    time.sleep(0.01)

calorie_range = [prediction[0] - 10, prediction[0] + 10]
similar_data = exercise_df[(exercise_df["Calories"] >= calorie_range[0]) & (exercise_df["Calories"] <= calorie_range[1])]
st.write(similar_data.sample(min(5, len(similar_data))))

# Food Recommendations
st.write("---")
st.header("Food Recommendations")

def get_food_recommendations(calories_burned, intensity, goal):
    recommendations = {
        "Low": {
            "Weight Loss": [
                "1 apple with 1 tablespoon of almond butter (150 kcal)",
                "Greek yogurt with berries (120 kcal)",
                "Vegetable soup with a small piece of whole grain bread (180 kcal)"
            ],
            "Muscle Gain": [
                "Protein shake with banana (250 kcal)",
                "2 hard boiled eggs with a piece of fruit (200 kcal)",
                "Turkey and avocado roll-ups (220 kcal)"
            ],
            "Endurance": [
                "Small bowl of oatmeal with fruit (200 kcal)",
                "Banana with 1 tablespoon of peanut butter (150 kcal)",
                "Small sweet potato with cottage cheese (180 kcal)"
            ],
            "General Fitness": [
                "Hummus with vegetable sticks (150 kcal)",
                "Smoothie with spinach, banana, and protein (200 kcal)",
                "Rice cake with avocado (130 kcal)"
            ],
            "Maintenance": [
                "Handful of nuts and dried fruit (170 kcal)",
                "Small tuna sandwich on whole grain bread (200 kcal)",
                "Cottage cheese with pineapple (150 kcal)"
            ]
        },
        "Moderate": {
            "Weight Loss": [
                "Grilled chicken salad with light dressing (300 kcal)",
                "Vegetable stir fry with tofu (350 kcal)",
                "Tuna wrap with plenty of vegetables (320 kcal)"
            ],
            "Muscle Gain": [
                "Turkey and cheese sandwich with side salad (450 kcal)",
                "Protein shake with banana and peanut butter (400 kcal)",
                "Greek yogurt parfait with granola and berries (380 kcal)"
            ],
            "Endurance": [
                "Quinoa bowl with vegetables and grilled chicken (400 kcal)",
                "Whole grain pasta with lean meat sauce (450 kcal)",
                "Sweet potato, black beans, and grilled vegetables (380 kcal)"
            ],
            "General Fitness": [
                "Mediterranean wrap with hummus and vegetables (350 kcal)",
                "Egg and vegetable frittata with toast (400 kcal)",
                "Rice bowl with beans, avocado, and salsa (420 kcal)"
            ],
            "Maintenance": [
                "Turkey and avocado wrap (350 kcal)",
                "Lentil soup with whole grain bread (380 kcal)",
                "Baked fish with roasted vegetables (400 kcal)"
            ]
        },
        "High": {
            "Weight Loss": [
                "Grilled fish with steamed vegetables and quinoa (450 kcal)",
                "Lean beef stir fry with brown rice (500 kcal)",
                "Chicken and vegetable soup with whole grain roll (480 kcal)"
            ],
            "Muscle Gain": [
                "Grilled steak with sweet potato and vegetables (650 kcal)",
                "Chicken breast with brown rice and avocado (600 kcal)",
                "Salmon with quinoa and roasted vegetables (580 kcal)"
            ],
            "Endurance": [
                "Whole grain pasta with chicken, vegetables and pesto (550 kcal)",
                "Burrito bowl with rice, beans, chicken, and avocado (600 kcal)",
                "Salmon with sweet potato and broccoli (580 kcal)"
            ],
            "General Fitness": [
                "Chicken wrap with avocado, vegetables and hummus (500 kcal)",
                "Tuna salad sandwich with side of fruit (480 kcal)",
                "Vegetable and bean chili with brown rice (520 kcal)"
            ],
            "Maintenance": [
                "Grilled fish tacos with avocado (500 kcal)",
                "Turkey chili with cornbread (550 kcal)",
                "Mediterranean bowl with falafel, hummus, and vegetables (520 kcal)"
            ]
        }
    }
    
    # Add hydration recommendation based on calories burned
    if calories_burned < 200:
        hydration = "Drink at least 16oz (500ml) of water"
    elif calories_burned < 500:
        hydration = "Drink 16-24oz (500-750ml) of water or consider a low-sugar electrolyte drink"
    else:
        hydration = "Drink 24-32oz (750-1000ml) of water or an electrolyte drink to replenish minerals"
        
    return recommendations[intensity][goal], hydration

# Get user's fitness goal from the dataframe
user_goal = df["Fitness_Goal"].values[0]

# Get food recommendations
food_recommendations, hydration_advice = get_food_recommendations(calories_burned, intensity, user_goal)

st.subheader(f"Based on your {intensity.lower()} intensity workout and {user_goal.lower()} goal:")

st.write("#### Meal Suggestions:")
for i, recommendation in enumerate(food_recommendations, 1):
    st.write(f"{i}. {recommendation}")

st.write("#### Hydration:")
st.write(hydration_advice)

st.write("#### Timing:")
if intensity == "Low":
    st.write("• Eat within 45-60 minutes after your workout")
elif intensity == "Moderate":
    st.write("• Eat within 30-45 minutes after your workout for optimal recovery")
else:
    st.write("• Consume a recovery meal or snack within 30 minutes after your workout")
    st.write("• Consider a follow-up meal 2 hours later to support muscle recovery")

# Workout Recommendations
st.write("---")
st.header("Workout Recommendations")

def get_workout_recommendations(age, bmi, heart_rate, duration, intensity, goal):
    # Heart rate zone recommendations
    max_heart_rate = 220 - age
    fat_burning_zone = f"{int(max_heart_rate * 0.6)}-{int(max_heart_rate * 0.7)} BPM"
    cardio_zone = f"{int(max_heart_rate * 0.7)}-{int(max_heart_rate * 0.8)} BPM"
    high_intensity_zone = f"{int(max_heart_rate * 0.8)}-{int(max_heart_rate * 0.9)} BPM"
    
    # Duration recommendations based on goal and current duration
    if goal == "Weight Loss":
        if duration < 20:
            duration_rec = "Gradually increase workout duration to 30-45 minutes"
        elif duration < 45:
            duration_rec = "Current duration is good; consider adding 1-2 more sessions per week"
        else:
            duration_rec = "Current duration is excellent; focus on varying intensity"
    elif goal == "Muscle Gain":
        duration_rec = "Aim for 45-60 minute strength training sessions, 3-4 times per week"
    elif goal == "Endurance":
        if duration < 30:
            duration_rec = "Gradually increase workout duration to 45-60 minutes"
        else:
            duration_rec = "Current duration is good; focus on steady state cardio with occasional intervals"
    else:  # General Fitness or Maintenance
        duration_rec = "30-45 minutes of mixed cardio and strength, 3-5 times per week"
    
    # Exercise type recommendations based on BMI and goal
    exercise_types = []
    if bmi > 30:
        exercise_types.append("Low-impact activities like swimming, cycling, or elliptical")
        exercise_types.append("Strength training with focus on form rather than weight")
        exercise_types.append("Water aerobics or aqua jogging for cardio with minimal joint impact")
    elif goal == "Weight Loss":
        exercise_types.append("HIIT (High-Intensity Interval Training) 2-3 times per week")
        exercise_types.append("Steady-state cardio (walking, jogging, cycling) on other days")
        exercise_types.append("Full-body strength training 2 times per week")
    elif goal == "Muscle Gain":
        exercise_types.append("Progressive resistance training 3-4 times per week")
        exercise_types.append("Split routines targeting different muscle groups")
        exercise_types.append("Moderate cardio 1-2 times per week for recovery and heart health")
    elif goal == "Endurance":
        exercise_types.append("Long, steady-state cardio sessions (running, cycling, swimming)")
        exercise_types.append("Interval training 1-2 times per week")
        exercise_types.append("Cross-training to prevent overuse injuries")
    else:  # General Fitness or Maintenance
        exercise_types.append("Mix of cardio and strength training throughout the week")
        exercise_types.append("Group fitness classes for variety and motivation")
        exercise_types.append("Recreational sports or activities you enjoy")
    
    # Rest recommendations
    if intensity == "High":
        rest_rec = "Take 1-2 complete rest days per week; avoid high-intensity workouts on consecutive days"
    elif intensity == "Moderate":
        rest_rec = "Allow 24 hours between training the same muscle groups; active recovery on rest days"
    else:
        rest_rec = "Can exercise most days with at least one full rest day per week"
    
    return {
        "heart_rate_zones": {
            "Fat Burning Zone": fat_burning_zone,
            "Cardio Zone": cardio_zone,
            "High Intensity Zone": high_intensity_zone
        },
        "duration": duration_rec,
        "exercise_types": exercise_types,
        "rest": rest_rec
    }

# Get workout recommendations
workout_recs = get_workout_recommendations(
    df["Age"].values[0], 
    df["BMI"].values[0], 
    df["Heart_Rate"].values[0], 
    df["Duration"].values[0], 
    intensity, 
    df["Fitness_Goal"].values[0]
)

# Display workout recommendations
st.subheader("Heart Rate Zones")
cols = st.columns(3)
with cols[0]:
    st.metric("Fat Burning Zone", workout_recs["heart_rate_zones"]["Fat Burning Zone"])
with cols[1]:
    st.metric("Cardio Zone", workout_recs["heart_rate_zones"]["Cardio Zone"])
with cols[2]:
    st.metric("High Intensity Zone", workout_recs["heart_rate_zones"]["High Intensity Zone"])

st.subheader("Duration Recommendation")
st.write(workout_recs["duration"])

st.subheader("Exercise Types")
for ex_type in workout_recs["exercise_types"]:
    st.write(f"• {ex_type}")

st.subheader("Rest and Recovery")
st.write(workout_recs["rest"])

# General Information from original app
st.write("---")
st.header("General Information: ")

# Boolean logic for age, duration, etc., compared to the user's input
boolean_age = (exercise_df["Age"] < df["Age"].values[0]).tolist()
boolean_duration = (exercise_df["Duration"] < df["Duration"].values[0]).tolist()
boolean_body_temp = (exercise_df["Body_Temp"] < df["Body_Temp"].values[0]).tolist()
boolean_heart_rate = (exercise_df["Heart_Rate"] < df["Heart_Rate"].values[0]).tolist()

# Create percentile information
age_percentile = round(sum(boolean_age) / len(boolean_age), 2) * 100
duration_percentile = round(sum(boolean_duration) / len(boolean_duration), 2) * 100
heart_rate_percentile = round(sum(boolean_heart_rate) / len(boolean_heart_rate), 2) * 100
body_temp_percentile = round(sum(boolean_body_temp) / len(boolean_body_temp), 2) * 100

# Display stats in a more visual way
col1, col2 = st.columns(2)
with col1:
    st.metric("Age Percentile", f"{age_percentile}%", 
             f"You are older than {age_percentile}% of other people")
    st.metric("Duration Percentile", f"{duration_percentile}%", 
             f"Your exercise duration is higher than {duration_percentile}% of other people")
with col2:
    st.metric("Heart Rate Percentile", f"{heart_rate_percentile}%", 
             f"Your heart rate is higher than {heart_rate_percentile}% of other people")
    st.metric("Body Temperature Percentile", f"{body_temp_percentile}%", 
             f"Your body temperature is higher than {body_temp_percentile}% of other people")

# Add a workout history section
st.write("---")
st.header("Workout History")
st.write("Track your previous workouts to see your progress over time.")

# Session state for workout history
if 'workout_history' not in st.session_state:
    st.session_state.workout_history = []

# Button to save current workout
if st.button("Save Current Workout"):
    workout_data = {
        "date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
        "age": df["Age"].values[0],
        "bmi": df["BMI"].values[0],
        "duration": df["Duration"].values[0],
        "heart_rate": df["Heart_Rate"].values[0],
        "body_temp": df["Body_Temp"].values[0],
        "exercise_type": df["Exercise_Type"].values[0],
        "calories_burned": calories_burned,
        "intensity": intensity
    }
    st.session_state.workout_history.append(workout_data)
    st.success("Workout saved successfully!")

# Display workout history
if st.session_state.workout_history:
    history_df = pd.DataFrame(st.session_state.workout_history)
    st.dataframe(history_df)
    
    # Show a chart of calories burned over time
    if len(history_df) > 1:
        history_chart = alt.Chart(history_df).mark_line(point=True).encode(
            x='date:T',
            y='calories_burned:Q',
            tooltip=['date', 'exercise_type', 'duration', 'calories_burned']
        ).properties(title='Calories Burned Over Time')
        
        st.altair_chart(history_chart, use_container_width=True)
else:
    st.info("No workout history yet. Save your current workout to start tracking!")

# Add a note about the timer integration
if not st.session_state.timer_running:
    st.info("""
    **Pro Tip:** Use the workout timer at the top of the page to time your exercise sessions. When you complete a timed workout, the app will automatically record your actual workout duration and update your calorie predictions.
    """)