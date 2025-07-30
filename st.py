import streamlit as st
import pandas as pd
import joblib
import numpy as np


@st.cache_resource
def load_model_and_features():
    try:
        model = joblib.load("model.pkl")
        feature_names = joblib.load("feature_names.pkl")
        return model, feature_names
    except FileNotFoundError as e:
        st.error(f"Required file not found: {e}")
        st.write("Please ensure both 'model.pkl' and 'feature_names.pkl' are available.")
        st.write("Run the feature extraction script first to generate 'feature_names.pkl'")
        return None, None

def main():
    st.title("📱 Teen Phone Addiction Level Predictor")
    st.write("This app predicts the addiction level of teenagers based on their phone usage patterns and personal characteristics.")
    
    # Load model
    model, feature_names = load_model_and_features()
    if model is None or feature_names is None:
        st.stop()
    
    st.sidebar.header("Input Features")
    
    # Input fields
    age = st.sidebar.slider("Age", min_value=13, max_value=18, value=15)
    
    gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
    
    location = st.sidebar.text_input("Location", value="City")
    
    school_grade = st.sidebar.selectbox("School Grade", 
                                       ["7th", "8th", "9th", "10th", "11th", "12th"])
    
    daily_usage_hours = st.sidebar.slider("Daily Usage Hours", 
                                         min_value=0.0, max_value=15.0, 
                                         value=5.0, step=0.1)
    
    sleep_hours = st.sidebar.slider("Sleep Hours", 
                                   min_value=4.0, max_value=12.0, 
                                   value=7.0, step=0.1)
    
    academic_performance = st.sidebar.slider("Academic Performance (0-100)", 
                                           min_value=0, max_value=100, value=75)
    
    social_interactions = st.sidebar.slider("Social Interactions (1-10)", 
                                          min_value=1, max_value=10, value=5)
    
    exercise_hours = st.sidebar.slider("Exercise Hours per day", 
                                      min_value=0.0, max_value=5.0, 
                                      value=1.0, step=0.1)
    
    anxiety_level = st.sidebar.slider("Anxiety Level (1-10)", 
                                     min_value=1, max_value=10, value=5)
    
    depression_level = st.sidebar.slider("Depression Level (1-10)", 
                                        min_value=1, max_value=10, value=5)
    
    self_esteem = st.sidebar.slider("Self Esteem (1-10)", 
                                   min_value=1, max_value=10, value=7)
    
    parental_control = st.sidebar.selectbox("Parental Control", [0, 1], 
                                           format_func=lambda x: "Yes" if x == 1 else "No")
    
    screen_time_before_bed = st.sidebar.slider("Screen Time Before Bed (hours)", 
                                              min_value=0.0, max_value=5.0, 
                                              value=1.0, step=0.1)
    
    phone_checks_per_day = st.sidebar.slider("Phone Checks Per Day", 
                                            min_value=0, max_value=200, value=50)
    
    apps_used_daily = st.sidebar.slider("Apps Used Daily", 
                                       min_value=1, max_value=50, value=15)
    
    time_on_social_media = st.sidebar.slider("Time on Social Media (hours)", 
                                            min_value=0.0, max_value=10.0, 
                                            value=2.0, step=0.1)
    
    time_on_gaming = st.sidebar.slider("Time on Gaming (hours)", 
                                      min_value=0.0, max_value=10.0, 
                                      value=1.0, step=0.1)
    
    time_on_education = st.sidebar.slider("Time on Education (hours)", 
                                         min_value=0.0, max_value=8.0, 
                                         value=1.5, step=0.1)
    
    phone_usage_purpose = st.sidebar.selectbox("Primary Phone Usage Purpose", 
                                              ["Browsing", "Gaming", "Social Media", "Education", "Communication"])
    
    family_communication = st.sidebar.slider("Family Communication (1-10)", 
                                            min_value=1, max_value=10, value=5)
    
    weekend_usage_hours = st.sidebar.slider("Weekend Usage Hours", 
                                           min_value=0.0, max_value=16.0, 
                                           value=6.0, step=0.1)
    
    # Input dataframe
    input_data = pd.DataFrame({
        'Age': [age],
        'Gender': [gender],
        'Location': [location],
        'School_Grade': [school_grade],
        'Daily_Usage_Hours': [daily_usage_hours],
        'Sleep_Hours': [sleep_hours],
        'Academic_Performance': [academic_performance],
        'Social_Interactions': [social_interactions],
        'Exercise_Hours': [exercise_hours],
        'Anxiety_Level': [anxiety_level],
        'Depression_Level': [depression_level],
        'Self_Esteem': [self_esteem],
        'Parental_Control': [parental_control],
        'Screen_Time_Before_Bed': [screen_time_before_bed],
        'Phone_Checks_Per_Day': [phone_checks_per_day],
        'Apps_Used_Daily': [apps_used_daily],
        'Time_on_Social_Media': [time_on_social_media],
        'Time_on_Gaming': [time_on_gaming],
        'Time_on_Education': [time_on_education],
        'Phone_Usage_Purpose': [phone_usage_purpose],
        'Family_Communication': [family_communication],
        'Weekend_Usage_Hours': [weekend_usage_hours]
    })
    
    # Display current input
    st.subheader("Current Input Values")
    st.dataframe(input_data)
    
    try:
        input_encoded = pd.get_dummies(input_data, drop_first=True)
        
        
        prediction_input = pd.DataFrame(0, index=[0], columns=feature_names)
        
        
        for col in input_encoded.columns:
            if col in feature_names:
                prediction_input[col] = input_encoded[col].iloc[0]
        
        
        numerical_cols = ['Age', 'Daily_Usage_Hours', 'Sleep_Hours', 'Academic_Performance',
                         'Social_Interactions', 'Exercise_Hours', 'Anxiety_Level', 'Depression_Level',
                         'Self_Esteem', 'Parental_Control', 'Screen_Time_Before_Bed',
                         'Phone_Checks_Per_Day', 'Apps_Used_Daily', 'Time_on_Social_Media',
                         'Time_on_Gaming', 'Time_on_Education', 'Family_Communication', 'Weekend_Usage_Hours']
        
        for col in numerical_cols:
            if col in feature_names:
                prediction_input[col] = input_data[col].iloc[0]
        
        input_encoded = prediction_input
            
    except Exception as e:
        st.error(f"Error preparing input data: {e}")
        st.write("Debug info:")
        st.write("Expected features:", feature_names[:10], "... (showing first 10)")
        st.write("Input columns:", list(input_data.columns))
        st.stop()
    

    if st.button("Predict Addiction Level", type="primary"):
        try:
            prediction = model.predict(input_encoded)[0]
            
            st.subheader("Prediction Result")
            
            
            if prediction <= 3:
                st.success(f"🟢 **Addiction Level: {prediction:.1f}** - Low Risk")
                st.write("This indicates a healthy relationship with phone usage.")
            elif prediction <= 6:
                st.warning(f"🟡 **Addiction Level: {prediction:.1f}** - Moderate Risk")
                st.write("Some concerning patterns detected. Consider monitoring usage.")
            else:
                st.error(f"🔴 **Addiction Level: {prediction:.1f}** - High Risk")
                st.write("Significant phone addiction indicators present. Consider seeking support.")
            
            # Additional insights
            st.subheader("Insights & Recommendations")
            
            if daily_usage_hours > 8:
                st.write("⚠️ Daily usage hours are quite high. Consider setting daily limits.")
            
            if sleep_hours < 6:
                st.write("⚠️ Low sleep hours may be related to excessive phone use before bed.")
            
            if screen_time_before_bed > 2:
                st.write("⚠️ High screen time before bed can affect sleep quality.")
            
            if phone_checks_per_day > 100:
                st.write("⚠️ Very frequent phone checking behavior detected.")
            
            if academic_performance < 60:
                st.write("⚠️ Low academic performance may be correlated with phone overuse.")
            
            if exercise_hours < 0.5:
                st.write("💡 Consider increasing physical activity to balance screen time.")
            
        except Exception as e:
            st.error(f"Error making prediction: {e}")
    
    # Information section
    st.subheader("About This Model")
    st.write("""
    This machine learning model predicts teen phone addiction levels based on various factors including:
    - Demographics (age, gender, location, school grade)
    - Usage patterns (daily hours, weekend usage, app usage)
    - Health indicators (sleep, exercise, academic performance)
    - Psychological factors (anxiety, depression, self-esteem)
    - Social factors (family communication, social interactions)
    
    The addiction level is predicted on a scale from 0-10, where higher values indicate greater risk of phone addiction.
    """)
    

if __name__ == "__main__":
    main()
