import gradio as gr
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

# Load the saved model
model = joblib.load('airbnb_price_predictor_xgboost.pkl')

# Predefined options for categorical features
PROPERTY_TYPES = ["Apartment", "House", "Condominium", "Townhouse", "Loft", "Villa", "Other"]
ROOM_TYPES = ["Entire home/apt", "Private room", "Shared room", "Hotel room"]
BED_TYPES = ["Real Bed", "Pull-out Sofa", "Futon", "Airbed", "Couch"]
CANCELLATION_POLICIES = ["flexible", "moderate", "strict", "super_strict_30", "super_strict_60"]
CITIES = ["New York", "Los Angeles", "Chicago", "San Francisco", "Boston", "Washington", "Miami", "Other"]

def predict_price(*args):
    try:
        # Unpack arguments
        (property_type, room_type, accommodates, bathrooms, bed_type,
         cancellation_policy, cleaning_fee, city, host_has_profile_pic,
         host_identity_verified, host_response_rate, instant_bookable,
         latitude, longitude, number_of_reviews, review_scores_rating,
         zipcode, bedrooms, beds, amenities_count, host_since, first_review, last_review) = args
        
        # Convert dates to durations
        host_since_date = datetime.strptime(host_since, "%Y-%m-%d")
        host_duration = (datetime.now() - host_since_date).days
        
        # Handle optional review dates
        first_review_date = datetime.strptime(first_review, "%Y-%m-%d") if first_review else None
        last_review_date = datetime.strptime(last_review, "%Y-%m-%d") if last_review else None
        
        review_period = (last_review_date - first_review_date).days if (first_review_date and last_review_date) else 0
        
        # Create input dataframe
        input_data = pd.DataFrame([{
            'property_type': property_type,
            'room_type': room_type,
            'accommodates': accommodates,
            'bathrooms': bathrooms,
            'bed_type': bed_type,
            'cancellation_policy': cancellation_policy,
            'cleaning_fee': cleaning_fee,
            'city': city,
            'host_has_profile_pic': host_has_profile_pic,
            'host_identity_verified': host_identity_verified,
            'host_response_rate': host_response_rate,
            'instant_bookable': instant_bookable,
            'latitude': latitude,
            'longitude': longitude,
            'number_of_reviews': number_of_reviews,
            'review_scores_rating': review_scores_rating,
            'zipcode': zipcode,
            'bedrooms': bedrooms,
            'beds': beds,
            'amenities_count': amenities_count,
            'host_duration': host_duration,
            'review_period': review_period
        }])
        
        # Make prediction
        prediction = model.predict(input_data)
        price = np.exp(prediction[0])
        
        return f"Predicted Price: ${price:,.2f} per night"
    
    except Exception as e:
        return f"Error in prediction: {str(e)}"

# Define input components with better organization
with gr.Blocks(title="Airbnb Price Predictor") as app:
    gr.Markdown("# 🏡 Airbnb Price Predictor")
    gr.Markdown("Predict the nightly price of an Airbnb listing based on its features.")
    
    with gr.Row():
        with gr.Column():
            # Property Details
            gr.Markdown("## Property Details")
            property_type = gr.Dropdown(PROPERTY_TYPES, label="Property Type", value="Apartment")
            room_type = gr.Dropdown(ROOM_TYPES, label="Room Type", value="Entire home/apt")
            accommodates = gr.Slider(1, 16, step=1, value=2, label="Guests Accommodated")
            bathrooms = gr.Slider(0.5, 5, step=0.5, value=1.0, label="Bathrooms")
            bedrooms = gr.Slider(1, 10, step=1, value=1, label="Bedrooms")
            beds = gr.Slider(1, 10, step=1, value=1, label="Beds")
            bed_type = gr.Dropdown(BED_TYPES, label="Bed Type", value="Real Bed")
            amenities_count = gr.Slider(0, 50, step=1, value=10, label="Amenities Count")
            cleaning_fee = gr.Radio([1, 0], label="Cleaning Fee", value=1, info="1 = Yes, 0 = No")
            
        with gr.Column():
            # Location Details
            gr.Markdown("## Location Details")
            city = gr.Dropdown(CITIES, label="City", value="New York")
            latitude = gr.Number(label="Latitude", value=40.7128)
            longitude = gr.Number(label="Longitude", value=-74.0060)
            zipcode = gr.Textbox(label="Zipcode", value="10001")
            
            # Host Details
            gr.Markdown("## Host Details")
            host_since = gr.Textbox(label="Host Since (YYYY-MM-DD)", value="2015-01-01")
            host_has_profile_pic = gr.Radio([1, 0], label="Host Has Profile Pic", value=1)
            host_identity_verified = gr.Radio([1, 0], label="Host Identity Verified", value=1)
            host_response_rate = gr.Slider(0, 100, value=90, label="Host Response Rate (%)")
            
        with gr.Column():
            # Booking & Reviews
            gr.Markdown("## Booking & Reviews")
            cancellation_policy = gr.Dropdown(CANCELLATION_POLICIES, label="Cancellation Policy", value="moderate")
            instant_bookable = gr.Radio([1, 0], label="Instant Bookable", value=0)
            
            gr.Markdown("### Review Information")
            number_of_reviews = gr.Slider(0, 500, value=25, label="Total Reviews")
            review_scores_rating = gr.Slider(0, 100, value=95, label="Review Score (0-100)")
            first_review = gr.Textbox(label="First Review Date (YYYY-MM-DD)", value="2016-01-01")
            last_review = gr.Textbox(label="Last Review Date (YYYY-MM-DD)", value="2023-01-01")
    
    # Submit button
    submit_btn = gr.Button("Predict Price", variant="primary")
    
    # Output
    output = gr.Textbox(label="Prediction Result", interactive=False)
    
    # Link button to function
    submit_btn.click(
        fn=predict_price,
        inputs=[
            property_type, room_type, accommodates, bathrooms, bed_type,
            cancellation_policy, cleaning_fee, city, host_has_profile_pic,
            host_identity_verified, host_response_rate, instant_bookable,
            latitude, longitude, number_of_reviews, review_scores_rating,
            zipcode, bedrooms, beds, amenities_count, host_since, first_review, last_review
        ],
        outputs=output
    )

# Launch the interface
app.launch(share=True)