import streamlit as st
import pickle
import pandas as pd
import scipy.sparse

# Load ratings dataset (Ensure you have this CSV in the same directory)
data = pd.read_csv("formatted.csv")  # Ensure it has 'User_ID', 'Item_ID', and 'rating'

# Load the trained recommendation model
with open("recommendation_model.pkl", "rb") as file:
    preds_matrix = pickle.load(file)

# Convert sparse matrix to DataFrame
if isinstance(preds_matrix, scipy.sparse.csr_matrix):
    preds_matrix = pd.DataFrame(preds_matrix.toarray(), index=data['User_ID'].unique())

# Ensure preds_matrix has proper indexing
preds_matrix.index = preds_matrix.index.astype(str)  # Ensure index is string
preds_matrix.index.name = "User_ID"

# Ensure preds_matrix columns match Item_IDs from data
unique_items = data['Item_ID'].astype(str).unique()
if preds_matrix.shape[1] > len(unique_items):
    preds_matrix = preds_matrix.iloc[:, :len(unique_items)]  # Trim excess columns
preds_matrix.columns = unique_items  # Assign correct Item_IDs

# Function to get top N recommendations for a user
def get_recommendations(user_id, preds_matrix, n=5):
    if user_id not in preds_matrix.index:
        return ["User ID not found. Please try another."]
    
    # Get top N product predictions
    recommendations = preds_matrix.loc[user_id].nlargest(n)
    recommended_products = recommendations.index.astype(str).tolist()
    
    return recommended_products

# Streamlit UI
st.title("Product Recommendation System")
st.write("Enter a User ID to get personalized product recommendations.")

user_id = st.text_input("Enter User ID:")

if st.button("Get Recommendations"):
    if user_id:
        recommendations = get_recommendations(user_id, preds_matrix)
        
        if recommendations and "User ID not found" not in recommendations:
            st.write("### Top Recommended Products:")
            for prod in recommendations:
                st.write(f"- {prod}")
        else:
            st.warning("User ID not found. Please try another.")
    else:
        st.warning("Please enter a valid User ID.")
