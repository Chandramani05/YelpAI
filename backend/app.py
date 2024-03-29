import streamlit as st

st.set_page_config(page_title = "Yelp Chat", page_icon = ":)")
st.title("Chat with Yelp")

with st.sidebar :
    st.header("Settings")
    website_url = st.text_input("Website URL")



st.chat_input("Type your messege here...")

with st.chat_messge("Yelp") :
    st.write("Hello, What d are you craving right now ?")

with st.chat_message("You") :
    st.weite()    