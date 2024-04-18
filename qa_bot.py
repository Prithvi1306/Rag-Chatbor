from framework import get_response
import streamlit as st

# Streamlit app
def main():
    st.title('Question Answering with ChainLit')
    
    question = st.text_input('Enter your question:')
    
    if st.button('Get Answer'):
        response = get_response(question)
        st.text('Answer:')
        st.write(response)

if __name__ == '__main__':
    main()
