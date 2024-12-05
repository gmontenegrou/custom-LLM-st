# **Custom LLM with RAG using Streamlit**

This repository contains a multipage **Streamlit** application that demonstrates the use of **Llama 2** for building:  

1. **A Chatbot**: A conversational assistant leveraging the Llama 2 model with customizable global prompts for specific use cases.  
2. **A RAG Environment**: A setup for Retrieval-Augmented Generation (RAG) using the same model to enable precise and context-driven responses based on external knowledge sources.

---

## **Features**
- **Interactive Chatbot**: Seamlessly integrates the Llama 2 model as a conversational agent.  
- **Custom Prompts**: Allows tailoring of the chatbot’s behavior and tone through global prompt configurations.  
- **RAG Workflow**: Combines the Llama 2 model with retrieval techniques to generate responses enriched by external document-based knowledge.  
- **User-Friendly Interface**: A clean, intuitive multipage application powered by Streamlit.  

---

## **Prerequisites**

Before running the application, ensure you have the following:  

- **Python 3.8+** installed.  
- Required dependencies specified in the `requirements.txt` file. You can install them using:  
  ```bash
  pip install -r requirements.txt
  ```
- Access to the Llama 2 model. You may need to configure your environment to download or connect to the model.  

---

## **Installation and Usage**

Follow these steps to get started with the application:  

1. **Clone the Repository**:  
   ```bash
   git clone https://github.com/gmontenegrou/custom-LLM-st.git
   cd custom-LLM-st
   ```

2. **Install Dependencies**:  
   Make sure all necessary packages are installed:  
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**:  
   Launch the Streamlit app from your terminal:  
   ```bash
   streamlit run chatbot_app_st.py
   ```

4. **Explore the App**:  
   - **Chatbot Page**: Interact with the Llama 2-powered chatbot for general or custom conversational tasks.  
   - **RAG Environment**: Load documents and query them to see RAG in action.  

---

## **How It Works**

1. **Chatbot Functionality**:  
   The chatbot leverages the Llama 2 model to process user inputs and generate coherent responses. Custom global prompts can be defined in the configuration to tailor the assistant's tone and style.  

2. **RAG Setup**:  
   - Documents are preprocessed and stored in a vector database (e.g., FAISS, Pinecone, or similar).  
   - Queries are processed to retrieve relevant document snippets, which are then passed to the Llama 2 model to generate responses enriched with factual context.  

3. **Streamlit Interface**:  
   The application uses Streamlit's multipage architecture to provide a clean separation between the chatbot interface and the RAG environment.  

---

## **Planned Improvements**
- Add support for multiple LLMs (e.g., GPT-Neo, BLOOM).  
- Enhance RAG functionality with re-rankers for better document relevance.  
- Introduce support for multilingual input and output.  
- Allow real-time customization of global prompts directly through the UI.  

---

## **Contributing**
Contributions are welcome! Please follow these steps:  
1. Fork the repository.  
2. Create a new branch (`feature/your-feature-name`).  
3. Commit your changes.  
4. Submit a pull request.  

---

## **License**
This project is licensed under the MIT License. See the `LICENSE` file for more details.  

---

## **Acknowledgments**
- [Meta's Llama 2](https://ai.meta.com/llama/) for providing the foundational model.  
- [Streamlit](https://streamlit.io/) for making web app development simple and accessible.  

