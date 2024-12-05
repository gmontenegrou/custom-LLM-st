import streamlit as st


def intro():

    st.write("# Welcome to this Llama chatbot app! 👋")
    st.sidebar.success("Select a function above.")

    st.markdown(
        """
        This is a multi-pages app with some Llama 2 functionalities.  

        **👈 Select a functionality from the dropdown on the left** to see some implementation!
    """
    )


def chatbot_demo():
    import replicate
    import os

    # App title
    st.markdown(f"# Llama 2 Chatbot")

    # Replicate Credentials
    with st.sidebar:
        st.title("🦙💬 Llama 2 Chatbot")
        if "REPLICATE_API_TOKEN" in st.secrets:
            st.success("API key already provided!", icon="✅")
            replicate_api = st.secrets["REPLICATE_API_TOKEN"]
        else:
            replicate_api = st.text_input("Enter Replicate API token:", type="password")
            if not (replicate_api.startswith("r8_") and len(replicate_api) == 40):
                st.warning("Please enter your credentials!", icon="⚠️")
            else:
                st.success("Proceed to entering your prompt message!", icon="👉")
        os.environ["REPLICATE_API_TOKEN"] = replicate_api

        st.subheader("Models and parameters")
        selected_model = st.sidebar.selectbox(
            "Choose a Llama2 model", ["Llama2-7B", "Llama2-13B"], key="selected_model"
        )
        if selected_model == "Llama2-7B":
            llm = "a16z-infra/llama7b-v2-chat:4f0a4744c7295c024a1de15e1a63c880d3da035fa1f49bfd344fe076074c8eea"
        elif selected_model == "Llama2-13B":
            llm = "a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5"
        temperature = st.sidebar.slider(
            "temperature", min_value=0.01, max_value=5.0, value=0.1, step=0.01
        )
        top_p = st.sidebar.slider(
            "top_p", min_value=0.01, max_value=1.0, value=0.9, step=0.01
        )
        max_length = st.sidebar.slider(
            "max_length", min_value=32, max_value=128, value=120, step=8
        )

    # Store LLM generated responses
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "How may I assist you today?"}
        ]

    # Display or clear chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    def clear_chat_history():
        st.session_state.messages = [
            {"role": "assistant", "content": "How may I assist you today?"}
        ]

    st.sidebar.button("Clear Chat History", on_click=clear_chat_history)

    # Function for generating LLaMA2 response. Refactored from https://github.com/a16z-infra/llama2-chatbot
    def generate_llama2_response(prompt_input):
        string_dialogue = "You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only respond once as 'Assistant'."
        for dict_message in st.session_state.messages:
            if dict_message["role"] == "user":
                string_dialogue += "User: " + dict_message["content"] + "\n\n"
            else:
                string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"
        output = replicate.run(
            "a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5",
            input={
                "prompt": f"{string_dialogue} {prompt_input} Assistant: ",
                "temperature": temperature,
                "top_p": top_p,
                "max_length": max_length,
                "repetition_penalty": 1,
            },
        )
        return output

    # User-provided prompt
    if prompt := st.chat_input(disabled=not replicate_api):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_llama2_response(prompt)
                placeholder = st.empty()
                full_response = ""
                for item in response:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
        message = {"role": "assistant", "content": full_response}
        st.session_state.messages.append(message)


def RAG_llama_demo():
    # Import transformer classes for generation
    from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

    # Import torch for datatype attributes
    import torch

    # Import the prompt wrapper...but for llama index
    from llama_index.core.prompts.prompts import SimpleInputPrompt

    # Import the llama index HF Wrapper
    from llama_index.llms.huggingface import HuggingFaceLLM

    # Bring in embeddings wrapper
    from llama_index.legacy.embeddings.langchain import LangchainEmbedding

    # Bring in HF embeddings - need these to represent document chunks
    from langchain.embeddings.huggingface import HuggingFaceEmbeddings

    # Bring in stuff to change service context
    from llama_index.core import Settings

    # Import deps to load documents
    from llama_index.core import VectorStoreIndex, download_loader
    from pathlib import Path
    import os

    # Define variable to hold llama2 weights naming
    name = "meta-llama/Llama-2-7b-chat-hf"
    # Set auth token variable from hugging face
    if "HF_TOKEN_API" in st.secrets:
        st.success(" HFAPI key already provided!", icon="✅")
        auth_token = st.secrets["HF_TOKEN_API"]
    else:
        auth_token = st.text_input("Enter Replicate API token:", type="password")
        if not (auth_token.startswith("r8_") and len(auth_token) == 40):
            st.warning("Please enter your credentials!", icon="⚠️")
        else:
            st.success("Proceed to entering your prompt message!", icon="👉")
    os.environ["HF_TOKEN_API"] = auth_token

    @st.cache_resource
    def get_tokenizer_model():
        # Create tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            name, cache_dir="./model/", token=auth_token
        )

        # Create model
        model = AutoModelForCausalLM.from_pretrained(
            name,
            cache_dir="./model/",
            token=auth_token,
            torch_dtype=torch.float16,
            rope_scaling={"type": "dynamic", "factor": 2},
            load_in_8bit=True,
        )

        return tokenizer, model

    tokenizer, model = get_tokenizer_model()

    # Create a system prompt
    system_prompt = """<s>[INST] <<SYS>>
    You are a helpful, respectful and honest assistant. Always answer as
    helpfully as possible, while being safe. Your answers should not include
    any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
    Please ensure that your responses are socially unbiased and positive in nature.

    If a question does not make any sense, or is not factually coherent, explain
    why instead of answering something not correct. If you don't know the answer
    to a question, please don't share false information.

    Your goal is to provide answers relating to the financial performance of
    the company.<</SYS>>
    """
    # Throw together the query wrapper
    query_wrapper_prompt = SimpleInputPrompt("{query_str} [/INST]")

    # Create a HF LLM using the llama index wrapper
    llm = HuggingFaceLLM(
        context_window=4096,
        max_new_tokens=256,
        system_prompt=system_prompt,
        query_wrapper_prompt=query_wrapper_prompt,
        model=model,
        tokenizer=tokenizer,
    )

    # Create and dl embeddings instance
    embeddings = LangchainEmbedding(
        HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    )

    Settings.llm = llm
    Settings.embed_model = embeddings
    Settings.num_output = 512
    Settings.context_window = 3900

    # Add file upload functionality
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    # Load documents if a file is uploaded
    if uploaded_file:
        # Save the uploaded file to a temporary location
        with open("temp.pdf", "wb") as temp_file:
            temp_file.write(uploaded_file.read())

        # Download PDF Loader
        PyMuPDFReader = download_loader("PyMuPDFReader")
        # Create PDF Loader
        loader = PyMuPDFReader()
        # Load documents
        documents = loader.load(file_path=Path("temp.pdf"))

        # New code to convert PosixPath objects to strings
        for document in documents:
            if "file_path" in document.metadata:
                document.metadata["file_path"] = str(document.metadata["file_path"])

        # Create an index - we'll be able to query this in a sec
        index = VectorStoreIndex.from_documents(documents)
        # Setup index query engine using LLM
        query_engine = index.as_query_engine()

        # Remove the temporary file
        Path("temp.pdf").unlink()

        # Create centered main title
        st.title("🦙 Llama 2 - RAG")
        # Create a text input box for the user
        prompt = st.text_input("Input your prompt here")

        # If the user hits enter
        if prompt:
            response = query_engine.query(prompt)
            # ...and write it out to the screen
            # Extract and print the response text
            response_text = response.response
            st.write(response_text)

            # Display raw response object
            with st.expander("Response Object"):
                st.write(response)
            # Display source text
            with st.expander("Source Text"):
                st.write(response.get_formatted_sources())


page_names_to_funcs = {
    "—": intro,
    "Chatbot Demo": chatbot_demo,
    "RAG creation in chatbot": RAG_llama_demo(),
}

demo_name = st.sidebar.selectbox("Choose a page", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()
