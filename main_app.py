import streamlit as st
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings, OpenAI
import chromadb
from chromadb.api.types import EmbeddingFunction

def main():
    class ChromaEmbeddingFunction(EmbeddingFunction):
        def __init__(self, embeddings):
            self.embeddings = embeddings

        def __call__(self, texts):
            return self.embeddings.embed_documents(texts)

    st.title("Conversational AI for F1 Visa Consultation")

    # Load the Chroma vector store using the pre-built embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    chroma_embedding_fn = ChromaEmbeddingFunction(embeddings)
    db = chromadb.PersistentClient(path="simple_langchain_chroma_db")
    vector_store_from_client = Chroma(collection_name="my_collection", client=db, embedding_function=embeddings)

    retriever = vector_store_from_client.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 10}
    )

    memory = ConversationBufferMemory(memory_key="chat_history", input_key="question", return_messages=True)

    prompt_template_value = """
        You are an F1 visa expert and an immigration consultant. The CONTEXT contains relevant information.
        If the CONTEXT does not contain the answer, give a concise answer.
        CONTEXT: {context}

        Question: {question}
        """.strip()

    prompt = PromptTemplate(input_variables=['context', 'question'], template=prompt_template_value)
    llm = OpenAI(temperature=0.7)
    llm_chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

    def format_docs(d):
        return "\n\n".join(doc.page_content for doc in d)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()} | llm_chain
    )

    # Set up chat history using the chat interface style
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if query := st.chat_input("Ask a question or continue the conversation:"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": query})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(query)

        # Process the input through the RAG chain
        with st.spinner("Processing..."):
            try:
                response = rag_chain.invoke(query)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                return

        ai_response = response['text']

        # Add AI response to chat history
        st.session_state.messages.append({"role": "assistant", "content": ai_response})
        # Display AI response in chat message container
        with st.chat_message("assistant"):
            st.write(ai_response)

if __name__ == "__main__":
    main()
#====================================================
#improved the chat surface.
# import streamlit as st
# from langchain.vectorstores import Chroma
# from langchain.prompts import PromptTemplate
# from langchain.schema.runnable import RunnablePassthrough
# from langchain.chains import LLMChain
# from langchain.memory import ConversationBufferMemory
# from langchain_openai import OpenAI
# from langchain_openai import OpenAIEmbeddings, OpenAI
# import chromadb
# from chromadb.api.types import EmbeddingFunction


# def main():
#     class ChromaEmbeddingFunction(EmbeddingFunction):
#         def __init__(self, embeddings):
#             self.embeddings = embeddings

#         def __call__(self, texts):
#             return self.embeddings.embed_documents(texts)

#     st.title("Conversational AI for F1 Visa Consultation")

#     # Load the Chroma vector store using the pre-built embeddings
#     embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
#     chroma_embedding_fn = ChromaEmbeddingFunction(embeddings)
#     db = chromadb.PersistentClient(path="simple_langchain_chroma_db")
#     vector_store_from_client = Chroma(collection_name="my_collection", client=db, embedding_function=embeddings)

#     retriever = vector_store_from_client.as_retriever(
#         search_type="similarity",
#         search_kwargs={"k": 10}
#     )

#     memory = ConversationBufferMemory(memory_key="chat_history", input_key="question", return_messages=True)

#     prompt_template_value = """
#         You are an F1 visa expert and an immigration consultant. The CONTEXT contains relevant information.
#         If the CONTEXT does not contain the answer, give a concise answer.
#         CONTEXT: {context}

#         Question: {question}
#         """.strip()

#     prompt = PromptTemplate(input_variables=['context', 'question'], template=prompt_template_value)
#     llm = OpenAI(temperature=0.7)
#     llm_chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

#     def format_docs(d):
#         return "\n\n".join(doc.page_content for doc in d)

#     rag_chain = (
#         {"context": retriever | format_docs, "question": RunnablePassthrough()} | llm_chain
#     )

#     if "conversation_history" not in st.session_state:
#         st.session_state.conversation_history = []

#     for i, (user_input, ai_response) in enumerate(st.session_state.conversation_history):
#         st.write(f"**You:** {user_input}")
#         st.write(f"**AI:** {ai_response}")

#     query = st.text_area("Ask a question or continue the conversation:", height=100)
#     button = st.button("Send")

#     if button and query:
#         with st.spinner("Processing..."):
#             try:
#                 response = rag_chain.invoke(query)
#             except Exception as e:
#                 st.error(f"An error occurred: {str(e)}")
#                 return

#             ai_response = response['text']
#             st.session_state.conversation_history.append((query, ai_response))

#             # Display the updated conversation
#             st.write(f"**You:** {query}")
#             st.write(f"**AI:** {ai_response}")

# if __name__ == "__main__":
#     main()
