from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
import streamlit as st

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def main():
    st.title("Ask questions and get answers from the Reddit F1Visa subreddit.")
    pdf_path = "reddit_posts_top_new.pdf"
    document_text = extract_text_from_pdf(pdf_path)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = text_splitter.split_text(document_text)
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    prompt_template_value = """
        You are an F1 visa expert and an immigration consultant. The CONTEXT contains relevant information.
        If the CONTEXT does not contain the answer, give a concise answer.
        CONTEXT: {context}

        Question: {question}
        """.strip()
    prompt = PromptTemplate(input_variables=['context','question'],template = prompt_template_value)
    llm = OpenAI(temperature=0.7)
    llm_chain = LLMChain(llm=llm, prompt=prompt,memory=memory)
    db = Chroma.from_documents(
        text_chunks,
        embeddings
    )
    retriever = db.as_retriever(
        search_type="similarity",
        search_kwargs={"k":4}
    )
    rag_chain = (
        {"context":retriever, "question": RunnablePassthrough()} | llm_chain
    )

    query = st.text_input("Ask a question: ")
    if query:
            with st.spinner("Processing..."):
                response = rag_chain({"question": query})
                st.write(response["answer"])
    response = rag_chain.invoke(query)
if __name__ == "__main__":
    main()
