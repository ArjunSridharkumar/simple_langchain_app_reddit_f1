from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings, OpenAI
# from langchain.chains import RunnableSequence
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from tqdm import tqdm
from langchain.schema.runnable import RunnablePassthrough
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from chromadb.api.types import EmbeddingFunction
import os
import streamlit as st
import chromadb
from chromadb.utils.batch_utils import create_batches

def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# create_batches(
#         api=chromadb.client(),
#         ids=[str(uuid.uuid4()) for _ in range(len(pages))],
#         metadatas=[t.metadata for t in pages],
#         documents=[t.page_content for t in pages],


def main():
    class ChromaEmbeddingFunction(EmbeddingFunction):
        def __init__(self, embeddings):
            self.embeddings = embeddings

        def __call__(self, texts):
            return self.embeddings.embed_documents(texts)

    # st.title("Ask questions and get answers from the Reddit F1Visa subreddit.")
    pdf_path = "reddit_posts_top_new.pdf"
    document_text = extract_text_from_pdf(pdf_path)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
    text_chunks = text_splitter.split_text(document_text)
    text_chunks = text_splitter.create_documents(text_chunks)
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    chroma_embedding_fn = ChromaEmbeddingFunction(embeddings)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # memory_key="chat_history", input_key="human_input"
    prompt_template_value = """
        You are an F1 visa expert and an immigration consultant. The CONTEXT contains relevant information.
        If the CONTEXT does not contain the answer, give a concise answer.
        CONTEXT: {context}

        Question: {question}
        """.strip()
    prompt = PromptTemplate(input_variables=['context','question'],template = prompt_template_value)
    memory = ConversationBufferMemory(memory_key="chat_history",input_key="question", return_messages=True)
    llm = OpenAI(temperature=0.7)
    llm_chain = LLMChain(llm=llm, prompt=prompt,memory=memory)
    # db = Chroma.from_documents(embedding_function=embedding_function, documents=text_chunks,persist_directory="./data_db/",
                            #    )
    # db.persist()
    db = chromadb.PersistentClient(path="simple_langchain_chroma_db")
    vector_store_from_client = Chroma(client=db)
    col = db.get_or_create_collection("reddit_collection",embedding_function=chroma_embedding_fn)
    # import pdb;pdb.set_trace()
    # db = Chroma.from_documents(
    #     embedding_function=embeddings,
    #     persist_directory="data_db_1/"
    # )
    for i, chunk in enumerate(text_chunks):
        if not chunk.metadata:
            # Assign default metadata if empty
            chunk.metadata = {"source": "reddit_posts", "chunk_id": i}
    batches = create_batches(
        api=db,
        documents=[t.page_content for t in text_chunks],
        metadatas=[t.metadata for t in text_chunks],
        ids=[str(i) for i in range(len(text_chunks))],
        # batch_size=5461  # Use the maximum batch size to avoid the error
    )
    # db = Chroma.from_documents(
    #     embeddings,
    #     persist_directory="data_db_1/"
    # )
    # import pdb;pdb.set_trace()
    # text_chunks = text_chunks[:5]
    for batch in tqdm(batches):
        # db._collection.upsert(
        #     documents=batch['documents'],
        #     metadatas=batch['metadatas'],
        #     ids=batch['ids']
        # )
        col.add(*batch)
    # import pdb;pdb.set_trace()
    # db.persist()


    # retriever = vector_store_from_client.as_retriever(
    #     search_type="similarity",
    #     search_kwargs={"k":10}
    # )


#     print (retriever)
#     rag_chain = (
#         {"context":retriever, "question": RunnablePassthrough()} | llm_chain
#     )

#     # query = st.text_area("Ask a question or continue the conversation:", height=100)
#     # button = st.button("Send")
#     if True:
#             with st.spinner("Processing..."):
#                 # response = rag_chain({"question": query})
#                 response = rag_chain.invoke(query)
#                 conversation_history += f"**You:** {query}\n\n**AI:** {response['text']}\n\n"
#                 st.write(conversation_history)
#                 print (response)
#                 st.write(response['text'])
#     # response = rag_chain.invoke(query)
# if __name__ == "__main__":
#     main()
