"""
tunkara_faq_bot.py

FAQ chatbot for my Etsy shop, TunkaraShop. Answers customer questions
about digital downloads, editing in Canva, printing, licensing, and the
specific products (budget planner, debt tracker, fitness tracker, resume
templates) using a small FAQ dataset I wrote myself based on questions
I've actually gotten as messages on Etsy.

Built with LangChain: it loads the FAQ rows as documents, embeds them,
stores them in a FAISS vector index, and when someone asks a question it
retrieves the most relevant FAQ entries and passes them to an LLM (GPT
through the OpenAI API) along with the question, so the answer stays
grounded in my actual FAQ instead of the model just making something up.

One thing about how I built this - I didn't always have reliable
internet/API access while working on it, so I added a fallback mode
that runs the retrieval part with a simple TF-IDF search (scikit-learn)
instead of OpenAI's embeddings, and skips the LLM step in favor of just
returning the closest matching FAQ answer directly. If langchain, faiss,
and an OPENAI_API_KEY are all available, it automatically uses the real
LangChain + GPT pipeline instead. Both paths use the same FAQ data and
the same retrieval logic conceptually, just swapping which tools do the
embedding/generation.

- Ousmana Tunkara
"""

import os
import sys
import pandas as pd

# --- try to import the real stack, fall back if it's not available ---
LANGCHAIN_OK = True
try:
    from langchain.schema import Document
    from langchain.embeddings.base import Embeddings
    from langchain_community.vectorstores import FAISS
    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate
except ImportError:
    LANGCHAIN_OK = False

# TF-IDF fallback always available since it's just scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

FAQ_PATH = "tunkarashop_faq.csv"
TOP_K = 3


# ---------------------------------------------------------
# a tiny custom embedding class so this can run without ever
# calling OpenAI's embedding endpoint (keeps API costs down while
# testing, and means the vector store still works offline)
# ---------------------------------------------------------
if LANGCHAIN_OK:
    class TfidfEmbeddings(Embeddings):
        """A LangChain-compatible embedding class backed by TF-IDF instead
        of a real embedding model. Not as good as OpenAI/HuggingFace
        embeddings semantically, but it's free, offline, and good enough
        for a small FAQ set like this one."""

        def __init__(self, corpus):
            self.vectorizer = TfidfVectorizer(stop_words="english")
            self.vectorizer.fit(corpus)

        def embed_documents(self, texts):
            return self.vectorizer.transform(texts).toarray().tolist()

        def embed_query(self, text):
            return self.vectorizer.transform([text]).toarray()[0].tolist()


def load_faq(path=FAQ_PATH):
    df = pd.read_csv(path)
    df["combined"] = df["question"] + " " + df["answer"]
    return df


def build_langchain_store(df):
    docs = [
        Document(
            page_content=row["combined"],
            metadata={"question": row["question"], "answer": row["answer"], "category": row["category"]},
        )
        for _, row in df.iterrows()
    ]
    embeddings = TfidfEmbeddings(df["combined"].tolist())
    store = FAISS.from_documents(docs, embeddings)
    return store


def get_llm():
    # only reached if LANGCHAIN_OK is True and someone actually has an
    # OPENAI_API_KEY set. swap this for ChatAnthropic if you'd rather
    # use Claude - same idea, different import.
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.2)


PROMPT_TEMPLATE = """You are a helpful customer support assistant for TunkaraShop,
an Etsy shop that sells digital planners, trackers, and resume templates.

Answer the customer's question using ONLY the FAQ context below. If the
context doesn't cover it, say you're not sure and suggest they message
the shop directly. Keep the answer short and friendly.

FAQ context:
{context}

Customer question: {question}
Answer:"""


def answer_with_langchain(store, question):
    results = store.similarity_search(question, k=TOP_K)
    context = "\n".join(f"- Q: {r.metadata['question']}\n  A: {r.metadata['answer']}" for r in results)
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    llm = get_llm()
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})
    return response.content, results


# ---------------------------------------------------------
# fallback path - no langchain / no API key needed, just TF-IDF search
# over the same FAQ rows and a direct return of the closest match
# ---------------------------------------------------------
def build_fallback_index(df):
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(df["combined"])
    return vectorizer, matrix


def answer_with_fallback(df, vectorizer, matrix, question):
    q_vec = vectorizer.transform([question])
    sims = cosine_similarity(q_vec, matrix)[0]
    top_idx = sims.argsort()[::-1][:TOP_K]
    top_rows = df.iloc[top_idx]
    best = top_rows.iloc[0]

    if sims[top_idx[0]] < 0.15:
        answer = "Hmm, I'm not finding a good match for that in the FAQ. Try rephrasing, or message the shop directly on Etsy."
    else:
        answer = best["answer"]

    return answer, top_rows


def chat_loop():
    df = load_faq()
    print(f"Loaded {len(df)} FAQ entries for TunkaraShop.\n")

    use_real_llm = LANGCHAIN_OK and os.environ.get("OPENAI_API_KEY")

    if use_real_llm:
        print("LangChain + OpenAI detected - using the full retrieval + GPT pipeline.\n")
        store = build_langchain_store(df)
    else:
        print("Running in offline fallback mode (TF-IDF search, no API calls).")
        print("To use the real LangChain + GPT pipeline: pip install langchain langchain-community")
        print("langchain-openai faiss-cpu, then set OPENAI_API_KEY before running this again.\n")
        vectorizer, matrix = build_fallback_index(df)

    print("Ask a question about TunkaraShop (or type 'quit' to exit)\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in ("quit", "exit"):
            print("Bot: Thanks for stopping by TunkaraShop! Bye for now.")
            break
        if not question:
            continue

        if use_real_llm:
            answer, sources = answer_with_langchain(store, question)
        else:
            answer, sources = answer_with_fallback(df, vectorizer, matrix, question)

        print(f"Bot: {answer}")
        matched_qs = ", ".join(f'"{s["question"] if isinstance(s, dict) else s.metadata["question"]}"' for s in
                                (sources.to_dict("records") if hasattr(sources, "to_dict") else sources))
        print(f"     (matched FAQ: {matched_qs})\n")


if __name__ == "__main__":
    chat_loop()
