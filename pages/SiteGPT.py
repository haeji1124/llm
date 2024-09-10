from langchain.document_loaders import SitemapLoader
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st
import requests, os


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


def is_valid(key):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}",
    }
    try:
        response = requests.get(
            "https://api.openai.com/v1/models", headers=headers)
        if response.status_code == 200:
            return True
        else:
            return False
    except:
        return False
    

answers_prompt = ChatPromptTemplate.from_template(
    """
    다음의 context만을 이용해서 질문에 답해야 합니다. 모르면 모른다고 대답하고, 꾸며내거나 과장하지 마세요.
    각 대답에는 0에서 5까지의 score를 부여하세요.
    대답이 정확할 수록 score는 높아야하고, 정확하지 않을 수록 score는 낮아야합니다.
    0점이라고 해도 대답의 score를 포함하세요.

    Context: {context}

    예시:

    question: 달은 얼마나 멀리 떨어져 있나요?
    answer: 달은 384,400 km 떨어져있습니다.
    score: 5

    question: 태양은 얼마나 멀리 떨어져 있나요?
    answer: 모릅니다.
    score: 0

    이제 당신의 차례입니다!

    question: {question}
"""
)


def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]
    # Map Re Rank의 과정은 출력하지 않기 위해 streaming 하지 않는 llm 사용
    answers_chain = answers_prompt | llm
    return {
        "question": question,
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {"question": question, "context": doc.page_content}
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"],
            }
            for doc in docs
        ],
    }


choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            아래에 제시되는 대답들만을 사용해서 질문에 최종적으로 대답하세요.
            score가 높고(도움이 되는) 최신의 대답을 사용하세요.
            대답의 출처와 lastmod를 덧붙이되, 수정하지 말고 그대로 제시하세요.

            answers: {answers}
            """,
        ),
        ("human", "{question}"),
    ]
)


def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]
    # 최종 결과는 출력을 위해 streaming 하는 llm_streaming 사용
    choose_chain = choose_prompt | llm_streaming
    condensed = "\n\n".join(
        f"{answer['answer']}\nSource:{answer['source']}\nDate:{answer['date']}\n"
        for answer in answers
    )
    return choose_chain.invoke(
        {
            "question": question,
            "answers": condensed,
        }
    )


def parse_page(soup):
    # Cloudflare의 경우 <main> 태그에 컨텐츠가 포함됨
    return (
        str(soup.find("main").get_text())
        .replace("\n", " ")
        .replace("\xa0", " ")
        .replace("Edit page   Cloudflare DashboardDiscordCommunityLearning CenterSupport Portal  Cookie Settings", "")
    )


@st.cache_resource(show_spinner="웹사이트 로딩 중...")
def load_website(url, key):
    # .cache 폴더가 없으면 생성해준다.
    file_folder = './.cache/embeddings/site'
    
    if not os.path.exists(file_folder):
        os.makedirs(file_folder)

    cache_dir = LocalFileStore(f"{file_folder}")

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )
    loader = SitemapLoader(
        url,
        parsing_function=parse_page,
        # 아래 3개의 URL만 대상으로 함
        filter_urls=[
            'https://developers.cloudflare.com/ai-gateway/',
            'https://developers.cloudflare.com/vectorize/',
            'https://developers.cloudflare.com/workers-ai/',
        ]
    )
    loader.requests_per_second = 2
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings(api_key=key)
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vector_store = FAISS.from_documents(docs, cached_embeddings)
    return vector_store.as_retriever()


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


st.set_page_config(
    page_title="SiteGPT",
    page_icon="🖥️",
)

st.title("🖥️ SiteGPT")

st.markdown(
    """
    # SiteGPT
            
    Ask questions about the content of a website.
            
    Start by writing the URL of the website on the sidebar.
"""
)

with st.sidebar:
    # Cloudflare를 위한 문서로 URL을 고정시킴
    url = "https://developers.cloudflare.com/sitemap-0.xml"

    key = st.text_input("OPEN_API_KEY", placeholder="OPENAI_API_KEY를 입력해주세요.", type="password")

    if key:
        # OPENAI_API_KEY 가 입력되면 파일 업로드 가능
        if is_valid(key):
            st.success("유효한 OPENAI_API_KEY 입니다.")
            
            # ChatCallbackHandler()에서 llm을 모니터링하면서 token을 추가해주는데, Map Re Rank의 중간과정은 출력해주고 싶지 않았음
            # 따라서, llm을 2개로 분리하여, get_answers()는 streaming을 하지 않고, choose_answer는 streaming 처리
            llm = ChatOpenAI(
                temperature=0.1,
                model="gpt-4o-mini-2024-07-18",
                # streaming=True,
                # callbacks=[ChatCallbackHandler(),],
                api_key=key
            )

            llm_streaming = ChatOpenAI(
                temperature=0.1,
                model="gpt-4o-mini-2024-07-18",
                streaming=True,
                callbacks=[ChatCallbackHandler(),],
                api_key=key
            )

            ############ 여기서는 URL을 고정시켰으므로, URL 입력 부분은 주석 처리
            # url = st.text_input(
            #     "URL을 입력하세요.", "",
            #     placeholder="https://example.com",
            # )
        else:
            st.warning("올바른 OPENAI_API_KEY를 입력하세요.")
            key = ""

if key:
    ############ URL의 구조 파악 및 parsing을 위한 테스트 코드
    # if url:
    #     loader = SitemapLoader(url,
    #         parsing_function=parse_page,
    #         filter_urls=[
    #             'https://developers.cloudflare.com/ai-gateway/',
    #             'https://developers.cloudflare.com/vectorize/',
    #             'https://developers.cloudflare.com/workers-ai/',
    #         ]
    #     )
    #     loader.requests_per_second = 1
    #     docs = loader.load()
    #     st.write(docs)

    ############ 여기서는 URL을 고정시켰으므로, Sitemap URL 여부를 체크하는 부분은 주석 처리
    # if ".xml" not in url:
    #     with st.sidebar:
    #         st.error("Sitemap URL로 작성해 주세요.")
    # else:
    retriever = load_website(url, key)

    send_message("Hi", "ai", save=False)
    paint_history()
    message = st.chat_input("please ask about document")
    if message:
        send_message(message, "human")
        chain = (
            {
                "docs": retriever,
                "question": RunnablePassthrough(),
            }
            | RunnableLambda(get_answers)
            | RunnableLambda(choose_answer)
        )
        with st.chat_message("ai"):
            chain.invoke(message).content.replace("$", "\$")
else:
    st.session_state["messages"] = []