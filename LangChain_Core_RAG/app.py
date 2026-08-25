import gradio as gr
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import re
import os

# ── 전역 변수 ─────────────────────────────────────────────────
# 체인과 벡터스토어를 전역으로 유지하여 PDF 업로드 후 재사용
chain = None
vectorstore = None

# ── 1. PDF 처리 및 RAG 체인 초기화 ───────────────────────────
def process_pdf(pdf_file, openai_api_key):
    """
    업로드된 PDF를 처리하여 RAG 체인을 초기화합니다.
    - PDF 로드 → 전처리 → 청크 분할 → 임베딩 → 벡터 DB 저장 → 체인 생성
    """
    global chain, vectorstore

    if not openai_api_key:
        return "❌ OpenAI API 키를 입력해주세요."
    if pdf_file is None:
        return "❌ PDF 파일을 업로드해주세요."

    os.environ["OPENAI_API_KEY"] = openai_api_key

    try:
        # 1-1. PDF 로드
        loader = PyMuPDFLoader(pdf_file.name)
        pages = loader.load()

        # 1-2. 텍스트 전처리: 특수문자·불필요한 공백 제거
        for page in pages:
            page.page_content = re.sub(
                r'[^\w\s]', '',
                page.page_content
                    .replace('\xa0', '')
                    .replace('\n\n', '\n')
            )

        # 1-3. 청크 분할 (chunk_size=1000, overlap=200)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        splits = text_splitter.split_documents(pages)

        # 1-4. 임베딩 모델 초기화
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

        # 1-5. 기존 벡터 DB 삭제 후 재생성 (중복 방지)
        db_name = "OpenAI_Embeddings_DB"
        if os.path.exists(db_name):
            Chroma(
                persist_directory=db_name,
                embedding_function=embeddings
            ).delete_collection()

        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=db_name
        )

        # 1-6. Retriever 생성
        retriever = vectorstore.as_retriever()

        # 1-7. 시스템 프롬프트 정의
        system_prompt = """'질의응답' 과정의 자료가 아래에 주어집니다. 질문에 대해 아래의 순서로 답변하세요. 한국어로 답변하세요.
1. 주어진 문서의 내용이 질문에 얼마나 적합한지 평가하세요.
2. 질문에 대한 답변을 출력하세요.
3. metadata page도 답하라
---
{summaries}
---
Answer:"""

        messages = [
            SystemMessagePromptTemplate.from_template(system_prompt),
            HumanMessagePromptTemplate.from_template("Question: {question}")
        ]
        prompt = ChatPromptTemplate.from_messages(messages)

        # 1-8. LLM 및 RAG 체인 구성
        llm = ChatOpenAI(
            model_name="gpt-4.1-nano",
            temperature=0.1,
            max_tokens=4096
        )

        chain = (
            {"summaries": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        doc_count = vectorstore._collection.count()
        return f"✅ PDF 처리 완료! 총 {len(pages)}페이지 / {doc_count}개 청크가 벡터 DB에 저장되었습니다."

    except Exception as e:
        return f"❌ 오류 발생: {str(e)}"


# ── 2. 질의응답 함수 ──────────────────────────────────────────
def answer_question(question, history):
    """
    사용자 질문을 받아 RAG 체인으로 답변을 생성합니다.
    history: Gradio Chatbot이 관리하는 대화 기록 리스트
    """
    global chain

    if chain is None:
        return history + [("⚠️ 먼저 PDF를 업로드하고 처리해주세요.", None)]
    if not question.strip():
        return history

    try:
        answer = chain.invoke(question)
        # (사용자 질문, 모델 답변) 튜플을 history에 추가
        history = history + [(question, answer)]
        return history
    except Exception as e:
        history = history + [(question, f"❌ 오류 발생: {str(e)}")]
        return history


# ── 3. Gradio UI 구성 ─────────────────────────────────────────
with gr.Blocks(
    title="📄 PDF RAG 질의응답",
    theme=gr.themes.Soft(),
    css="""
        /* 전체 컨테이너 최대 너비 제한 */
        .gradio-container { max-width: 900px; margin: auto; }
        /* 채팅 영역 높이 고정 */
        #chatbot { height: 480px; overflow-y: auto; }
        /* 업로드 버튼 강조 */
        #upload_btn { background-color: #2563eb; color: white; }
        /* 전송 버튼 강조 */
        #send_btn { background-color: #16a34a; color: white; }
    """
) as demo:

    # 상단 제목
    gr.Markdown(
        """
        # 📄 PDF RAG 질의응답 시스템
        PDF를 업로드하고 문서 내용에 대해 자유롭게 질문하세요.
        """
    )

    with gr.Row():
        # ── 왼쪽 패널: 설정 영역 ──────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ 설정")

            # OpenAI API 키 입력 (type="password"로 마스킹)
            api_key_input = gr.Textbox(
                label="🔑 OpenAI API Key",
                placeholder="sk-...",
                type="password",
                info="API 키는 서버에 저장되지 않습니다."
            )

            # PDF 파일 업로드
            pdf_input = gr.File(
                label="📂 PDF 파일 업로드",
                file_types=[".pdf"],
                type="filepath"
            )

            # 처리 버튼
            upload_btn = gr.Button(
                "🚀 PDF 처리 시작",
                variant="primary",
                elem_id="upload_btn"
            )

            # 처리 상태 메시지 출력
            status_output = gr.Textbox(
                label="📊 처리 상태",
                interactive=False,
                lines=3
            )

        # ── 오른쪽 패널: 채팅 영역 ────────────────────────────
        with gr.Column(scale=2):
            gr.Markdown("### 💬 질의응답")

            # 대화 기록 표시 영역
            chatbot = gr.Chatbot(
                label="대화 내용",
                elem_id="chatbot",
                bubble_full_width=False,
                show_copy_button=True  # 답변 복사 버튼 활성화
            )

            with gr.Row():
                # 질문 입력창 (Enter로 전송 가능)
                question_input = gr.Textbox(
                    label="질문 입력",
                    placeholder="문서에 대해 궁금한 점을 입력하세요...",
                    lines=2,
                    scale=4
                )
                # 전송 버튼
                send_btn = gr.Button(
                    "전송 ➤",
                    variant="primary",
                    scale=1,
                    elem_id="send_btn"
                )

            # 대화 초기화 버튼
            clear_btn = gr.ClearButton(
                [chatbot, question_input],
                value="🗑️ 대화 초기화"
            )

    # ── 4. 이벤트 연결 ────────────────────────────────────────
    # PDF 처리 버튼 클릭 시
    upload_btn.click(
        fn=process_pdf,
        inputs=[pdf_input, api_key_input],
        outputs=[status_output]
    )

    # 전송 버튼 클릭 시 답변 생성 → 입력창 초기화
    send_btn.click(
        fn=answer_question,
        inputs=[question_input, chatbot],
        outputs=[chatbot]
    ).then(
        fn=lambda: "",          # 질문 전송 후 입력창 자동 비우기
        outputs=[question_input]
    )

    # Enter 키로도 질문 전송 가능
    question_input.submit(
        fn=answer_question,
        inputs=[question_input, chatbot],
        outputs=[chatbot]
    ).then(
        fn=lambda: "",
        outputs=[question_input]
    )

    # 사용 안내
    gr.Markdown(
        """
        ---
        **사용 방법**
        1. OpenAI API 키를 입력하세요.
        2. PDF 파일을 업로드하고 **PDF 처리 시작** 버튼을 클릭하세요.
        3. 처리 완료 메시지 확인 후 질문을 입력하세요.
        """
    )

# ── 5. 앱 실행 ────────────────────────────────────────────────
# share=True  → 외부 공유 링크 생성 (Colab 등 원격 환경에서 유용)
# share=False → 로컬 환경에서만 접근 가능
if __name__ == "__main__":
    demo.launch(share=False)