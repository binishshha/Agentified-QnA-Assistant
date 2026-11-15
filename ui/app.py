import os
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
st.set_page_config(page_title="Production RAG Assistant", page_icon="🤖")
st.title("🤖 Production RAG Assistant")
st.caption("Upload documents, then ask grounded questions about them.")

with st.sidebar:
    st.header("Documents")
    uploaded = st.file_uploader(
        "PDF, Markdown, or text",
        type=["pdf", "md", "txt"],
        accept_multiple_files=True,
    )
    if uploaded and st.button("Upload documents"):
        try:
            files = [("files", (document.name, document.getvalue())) for document in uploaded]
            response = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=300)
            response.raise_for_status()
            result = response.json()
            for item in result.get("uploaded", []):
                st.success(f"Indexed {item['file']} ({item['chunks']} chunks)")
            for item in result.get("failed", []):
                st.error(f"Upload failed for {item['file']}: {item['error']}")
            st.info(f"Indexed {len(result.get('uploaded', []))} document(s)")
        except requests.HTTPError:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            st.error(f"Batch upload failed ({response.status_code}): {detail}")
        except requests.RequestException as error:
            st.error(f"Batch upload failed: {error}")
    top_k = st.slider("Retrieved passages", 1, 10, 5)
    research_mode = st.checkbox("Research mode", help="Show the evidence-driven tool trajectory.")

if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question := st.chat_input("Ask a question about your documents"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            endpoint = "/research" if research_mode else "/ask"
            payload = {"question": question} if research_mode else {"question": question, "top_k": top_k}
            response = requests.post(
                f"{BACKEND_URL}{endpoint}",
                json=payload,
                timeout=300,
            )
            if response.status_code == 429:
                answer = "Rate limit reached. Please try again shortly."
            else:
                response.raise_for_status()
                data = response.json()
                meta = data.get("meta") or {}
                if research_mode:
                    answer = data["answer"] + f"\n\n*Status: `{data.get('status')}` | Steps: {data.get('steps')} | Tokens: {data.get('tokens')}*"
                    with st.expander("Research trace"):
                        st.json(data.get("trace", []))
                else:
                    answer = data["answer"] + f"\n\n*Model: `{meta.get('model', 'unknown')}` | Provider: {meta.get('provider', 'unknown')} | Cached: {meta.get('cached', False)}*"
                if data.get("sources"):
                    answer += "\n\n**Sources:** " + ", ".join(data["sources"])
        except requests.RequestException as error:
            answer = f"The backend is unavailable right now: {error}"
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
