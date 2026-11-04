import streamlit as st
import streamlit.components.v1 as components

NEW_URL = "https://huggingface.co/spaces/ユーザー名/スペース名"

st.set_page_config(page_title="移転しました", page_icon="🧠")

# 自動で移動を試みる
components.html(
    f"""
    <script>
      try {{ window.top.location.href = "{NEW_URL}"; }}
      catch (e) {{ window.parent.location.href = "{NEW_URL}"; }}
    </script>
    """,
    height=0,
)

# 自動で飛ばなかった場合のためのボタン
st.title("四脳分類 顔診断")
st.info("このツールは新しい場所に移転しました。自動で移動しない場合は下のボタンを押してください。")
st.link_button("新しいページへ移動する", NEW_URL, type="primary")