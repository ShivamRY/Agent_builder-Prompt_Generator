import streamlit as st
from backend import prompt_generator

# --- PAGE CONFIG ---
st.set_page_config(page_title="Agent Builder", page_icon="🤖", layout="centered")

# --- HEADER / TITLE ---
st.markdown(
    """
    <div style="text-align:center;">
        <h1 style="color:#4CAF50;"> Agent Builder – Prompt Generator</h1>
        <p style="font-size:17px; color:gray;">
            Build intelligent AI calling agents by just filling a few fields 🎯
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# --- SIDEBAR HELP / ABOUT ---
with st.sidebar:
    st.header("⚙️ Settings")
    st.info("Fill details & generate a call-agent prompt instantly.")
    st.write("🌐 Supports multilingual prompt generation.")
    st.write("📞 Structured inputs only (no raw metadata).")
    st.markdown("---")
    st.caption("🔧 built using Streamlit")

# --- MAIN UI ---
st.subheader("Company Details")
with st.container():
    col1, col2 = st.columns(2)

    with col1:
        company_name = st.text_input("Company Name")
        product_category = st.text_input("Product Category")
        customer_care = st.text_input("Customer Care Number")

    with col2:
        collection_stage = st.selectbox("Stage of Collection", ["Pre-Due", "Post-Due"])
        support_email = st.text_input("Support Email")
        website = st.text_input("Company Website URL (optional)")

st.markdown("---")

st.subheader("Agent Details")
with st.container():
    colA, colB = st.columns(2)

    with colA:
        agent_name = st.text_input("Agent Name")

    with colB:
        agent_description = st.text_area(
            "Agent Description (voice tone, persona, behavior)",
            placeholder="Example: Polite, empathetic, metro customer language tone..."
        )

st.markdown("---")

language = st.selectbox(
    "🌍 Target Language",
    ["English", "Hindi", "Marathi"]
)

# ---- Generate Button ----
generated = False
result = None

if st.button("🚀 Generate Prompt", use_container_width=True):
    metadata_list = []

    if company_name:
        metadata_list.append(f"{{{{company_name}}}} = {company_name}")
    if agent_name:
        metadata_list.append(f"{{{{agent_name}}}} = {agent_name}")
    if product_category:
        metadata_list.append(f"{{{{product_category}}}} = {product_category}")
    if collection_stage:
        metadata_list.append(f"{{{{collection_stage}}}} = {collection_stage}")
    if customer_care:
        metadata_list.append(f"{{{{customer_care_number}}}} = {customer_care}")
    if support_email:
        metadata_list.append(f"{{{{support_email}}}} = {support_email}")
    if website:
        metadata_list.append(f"{{{{website}}}} = {website}")

    final_metadata = "\n".join(metadata_list)

    if not final_metadata.strip():
        st.error("⚠️ Please fill at least one field before generating.")
    else:
        with st.spinner("✨ Generating AI Prompt..."):
            try:
                result = prompt_generator(final_metadata, language)
                generated = True
            except Exception as e:
                st.error(f"❌ Error occurred: {e}")

# ---- DISPLAY RESULT ----
if generated and result:
    st.success("🎉 Prompt Generated Successfully!")
    st.text_area("📌 Final Prompt Output", result, height=400)

    st.download_button(
        "⬇️ Download Prompt",
        data=result,
        file_name="agent_prompt.txt"
    )
