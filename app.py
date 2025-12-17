import streamlit as st
import requests

st.set_page_config(page_title="SHL Assessment AI", layout="centered")

st.title("GENAI SHL Assessment Recommender")
st.markdown("""
    Find the best assessments for your hiring needs using **Semantic Vector Search**. 
    Enter a job description below to see matches.
""")

query = st.text_area(
    "Hiring Requirement / Job Description",
    placeholder="e.g., Senior Java Developer with strong leadership and problem-solving skills",
    height=150
)

num_results = st.sidebar.slider("Number of recommendations", 5, 10, 10)

if st.button("Generate Recommendations", type="primary"):
    if not query.strip():
        st.warning("Please enter some requirements first.")
    else:
        with st.spinner("AI is analyzing assessment catalog..."):
            try:
                response = requests.post(
                    "http://127.0.0.1:8000/recommend",
                    json={"query": query, "top_k": num_results},
                    timeout=15
                )
                response.raise_for_status()
                results = response.json().get("recommendations", [])

                if not results:
                    st.info("No close matches found. Try broadening your query.")
                else:
                    st.subheader(f"Top {len(results)} Matches")
                    for i, r in enumerate(results, 1):
                        with st.expander(f"{i}. {r['name']} (Match: {int(r['score']*100)}%)"):
                            st.write(f"**Type:** {r['type']} | **Duration:** {r['duration']} mins")
                            st.write(f"**Description:** {r['description']}")
                            st.link_button("View on SHL Website", r['url'])

            except requests.exceptions.ConnectionError:
                st.error("API Server is offline. Please run 'python api.py' first.")
            except Exception as e:
                st.error(f"An error occurred: {e}")

st.sidebar.markdown("---")
st.sidebar.info("This system uses Sentence-Transformers (all-MiniLM-L6-v2) and FAISS for sub-millisecond similarity search.")