import streamlit as st
import pandas as pd
import numpy as np
import torch
import math
from sentence_transformers import SentenceTransformer, util
from typing import List

@st.cache_resource
def load_model():
    return SentenceTransformer('all-mpnet-base-v2')

model = load_model()

#wm database
@st.cache_data
def load_wm_courses():
    wm_df = pd.read_csv("wm_courses_2025.csv", encoding='latin1')
    wm_df = wm_df.dropna(subset=['course_description', 'course_title'])

    # Generate embeddings for both title and description
    wm_df['desc_embedding'] = wm_df['course_description'].apply(
        lambda x: model.encode(str(x), convert_to_tensor=True)
    )
    wm_df['title_embedding'] = wm_df['course_title'].apply(
        lambda x: model.encode(str(x), convert_to_tensor=True)
    )
    return wm_df

wm_df = load_wm_courses()


st.title("TransfersAI")
st.markdown("Enter any number of courses below. We'll return the most similar William & Mary course(s)")

#input
with st.form("course_form"):
    num_courses = st.number_input("Number of courses to compare:", min_value=1, max_value=50, value=3)
    input_courses = []
    for i in range(num_courses):
        st.markdown(f"### Course {i+1}")
        input_title = st.text_input(f"Course Title {i+1}", key=f"title_{i}")
        input_desc = st.text_area(f"Course Description {i+1}", key=f"desc_{i}")
        input_courses.append((input_title.strip(), input_desc.strip()))
    submitted = st.form_submit_button("Compare Courses")

if submitted:
    results = []
    for input_title, input_desc in input_courses:
        if not input_desc:
            st.warning(f"Missing description for '{input_title}'. Skipping.")
            continue

        desc_embedding = model.encode(input_desc, convert_to_tensor=True)
        title_embedding = model.encode(input_title, convert_to_tensor=True)

        wm_desc_tensor = torch.stack(wm_df['desc_embedding'].tolist())
        wm_title_tensor = torch.stack(wm_df['title_embedding'].tolist())

        desc_sim_tensor = util.cos_sim(desc_embedding, wm_desc_tensor)[0]
        title_sim_tensor = util.cos_sim(title_embedding, wm_title_tensor)[0]

        desc_sim_np = desc_sim_tensor.cpu().numpy()
        title_sim_np = title_sim_tensor.cpu().numpy()

        combined_scores = 1 / (1 + np.exp(-(-5.888 + 8.2 * desc_sim_np + 2.71 * title_sim_np)))
        best_idx = np.argmax(combined_scores)
        best_row = wm_df.iloc[best_idx]

        results.append({
            "Input Course Title": input_title,
            "Most Similar W&M Course": best_row['course_title'],
            "Matched Description": best_row['course_description'],
            "Description Similarity": round(desc_sim_np[best_idx], 2),
            "Title Similarity": round(title_sim_np[best_idx], 2),
            "Combined Score": round(combined_scores[best_idx], 2),
            "Transferable": "✅ Yes" if combined_scores[best_idx] >= 0.6 else "❌ No"
        })

    if results:
        st.success("Matching complete!")

        st.subheader("Transfer Summary")
        st.markdown(f"**Courses Submitted:** {len(results)}")
        st.markdown(f"**Likely Transferable:** {len([r for r in results if r['Transferable'] == 'Yes'])}")
        st.markdown(f"**Not Likely Transferable:** {len([r for r in results if r['Transferable'] == 'No'])}")

        st.subheader("Transfer Report Card")
        report_df = pd.DataFrame([{
            "Input Title": r["Input Course Title"],
            "Matched W&M Course": r["Most Similar W&M Course"],
            "Score": r["Combined Score"],
            "Transferable": r["Transferable"]
        } for r in results])

        st.dataframe(report_df.style.applymap(
            lambda v: 'background-color: #d1e7dd' if v == '✅ Yes' else 'background-color: #f8d7da',
            subset=["Transferable"]
        ))

        st.download_button("Download Transfer Report", report_df.to_csv(index=False), file_name="transfer_report.csv")

        with st.expander("🧠 Full Similarity Details"):
            for r in results:
                st.markdown(f"**{r['Input Course Title']}** ➝ **{r['Most Similar W&M Course']}**")
                st.markdown(f"**Description Similarity:** {r['Description Similarity']} | **Title Similarity:** {r['Title Similarity']}")
                st.markdown(f"**Combined Score:** {r['Combined Score']}")
                st.markdown(f"**Matched Description:** {r['Matched Description']}")
                st.markdown("---")
