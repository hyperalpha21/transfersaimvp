TransfersAI - Course Matcher for Transfers

Created by Neel Davuluri and Garrett Bellin

This is a private MVP app that matches transfer courses to William & Mary equivalents using sentence similarity and logistic regression.

How It Works

- Uses MPNet sentence embeddings to compare course descriptions and titles.
- Calculates a combined score using a logistic regression model.
- Flags likely transferable courses and outputs a transfer report card.

Tech Stack
- Streamlit
- SentenceTransformers (all-mpnet-base-v2)
- PyTorch
- pandas, numpy

Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
