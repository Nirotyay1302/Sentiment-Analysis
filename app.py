import streamlit as st
import pandas as pd
import joblib
import snscrape.modules.twitter as sntwitter
from youtube_comment_downloader import YoutubeCommentDownloader
import os
import platform
import re
from textblob import TextBlob
import io
from threading import Thread, Event
from http.server import HTTPServer, BaseHTTPRequestHandler
import json

# (OCR support removed for simpler hosting; app no longer requires Tesseract)

# Load sentiment model for dataset/social media
pipe = joblib.load("model.joblib")
labels = {0: "Negative", 1: "Neutral", 2: "Positive"}

 # Health-check server using builtin http.server so hosting platforms can probe readiness
ready_event = Event()


class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            payload = {"ready": ready_event.is_set()}
            self.wfile.write(json.dumps(payload).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()


def run_health_server(port: int = 8502):
    try:
        server = HTTPServer(("0.0.0.0", port), _HealthHandler)
        server.serve_forever()
    except Exception:
        # If the platform restricts starting a TCP server, skip silently
        return


# After model and resources loaded, set readiness
ready_event.set()

# Start health server in background thread (non-blocking)
try:
    health_port = int(os.environ.get("HEALTH_PORT", "8502"))
    t = Thread(target=run_health_server, args=(health_port,), daemon=True)
    t.start()
except Exception:
    pass


def get_model_classes(model):
    """Try to retrieve class labels from the fitted model/pipeline."""
    try:
        # common: pipeline with classifier named 'clf' or last step
        if hasattr(model, 'classes_'):
            return list(model.classes_)
        if hasattr(model, 'named_steps'):
            # take last estimator
            last = list(model.named_steps.values())[-1]
            if hasattr(last, 'classes_'):
                return list(last.classes_)
        # fallback: unknown
    except Exception:
        pass
    return None


def read_csv_with_header_detection(uploaded):
    """Read a CSV and try to detect if the real header is on a later row (common with exported CSVs that include a title row).

    Returns (df, header_row_index_or_None)
    """
    try:
        # Create a text buffer for preview
        if hasattr(uploaded, 'read'):
            uploaded.seek(0)
            sample_text = uploaded.read().decode('utf-8', errors='ignore')
            preview_buf = io.StringIO(sample_text)
        else:
            preview_buf = uploaded

        preview = pd.read_csv(preview_buf, header=None, nrows=10, dtype=str, keep_default_na=False)
        header_row = None
        for i, row in preview.iterrows():
            row_vals = ' '.join([str(x).lower() for x in row.tolist()])
            if re.search(r'\btext\b', row_vals):
                header_row = i
                break

        # Fallback: if first row contains title-like text, assume header is second row
        if header_row is None:
            first0 = str(preview.iloc[0, 0]).lower()
            if 'social media' in first0 or 'sentiments' in first0 or 'social' in first0:
                header_row = 1

        # Read the full CSV using detected header
        if hasattr(uploaded, 'read'):
            uploaded.seek(0)
            if header_row is not None:
                df = pd.read_csv(uploaded, header=header_row, dtype=str, keep_default_na=False)
            else:
                uploaded.seek(0)
                df = pd.read_csv(uploaded, header=0, dtype=str, keep_default_na=False)
        else:
            if header_row is not None:
                df = pd.read_csv(uploaded, header=header_row, dtype=str, keep_default_na=False)
            else:
                df = pd.read_csv(uploaded, header=0, dtype=str, keep_default_na=False)

        return df, header_row
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return None, None

st.set_page_config(page_title="Social Media Sentiment Analyzer", layout="centered", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    body {background-color: #f5f6fa;}
    .main {background-color: #fff; border-radius: 12px; padding: 2rem;}
    .stButton>button {background-color: #0066cc; color: white; border-radius: 6px;}
    .stFileUploader {border-radius: 6px;}
    .stTextInput>div>div>input {border-radius: 6px;}
    .stTextArea>div>textarea {border-radius: 6px;}
    .stDataFrame {background-color: #f9f9f9; border-radius: 6px;}
    .stAlert {border-radius: 6px;}
    .st-bb {background: #0066cc !important; color: white !important;}
    .stApp {padding-bottom: 60px;}
    .footer {position: fixed; left: 0; bottom: 0; width: 100%; background: #f5f6fa; color: #888; text-align: center; padding: 10px 0; font-size: 0.9rem;}
    .branding {display: flex; align-items: center; gap: 1rem; margin-bottom: 1.5rem;}
    .branding img {height: 48px;}
    .branding-title {font-size: 2.1rem; font-weight: 700; color: #0066cc; letter-spacing: 1px;}
    </style>
    """,
    unsafe_allow_html=True
)



st.sidebar.header("Choose Mode")
mode = st.sidebar.selectbox(
    "",
    [
        "Analyze Dataset",
        "Analyze Social Media Link",
        "Manual Text Input"
    ],
    key="mode_selectbox"
)
st.markdown("<h1 style='text-align: center; color: #0066cc; margin-bottom: 0.5em;'>📊 Social Media Sentiment Analyzer</h1>", unsafe_allow_html=True)

def clean_text(text):
    text = re.sub(r'[^A-Za-z0-9\s.,!?]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def analyze_sentiment_textblob(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.05:
        return "Positive"
    elif polarity < -0.05:
        return "Negative"
    else:
        return "Neutral"


def render_pie_chart(sentiment_counts, title="Sentiment Distribution", colors=None):
    """Render a pie chart. Try Plotly, fall back to Matplotlib, else show a table.

    sentiment_counts: pandas Series (index=labels, values=counts)
    colors: list of colors aligned with sentiment_counts.index
    """
    try:
        import plotly.graph_objects as go

        fig = go.Figure(
            data=[
                go.Pie(
                    labels=sentiment_counts.index.tolist(),
                    values=sentiment_counts.values.tolist(),
                    hole=0.3,
                    marker=dict(colors=colors if colors is not None else ["#2ecc71", "#f1c40f", "#e74c3c"]),
                )
            ]
        )
        fig.update_layout(title=title, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
        return
    except Exception:
        # Plotly not available or error occurred - try matplotlib
        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            ax.pie(
                sentiment_counts.values.tolist(),
                labels=sentiment_counts.index.tolist(),
                autopct="%1.1f%%",
                colors=colors if colors is not None else ["#2ecc71", "#f1c40f", "#e74c3c"],
            )
            ax.axis("equal")
            plt.title(title)
            st.pyplot(fig)
            return
        except Exception:
            # Fallback: textual summary
            st.write(f"**{title}**")
            st.write(sentiment_counts)
            return

def fetch_twitter_replies(url, limit=100):
    tweet_id = url.split("/")[-1]
    replies = []
    try:
        for i, tweet in enumerate(sntwitter.TwitterSearchScraper(f'conversation_id:{tweet_id}').get_items()):
            if i >= limit:
                break
            if tweet.inReplyToTweetId == int(tweet_id):
                replies.append(tweet.content)
    except Exception as e:
        pass
    return replies

def fetch_youtube_comments(url, limit=100):
    downloader = YoutubeCommentDownloader()
    comments = []
    try:
        for i, comment in enumerate(downloader.get_comments_from_url(url, sort_by=0)):
            if i >= limit:
                break
            comments.append(comment["text"])
    except Exception as e:
        pass
    return comments

# ----------- Mode 1: Dataset Analyzer -----------
if mode == "Analyze Dataset":
    st.subheader("📂 Batch Dataset Sentiment Analysis")
    st.markdown("""
    Upload a CSV file containing a column with text data. The app will perform a quick scan of up to 100 rows by default (fast), and you can opt to analyze the full dataset.
    """)
    uploaded_file = st.file_uploader("Upload a dataset (CSV)", type=["csv"], key="dataset_uploader")
    if uploaded_file is not None:
        df, header_row = read_csv_with_header_detection(uploaded_file)
        if df is None:
            st.error("Could not parse uploaded CSV.")
        else:
            if header_row is not None:
                st.info(f"Detected header row at line {header_row + 1} (0-indexed: {header_row}). If this is wrong, override below.")
            # Allow user to override the header row if detection fails
            override_header = st.checkbox("Override detected header row / specify header row manually", value=False, key="override_header_checkbox")
            if override_header:
                hdr = st.number_input("Header row index (0-based)", min_value=0, max_value=1000, value=header_row if header_row is not None else 0, step=1, key="header_row_input")
                try:
                    # Re-read uploaded file with user-specified header
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, header=int(hdr), dtype=str, keep_default_na=False)
                    st.success(f"Re-read CSV with header row = {hdr}")
                except Exception as e:
                    st.error(f"Failed to re-read CSV with header={hdr}: {e}")

        if df is not None:
            # Let the user pick which column contains the text to analyze
            # Heuristic: prefer a column with 'text' in the name; otherwise pick the column with the highest average token length
            cols = list(df.columns)
            detected_col = None
            for c in cols:
                if 'text' in str(c).lower():
                    detected_col = c
                    break
            if detected_col is None:
                # score columns by average length of string values (ignoring empty cells)
                scores = {}
                for c in cols:
                    try:
                        vals = df[c].astype(str).replace('nan','').tolist()
                        lengths = [len(re.sub(r'[^A-Za-z0-9\s]', '', v).strip()) for v in vals if v and v.strip()]
                        scores[c] = sum(lengths)/len(lengths) if lengths else 0
                    except Exception:
                        scores[c] = 0
                detected_col = max(scores, key=scores.get)

            text_col = st.selectbox("Select text column for sentiment analysis", options=cols, index=cols.index(detected_col))
            # warn if the selected column looks mostly numeric or very short
            sample_vals = df[text_col].astype(str).head(20).tolist()
            num_numeric = sum(1 for v in sample_vals if re.fullmatch(r"\s*\d+\s*", v))
            if num_numeric > 5:
                st.warning(f"The selected column '{text_col}' appears to contain many numeric or index-like values. If this is incorrect, pick a different column.")
            total_rows = len(df)
            st.write(f"Rows in dataset: **{total_rows}** — using column: **{text_col}**")

            # Quick scan sample size (user-configurable)
            default_sample = min(100, total_rows)
            max_sample = min(10000, total_rows)
            sample_size = st.number_input(
                "Quick scan sample size (rows)",
                min_value=1,
                max_value=max_sample,
                value=default_sample,
                step=1,
                key="sample_size_input",
            )
            st.info(f"Quick scan will analyze {sample_size} row(s). Toggle below to analyze the full dataset.")

            analyze_full = st.checkbox("Analyze full dataset (may take longer)", value=False, key="analyze_full_checkbox")

            if analyze_full:
                texts = df[text_col].astype(str).apply(clean_text).tolist()
                preds = pipe.predict(texts)
                df["Predicted"] = preds
                df["Predicted_Label"] = df["Predicted"].map(labels)
                result_df = df
                st.success("Full dataset analysis complete")
            else:
                # Use first N rows (deterministic) instead of random sampling
                head_df = df.head(int(sample_size)).copy()
                head_texts = head_df[text_col].astype(str).apply(clean_text).tolist()
                head_preds = pipe.predict(head_texts)
                head_df["Predicted"] = head_preds
                head_df["Predicted_Label"] = head_df["Predicted"].map(labels)
                result_df = head_df
                st.success(f"Quick scan (first {sample_size} rows) complete")

            # Allow filtering by sentiment (based on text predictions)
            sentiment_choices = ["Positive", "Neutral", "Negative"]
            selected_sentiments = st.multiselect("Filter results by sentiment (text) — leave empty to show all", options=sentiment_choices, default=sentiment_choices, key="dataset_sentiment_filter")

            # Show sample of results
            st.markdown("**Sample results:**")
            filtered_df = result_df[result_df["Predicted_Label"].isin(selected_sentiments)] if selected_sentiments else result_df
            st.dataframe(filtered_df.head(20))

            # Sentiment distribution (pie + bar)
            # Normalize label order to Positive, Neutral, Negative for consistent colors
            ordered_labels = ["Positive", "Neutral", "Negative"]
            sentiment_counts = filtered_df["Predicted_Label"].value_counts().reindex(ordered_labels, fill_value=0) if not filtered_df.empty else pd.Series([0,0,0], index=ordered_labels)
            if sentiment_counts.sum() > 0:
                render_pie_chart(sentiment_counts, colors=["#2ecc71", "#f1c40f", "#e74c3c"])
            else:
                st.info("No rows match the selected sentiment filters.")

            # Bar chart for counts
            st.markdown("**Counts (bar):**")
            st.bar_chart(sentiment_counts)

            # Show top positive / negative examples (if available)
            st.markdown("**Top examples:**")
            try:
                pos_examples = result_df[result_df["Predicted_Label"] == "Positive"][text_col].head(5)
                neg_examples = result_df[result_df["Predicted_Label"] == "Negative"][text_col].head(5)
                st.write("Top Positive examples:")
                for i, v in enumerate(pos_examples, 1):
                    st.write(f"{i}. {v}")
                st.write("Top Negative examples:")
                for i, v in enumerate(neg_examples, 1):
                    st.write(f"{i}. {v}")
            except Exception:
                pass

            # Download results (filtered view)
            try:
                download_df = filtered_df if not filtered_df.empty else result_df
                csv_bytes = download_df.to_csv(index=False).encode("utf-8")
                st.download_button("Download results as CSV", data=csv_bytes, file_name="sentiment_results.csv", mime="text/csv", key="download_results")
            except Exception as e:
                st.error(f"Could not prepare download: {e}")

# ----------- Mode 2: Social Media Analyzer (Twitter/YouTube) -----------
elif mode == "Analyze Social Media Link":
    link = st.text_input("Paste a Twitter or YouTube link:")
    if st.button("Fetch & Analyze"):
        comments = []

        # Twitter support
        if "twitter.com" in link:
            comments = fetch_twitter_replies(link, limit=100)

        # YouTube support
        elif "youtube.com" in link or "youtu.be" in link:
            comments = fetch_youtube_comments(link, limit=100)

        if comments:
            cleaned_comments = [clean_text(c) for c in comments]
            preds = pipe.predict(cleaned_comments)
            df = pd.DataFrame({"Comment": comments, "Predicted": preds})
            df["Sentiment"] = df["Predicted"].map(labels)
            # Sentiment filter
            sentiment_choices = ["Positive", "Neutral", "Negative"]
            selected_sentiments = st.multiselect("Filter comments by sentiment (text)", options=sentiment_choices, default=sentiment_choices, key="social_sentiment_filter")

            filtered_comments = df[df["Sentiment"].isin(selected_sentiments)] if selected_sentiments else df

            st.dataframe(filtered_comments.head(20))
            ordered_labels = ["Positive", "Neutral", "Negative"]
            counts = filtered_comments["Sentiment"].value_counts().reindex(ordered_labels, fill_value=0)
            if counts.sum() > 0:
                render_pie_chart(counts, colors=["#2ecc71", "#f1c40f", "#e74c3c"])
                st.markdown("**Counts (bar):**")
                st.bar_chart(counts)
            else:
                st.info("No comments match the selected sentiment filters.")

            # Download filtered comments
            try:
                csv_bytes = filtered_comments.to_csv(index=False).encode("utf-8")
                st.download_button("Download comments as CSV", data=csv_bytes, file_name="social_comments.csv", mime="text/csv", key="download_social")
            except Exception as e:
                st.error(f"Could not prepare download: {e}")
        else:
            st.error("No comments found or could not fetch comments.")

# (Screenshot / OCR mode removed for simplified hosting)

# ----------- Mode 4: Manual Text Input -----------
elif mode == "Manual Text Input":
    text = st.text_area("Enter text:")
    if st.button("Analyze Text"):
        if text.strip():
            cleaned = clean_text(text)
            pred = pipe.predict([cleaned])[0]
            st.success(f"Predicted Sentiment: {labels[pred]}")