from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
import wikipedia
import re
import feedparser
import spacy
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)
CORS(app)

# Load NLP and ML models
nli = pipeline("text-classification", model="roberta-large-mnli", top_k=None)
nlp = spacy.load("en_core_web_sm")
sbert = SentenceTransformer('all-MiniLM-L6-v2')

# --- Helper: Extract longer, meaningful claims ---
def extract_claims(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    # Ensure that the extracted claim contains enough information and context
    return [s.strip() for s in sentences if len(s.split()) > 5 and not s.lower().startswith('according')]

# --- Helper: Extract keywords for querying ---
def extract_keywords(text):
    doc = nlp(text)
    # Now include more general nouns along with entities
    entities = [ent.text for ent in doc.ents if ent.label_ in {"ORG", "GPE", "PRODUCT", "LAW"}]
    if not entities:
        # Extract general nouns and important keywords as fallback
        tokens = re.findall(r'\b\w+\b', text)
        stopwords = {'is', 'was', 'the', 'in', 'and', 'at', 'of', 'from', 'to', 'on'}
        entities = [word for word in tokens if word.lower() not in stopwords and len(word) > 2]
    return ' '.join(entities[:6])  # Limit to first 6 keywords

def is_relevant(claim, summary):
    try:
        # Lowering similarity threshold to 0.4 for more flexible matching
        sim = util.cos_sim(
            sbert.encode(claim, convert_to_tensor=True),
            sbert.encode(summary, convert_to_tensor=True)
        )
        return sim.item() > 0.4  # Lower threshold
    except Exception as e:
        print(f"Error in relevance check: {e}")
        return False

def fetch_wikipedia_evidence(claim):
    try:
        # First try to extract a PERSON name
        doc = nlp(claim)
        person_names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]

        # Try to fetch exact Wikipedia page for the person
        if person_names:
            for name in person_names:
                try:
                    page = wikipedia.page(name, auto_suggest=False)
                    summary = page.summary[:800]
                    if is_relevant(claim, summary):
                        return summary, name
                except Exception:
                    continue  # If the page doesn't exist, skip

        # Fall back to general keyword search
        keywords = extract_keywords(claim)
        search_results = wikipedia.search(keywords)
        if not search_results:
            return "", ""

        for title in search_results:
            try:
                page = wikipedia.page(title, auto_suggest=False)
                summary = page.summary[:800]
                if is_relevant(claim, summary):
                    return summary, title
            except Exception:
                continue
        return "", ""
    except Exception as e:
        print(f"Wikipedia error: {e}")
        return "", ""

# --- Google News evidence fetch ---
def fetch_google_news(claim):
    try:
        keywords = extract_keywords(claim)
        query = '+'.join(keywords.split())
        url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
        feed = feedparser.parse(url)

        for entry in feed.entries[:10]:  # Increased number of entries checked
            title = entry.title
            summary = entry.summary if hasattr(entry, 'summary') else title
            content = f"{title}. {summary}"
            if is_relevant(claim, content):
                return content
        return ""
    except Exception as e:
        print(f"Google News error: {e}")
        return ""

# --- Main route to verify claims ---
@app.route("/verify", methods=["POST"])
def verify():
    data = request.json
    text = data.get("text", "")
    claims = extract_claims(text)
    results = []

    for claim in claims:
        evidence, title = fetch_wikipedia_evidence(claim)
        source = f"Wikipedia - {title}" if evidence and title else ""

        if not evidence:
            evidence = fetch_google_news(claim)
            if evidence:
                source = "Google News"

        if not evidence:
            label = "Not Enough Evidence"
            score = 0
        else:
            try:
                result = nli({"text": evidence, "text_pair": claim})
                result = result[0] if isinstance(result, list) and isinstance(result[0], list) else result
                sorted_result = sorted(result, key=lambda x: x['score'], reverse=True)
                top = sorted_result[0]

                raw_label = top['label'].upper()
                score = round(top['score'], 2)

                if raw_label == "ENTAILMENT":
                    label = "True"
                elif raw_label == "CONTRADICTION":
                    label = "False"
                else:  # NEUTRAL or others
                    label = "True" if score >= 0.75 else "False"

                # Low confidence label based on score and confidence
                if 0.5 <= score < 0.8 and label in ["True", "False"]:
                    label = "Low Confidence – Further Verification Recommended"

            except Exception as e:
                print(f"NLI model error: {e}")
                label = "Error Evaluating Claim"
                score = 0

        results.append({
            "claim": claim,
            "label": label,
            "score": score,
            "evidence": evidence,
            "source": source
        })

    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)
