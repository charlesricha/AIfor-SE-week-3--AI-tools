# task3-spaCy.py

import spacy

# Load English spaCy model
nlp = spacy.load("en_core_web_sm")

# 🔤 Ask the user for a sentence or review
review = input("📝 Enter your product review: ")

# Run spaCy NLP pipeline
doc = nlp(review)

# 🧾 Named Entity Recognition (NER)
print("\n📦 Named Entities (Brands, Products, Orgs):")
for ent in doc.ents:
    if ent.label_ in ["ORG", "PRODUCT"]:
        print(f" - {ent.text} ({ent.label_})")

# 📊 Rule-Based Sentiment Analysis
positive_words = ["amazing", "super", "impressed", "solid", "great", "love", "awesome", "excellent"]
negative_words = ["bad", "terrible", "worst", "high", "poor", "disappointed", "hate", "expensive"]

# Lowercase tokens
tokens = [token.text.lower() for token in doc]

# Count positive and negative words
pos_count = sum(word in tokens for word in positive_words)
neg_count = sum(word in tokens for word in negative_words)

# 🧠 Sentiment Result
print("\n🧠 Sentiment Analysis:")
print(f" - Positive words found: {pos_count}")
print(f" - Negative words found: {neg_count}")

if pos_count > neg_count:
    print(" → Overall Sentiment: 👍 Positive")
elif neg_count > pos_count:
    print(" → Overall Sentiment: 👎 Negative")
else:
    print(" → Overall Sentiment: 😐 Neutral")
