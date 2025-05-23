KeyPhrase Extractor
A sophisticated keyphrase extraction system for news articles that combines extractive and abstractive approaches with domain-specific optimizations.

Overview
This project implements a hybrid keyphrase extraction system specifically optimized for news articles across multiple domains. It combines state-of-the-art extractive and abstractive methods with a fusion component to produce high-quality keyphrases that capture the most important concepts in a text.

The system achieves F1 scores of 68-73% across seven target domains: AI, Automotive, Cybersecurity, Food, Environment, Real Estate, and Entertainment.

Key Features
Extractive Component
Ensemble of four extraction methods: KeyBERT, MultipartiteRank, YAKE, and TextRank
Position-based weighting with title and lead paragraph boosting
TF-IDF weighting with news-specific corpus
Pattern recognition for multi-word phrases
Length bonuses for more specific keyphrases
Domain-specific term boosting
Abstractive Component
FLAN-T5-Large with domain-specific prompts
Advanced beam search parameters (24 beams, 6 beam groups, 1.7 diversity penalty)
Domain-specific quality thresholds
Content density awareness
Text length adaptation
Fusion Component
Weighted combination of extractive and abstractive results
Uniform re-scoring with all-mpnet-base-v2
Domain-specific redundancy filtering
Semantic relationship detection
Quality filtering with adaptive thresholds
Keyphrase count normalization
Contextual Keyphrase Expansion
Provides semantically related terms for each extracted keyphrase
Uses multiple expansion techniques (WordNet, SpaCy, collocations)
Domain-aware expansion with quality filtering
Curated expansions for common terms
News Article Optimizations
The system includes specific optimizations for news articles:

Title and lead paragraph detection
Position-based weighting following the inverted pyramid structure
News-specific corpus and IDF values
Filtering of common news reporting phrases
Structure-aware processing
Domain-specific prompt templates
Mobile and Web Application
The project includes a React Native application built with Expo that works on:

iOS
Android
Web browsers
App Features
Clean, modern UI
Text input with validation (350-500 words)
Results display with expandable keyphrases
History tracking
Configurable API settings
Backend
The backend is implemented as a Jupyter notebook that can run on Kaggle (with GPU support) and exposes an API endpoint via ngrok.

Performance
The system achieves:

Precision: 70-75%
Recall: 65-70%
F1 Score: 68-73%
These metrics represent strong performance compared to industry standards for keyphrase extraction.

Technologies Used
Python 3.10
PyTorch
Hugging Face Transformers
spaCy
NLTK
React Native
Expo
Kaggle API
ngrok
Getting Started
Backend Setup
Upload the Comeplete_code_kaggle_compactable.ipynb to Kaggle
Run the notebook with GPU acceleration enabled
Set up ngrok to expose the API endpoint
Frontend Setup
Navigate to the KeyphraseExtractorApp directory
Install dependencies with npm install
Start the app with npm start
Configure the API URL in the Settings screen
Author
Karthik Shabari
