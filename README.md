##✨ Overview
This project combines extractive and abstractive approaches to extract high-quality keyphrases from news articles across multiple domains. The system is optimized for seven target domains: AI, Automotive, Cybersecurity, Food, Environment, Real Estate, and Entertainment.

##Keyphrase Extraction Demo

##🚀 Features
#🧠 Smart Extraction Engine
Hybrid Approach: Combines the best of extractive and abstractive methods
Domain Awareness: Optimized for specific domains with custom parameters
News Article Focus: Special handling for title, lead paragraph, and news structure
High Performance: Achieves F1 scores of 68-73% across all domains
#📱 Cross-Platform App
Works Everywhere: iOS, Android, and Web from a single codebase
Modern UI: Clean, intuitive interface with smooth animations
History Tracking: Save and revisit previous extractions
Customizable: Configure API settings to your needs
#🔍 Contextual Expansion
Related Terms: Discover semantically related concepts
Domain-Specific: Expansions tailored to each domain
Quality Filtered: Only high-quality, relevant suggestions
#📊 Performance
Metric	Score
Precision	70-75%
Recall	65-70%
F1 Score	68-73%
#🛠️ Technology Stack
Backend
Python with PyTorch
FLAN-T5-Large for abstractive extraction
KeyBERT, YAKE, TextRank for extractive methods
all-mpnet-base-v2 for semantic similarity
Kaggle for GPU acceleration
ngrok for API exposure
Frontend
React Native with Expo
Paper UI components
AsyncStorage for local data
Axios for API communication

##📝 How It Works
Input: Enter a news article (350-500 words)
Processing: The system analyzes the text using multiple AI models
Extraction: Key concepts are identified and scored
Expansion: Related terms are suggested for each keyphrase
Results: View and explore the extracted keyphrases

##👨‍💻 Author
Karthik Shabari
