# AI-Powered Semantic Search Engine

**Production-Ready Semantic Search with Vector Embeddings**

[![Deploy to Streamlit Cloud](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy)

A complete semantic search engine implementation following DeepLearning.AI best practices for "Quality and Safety for LLM Applications," "Vector Databases: from Embeddings to Applications," and "Building Applications with Vector Databases."

## 🚀 Quick Deploy to Streamlit Cloud

### 1. Fork this Repository
Click the "Fork" button at the top of this GitHub page.

### 2. Get Your Pinecone API Key
1. Sign up for free at [Pinecone.io](https://app.pinecone.io/)
2. Create a new project
3. Copy your API key from the console

### 3. Deploy to Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io/)
2. Click "New app"
3. Connect your GitHub account
4. Select your forked repository
5. Set main file path: `app.py`
6. Click "Advanced settings" → "Secrets"
7. Add your secrets:
   ```toml
   PINECONE_API_KEY = "your-actual-pinecone-api-key"
   PINECONE_ENVIRONMENT = "us-west1-gcp"  
   PINECONE_INDEX_NAME = "semantic-search"
   ```
8. Click "Deploy"

Your app will be live in 2-3 minutes! 🎉

## 🏗️ Local Development

### Prerequisites
- Python 3.8+
- Git

### Setup
```bash
# Clone your forked repository
git clone https://github.com/YOUR_USERNAME/semantic-search-engine.git
cd semantic-search-engine

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables for local development
export PINECONE_API_KEY="your-api-key"
# Or create .env file (see .env.example)

# Run locally
streamlit run app.py
```

## 🎯 Features

- **🧠 AI-Powered Search**: Uses sentence transformers for semantic understanding
- **⚡ Real-time Results**: Instant search with confidence scores
- **🛡️ Production Safe**: Input validation, error handling, content filtering
- **☁️ Cloud Scale**: Powered by Pinecone vector database
- **📱 Responsive UI**: Professional Streamlit interface
- **🔒 Secure**: Proper secrets management for deployment

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **ML Model**: Sentence Transformers (all-MiniLM-L6-v2)
- **Vector Database**: Pinecone
- **Language**: Python 3.8+
- **Deployment**: Streamlit Cloud

## 📊 Sample Queries

Try these example searches:
- "What is machine learning?"
- "How do neural networks work?"
- "Explain vector databases" 
- "Python programming basics"
- "Deep learning applications"

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit UI  │────│  Sentence Trans- │────│  Pinecone Vec-  │
│   (Frontend)    │    │  formers Model   │    │  tor Database   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
        │                        │                        │
        ├── Query Input          ├── Embeddings           ├── Similarity Search
        ├── Results Display      ├── 384D Vectors         ├── Cosine Distance
        └── Safety Validation    └── Normalization        └── Top-K Results
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `PINECONE_API_KEY` | Your Pinecone API key (required) | - |
| `PINECONE_ENVIRONMENT` | Pinecone environment | `us-west1-gcp` |
| `PINECONE_INDEX_NAME` | Index name | `semantic-search` |
| `LOG_LEVEL` | Logging level | `INFO` |

### For Streamlit Cloud
Add secrets in your app's settings dashboard:
```toml
PINECONE_API_KEY = "your-actual-api-key"
```

### For Local Development  
Create `.env` file:
```env
PINECONE_API_KEY=your-actual-api-key
```

## 🔒 Security & Safety

- ✅ Input validation and sanitization
- ✅ Content filtering for inappropriate queries
- ✅ Error handling with user-friendly messages  
- ✅ API key protection (never in code)
- ✅ Rate limiting ready architecture

## 📈 Performance & Scaling

- **Search Speed**: Sub-second semantic search
- **Accuracy**: High-quality sentence transformer embeddings
- **Scalability**: Cloud vector database handles millions of documents
- **Monitoring**: Comprehensive logging for production insights

## 🧪 Testing

The app includes built-in test queries and error handling:
- Try empty queries (validation test)
- Try very long queries (length limit test)  
- Try different semantic variations (AI understanding test)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

### Common Issues

**"Invalid API Key" Error**
- Ensure your Pinecone API key is correctly set in Streamlit Cloud secrets
- Verify the key is active in your Pinecone console

**"Model Loading Failed"**
- Usually resolves automatically on retry
- Check internet connection for model download

**"No Search Results"**
- Verify embeddings uploaded successfully  
- Try different query phrasing

### Getting Help
- Check the logs in your Streamlit Cloud app
- Review Pinecone console for index status
- Open an issue with full error details

---

**🚀 Built with DeepLearning.AI best practices for production AI applications**

[![Powered by Streamlit](https://img.shields.io/badge/Powered%20by-Streamlit-FF6B6B?style=flat-square)](https://streamlit.io/)
[![Powered by Pinecone](https://img.shields.io/badge/Powered%20by-Pinecone-00D4AA?style=flat-square)](https://pinecone.io/)
[![Powered by Hugging Face](https://img.shields.io/badge/Powered%20by-🤗%20Hugging%20Face-FFD700?style=flat-square)](https://huggingface.co/)