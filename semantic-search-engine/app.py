"""Main Streamlit application - Production Ready with Enhanced Debugging."""

import streamlit as st
import logging
import time
import traceback
from typing import List, Dict
import numpy as np

# Import custom modules
from embeddings import EmbeddingGenerator
from vector_db import PineconeDB
from dataset import dataset_manager
from safety import input_sanitizer, content_filter
from config import Config

# Configure logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def show_deployment_info():
    """Show deployment information for debugging."""
    if st.sidebar.button("🔧 Show Config Info"):
        info = Config.get_deployment_info()
        st.sidebar.json(info)

def initialize_search_engine():
    """Initialize the semantic search engine components with comprehensive error handling and debugging."""
    try:
        if "search_initialized" not in st.session_state:
            
            # Validate configuration first
            if not Config.validate():
                st.error("❌ **Configuration Error**")
                st.error("Pinecone API key is missing or invalid.")
                
                # Show different instructions based on deployment type
                if hasattr(st, 'secrets'):
                    st.info("""
                    **For Streamlit Cloud:**
                    1. Go to your app settings
                    2. Click on "Secrets" 
                    3. Add: `PINECONE_API_KEY = "your-actual-api-key"`
                    4. Redeploy the app
                    """)
                else:
                    st.info("""
                    **For Local Development:**
                    1. Get your API key from [Pinecone Console](https://app.pinecone.io/)
                    2. Set environment variable: `export PINECONE_API_KEY="your-key"`
                    3. Or create `.env` file with: `PINECONE_API_KEY=your-key`
                    4. Restart the app
                    """)
                
                st.stop()
            
            with st.spinner("🚀 Initializing AI-powered semantic search engine..."):
                
                # Load dataset with detailed logging
                progress_bar = st.progress(0)
                st.write("📚 Loading document dataset...")
                
                try:
                    ids, documents, titles, metadata = dataset_manager.load_dataset()
                    print(f"[APP] Loaded {len(ids)} documents")
                    print(f"[APP] First 3 titles: {titles[:3]}")
                    
                    logger.info(f"Successfully loaded {len(documents)} documents")
                    st.write(f"✅ Loaded {len(documents)} documents successfully")
                    
                except Exception as e:
                    logger.error(f"Dataset loading failed: {e}")
                    st.error(f"Failed to load dataset: {str(e)}")
                    print(f"[APP] ERROR: Dataset loading failed: {e}")
                    print(traceback.format_exc())
                    st.stop()
                
                progress_bar.progress(20)
                
                # Initialize embedding generator with debugging
                st.write("🧠 Loading AI embedding model...")
                try:
                    embedder = EmbeddingGenerator()
                    st.write("✅ Embedding model loaded successfully")
                except Exception as e:
                    logger.error(f"Embedder initialization failed: {e}")
                    st.error(f"Failed to load embedding model: {str(e)}")
                    print(f"[APP] ERROR: Embedder initialization failed: {e}")
                    print(traceback.format_exc())
                    st.stop()
                
                progress_bar.progress(40)
                
                # Initialize vector database with debugging
                st.write("🗄️ Connecting to vector database...")
                try:
                    vector_db = PineconeDB()
                    st.write("✅ Connected to Pinecone successfully")
                except Exception as e:
                    logger.error(f"Vector DB initialization failed: {e}")
                    st.error(f"Failed to connect to Pinecone: {str(e)}")
                    print(f"[APP] ERROR: Vector DB initialization failed: {e}")
                    print(traceback.format_exc())
                    st.stop()
                
                progress_bar.progress(60)
                
                # Generate embeddings with comprehensive debugging
                st.write("⚡ Generating semantic embeddings...")
                try:
                    doc_embeddings = embedder.encode(documents, batch_size=32)
                    print(f"[APP] Generated {len(doc_embeddings)} embeddings, shape: {doc_embeddings.shape}")
                    
                    # Validate embedding dimensions
                    assert len(ids) == len(doc_embeddings), f"[APP] ERROR: IDs ({len(ids)}) and embeddings ({len(doc_embeddings)}) count mismatch!"
                    
                    st.write(f"✅ Generated {len(doc_embeddings)} embeddings")
                    
                except Exception as e:
                    logger.error(f"Embedding generation failed: {e}")
                    st.error(f"Failed to generate embeddings: {str(e)}")
                    print(f"[APP] ERROR: Embedding generation failed: {e}")
                    print(traceback.format_exc())
                    st.stop()
                
                progress_bar.progress(80)
                
                # Store in vector database with batch processing and detailed logging
                st.write("💾 Storing embeddings in cloud database...")
                try:
                    # Prepare metadata with content
                    metadata_with_content = []
                    for i, meta in enumerate(metadata):
                        meta_copy = meta.copy()
                        meta_copy['content'] = documents[i]
                        metadata_with_content.append(meta_copy)
                    
                    print(f"[APP] Prepared {len(metadata_with_content)} metadata entries")
                    
                    # Batch upload with detailed progress tracking
                    batch_size = 100
                    n_uploaded = 0
                    
                    for i in range(0, len(ids), batch_size):
                        batch_ids = ids[i:i+batch_size]
                        batch_embs = doc_embeddings[i:i+batch_size]
                        batch_meta = metadata_with_content[i:i+batch_size]
                        
                        batch_num = (i // batch_size) + 1
                        total_batches = (len(ids) - 1) // batch_size + 1
                        
                        print(f"[APP] Uploading batch {batch_num}/{total_batches}: {len(batch_ids)} vectors")
                        st.write(f"📤 Uploading batch {batch_num}/{total_batches}...")
                        
                        try:
                            success = vector_db.add_embeddings(
                                ids=batch_ids,
                                embeddings=batch_embs,
                                metadata_list=batch_meta
                            )
                            
                            if success:
                                n_uploaded += len(batch_ids)
                                print(f"[APP] Successfully uploaded batch {batch_num}: {len(batch_ids)} vectors")
                            else:
                                print(f"[APP] ERROR: Upload failed for batch {batch_num}")
                                st.error(f"❌ Failed to upload batch {batch_num}")
                                st.stop()
                                
                        except Exception as batch_error:
                            print(f"[APP] EXCEPTION: Exception during batch {batch_num}:")
                            print(traceback.format_exc())
                            st.error(f"❌ Exception during batch {batch_num}: {str(batch_error)}")
                            st.stop()
                    
                    print(f"[APP] Upload complete! Total vectors uploaded: {n_uploaded}")
                    st.write(f"✅ Successfully uploaded {n_uploaded} embeddings to Pinecone")
                    
                    if n_uploaded != len(ids):
                        st.warning(f"⚠️ Upload count mismatch: expected {len(ids)}, got {n_uploaded}")
                        
                except Exception as e:
                    logger.error(f"Vector storage failed: {e}")
                    st.error(f"Failed to store vectors: {str(e)}")
                    print(f"[APP] ERROR: Vector storage failed: {e}")
                    print(traceback.format_exc())
                    st.stop()
                
                progress_bar.progress(100)
                
                # Store in session state
                st.session_state.embedder = embedder
                st.session_state.vector_db = vector_db
                st.session_state.dataset_info = {
                    'total_docs': len(documents),
                    'categories': list(set(meta.get('category', 'General') for meta in metadata))
                }
                st.session_state.search_initialized = True
                
                st.success("✅ Search engine initialized successfully!")
                print(f"[APP] Initialization complete! Ready for search queries.")
                time.sleep(1)
                st.rerun()
    
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        st.error(f"❌ **Initialization Failed**")
        st.error(f"Error: {str(e)}")
        print(f"[APP] FATAL ERROR: Initialization failed: {str(e)}")
        print(traceback.format_exc())
        
        # Show helpful error messages
        error_str = str(e).lower()
        if "unauthorized" in error_str or "invalid api key" in error_str:
            st.error("🔑 **API Key Issue**: Your Pinecone API key is invalid or missing")
            if hasattr(st, 'secrets'):
                st.info("Check your Streamlit Cloud secrets configuration")
            else:
                st.info("Check your environment variables or .env file")
        elif "not found" in error_str:
            st.error("🌍 **Region Issue**: The specified region is not available")
            st.info("This is automatically handled - the app will try alternative regions")
        elif "network" in error_str or "connection" in error_str:
            st.error("🌐 **Network Issue**: Cannot connect to Pinecone")
            st.info("Check your internet connection and try again")
        elif "model" in error_str:
            st.error("🤖 **Model Issue**: Cannot load the AI model")
            st.info("This usually resolves on retry. Refresh the page.")
        elif "title" in error_str or "dataset" in error_str:
            st.error("📁 **Dataset Issue**: Problem with dataset structure")
            st.info("The dataset format might be incorrect. Check data.json file.")
        
        # Show deployment info for debugging
        with st.expander("🔧 Debug Information"):
            st.json(Config.get_deployment_info())
        
        st.stop()

def render_sidebar():
    """Render informative sidebar."""
    st.sidebar.title("ℹ️ About This App")
    
    st.sidebar.markdown(f"""
    **{Config.APP_TITLE}**
    
    {Config.APP_DESCRIPTION}
    
    ### 🎯 How It Works:
    1. **AI Understanding**: Uses sentence transformers to understand meaning
    2. **Vector Search**: Finds similar documents using neural embeddings  
    3. **Cloud Scale**: Powered by Pinecone vector database
    4. **Safe & Secure**: Input validation and content filtering
    
    ### 🛠️ Technology:
    - **Model**: {Config.EMBEDDING_MODEL.split('/')[-1]}
    - **Dimensions**: {Config.EMBEDDING_DIMENSION}D vectors
    - **Similarity**: {Config.SIMILARITY_METRIC} distance
    - **Database**: Pinecone Cloud
    """)
    
    # Dataset stats
    if "dataset_info" in st.session_state:
        info = st.session_state.dataset_info
        st.sidebar.markdown("### 📊 Dataset Info:")
        st.sidebar.metric("Documents", info['total_docs'])
        categories = info['categories'][:8]  # Show first 8 categories
        if len(info['categories']) > 8:
            categories.append("...")
        st.sidebar.write(f"**Categories**: {', '.join(categories)}")
    
    # Example queries - Enhanced for new dataset
    st.sidebar.markdown("### 🔍 Try These Queries:")
    examples = [
        "What is machine learning?",
        "How do neural networks work?", 
        "Explain vector databases",
        "Python programming guide",
        "Deep learning applications",
        "Best coffee makers",
        "Smart home devices",
        "Investment strategies",
        "Sustainable fashion brands",
        "Cybersecurity threats"
    ]
    
    for example in examples:
        if st.sidebar.button(f"🔍 '{example}'", key=f"ex_{hash(example)}", use_container_width=True):
            st.session_state.example_query = example
            st.rerun()
    
    # Debug info (only show in development)
    if not hasattr(st, 'secrets') or st.sidebar.checkbox("Show Debug Info"):
        show_deployment_info()

def perform_semantic_search(query: str) -> List[Dict]:
    """Perform semantic search with complete error handling."""
    try:
        embedder = st.session_state.embedder
        vector_db = st.session_state.vector_db
        
        with st.spinner("🔍 Searching for semantically similar content..."):
            # Generate query embedding
            query_embedding = embedder.encode_single(query)
            
            # Perform vector search
            results = vector_db.search(
                query_embedding=query_embedding,
                top_k=Config.DEFAULT_TOP_K,
                include_metadata=True
            )
        
        # Apply safety filters
        filtered_results = content_filter.filter_results(results)
        return filtered_results
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        st.error(f"Search failed: {str(e)}")
        print(f"[APP] ERROR: Search failed: {e}")
        return []

def display_search_results(results: List[Dict], query: str):
    """Display search results with enhanced formatting for comprehensive dataset."""
    if not results:
        st.info("🤔 No similar documents found. Try rephrasing your query or using different keywords.")
        st.markdown("**💡 Tips:**")
        st.markdown("- Use descriptive terms")
        st.markdown("- Try synonyms or related concepts") 
        st.markdown("- Check the example queries in the sidebar")
        return
    
    st.subheader(f"🎯 Results for: *\"{query}\"*")
    st.markdown(f"Found **{len(results)}** semantically similar documents:")
    
    for i, result in enumerate(results, 1):
        with st.container():
            col1, col2 = st.columns([4, 1])
            
            with col1:
                metadata = result.get('metadata', {})
                title = metadata.get('title', f"Document {result['id']}")
                content = metadata.get('content', 'No content available')
                category = metadata.get('category', 'General')
                
                # Result header
                st.markdown(f"### {i}. {title}")
                
                # Enhanced category icons for comprehensive dataset
                category_colors = {
                    'Programming': '💻',
                    'AI/ML': '🤖', 
                    'Database': '🗄️',
                    'Open Source': '🔓',
                    'Cloud Computing': '☁️',
                    'Machine Learning': '🧠',
                    'Cybersecurity': '🔐',
                    'Coffee Equipment': '☕',
                    'Smart Home': '🏠',
                    'Sustainable Fashion': '🌱',
                    'Gaming': '🎮',
                    'Tea': '🍵',
                    'History': '📜',
                    'Education': '🎓',
                    'Psychology': '🧠',
                    'Philosophy': '💭',
                    'Literature': '📚',
                    'Yoga': '🧘',
                    'Health Supplements': '💊',
                    'Mental Health': '💚',
                    'Recipes': '🍳',
                    'Meditation': '🧘‍♀️',
                    'Investment Strategies': '💰',
                    'Business Management': '📊',
                    'Cryptocurrency': '₿',
                    'Venture Capital': '💼',
                    'Marketing': '📈',
                    'Architecture': '🏛️',
                    'Cocktails': '🍸',
                    'Photography': '📸',
                    'Public Speaking': '🎤',
                    'Musical Instruments': '🎵',
                    'General': '📄'
                }
                icon = category_colors.get(category, '📄')
                st.markdown(f"{icon} **{category}**")
                
                # Content preview
                display_content = content[:300] + "..." if len(content) > 300 else content
                st.markdown(f"*{display_content}*")
                
                # Tags
                tags = metadata.get('tags', [])
                if tags:
                    tag_badges = " ".join([f"`{tag}`" for tag in tags[:5]])
                    st.markdown(f"**Tags:** {tag_badges}")
            
            with col2:
                # Similarity score with visual indicator
                score = result['score']
                score_percent = score * 100
                
                # Color coding
                if score >= 0.8:
                    color = "🟢"
                    quality = "Excellent"
                elif score >= 0.6:
                    color = "🟡" 
                    quality = "Good"
                else:
                    color = "🟠"
                    quality = "Fair"
                
                st.metric(
                    label="Relevance",
                    value=f"{score_percent:.1f}%",
                    help=f"{quality} semantic similarity"
                )
                
                st.markdown(f"{color} **{score:.3f}**")
        
        st.divider()

def main():
    """Main application function."""
    # Page configuration
    st.set_page_config(
        page_title=Config.APP_TITLE,
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for professional appearance
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .search-box {
        background: #f8fafc;
        padding: 2rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        margin-bottom: 2rem;
    }
    
    .stAlert > div {
        padding: 1rem;
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown(f"""
    <div class="main-header">
        <h1>🔍 {Config.APP_TITLE}</h1>
        <p style="font-size: 1.2rem; margin: 0;">{Config.APP_DESCRIPTION}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize application
    if "search_initialized" not in st.session_state:
        initialize_search_engine()
        return
    
    # Render sidebar
    render_sidebar()
    
    # Search interface
    st.markdown('<div class="search-box">', unsafe_allow_html=True)
    st.markdown("### 🎯 Enter your search query:")
    
    # Handle example query from sidebar
    default_query = ""
    if "example_query" in st.session_state:
        default_query = st.session_state.example_query
        del st.session_state.example_query
    
    # Search input
    query = st.text_input(
        "Ask anything about the documents...",
        value=default_query,
        placeholder="e.g., 'smart home devices' or 'investment strategies'",
        help="Enter a natural language question or topic. The AI will find relevant documents based on meaning, not just keywords.",
        label_visibility="collapsed"
    )
    
    # Search and clear buttons
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        search_clicked = st.button("🔍 **Search**", type="primary", use_container_width=True)
    with col2:
        clear_clicked = st.button("🗑️ Clear", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Handle clear
    if clear_clicked:
        if "search_results" in st.session_state:
            del st.session_state.search_results
        if "last_query" in st.session_state:
            del st.session_state.last_query
        st.rerun()
    
    # Handle search
    if search_clicked and query:
        # Validate and sanitize input
        clean_query, errors = input_sanitizer.sanitize_query(query)
        
        if errors:
            st.error(f"❌ Input validation failed: {'; '.join(errors)}")
        else:
            # Perform search
            results = perform_semantic_search(clean_query)
            st.session_state.search_results = results
            st.session_state.last_query = clean_query
    
    # Display results
    if "search_results" in st.session_state and st.session_state.search_results:
        display_search_results(
            st.session_state.search_results,
            st.session_state.get("last_query", query)
        )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #64748b; padding: 1rem;">
        <p>🚀 <strong>Deployed on Streamlit Cloud</strong></p>
        <p>Powered by Sentence Transformers • Pinecone Vector Database • GitHub</p>
        <p style="font-size: 0.9rem;">Production-ready AI semantic search with comprehensive debugging</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()