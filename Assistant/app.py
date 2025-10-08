"""Main Streamlit application for LMS AI Assistant System."""

import streamlit as st
import logging
import json
import time
from typing import Dict, List, Any
import numpy as np

# Import our modules
from config import Config
from vector_db import VectorDBManager
from embeddings import EmbeddingGenerator
from recommendation import RecommendationEngine
from faq_system import FAQSystem

import os
import json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # This is /mount/src/preetham/Assistant if code is inside that folder

# Since Streamlit runs in /mount/src/preetham, files are in 'Assistant' subfolder
# So need to prepend 'Assistant' if code runs from parent directory or deal with different runtime cwd

# Proposed:
#DATA_DIR = os.path.join(BASE_DIR)  # If code runs from Assistant folder, keep as is
# or
DATA_DIR = os.path.join(BASE_DIR, "Assistant")  # If code runs from parent folder, add Assistant

# Since BASE_DIR is already Assistant, your files are directly under BASE_DIR:
courses_path = os.path.join(DATA_DIR, "courses.json")
users_path = os.path.join(DATA_DIR, "users.json")
faq_path = os.path.join(DATA_DIR, "faq.json")

def load_sample_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

courses_data = load_sample_data(courses_path)
users_data = load_sample_data(users_path)
faq_data = load_sample_data(faq_path)


### Configure logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL, "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="LMS AI Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: #f4f4f4;
        padding: 2rem;
        border-radius: 10px;
        color: #222;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .feature-card, .recommendation-card {
        background: #fff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #888;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.07);
        color: #222;
    }
    .chat-message {
        background: #f6f6f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #111;
        color: #222;
    }
    .metric-card {
        background: #f8f8f8;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #bbb;
        color: #222;
    }
    .debug-info {
        background: #f6f6f6;
        border: 1px solid #ccc;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
        font-family: monospace;
        font-size: 0.9rem;
        color: #111;
    }
    .st-cg {background: #fff;}
    /* Remove all colored buttons and make them grayscale */
    .stButton > button {
        background: #e0e0e0;
        color: #1a1a1a;
        border: 1px solid #888;
        border-radius: 5px;
    }
    .stButton > button:hover {
        background: #222;
        color: #fff;
    }
    /* Remove Streamlit colored header underline for tabs */
    .stTabs [data-baseweb="tab-list"] button {
        border-bottom: 2px solid #222 !important;
        color: #222 !important;
        background: #fff !important;
    }
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        border-bottom: 4px solid #222 !important;
        color: #111 !important;
        background: #f4f4f4 !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_system():
    """Initialize the LMS AI system components."""
    try:
        with st.spinner("🚀 Initializing LMS AI Assistant System..."):
            progress_bar = st.progress(0)
            
            # Validate configuration
            st.write("🔧 Validating configuration...")
            if not Config.validate():
                st.error("❌ Configuration validation failed. Please check your Pinecone API key.")
                st.stop()
            progress_bar.progress(10)
            
            # Initialize vector database
            st.write("🗄️ Connecting to vector database...")
            vector_db = VectorDBManager()
            progress_bar.progress(30)
            
            # Initialize embedding generator
            st.write("🧠 Loading AI models...")
            embedding_gen = EmbeddingGenerator()
            progress_bar.progress(50)
            
            # Load sample data
            st.write("📚 Loading sample data...")
            courses_data = load_sample_data('courses.json')
            users_data = load_sample_data('users.json')
            faq_data = load_sample_data('faq.json')
            progress_bar.progress(60)
            
            # Generate embeddings and populate vector database
            st.write("⚡ Generating embeddings...")
            
            # Content embeddings
            content_embeddings, content_items = embedding_gen.generate_content_embeddings(courses_data)
            vector_db.upsert_content_embeddings(content_items, content_embeddings)
            progress_bar.progress(70)
            
            # User embeddings
            user_embeddings, user_items = embedding_gen.generate_user_embeddings(users_data)
            vector_db.upsert_user_embeddings(user_items, user_embeddings)
            progress_bar.progress(80)
            
            # FAQ embeddings
            faq_embeddings, faq_items = embedding_gen.generate_faq_embeddings(faq_data)
            vector_db.upsert_faq_embeddings(faq_items, faq_embeddings)
            progress_bar.progress(90)
            
            # Initialize recommendation engine and FAQ system
            st.write("🎯 Initializing intelligent systems...")
            recommendation_engine = RecommendationEngine(vector_db, embedding_gen)
            faq_system = FAQSystem(vector_db, embedding_gen)
            progress_bar.progress(100)
            
            st.success("✅ LMS AI Assistant initialized successfully!")
            time.sleep(1)
            
            return {
                'vector_db': vector_db,
                'embedding_gen': embedding_gen,
                'recommendation_engine': recommendation_engine,
                'faq_system': faq_system,
                'sample_data': {
                    'courses': courses_data,
                    'users': users_data,
                    'faq': faq_data
                }
            }
            
    except Exception as e:
        st.error(f"❌ System initialization failed: {str(e)}")
        logger.error(f"System initialization failed: {e}")
        st.stop()

@st.cache_data
def load_sample_data(filename: str) -> List[Dict]:
    """Load sample data from JSON files."""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load {filename}: {e}")
        return []

def render_header():
    """Render the main header."""
    st.markdown("""
    <div class="main-header">
        <h1>🎓 LMS AI Assistant System</h1>
        <p style="font-size: 1.2rem; margin: 0;">Semantic Search • Personalized Recommendations • Intelligent FAQ</p>
    </div>
    """, unsafe_allow_html=True)

def render_sidebar(system_components):
    """Render the sidebar with user selection and system info."""
    st.sidebar.title("🔧 System Controls")
    
    # User selection for personalization
    users_data = system_components['sample_data']['users']
    user_options = {user['username']: user['user_id'] for user in users_data}
    selected_username = st.sidebar.selectbox(
        "👤 Select User Profile",
        options=list(user_options.keys()),
        help="Choose a user profile for personalized recommendations"
    )
    selected_user_id = user_options[selected_username]
    
    # Display user info
    selected_user = next(u for u in users_data if u['user_id'] == selected_user_id)
    profile = selected_user.get('profile', {})
    
    st.sidebar.markdown("### 📊 User Profile")
    st.sidebar.markdown(f"**Name:** {selected_user.get('full_name', 'N/A')}")
    st.sidebar.markdown(f"**Major:** {profile.get('major', 'N/A')}")
    st.sidebar.markdown(f"**Level:** {profile.get('preferred_difficulty', 'N/A')}")
    
    interests = profile.get('interests', [])
    if interests:
        st.sidebar.markdown(f"**Interests:** {', '.join(interests[:3])}")
    
    # System health
    st.sidebar.markdown("### 🏥 System Health")
    vector_db = system_components['vector_db']
    health = vector_db.health_check()
    
    for index_type, is_healthy in health.items():
        status = "🟢" if is_healthy else "🔴"
        st.sidebar.markdown(f"{status} {index_type.title()} Index")
    
    # Quick stats
    content_stats = vector_db.get_index_stats('content')
    if content_stats:
        st.sidebar.metric("📚 Content Items", content_stats.get('total_vectors', 0))
    
    return selected_user_id, selected_user

def semantic_search_tab(system_components, user_id):
    """Render the semantic search interface."""
    st.header("🔍 Semantic Search")
    st.markdown("Search for learning content using natural language queries.")
    
    # Search interface
    col1, col2 = st.columns([3, 1])
    with col1:
        query = st.text_input(
            "Search Query",
            placeholder="e.g., 'machine learning algorithms for beginners'",
            help="Enter your search query in natural language"
        )
    
    with col2:
        search_button = st.button("🔍 Search", type="primary")
    
    # Filters
    with st.expander("🔧 Search Filters"):
        col1, col2, col3 = st.columns(3)
        with col1:
            subject_filter = st.selectbox("Subject Area", ["All"] + Config.SUBJECT_AREAS)
        with col2:
            difficulty_filter = st.selectbox("Difficulty Level", ["All"] + Config.DIFFICULTY_LEVELS)
        with col3:
            content_type_filter = st.selectbox("Content Type", ["All"] + Config.CONTENT_TYPES)
    
    if search_button and query:
        with st.spinner("🔍 Searching..."):
            try:
                # Generate query embedding
                embedding_gen = system_components['embedding_gen']
                vector_db = system_components['vector_db']
                
                query_embedding = embedding_gen.generate_query_embedding(query)
                
                # Build filters
                filters = {}
                if subject_filter != "All":
                    filters['subject_area'] = subject_filter
                if difficulty_filter != "All":
                    filters['difficulty_level'] = difficulty_filter
                if content_type_filter != "All":
                    filters['content_type'] = content_type_filter
                
                # Perform search
                results = vector_db.search_content(
                    query_embedding=query_embedding,
                    top_k=8,
                    filters=filters if filters else None
                )
                
                if results:
                    st.success(f"✅ Found {len(results)} relevant results")
                    
                    for i, result in enumerate(results, 1):
                        metadata = result.get('metadata', {})
                        score = result.get('score', 0)
                        
                        with st.container():
                            col1, col2 = st.columns([4, 1])
                            
                            with col1:
                                st.markdown(f"### {i}. {metadata.get('title', 'Untitled')}")
                                st.markdown(f"**Course:** {metadata.get('course_id', 'N/A')}")
                                st.markdown(f"**Type:** {metadata.get('content_type', 'N/A')} | **Subject:** {metadata.get('subject_area', 'N/A')} | **Level:** {metadata.get('difficulty_level', 'N/A')}")
                                
                                # Show content preview if available
                                content_preview = metadata.get('content', '')
                                if content_preview:
                                    preview = content_preview[:200] + "..." if len(content_preview) > 200 else content_preview
                                    st.markdown(f"*{preview}*")
                            
                            with col2:
                                # Relevance score
                                score_percent = score * 100
                                st.metric("Relevance", f"{score_percent:.1f}%")
                                
                                if score >= 0.8:
                                    st.success("🎯 Excellent match")
                                elif score >= 0.6:
                                    st.info("✅ Good match")
                                else:
                                    st.warning("⚠️ Partial match")
                        
                        st.divider()
                else:
                    st.info("🤔 No results found. Try different keywords or adjust filters.")
                    
            except Exception as e:
                st.error(f"Search failed: {str(e)}")
                logger.error(f"Search error: {e}")

def recommendations_tab(system_components, user_id, user_data):
    """Render the recommendations interface."""
    st.header("🎯 Personalized Recommendations")
    st.markdown("Get content recommendations tailored to your learning profile.")
    
    recommendation_engine = system_components['recommendation_engine']
    
    # Recommendation types
    tab1, tab2, tab3, tab4 = st.tabs([
        "📚 Content for You", 
        "🔥 Trending", 
        "📈 Learning Path", 
        "👥 Similar Users"
    ])
    
    with tab1:
        st.subheader("Recommended Content")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            num_recs = st.slider("Number of recommendations", 3, 10, 5)
        with col2:
            if st.button("🔄 Refresh Recommendations"):
                st.rerun()
        
        try:
            recommendations = recommendation_engine.get_content_recommendations(
                user_id=user_id,
                num_recommendations=num_recs
            )
            
            if recommendations:
                for i, rec in enumerate(recommendations, 1):
                    metadata = rec.get('metadata', {})
                    reason = rec.get('recommendation_reason', 'Recommended for you')
                    
                    st.markdown(f"""
                    <div class="recommendation-card">
                        <h4>{i}. {metadata.get('title', 'Untitled')}</h4>
                        <p><strong>Subject:</strong> {metadata.get('subject_area', 'N/A')} | 
                        <strong>Level:</strong> {metadata.get('difficulty_level', 'N/A')} | 
                        <strong>Type:</strong> {metadata.get('content_type', 'N/A')}</p>
                        <p><em>💡 {reason}</em></p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No personalized recommendations available at this time.")
                
        except Exception as e:
            st.error(f"Failed to get recommendations: {str(e)}")
    
    with tab2:
        st.subheader("Trending Content")
        try:
            trending = recommendation_engine.get_trending_content(num_items=8)
            
            if trending:
                for i, item in enumerate(trending, 1):
                    metadata = item.get('metadata', {})
                    st.markdown(f"""
                    <div class="feature-card">
                        <h4>{i}. {metadata.get('title', 'Untitled')}</h4>
                        <p><strong>Subject:</strong> {metadata.get('subject_area', 'N/A')} | 
                        <strong>Level:</strong> {metadata.get('difficulty_level', 'N/A')}</p>
                        <p>🔥 Popular on the platform</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No trending content available.")
                
        except Exception as e:
            st.error(f"Failed to get trending content: {str(e)}")
    
    with tab3:
        st.subheader("Your Learning Path")
        try:
            learning_path = recommendation_engine.get_learning_path_recommendations(user_id)
            
            if learning_path:
                st.info("📈 Recommended next steps in your learning journey:")
                for i, item in enumerate(learning_path, 1):
                    metadata = item.get('metadata', {})
                    reason = item.get('recommendation_reason', 'Next step in your learning')
                    
                    st.markdown(f"""
                    <div class="recommendation-card">
                        <h4>{i}. {metadata.get('title', 'Untitled')}</h4>
                        <p><strong>Subject:</strong> {metadata.get('subject_area', 'N/A')} | 
                        <strong>Level:</strong> {metadata.get('difficulty_level', 'N/A')}</p>
                        <p><em>📈 {reason}</em></p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("Learning path recommendations will be available as you progress.")
                
        except Exception as e:
            st.error(f"Failed to get learning path: {str(e)}")
    
    with tab4:
        st.subheader("Connect with Similar Learners")
        try:
            similar_users = recommendation_engine.get_user_recommendations(user_id, 5)
            
            if similar_users:
                for user in similar_users:
                    metadata = user.get('metadata', {})
                    st.markdown(f"""
                    <div class="feature-card">
                        <h4>👤 {metadata.get('username', 'Anonymous')}</h4>
                        <p><strong>Major:</strong> {metadata.get('major', 'N/A')} | 
                        <strong>Interests:</strong> {', '.join(metadata.get('interests', [])[:3])}</p>
                        <p>Similar learning interests and goals</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No similar users found at this time.")
                
        except Exception as e:
            st.error(f"Failed to get similar users: {str(e)}")

def faq_tab(system_components):
    """Render the FAQ and chatbot interface."""
    st.header("💬 AI FAQ Assistant")
    st.markdown("Get instant answers to your questions using our intelligent FAQ system.")
    
    faq_system = system_components['faq_system']
    
    # Tabs for different FAQ interfaces
    tab1, tab2, tab3 = st.tabs(["💬 Ask a Question", "📋 Browse FAQs", "🔥 Popular Questions"])
    
    with tab1:
        st.subheader("Ask the AI Assistant")
        
        # Initialize chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Chat interface
        with st.form("chat_form"):
            user_question = st.text_input(
                "Your Question:",
                placeholder="e.g., How do I reset my password?",
                help="Ask any question about the platform"
            )
            submitted = st.form_submit_button("💬 Ask")
        
        if submitted and user_question:
            with st.spinner("🤔 Thinking..."):
                try:
                    response = faq_system.chatbot_response(
                        user_question, 
                        st.session_state.chat_history
                    )
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        'user': user_question,
                        'assistant': response['response'],
                        'type': response['type']
                    })
                    
                    # Display response
                    st.markdown(f"""
                    <div class="chat-message">
                        <strong>🤖 Assistant:</strong><br/>
                        {response['response']}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show suggestions if available
                    suggestions = response.get('suggestions', [])
                    if suggestions:
                        st.markdown("**💡 Related questions:**")
                        for suggestion in suggestions:
                            if st.button(suggestion, key=f"suggestion_{hash(suggestion)}"):
                                st.rerun()
                
                except Exception as e:
                    st.error(f"Failed to get answer: {str(e)}")
        
        # Display chat history
        if st.session_state.chat_history:
            st.subheader("💭 Conversation History")
            for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):
                st.markdown(f"**👤 You:** {chat['user']}")
                st.markdown(f"**🤖 Assistant:** {chat['assistant']}")
                st.divider()
            
            if st.button("🗑️ Clear History"):
                st.session_state.chat_history = []
                st.rerun()
    
    with tab2:
        st.subheader("Browse FAQ Categories")
        
        categories = faq_system.get_categories()
        selected_category = st.selectbox("Choose a category:", ["All"] + categories)
        
        if selected_category != "All":
            try:
                category_faqs = faq_system.get_category_faqs(selected_category)
                
                if category_faqs:
                    for faq in category_faqs:
                        with st.expander(f"❓ {faq['question']}"):
                            st.markdown(faq['answer'])
                            
                            # Feedback buttons
                            col1, col2, col3 = st.columns([1, 1, 3])
                            with col1:
                                if st.button("👍 Helpful", key=f"helpful_{faq['faq_id']}"):
                                    faq_system.record_feedback(faq['faq_id'], faq['question'], True)
                                    st.success("Thanks for your feedback!")
                            with col2:
                                if st.button("👎 Not Helpful", key=f"not_helpful_{faq['faq_id']}"):
                                    faq_system.record_feedback(faq['faq_id'], faq['question'], False)
                                    st.info("Thanks! We'll improve this answer.")
                else:
                    st.info(f"No FAQs found in the {selected_category} category.")
                    
            except Exception as e:
                st.error(f"Failed to load category FAQs: {str(e)}")
    
    with tab3:
        st.subheader("Popular Questions")
        
        try:
            popular_faqs = faq_system.get_popular_faqs(8)
            
            if popular_faqs:
                for i, faq in enumerate(popular_faqs, 1):
                    popularity = faq.get('popularity_score', 0)
                    
                    with st.expander(f"🔥 {i}. {faq['question']} (Score: {popularity:.1f})"):
                        st.markdown(faq['answer'])
                        st.markdown(f"**Category:** {faq.get('category', 'General')}")
            else:
                st.info("No popular FAQs available.")
                
        except Exception as e:
            st.error(f"Failed to load popular FAQs: {str(e)}")

def analytics_tab(system_components):
    """Render system analytics and insights."""
    st.header("📊 System Analytics")
    st.markdown("Overview of the LMS AI Assistant performance and usage.")
    
    vector_db = system_components['vector_db']
    
    # System health metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        content_stats = vector_db.get_index_stats('content')
        st.markdown(f"""
        <div class="metric-card">
            <h3>📚 Content Index</h3>
            <h2>{content_stats.get('total_vectors', 0)}</h2>
            <p>Total Items</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        user_stats = vector_db.get_index_stats('users')
        st.markdown(f"""
        <div class="metric-card">
            <h3>👥 Users Index</h3>
            <h2>{user_stats.get('total_vectors', 0)}</h2>
            <p>User Profiles</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        faq_stats = vector_db.get_index_stats('faq')
        st.markdown(f"""
        <div class="metric-card">
            <h3>❓ FAQ Index</h3>
            <h2>{faq_stats.get('total_vectors', 0)}</h2>
            <p>FAQ Items</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Sample data overview
    st.subheader("📋 Sample Data Overview")
    
    sample_data = system_components['sample_data']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📚 Courses by Subject**")
        courses = sample_data['courses']
        subjects = {}
        for course in courses:
            subject = course.get('subject_area', 'Unknown')
            subjects[subject] = subjects.get(subject, 0) + 1
        
        for subject, count in subjects.items():
            st.markdown(f"• {subject}: {count} courses")
    
    with col2:
        st.markdown("**👥 Users by Role**")
        users = sample_data['users']
        roles = {}
        for user in users:
            role = user.get('role', 'Unknown')
            roles[role] = roles.get(role, 0) + 1
        
        for role, count in roles.items():
            st.markdown(f"• {role.replace('_', ' ').title()}: {count} users")
    
    # Configuration info
    st.subheader("⚙️ Configuration")
    config_info = Config.get_deployment_info()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.json({
            "Pinecone Configured": config_info['pinecone_configured'],
            "Environment": config_info['environment'],
            "Embedding Model": config_info['embedding_model'],
            "Embedding Dimension": config_info['embedding_dimension']
        })
    
    with col2:
        st.json({
            "Content Index": config_info['content_index'],
            "User Index": config_info['user_index'], 
            "FAQ Index": config_info['faq_index']
        })

def main():
    """Main application function."""
    render_header()
    
    # Initialize system
    if 'system_initialized' not in st.session_state:
        system_components = initialize_system()
        st.session_state.system_components = system_components
        st.session_state.system_initialized = True
        st.rerun()
    
    system_components = st.session_state.system_components
    
    # Render sidebar
    selected_user_id, selected_user = render_sidebar(system_components)
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Semantic Search",
        "🎯 Recommendations", 
        "💬 FAQ Assistant",
        "📊 Analytics"
    ])
    
    with tab1:
        semantic_search_tab(system_components, selected_user_id)
    
    with tab2:
        recommendations_tab(system_components, selected_user_id, selected_user)
    
    with tab3:
        faq_tab(system_components)
    
    with tab4:
        analytics_tab(system_components)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>🎓 <strong>LMS AI Assistant System</strong></p>
        <p>Semantic Search • Personalized Recommendations • Intelligent FAQ</p>
        <p><em>Proof of Concept - Ready for LMS Integration</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
