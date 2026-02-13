"""
GraphGuard - AI-Powered Dynamic Authentication System
Streamlit App with LLM Integration and Fraud Detection
Week 4 Capstone Integration
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import hashlib
import secrets
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any, Optional, Tuple
import random
import json
from dataclasses import dataclass
from enum import Enum
import csv
import os

# Page configuration
st.set_page_config(
    page_title="GraphGuard - AI Auth",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CONFIGURATION
# ============================================================================

class SecurityLevel(Enum):
    LOW = 'low'
    MEDIUM = 'medium'
    HIGH = 'high'
    CRITICAL = 'critical'

# Logging configuration
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "product_metrics.csv")

# ============================================================================
# LOGGING SYSTEM
# ============================================================================

def initialize_logging():
    """Initialize logging directory and file"""
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp',
                'user_task_type',
                'user_id',
                'retrieval_configuration',
                'latency_ms',
                'evidence_ids',
                'confidence_score',
                'faithfulness_indicator',
                'status'
            ])

def log_interaction(
    user_task_type: str,
    user_id: str,
    retrieval_config: str,
    latency_ms: float,
    evidence_ids: List[str],
    confidence_score: float,
    faithfulness_indicator: str,
    status: str
):
    """Log a single interaction to the metrics file"""
    try:
        with open(LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(),
                user_task_type,
                user_id,
                retrieval_config,
                round(latency_ms, 2),
                ','.join(evidence_ids),
                round(confidence_score, 2),
                faithfulness_indicator,
                status
            ])
    except Exception as e:
        st.error(f"Logging error: {str(e)}")

# ============================================================================
# LLM INTEGRATION (Claude API)
# ============================================================================

def call_claude_api(prompt: str, max_tokens: int = 1000) -> str:
    """Call Claude API for intelligent question generation and analysis"""
    try:
        # Try to get API key from secrets or environment
        api_key = st.secrets.get("ANTHROPIC_API_KEY", None)

        if not api_key:
            # Fallback to demo mode
            return generate_demo_response(prompt)

        import anthropic

        client = anthropic.Anthropic(api_key=api_key)

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=max_tokens,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return message.content[0].text

    except Exception as e:
        st.warning(f"API call failed, using demo mode: {str(e)}")
        return generate_demo_response(prompt)

def generate_demo_response(prompt: str) -> str:
    """Generate demo responses when API is not available"""
    if "authentication questions" in prompt.lower():
        return json.dumps({
            "questions": [
                {
                    "question": "What was the merchant name of your last coffee shop visit?",
                    "options": ["Starbucks", "Dunkin", "Blue Bottle", "Peets Coffee"],
                    "correct_answer": "Starbucks",
                    "difficulty": "medium"
                },
                {
                    "question": "What was the approximate amount of your most recent grocery purchase?",
                    "options": ["$45-55", "$85-95", "$125-135", "$165-175"],
                    "correct_answer": "$125-135",
                    "difficulty": "medium"
                },
                {
                    "question": "In which city did you make your last electronics purchase?",
                    "options": ["San Francisco, CA", "New York, NY", "Los Angeles, CA", "Seattle, WA"],
                    "correct_answer": "San Francisco, CA",
                    "difficulty": "easy"
                }
            ]
        })
    elif "fraud analysis" in prompt.lower():
        return json.dumps({
            "fraud_score": 35.5,
            "risk_level": "MEDIUM",
            "insights": [
                "Transaction amount is within normal range for this user",
                "Merchant is familiar - visited 3 times in past 30 days",
                "Location matches user's typical transaction pattern",
                "Time of transaction (2:30 PM) is consistent with user behavior"
            ],
            "recommendations": [
                "No immediate action required",
                "Continue monitoring for pattern changes",
                "Consider step-up authentication for amounts > $500"
            ]
        })
    return "{}"

# ============================================================================
# DATA LOADING
# ============================================================================

@st.cache_data
def load_transaction_data(csv_path: str = None) -> pd.DataFrame:
    """Load transaction data from CSV file"""
    try:
        if csv_path and os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # Convert transaction date to datetime
            df['TRANSACTION_DATE'] = pd.to_datetime(df['TRANSACTION_DATE'])
            return df
        else:
            st.warning("CSV file not found. Using demo data.")
            return generate_demo_data()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return generate_demo_data()

@st.cache_data
def generate_demo_data(num_users: int = 10, transactions_per_user: int = 30) -> pd.DataFrame:
    """Generate realistic synthetic transaction data (fallback)"""

    merchants_by_category = {
        'Coffee Shops': ['Starbucks', 'Dunkin', 'Blue Bottle', 'Peets Coffee', 'Philz Coffee'],
        'Gas Stations': ['Shell', 'Chevron', 'Exxon', 'BP', '76'],
        'Restaurants': ["McDonald's", 'Chipotle', 'Panera Bread', 'Olive Garden', 'Subway'],
        'Grocery': ['Whole Foods', "Trader Joe's", 'Safeway', 'Kroger', 'Albertsons'],
        'Electronics': ['Best Buy', 'Apple Store', 'Amazon', 'Micro Center', 'B&H Photo'],
        'Jewelry': ['Tiffany & Co', 'Cartier', 'Zales', 'Kay Jewelers', 'Jared'],
        'Retail': ['Target', 'Walmart', 'Costco', 'Home Depot', 'CVS'],
        'Clothing': ['Nordstrom', 'Macy\'s', 'H&M', 'Zara', 'Gap']
    }

    locations = [
        'San Francisco, CA', 'New York, NY', 'Los Angeles, CA',
        'Chicago, IL', 'Seattle, WA', 'Boston, MA', 'Austin, TX'
    ]

    transactions = []

    for user_idx in range(num_users):
        user_id = f'USER_{user_idx + 1:03d}'
        user_location = random.choice(locations)

        favorite_categories = random.sample(list(merchants_by_category.keys()), 3)

        for txn_idx in range(transactions_per_user):
            if random.random() < 0.8:
                category = random.choice(favorite_categories)
            else:
                category = random.choice(list(merchants_by_category.keys()))

            merchant = random.choice(merchants_by_category[category])

            amount_ranges = {
                'Coffee Shops': (3, 15),
                'Gas Stations': (30, 80),
                'Grocery': (50, 250),
                'Electronics': (100, 2500),
                'Jewelry': (500, 5000),
                'Restaurants': (15, 150),
                'Retail': (20, 300),
                'Clothing': (40, 400)
            }

            min_amt, max_amt = amount_ranges.get(category, (10, 100))
            amount = round(random.uniform(min_amt, max_amt), 2)

            location = user_location if random.random() < 0.9 else random.choice(locations)

            days_ago = random.randint(0, 90)
            hours = random.randint(6, 22)
            minutes = random.randint(0, 59)

            transaction_date = datetime.now() - timedelta(days=days_ago, hours=hours, minutes=minutes)

            txn_id = hashlib.md5(f"{user_id}{transaction_date}{merchant}{amount}".encode()).hexdigest()[:12].upper()

            transactions.append({
                'TRANSACTION_ID': txn_id,
                'USER_ID': user_id,
                'MERCHANT_NAME': merchant,
                'CATEGORY': category,
                'AMOUNT': amount,
                'LOCATION': location,
                'TRANSACTION_DATE': transaction_date,
                'STATUS': 'COMPLETED'
            })

    df = pd.DataFrame(transactions)
    df = df.sort_values('TRANSACTION_DATE', ascending=False).reset_index(drop=True)

    return df

# ============================================================================
# AUTHENTICATION SYSTEM
# ============================================================================

@dataclass
class AuthQuestion:
    question: str
    options: List[str]
    correct_answer: str
    difficulty: str = 'medium'

class DynamicAuthenticator:
    """AI-powered dynamic authentication system"""

    def __init__(self, transaction_data: pd.DataFrame):
        self.data = transaction_data

    def generate_questions(self, user_id: str, num_questions: int = 3,
                          security_level: SecurityLevel = SecurityLevel.MEDIUM) -> List[AuthQuestion]:
        """Generate dynamic authentication questions based on user history"""
        
        start_time = datetime.now()
        
        user_txns = self.data[self.data['USER_ID'] == user_id].sort_values(
            'TRANSACTION_DATE', ascending=False
        )

        if len(user_txns) == 0:
            log_interaction(
                "authentication_generation",
                user_id,
                f"security_{security_level.value}",
                (datetime.now() - start_time).total_seconds() * 1000,
                [],
                0.0,
                "no_data",
                "failed"
            )
            return []

        # Create context for LLM
        recent_txns = user_txns.head(10).to_dict('records')
        txn_summary = [
            f"- {t['MERCHANT_NAME']} (${t['AMOUNT']:.2f}) in {t['LOCATION']} on {t['TRANSACTION_DATE'].strftime('%Y-%m-%d')}"
            for t in recent_txns
        ]

        prompt = f"""Based on this user's recent transaction history, generate {num_questions} authentication questions.
Security Level: {security_level.value.upper()}

Recent Transactions:
{chr(10).join(txn_summary)}

Generate questions in this exact JSON format:
{{
    "questions": [
        {{
            "question": "question text",
            "options": ["option1", "option2", "option3", "option4"],
            "correct_answer": "option1",
            "difficulty": "easy|medium|hard"
        }}
    ]
}}

Make questions specific to their actual transactions. Include merchant names, amounts, locations, and dates."""

        try:
            response = call_claude_api(prompt)
            data = json.loads(response)
            
            questions = []
            evidence_ids = []
            
            for q in data.get('questions', [])[:num_questions]:
                questions.append(AuthQuestion(
                    question=q['question'],
                    options=q['options'],
                    correct_answer=q['correct_answer'],
                    difficulty=q.get('difficulty', 'medium')
                ))
                # Track which transactions were used as evidence
                evidence_ids.append(recent_txns[0]['TRANSACTION_ID'])
            
            latency = (datetime.now() - start_time).total_seconds() * 1000
            
            log_interaction(
                "authentication_generation",
                user_id,
                f"security_{security_level.value}",
                latency,
                evidence_ids,
                0.85,  # Confidence based on LLM generation
                "high",
                "success"
            )
            
            return questions

        except Exception as e:
            st.error(f"Question generation failed: {str(e)}")
            latency = (datetime.now() - start_time).total_seconds() * 1000
            log_interaction(
                "authentication_generation",
                user_id,
                f"security_{security_level.value}",
                latency,
                [],
                0.0,
                "error",
                "failed"
            )
            return []

    def verify_answers(self, questions: List[AuthQuestion], user_answers: List[str]) -> Tuple[bool, float]:
        """Verify user answers and return success status and score"""
        
        start_time = datetime.now()
        
        if len(questions) != len(user_answers):
            return False, 0.0

        correct = sum(1 for q, a in zip(questions, user_answers) if q.correct_answer == a)
        score = (correct / len(questions)) * 100
        passed = score >= 60  # 60% threshold

        latency = (datetime.now() - start_time).total_seconds() * 1000
        
        log_interaction(
            "authentication_verification",
            "unknown",  # User ID not available in verification
            "answer_check",
            latency,
            [],
            score / 100,
            "high" if passed else "low",
            "success" if passed else "failed"
        )

        return passed, score

# ============================================================================
# FRAUD DETECTION SYSTEM
# ============================================================================

class AIFraudDetector:
    """AI-powered fraud detection system"""

    def __init__(self, transaction_data: pd.DataFrame):
        self.data = transaction_data

    def analyze_transaction(self, user_id: str, transaction_id: str) -> Dict[str, Any]:
        """Analyze a transaction for fraud indicators"""
        
        start_time = datetime.now()
        
        txn = self.data[self.data['TRANSACTION_ID'] == transaction_id]

        if len(txn) == 0:
            return {"error": "Transaction not found"}

        txn = txn.iloc[0]

        # Get user history
        user_history = self.data[
            (self.data['USER_ID'] == user_id) &
            (self.data['TRANSACTION_DATE'] < txn['TRANSACTION_DATE'])
        ].tail(20)

        # Build context for AI
        context = f"""Analyze this transaction for fraud:

Current Transaction:
- ID: {txn['TRANSACTION_ID']}
- Merchant: {txn['MERCHANT_NAME']}
- Category: {txn['CATEGORY']}
- Amount: ${txn['AMOUNT']:.2f}
- Location: {txn['LOCATION']}
- Date: {txn['TRANSACTION_DATE']}

User Profile:
- Average transaction: ${user_history['AMOUNT'].mean():.2f}
- Typical locations: {', '.join(user_history['LOCATION'].unique())}
- Frequent categories: {', '.join(user_history['CATEGORY'].value_counts().head(3).index)}
- Recent activity: {len(user_history)} transactions in last 90 days

Provide fraud analysis in this JSON format:
{{
    "fraud_score": 0-100,
    "risk_level": "LOW|MEDIUM|HIGH|CRITICAL",
    "insights": ["insight1", "insight2", ...],
    "recommendations": ["rec1", "rec2", ...]
}}"""

        try:
            response = call_claude_api(context)
            analysis = json.loads(response)
            
            latency = (datetime.now() - start_time).total_seconds() * 1000
            
            log_interaction(
                "fraud_detection",
                user_id,
                "ai_analysis",
                latency,
                [transaction_id],
                1.0 - (analysis.get('fraud_score', 50) / 100),  # Inverse of fraud score
                analysis.get('risk_level', 'UNKNOWN'),
                "success"
            )
            
            return analysis

        except Exception as e:
            st.error(f"Fraud analysis failed: {str(e)}")
            latency = (datetime.now() - start_time).total_seconds() * 1000
            log_interaction(
                "fraud_detection",
                user_id,
                "ai_analysis",
                latency,
                [transaction_id],
                0.0,
                "error",
                "failed"
            )
            return {"error": str(e)}

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application"""
    
    # Initialize logging
    initialize_logging()

    # Header
    st.markdown("""
    # 🔐 GraphGuard
    ## AI-Powered Dynamic Authentication & Fraud Detection System
    ### Week 4 Capstone Integration
    """)

    # Sidebar
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")

        data_source = st.radio(
            "Data Source",
            ["Load CSV File", "Demo Data"],
            help="Choose between real transaction data or generated demo data"
        )

        # Option to upload CSV or use default path
        if data_source == "Load CSV File":
            csv_path = st.text_input(
                "CSV File Path",
                value="transactions_export_20260213_000349.csv",
                help="Path to transaction CSV file"
            )
            if st.button("Load Data"):
                st.session_state.data_loaded = True
                st.session_state.csv_path = csv_path
        else:
            st.session_state.data_loaded = True
            st.session_state.csv_path = None

        st.markdown("---")
        st.markdown("### 📊 System Status")
        st.info(f"**Log File:** `{LOG_FILE}`")
        
        if os.path.exists(LOG_FILE):
            log_df = pd.read_csv(LOG_FILE)
            st.metric("Total Interactions", len(log_df))
            st.metric("Success Rate", f"{(log_df['status'] == 'success').mean() * 100:.1f}%")

    # Load data
    if st.session_state.get('data_loaded', False):
        csv_path = st.session_state.get('csv_path')
        df = load_transaction_data(csv_path)

        # Initialize session state
        if 'auth_questions' not in st.session_state:
            st.session_state.auth_questions = None
        if 'auth_result' not in st.session_state:
            st.session_state.auth_result = None
        if 'fraud_analysis' not in st.session_state:
            st.session_state.fraud_analysis = None

        # Main tabs
        tabs = st.tabs(["🔑 Authentication", "🚨 Fraud Detection", "📊 Dashboard", "🔍 Explorer"])

        # TAB 1: AUTHENTICATION
        with tabs[0]:
            st.markdown("### 🔑 Dynamic Authentication System")

            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown("#### 👤 User Selection")

                auth_user = st.selectbox("Select User", df['USER_ID'].unique())

                user_txns = df[df['USER_ID'] == auth_user].sort_values('TRANSACTION_DATE', ascending=False)

                st.dataframe(
                    user_txns[['TRANSACTION_DATE', 'MERCHANT_NAME', 'AMOUNT', 'LOCATION']].head(10),
                    use_container_width=True,
                    hide_index=True
                )

                security_level = st.select_slider(
                    "Security Level",
                    options=['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'],
                    value='MEDIUM'
                )

                if st.button("🎲 Generate Questions", type="primary", use_container_width=True):
                    with st.spinner("🤖 AI is generating questions..."):
                        authenticator = DynamicAuthenticator(df)
                        security_enum = SecurityLevel[security_level]
                        questions = authenticator.generate_questions(auth_user, 3, security_enum)
                        st.session_state.auth_questions = questions
                        st.session_state.auth_result = None
                        st.rerun()

            with col2:
                st.markdown("#### 📝 Authentication Challenge")

                if st.session_state.auth_questions:
                    questions = st.session_state.auth_questions
                    user_answers = []

                    for i, q in enumerate(questions, 1):
                        st.markdown(f"**Question {i}** ({q.difficulty.upper()})")
                        st.markdown(f"*{q.question}*")

                        answer = st.radio(
                            "Select your answer:",
                            q.options,
                            key=f"q_{i}",
                            label_visibility="collapsed"
                        )
                        user_answers.append(answer)
                        st.markdown("---")

                    if st.button("✅ Submit Answers", type="primary", use_container_width=True):
                        authenticator = DynamicAuthenticator(df)
                        passed, score = authenticator.verify_answers(questions, user_answers)
                        st.session_state.auth_result = {"passed": passed, "score": score}
                        st.rerun()

                    if st.session_state.auth_result:
                        result = st.session_state.auth_result

                        if result['passed']:
                            st.success(f"✅ Authentication Successful! Score: {result['score']:.1f}%")
                        else:
                            st.error(f"❌ Authentication Failed. Score: {result['score']:.1f}%")

                        st.progress(result['score'] / 100)
                else:
                    st.info("👈 Generate authentication questions to begin")

        # TAB 2: FRAUD DETECTION
        with tabs[1]:
            st.markdown("### 🚨 AI-Powered Fraud Detection")

            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown("#### 🔍 Select Transaction")

                fraud_user = st.selectbox("User", df['USER_ID'].unique(), key="fraud_user")
                fraud_txns = df[df['USER_ID'] == fraud_user].sort_values('TRANSACTION_DATE', ascending=False)

                st.dataframe(
                    fraud_txns[['TRANSACTION_ID', 'TRANSACTION_DATE', 'MERCHANT_NAME', 'AMOUNT']].head(20),
                    use_container_width=True,
                    hide_index=True
                )

                fraud_txn_id = st.selectbox(
                    "Transaction ID",
                    fraud_txns['TRANSACTION_ID'].tolist()
                )

                if st.button("🔬 Analyze with AI", type="primary", use_container_width=True):
                    with st.spinner("🤖 AI is analyzing..."):
                        detector = AIFraudDetector(df)
                        analysis = detector.analyze_transaction(fraud_user, fraud_txn_id)
                        st.session_state.fraud_analysis = analysis
                        st.rerun()

            with col2:
                st.markdown("#### 📋 Analysis Report")

                if st.session_state.fraud_analysis:
                    analysis = st.session_state.fraud_analysis

                    if 'error' not in analysis:
                        fraud_score = analysis.get('fraud_score', 0)
                        risk_level = analysis.get('risk_level', 'UNKNOWN')

                        st.metric("Fraud Score", f"{fraud_score:.1f}/100")
                        st.metric("Risk Level", risk_level)
                        st.progress(fraud_score / 100)

                        st.markdown("---")
                        st.markdown("**🧠 AI Insights:**")
                        for insight in analysis.get('insights', []):
                            st.markdown(f"• {insight}")

                        st.markdown("---")
                        st.markdown("**📌 Recommendations:**")
                        for rec in analysis.get('recommendations', []):
                            st.markdown(f"✓ {rec}")
                    else:
                        st.error(analysis['error'])
                else:
                    st.info("👈 Select a transaction to analyze")

        # TAB 3: DASHBOARD
        with tabs[2]:
            st.markdown("### 📊 System Dashboard")

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Transactions", f"{len(df):,}")
            with col2:
                st.metric("Total Users", f"{df['USER_ID'].nunique():,}")
            with col3:
                st.metric("Total Volume", f"${df['AMOUNT'].sum():,.2f}")
            with col4:
                st.metric("Avg Transaction", f"${df['AMOUNT'].mean():.2f}")

            st.markdown("---")

            col1, col2 = st.columns(2)

            with col1:
                category_data = df['CATEGORY'].value_counts()
                fig = px.pie(
                    values=category_data.values,
                    names=category_data.index,
                    title="Transaction Distribution by Category"
                )
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                daily_data = df.groupby(df['TRANSACTION_DATE'].dt.date)['AMOUNT'].sum().reset_index()
                daily_data.columns = ['Date', 'Amount']
                fig = px.line(daily_data, x='Date', y='Amount', title="Daily Transaction Volume")
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("### 🏪 Top Merchants")
            top_merchants = df.groupby('MERCHANT_NAME').agg({
                'TRANSACTION_ID': 'count',
                'AMOUNT': 'sum'
            }).sort_values('AMOUNT', ascending=False).head(10)
            top_merchants.columns = ['Transactions', 'Total Revenue']
            st.dataframe(top_merchants, use_container_width=True)

        # TAB 4: EXPLORER
        with tabs[3]:
            st.markdown("### 🔍 Transaction Explorer")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                filter_users = st.multiselect("Users", df['USER_ID'].unique())
            with col2:
                filter_categories = st.multiselect("Categories", df['CATEGORY'].unique())
            with col3:
                filter_locations = st.multiselect("Locations", df['LOCATION'].unique())
            with col4:
                min_amt, max_amt = st.slider(
                    "Amount Range ($)",
                    float(df['AMOUNT'].min()),
                    float(df['AMOUNT'].max()),
                    (float(df['AMOUNT'].min()), float(df['AMOUNT'].max()))
                )

            filtered_df = df.copy()

            if filter_users:
                filtered_df = filtered_df[filtered_df['USER_ID'].isin(filter_users)]
            if filter_categories:
                filtered_df = filtered_df[filtered_df['CATEGORY'].isin(filter_categories)]
            if filter_locations:
                filtered_df = filtered_df[filtered_df['LOCATION'].isin(filter_locations)]

            filtered_df = filtered_df[
                (filtered_df['AMOUNT'] >= min_amt) &
                (filtered_df['AMOUNT'] <= max_amt)
            ]

            st.markdown(f"**Showing {len(filtered_df):,} of {len(df):,} transactions**")

            st.dataframe(
                filtered_df.sort_values('TRANSACTION_DATE', ascending=False),
                use_container_width=True,
                hide_index=True
            )

            csv = filtered_df.to_csv(index=False)
            st.download_button(
                "📥 Download CSV",
                csv,
                f"transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv"
            )

    else:
        st.warning("⚠️ Please configure data source in sidebar")
        st.info("""
        **Getting Started:**
        1. Select "Load CSV File" and provide path to your transaction data
        2. Or select "Demo Data" for instant testing
        3. Optional: Add ANTHROPIC_API_KEY in `.streamlit/secrets.toml` for full AI features
        """)

if __name__ == "__main__":
    main()
