import os
import sys
import time
import json
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Add workspace to path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.pipeline.predictor import NetworkAttackPredictor
from src.data.dataset_generator import NetworkDatasetGenerator
from src.analysis.eda import NetworkEDAAnalyzer

# Page configuration
st.set_page_config(
    page_title="Cybersecurity — Network Attack Detection",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Cyberpunk CSS Styling
st.markdown("""
<style>
    .main {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    .stMetric {
        background: rgba(22, 27, 34, 0.8);
        border: 1px solid #30363d;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    .normal-card {
        background: rgba(46, 160, 67, 0.15);
        border: 1px solid #2ea043;
        border-radius: 8px;
        padding: 15px;
        color: #3fb950;
    }
    .attack-card {
        background: rgba(248, 81, 73, 0.15);
        border: 1px solid #f85149;
        border-radius: 8px;
        padding: 15px;
        color: #ff7b72;
    }
    .stButton>button {
        background-color: #238636;
        color: white;
        border-radius: 6px;
        font-weight: bold;
        border: none;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background-color: #2ea043;
        box-shadow: 0 0 10px rgba(46, 160, 67, 0.5);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_predictor():
    models_dir = os.path.join('models', 'saved')
    if os.path.exists(os.path.join(models_dir, 'preprocessor.joblib')):
        return NetworkAttackPredictor(models_dir=models_dir)
    return None

@st.cache_data
def load_data():
    train_path = os.path.join('data', 'raw', 'network_traffic.csv')
    if os.path.exists(train_path):
        return pd.read_csv(train_path)
    # Generate on the fly if missing
    gen = NetworkDatasetGenerator()
    return gen.generate_dataset(3000)

predictor = load_predictor()
df_data = load_data()

# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/color/96/cyber-security.png", width=70)
st.sidebar.title("🔐 Cyber Intelligence")
st.sidebar.markdown("**Network Intrusion & Threat Classification System**")

selected_tab = st.sidebar.radio(
    "Navigation Menu",
    [
        "📊 EDA & Traffic Analysis",
        "⚡ Live Monitor & Simulator",
        "🎯 Packet & Batch Inspector",
        "🧠 Model Studio & Benchmarks",
        "🔌 API Documentation"
    ]
)

model_choice = st.sidebar.selectbox(
    "Active Inference Engine",
    ["Random Forest", "XGBoost", "1D CNN (PyTorch)", "LSTM (PyTorch)"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("🛡️ **System Status**: `OPERATIONAL`")
st.sidebar.markdown("⚡ **Threat Sensor**: `ACTIVE (10 Gbps)`")

# -------------------------------------------------------------
# TAB 1: EDA & Traffic Analysis
# -------------------------------------------------------------
if selected_tab == "📊 EDA & Traffic Analysis":
    st.title("📊 Phase 4: EDA & Network Traffic Analysis")
    st.markdown("Statistical breakdown of normal network packets versus malicious cyber attack flows.")

    col1, col2, col3, col4 = st.columns(4)
    total_records = len(df_data)
    attack_count = int(df_data['is_attack'].sum()) if 'is_attack' in df_data.columns else 0
    normal_count = total_records - attack_count
    attack_pct = round((attack_count / total_records) * 100, 2)

    col1.metric("Total Traffic Flows", f"{total_records:,}")
    col2.metric("Normal Traffic (🟢)", f"{normal_count:,}", delta=f"{100-attack_pct:.1f}%")
    col3.metric("Malicious Attacks (🔴)", f"{attack_count:,}", delta=f"-{attack_pct:.1f}%", delta_color="inverse")
    col4.metric("Attack Types Identified", len(df_data['label'].unique()))

    st.markdown("---")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("📌 Attack Category Distribution")
        label_counts = df_data['label'].value_counts().reset_index()
        label_counts.columns = ['Attack Type', 'Count']
        fig_bar = px.bar(
            label_counts, x='Attack Type', y='Count',
            color='Attack Type', color_discrete_sequence=px.colors.qualitative.Vivid,
            template="plotly_dark", title="Traffic Breakdown by Attack Class"
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with c2:
        st.subheader("🔥 Feature Correlation Heatmap")
        num_cols = ['duration_sec', 'fwd_packets', 'bwd_packets', 'total_packets',
                    'packet_size_mean', 'flow_bytes_per_sec', 'flow_packets_per_sec', 'syn_flag', 'ack_flag']
        corr = df_data[num_cols].corr()
        fig_heat = px.imshow(
            corr, text_auto=True, color_continuous_scale="Viridis",
            template="plotly_dark", title="Pearson Feature Correlation Matrix"
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("---")

    c3, c4 = st.columns(2)

    with c3:
        st.subheader("📈 Packet Size Distribution: Normal vs Attack")
        fig_hist = px.histogram(
            df_data, x="packet_size_mean", color="label",
            marginal="box", template="plotly_dark",
            title="Distribution of Packet Size Mean across Attack Vectors"
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with c4:
        st.subheader("📦 Outlier Analysis: Flow Duration (IQR)")
        fig_box = px.box(
            df_data, x="label", y="duration_sec", color="label",
            points="outliers", template="plotly_dark",
            title="Connection Duration Outliers per Attack Category"
        )
        st.plotly_chart(fig_box, use_container_width=True)


# -------------------------------------------------------------
# TAB 2: Live Monitor & Simulator
# -------------------------------------------------------------
elif selected_tab == "⚡ Live Monitor & Simulator":
    st.title("⚡ Live Network Traffic Monitor & Threat Simulator")
    st.markdown("Real-time telemetry monitoring incoming packet flows and alerting on cyber attacks.")

    if 'simulation_logs' not in st.session_state:
        st.session_state.simulation_logs = []

    m1, m2, m3, m4 = st.columns(4)

    total_sim = len(st.session_state.simulation_logs)
    attack_sim = sum(1 for log in st.session_state.simulation_logs if log['is_attack'])
    normal_sim = total_sim - attack_sim

    m1.metric("Simulated Packets", total_sim)
    m2.metric("Normal Traffic", normal_sim)
    m3.metric("Attacks Detected (🔴)", attack_sim, delta_color="inverse")
    m4.metric("Active Model Engine", model_choice)

    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 4])
    with col_btn1:
        sim_trigger = st.button("🚀 Inject Packet Batch", use_container_width=True)
    with col_btn2:
        if st.button("🧹 Clear Logs", use_container_width=True):
            st.session_state.simulation_logs = []
            st.rerun()

    if sim_trigger and predictor:
        gen = NetworkDatasetGenerator()
        batch_samples = gen.generate_dataset(5)
        preds = predictor.predict_batch(batch_samples, model_type=model_choice)
        
        for p in preds:
            st.session_state.simulation_logs.insert(0, p)

    if st.session_state.simulation_logs:
        st.subheader("🚨 Real-Time Threat Stream Feed")
        
        # Display latest packet alert
        latest = st.session_state.simulation_logs[0]
        if latest['is_attack']:
            st.markdown(f"""
            <div class="attack-card">
                <h3>🔴 THREAT DETECTED: {latest['attack_type']}</h3>
                <p><strong>Confidence:</strong> {latest['confidence_percent']}% | 
                   <strong>Anomaly Score:</strong> {latest['anomaly_score']} | 
                   <strong>Source IP:</strong> {latest['flow_info']['src_ip']} ➔ 
                   <strong>Dest IP:</strong> {latest['flow_info']['dst_ip']}:{latest['flow_info']['dst_port']}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="normal-card">
                <h3>🟢 NORMAL TRAFFIC APPROVED</h3>
                <p><strong>Confidence:</strong> {latest['confidence_percent']}% | 
                   <strong>Source IP:</strong> {latest['flow_info']['src_ip']} ➔ 
                   <strong>Dest IP:</strong> {latest['flow_info']['dst_ip']}:{latest['flow_info']['dst_port']}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### 📋 Packet Detection History Log")
        log_df = pd.DataFrame([
            {
                "Status": log['status'],
                "Attack Type": log['attack_type'],
                "Confidence": f"{log['confidence_percent']}%",
                "Anomaly Score": log['anomaly_score'],
                "Source IP": log['flow_info']['src_ip'],
                "Dest IP": log['flow_info']['dst_ip'],
                "Dest Port": log['flow_info']['dst_port'],
                "Protocol": log['flow_info']['protocol']
            }
            for log in st.session_state.simulation_logs
        ])
        st.dataframe(log_df, use_container_width=True)
    else:
        st.info("Click **Inject Packet Batch** above to simulate live network traffic packet flows.")


# -------------------------------------------------------------
# TAB 3: Packet & Batch Inspector
# -------------------------------------------------------------
elif selected_tab == "🎯 Packet & Batch Inspector":
    st.title("🎯 Single Packet & Batch Flow Inspector")
    
    sub_tab1, sub_tab2 = st.tabs(["🔍 Manual Packet Tester", "📁 CSV Batch Classifier"])

    with sub_tab1:
        st.subheader("Configure Packet Parameters for Threat Analysis")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            src_ip = st.text_input("Source IP", "192.168.1.100")
            dst_ip = st.text_input("Destination IP", "10.0.0.1")
            protocol = st.selectbox("Protocol", ["TCP", "UDP", "ICMP"])
            src_port = st.number_input("Source Port", 1024, 65535, 54321)
            dst_port = st.number_input("Destination Port", 1, 65535, 80)
        
        with c2:
            duration_sec = st.number_input("Duration (sec)", 0.0, 300.0, 0.05)
            fwd_packets = st.number_input("Forward Packets", 1, 10000, 2000)
            bwd_packets = st.number_input("Backward Packets", 0, 10000, 5)
            pkt_mean = st.number_input("Packet Size Mean (B)", 40.0, 1500.0, 64.0)

        with c3:
            syn_flag = st.selectbox("TCP SYN Flag", [1, 0])
            ack_flag = st.selectbox("TCP ACK Flag", [0, 1])
            rst_flag = st.selectbox("TCP RST Flag", [0, 1])
            fin_flag = st.selectbox("TCP FIN Flag", [0, 1])
            psh_flag = st.selectbox("TCP PSH Flag", [0, 1])

        if st.button("⚡ Inspect & Predict Threat", use_container_width=True):
            if predictor:
                sample_dict = {
                    'src_ip': src_ip,
                    'dst_ip': dst_ip,
                    'src_port': src_port,
                    'dst_port': dst_port,
                    'protocol': protocol,
                    'duration_sec': duration_sec,
                    'fwd_packets': fwd_packets,
                    'bwd_packets': bwd_packets,
                    'total_packets': fwd_packets + bwd_packets,
                    'packet_size_mean': pkt_mean,
                    'packet_size_min': max(40.0, pkt_mean - 20),
                    'packet_size_max': pkt_mean + 20,
                    'packet_size_std': 5.0,
                    'flow_bytes_per_sec': (fwd_packets + bwd_packets) * pkt_mean / max(duration_sec, 1e-4),
                    'flow_packets_per_sec': (fwd_packets + bwd_packets) / max(duration_sec, 1e-4),
                    'syn_flag': syn_flag,
                    'ack_flag': ack_flag,
                    'rst_flag': rst_flag,
                    'fin_flag': fin_flag,
                    'psh_flag': psh_flag
                }
                
                res = predictor.predict_single(sample_dict, model_type=model_choice)
                
                st.markdown("---")
                p1, p2, p3 = st.columns(3)
                p1.metric("Threat Status", res['status'])
                p2.metric("Predicted Attack Type", res['attack_type'])
                p3.metric("Prediction Confidence", f"{res['confidence_percent']}%")

                st.progress(float(res['confidence_percent']) / 100.0)
                
                if res['is_attack']:
                    st.error(f"⚠️ HIGH RISK ALERT: {res['attack_type']} attack pattern detected with {res['confidence_percent']}% confidence.")
                else:
                    st.success("✅ Traffic pattern is classified as Normal and safe.")

    with sub_tab2:
        st.subheader("Upload CSV File for Batch Intrusion Detection")
        uploaded_file = st.file_uploader("Upload raw network traffic CSV", type=["csv"])
        
        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)
            st.write(f"Uploaded File: {len(batch_df)} rows")
            
            if st.button("🚀 Run Batch Prediction", use_container_width=True) and predictor:
                results = predictor.predict_batch(batch_df, model_type=model_choice)
                res_df = pd.DataFrame(results)
                
                # Expand nested dicts
                res_df['status'] = res_df['status']
                res_df['attack_type'] = res_df['attack_type']
                res_df['confidence'] = res_df['confidence_percent']
                
                st.dataframe(res_df[['status', 'attack_type', 'confidence', 'anomaly_score']], use_container_width=True)
                
                csv_download = res_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Threat Classification Report CSV",
                    data=csv_download,
                    file_name="cyber_attack_predictions.csv",
                    mime="text/csv"
                )


# -------------------------------------------------------------
# TAB 4: Model Studio & Benchmarks
# -------------------------------------------------------------
elif selected_tab == "🧠 Model Studio & Benchmarks":
    st.title("🧠 Phase 8: Model Performance & Security Benchmarks")
    st.markdown("Comprehensive comparison across Random Forest, XGBoost, 1D CNN, LSTM, and Autoencoder.")

    report_path = os.path.join('reports', 'evaluation_metrics.json')
    if os.path.exists(report_path):
        with open(report_path, 'r') as f:
            metrics_data = json.load(f)

        models_list = list(metrics_data.keys())
        acc_list = [metrics_data[m]['accuracy'] * 100 for m in models_list]
        rec_list = [metrics_data[m]['recall'] * 100 for m in models_list]
        f1_list = [metrics_data[m]['f1_score'] * 100 for m in models_list]

        df_bench = pd.DataFrame({
            'Model': models_list,
            'Accuracy (%)': acc_list,
            'Recall (%)': rec_list,
            'F1-Score (%)': f1_list
        })

        st.subheader("📊 Side-by-Side Model Comparison (Recall Priority)")
        st.dataframe(df_bench, use_container_width=True)

        fig_bench = px.bar(
            df_bench, x='Model', y=['Accuracy (%)', 'Recall (%)', 'F1-Score (%)'],
            barmode='group', template='plotly_dark',
            title='Model Performance Metrics Comparison'
        )
        st.plotly_chart(fig_bench, use_container_width=True)

        selected_model = st.selectbox("Select Model to Inspect Confusion Matrix", models_list)
        if selected_model in metrics_data:
            cm = np.array(metrics_data[selected_model]['confusion_matrix'])
            labels = ['Normal', 'DDoS', 'DoS', 'Brute Force', 'Botnet', 'PortScan', 'Infiltration']
            
            fig_cm = px.imshow(
                cm, x=labels[:cm.shape[1]], y=labels[:cm.shape[0]], text_auto=True,
                color_continuous_scale='Blues', template='plotly_dark',
                title=f'Confusion Matrix — {selected_model}'
            )
            st.plotly_chart(fig_cm, use_container_width=True)
    else:
        st.info("Run `python -m src.evaluation.evaluate_all` to generate benchmark metrics.")


# -------------------------------------------------------------
# TAB 5: API Documentation
# -------------------------------------------------------------
elif selected_tab == "🔌 API Documentation":
    st.title("🔌 Phase 10: FastAPI REST Backend Integration")
    st.markdown("The system exposes production REST API endpoints for seamless SIEM integration.")

    st.subheader("Available API Endpoints")
    st.markdown("""
    - `GET /health`: System health & model readiness probe
    - `POST /predict`: Classify single packet flow
    - `POST /predict/batch`: Process bulk CSV packet streams
    - `GET /simulate`: Generate & predict on live packet stream
    """)

    st.code("""
# Python API Integration Example
import requests

url = "http://localhost:8000/predict"
payload = {
    "src_ip": "192.168.1.50",
    "dst_ip": "10.0.0.5",
    "dst_port": 80,
    "protocol": "TCP",
    "fwd_packets": 2000,
    "bwd_packets": 2,
    "duration_sec": 0.01,
    "packet_size_mean": 64.0,
    "syn_flag": 1
}

response = requests.post(url, json=payload)
print(response.json())
    """, language="python")

    st.info("Launch API server locally via: `uvicorn src.api.main:app --port 8000`")
