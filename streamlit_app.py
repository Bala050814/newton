
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def apply_custom_css():
    """Apply custom CSS for professional styling"""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .main-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .header-container {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(79, 172, 254, 0.3);
    }
    
    .header-title {
        font-family: 'Inter', sans-serif;
        font-size: 3rem;
        font-weight: 700;
        color: white;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .header-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.2rem;
        color: rgba(255,255,255,0.9);
        margin-top: 0.5rem;
        font-weight: 300;
    }
    
    .section-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 25px rgba(0,0,0,0.08);
        border: 1px solid rgba(0,0,0,0.05);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .section-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 35px rgba(0,0,0,0.12);
    }
    
    .section-title {
        font-family: 'Inter', sans-serif;
        font-size: 1.4rem;
        font-weight: 600;
        color: #2d3748;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border-radius: 12px;
        padding: 1.5rem;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 8px 20px rgba(240, 147, 251, 0.3);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    
    .success-card {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        border-radius: 12px;
        padding: 1rem;
        border-left: 4px solid #48bb78;
        color: #2d3748;
        font-weight: 500;
    }
    
    .warning-card {
        background: linear-gradient(135deg, #ffeaa7 0%, #fab1a0 100%);
        border-radius: 12px;
        padding: 1rem;
        border-left: 4px solid #ed8936;
        color: #2d3748;
        font-weight: 500;
    }
    
    .error-card {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        border-radius: 12px;
        padding: 1rem;
        border-left: 4px solid #e53e3e;
        color: #2d3748;
        font-weight: 500;
    }
    
    .matrix-container {
        background: #f8fafc;
        border-radius: 10px;
        padding: 1rem;
        border: 2px dashed #cbd5e0;
        margin: 1rem 0;
    }
    
    .solution-table {
        background: white;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
    }
    
    .stSelectbox > div > div {
        border-radius: 8px;
        border: 2px solid #e2e8f0;
    }
    
    .stNumberInput > div > div > input {
        border-radius: 8px;
        border: 2px solid #e2e8f0;
        font-weight: 500;
    }
    
    .explanation-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
    }
    
    .explanation-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    .formula-card {
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        backdrop-filter: blur(5px);
        border: 1px solid rgba(255,255,255,0.2);
        font-family: 'Courier New', monospace;
        font-size: 1.1rem;
        text-align: center;
    }
    
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }
    
    .feature-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 6px 20px rgba(0,0,0,0.08);
        border-top: 4px solid #667eea;
        transition: transform 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
    }
    
    .feature-icon {
        font-size: 2rem;
        margin-bottom: 1rem;
    }
    
    .stExpander {
        background: white;
        border-radius: 15px;
        box-shadow: 0 6px 20px rgba(0,0,0,0.08);
        border: none;
    }
    
    </style>
    """, unsafe_allow_html=True)

def create_header():
    """Create professional header"""
    st.markdown("""
    <div class="header-container">
        <h1 class="header-title">⚡ Gauss-Seidel Calculator</h1>
        <p class="header-subtitle">Professional Linear Equation Solver with Advanced Analytics</p>
    </div>
    """, unsafe_allow_html=True)

def create_explanation_section():
    """Create modern explanation section"""
    with st.expander("🧠 Understanding the Gauss-Seidel Method", expanded=False):
        st.markdown("""
        <div class="explanation-card">
            <h3 class="explanation-title">🎯 What is Gauss-Seidel?</h3>
            <p>An elegant iterative method for solving systems of linear equations <strong>Ax = b</strong>. 
            Perfect for large sparse matrices where direct methods become computationally expensive.</p>
            
            <div class="formula-card">
                <strong>Core Formula:</strong><br>
                x₍ᵢ₎⁽ᵏ⁺¹⁾ = (bᵢ - Σⱼ₌₁ⁱ⁻¹ aᵢⱼx₍ⱼ₎⁽ᵏ⁺¹⁾ - Σⱼ₌ᵢ₊₁ⁿ aᵢⱼx₍ⱼ₎⁽ᵏ⁾) / aᵢᵢ
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-icon">🚀</div>
                <h4>Fast Convergence</h4>
                <p>Rapidly converges for diagonally dominant matrices</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-icon">💾</div>
                <h4>Memory Efficient</h4>
                <p>No need to store additional matrices during computation</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="feature-card">
                <div class="feature-icon">⚡</div>
                <h4>Real-time Updates</h4>
                <p>Uses most recent values immediately in calculations</p>
            </div>
            """, unsafe_allow_html=True)

def gauss_seidel(A, b, x0=None, max_iter=100, tolerance=1e-6):
    """Enhanced Gauss-Seidel implementation with detailed tracking"""
    n = len(A)
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()
    
    iterations = []
    errors = []
    convergence_rate = []
    
    for iteration in range(max_iter):
        x_old = x.copy()
        
        for i in range(n):
            sum1 = sum(A[i][j] * x[j] for j in range(i))
            sum2 = sum(A[i][j] * x_old[j] for j in range(i + 1, n))
            x[i] = (b[i] - sum1 - sum2) / A[i][i]
        
        error = np.max(np.abs(x - x_old))
        errors.append(error)
        iterations.append(x.copy())
        
        # Calculate convergence rate
        if len(errors) > 1 and errors[-2] > 0:
            rate = errors[-1] / errors[-2]
            convergence_rate.append(rate)
        
        if error < tolerance:
            return x, iterations, errors, convergence_rate, iteration + 1
    
    return x, iterations, errors, convergence_rate, max_iter

def is_diagonally_dominant(A):
    """Check diagonal dominance with detailed analysis"""
    n = len(A)
    dominance_ratios = []
    
    for i in range(n):
        diagonal = abs(A[i][i])
        row_sum = sum(abs(A[i][j]) for j in range(n) if j != i)
        if diagonal > 0:
            ratio = diagonal / (row_sum + 1e-10)
            dominance_ratios.append(ratio)
        else:
            dominance_ratios.append(0)
    
    is_dominant = all(ratio > 1 for ratio in dominance_ratios)
    return is_dominant, dominance_ratios

def create_matrix_input_section(n):
    """Create professional matrix input section"""
    st.markdown("""
    <div class="section-card">
        <h3 class="section-title">📊 System Configuration</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Preset examples
    presets = {
        2: {"A": [[4, -1], [2, 6]], "b": [3, 2], "name": "2×2 Diagonally Dominant"},
        3: {"A": [[10, -1, 2], [3, 9, -1], [2, -2, 8]], "b": [6, 30, 12], "name": "3×3 Well-Conditioned"},
        4: {"A": [[10, -1, 2, 0], [-1, 11, -1, 3], [2, -1, 10, -1], [0, 3, -1, 8]], 
             "b": [6, 25, -11, 15], "name": "4×4 Sparse Matrix"},
        5: {"A": [[8, -2, 1, 0, 0], [-2, 9, -1, 2, 0], [1, -1, 7, -2, 1], [0, 2, -2, 10, -1], [0, 0, 1, -1, 6]], 
             "b": [5, 3, 4, 2, 1], "name": "5×5 Engineering System"}
    }
    
    default_preset = presets[n]
    st.info(f"📋 **Loaded Preset:** {default_preset['name']}")
    
    return default_preset["A"], default_preset["b"]

def create_results_dashboard(solution, iterations, errors, convergence_rate, num_iter, A_matrix, b_vector, n):
    """Create comprehensive results dashboard"""
    st.markdown("""
    <div class="section-card">
        <h3 class="section-title">📈 Results Dashboard</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{num_iter}</p>
            <p class="metric-label">Iterations</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{errors[-1]:.2e}</p>
            <p class="metric-label">Final Error</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        A_np = np.array(A_matrix)
        cond_num = np.linalg.cond(A_np)
        st.markdown(f"""
        <div class="metric-card">
            <p class="metric-value">{cond_num:.1f}</p>
            <p class="metric-label">Condition #</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        if convergence_rate:
            avg_rate = np.mean(convergence_rate[-5:]) if len(convergence_rate) >= 5 else np.mean(convergence_rate)
            st.markdown(f"""
            <div class="metric-card">
                <p class="metric-value">{avg_rate:.3f}</p>
                <p class="metric-label">Convergence Rate</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Solution display
    st.markdown("### 🎯 Solution Vector")
    sol_cols = st.columns(min(n, 6))
    for i in range(n):
        col_idx = i % len(sol_cols)
        with sol_cols[col_idx]:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        color: white; padding: 1rem; border-radius: 10px; text-align: center; margin: 0.25rem;">
                <strong>x₍{i+1}₎</strong><br>
                <span style="font-size: 1.2rem; font-weight: bold;">{solution[i]:.6f}</span>
            </div>
            """, unsafe_allow_html=True)

def create_advanced_plots(errors, iterations, convergence_rate, n):
    """Create advanced visualization plots"""
    st.markdown("### 📊 Advanced Analytics")
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Convergence Analysis', 'Error Reduction Rate', 
                       'Solution Evolution', 'Convergence Rate'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Convergence plot
    fig.add_trace(
        go.Scatter(x=list(range(1, len(errors) + 1)), y=errors,
                  mode='lines+markers', name='Error',
                  line=dict(color='#667eea', width=3),
                  marker=dict(size=6, color='#764ba2')),
        row=1, col=1
    )
    
    # Error reduction rate
    if len(errors) > 1:
        reduction_rates = [errors[i]/errors[i-1] for i in range(1, len(errors))]
        fig.add_trace(
            go.Bar(x=list(range(2, len(errors) + 1)), y=reduction_rates,
                  name='Reduction Rate', marker_color='#4facfe'),
            row=1, col=2
        )
    
    # Solution evolution (first variable)
    solution_values = [iter_sol[0] for iter_sol in iterations]
    fig.add_trace(
        go.Scatter(x=list(range(1, len(solution_values) + 1)), y=solution_values,
                  mode='lines+markers', name='x₁ Evolution',
                  line=dict(color='#f093fb', width=3),
                  marker=dict(size=6)),
        row=2, col=1
    )
    
    # Convergence rate
    if convergence_rate:
        fig.add_trace(
            go.Scatter(x=list(range(2, len(convergence_rate) + 2)), y=convergence_rate,
                      mode='lines+markers', name='Convergence Rate',
                      line=dict(color='#a8edea', width=3),
                      marker=dict(size=6)),
            row=2, col=2
        )
    
    fig.update_layout(
        height=800,
        showlegend=True,
        template='plotly_white',
        title_text="<b>Comprehensive Analysis Dashboard</b>",
        title_x=0.5
    )
    
    # Update y-axes to log scale for error plots
    fig.update_yaxes(type="log", row=1, col=1)
    
    st.plotly_chart(fig, use_container_width=True)

def main():
    st.set_page_config(
        page_title="Gauss-Seidel Pro Calculator", 
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    apply_custom_css()
    create_header()
    create_explanation_section()
    
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    
    # Main interface
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.markdown("""
        <div class="section-card">
            <h3 class="section-title">⚙️ Configuration Panel</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # System size
        n = st.selectbox("🔢 **System Size:**", options=[2, 3, 4, 5], index=1)
        
        # Get matrix inputs
        default_A, default_b = create_matrix_input_section(n)
        
        # Matrix input with enhanced styling
        st.markdown("#### 📋 Coefficient Matrix (A)")
        st.markdown('<div class="matrix-container">', unsafe_allow_html=True)
        
        A_matrix = []
        for i in range(n):
            row = []
            cols = st.columns(n)
            for j in range(n):
                with cols[j]:
                    val = st.number_input(
                        f"a[{i+1},{j+1}]", 
                        value=float(default_A[i][j]), 
                        key=f"a_{i}_{j}",
                        step=0.1,
                        format="%.2f"
                    )
                    row.append(val)
            A_matrix.append(row)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Constants vector
        st.markdown("#### 📋 Constants Vector (b)")
        st.markdown('<div class="matrix-container">', unsafe_allow_html=True)
        
        b_vector = []
        b_cols = st.columns(min(n, 4))
        for i in range(n):
            col_idx = i % len(b_cols)
            with b_cols[col_idx]:
                val = st.number_input(
                    f"b[{i+1}]", 
                    value=float(default_b[i]), 
                    key=f"b_{i}",
                    step=0.1,
                    format="%.2f"
                )
                b_vector.append(val)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Advanced parameters
        with st.expander("🔧 Advanced Parameters"):
            col_param1, col_param2 = st.columns(2)
            with col_param1:
                max_iter = st.number_input("Max Iterations:", min_value=10, max_value=1000, value=100)
                tolerance = st.selectbox("Tolerance:", 
                                       options=[1e-3, 1e-6, 1e-9, 1e-12], 
                                       index=1, 
                                       format_func=lambda x: f"{x:.0e}")
            with col_param2:
                use_zero_initial = st.toggle("Zero Initial Guess", value=True)
                show_iterations = st.toggle("Show Iteration Details", value=False)
        
        # Solve button with enhanced styling
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚀 **Solve System**", type="primary", use_container_width=True):
            st.session_state.solve_clicked = True
    
    with col2:
        if hasattr(st.session_state, 'solve_clicked') and st.session_state.solve_clicked:
            # Matrix analysis
            is_dd, dominance_ratios = is_diagonally_dominant(A_matrix)
            A_np = np.array(A_matrix)
            
            st.markdown("""
            <div class="section-card">
                <h3 class="section-title">🔍 Matrix Analysis</h3>
            </div>
            """, unsafe_allow_html=True)
            
            if is_dd:
                st.markdown("""
                <div class="success-card">
                    ✅ <strong>Matrix is diagonally dominant!</strong> Convergence is guaranteed.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="warning-card">
                    ⚠️ <strong>Matrix is NOT diagonally dominant.</strong> Convergence not guaranteed.
                </div>
                """, unsafe_allow_html=True)
            
            # Check for computational issues
            zero_diagonal = any(A_matrix[i][i] == 0 for i in range(n))
            
            if zero_diagonal:
                st.markdown("""
                <div class="error-card">
                    ❌ <strong>Error:</strong> Zero diagonal element detected! Cannot proceed.
                </div>
                """, unsafe_allow_html=True)
            else:
                try:
                    # Solve the system
                    x0 = np.zeros(n) if use_zero_initial else None
                    solution, iterations, errors, convergence_rate, num_iter = gauss_seidel(
                        A_matrix, b_vector, x0, max_iter, tolerance
                    )
                    
                    # Create results dashboard
                    create_results_dashboard(solution, iterations, errors, convergence_rate, 
                                           num_iter, A_matrix, b_vector, n)
                    
                    # Verification
                    residual = A_np @ solution - np.array(b_vector)
                    residual_norm = np.linalg.norm(residual)
                    
                    if residual_norm < 1e-6:
                        st.markdown("""
                        <div class="success-card">
                            ✅ <strong>Solution Verified!</strong> Residual norm: {:.2e}
                        </div>
                        """.format(residual_norm), unsafe_allow_html=True)
                    
                    # Advanced visualizations
                    create_advanced_plots(errors, iterations, convergence_rate, n)
                    
                    # Iteration details
                    if show_iterations:
                        st.markdown("### 📋 Iteration History")
                        iter_data = []
                        for i, (iter_sol, error) in enumerate(zip(iterations[-10:], errors[-10:])):
                            row = {"Iteration": len(errors) - 10 + i + 1, "Error": f"{error:.2e}"}
                            for j, val in enumerate(iter_sol):
                                row[f"x₍{j+1}₎"] = f"{val:.6f}"
                            iter_data.append(row)
                        
                        df = pd.DataFrame(iter_data)
                        st.dataframe(df, use_container_width=True, hide_index=True)
                
                except Exception as e:
                    st.markdown(f"""
                    <div class="error-card">
                        ❌ <strong>Calculation Error:</strong> {str(e)}
                    </div>
                    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer with tips
    st.markdown("""
    <div class="section-card" style="margin-top: 2rem;">
        <h3 class="section-title">💡 Pro Tips</h3>
        <div class="feature-grid">
            <div class="feature-card">
                <h4>🎯 Ensure Convergence</h4>
                <p>Rearrange equations to maximize diagonal dominance</p>
            </div>
            <div class="feature-card">
                <h4>⚡ Optimize Performance</h4>
                <p>Use good initial guesses when available</p>
            </div>
            <div class="feature-card">
                <h4>🔍 Monitor Condition</h4>
                <p>Lower condition numbers indicate better stability</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
