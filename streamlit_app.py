import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# -------------------- CUSTOM CSS --------------------
def apply_custom_css():
    st.markdown("""
    <style>
    /* Global font */
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    /* Title */
    h1, h2, h3 {
        color: #4facfe;
        font-weight: 700;
    }

    /* Button styling */
    div.stButton > button:first-child {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
    }
    div.stButton > button:first-child:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6) !important;
    }

    /* Number Input Styling */
    div[data-baseweb="input"] > input {
        border-radius: 8px !important;
        border: 2px solid #764ba2 !important;
        font-weight: 500 !important;
        padding: 0.4rem !important;
    }

    /* Selectbox Styling */
    div[data-baseweb="select"] > div {
        border-radius: 8px !important;
        border: 2px solid #764ba2 !important;
        font-weight: 500 !important;
    }

    /* Expander Styling */
    div.streamlit-expanderHeader {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%) !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
    }
    </style>
    """, unsafe_allow_html=True)


# -------------------- GAUSS-SEIDEL FUNCTION --------------------
def gauss_seidel(A, b, x0=None, tol=1e-6, max_iter=100):
    n = len(b)
    if x0 is None:
        x0 = np.zeros_like(b, dtype=float)
    
    x = x0.copy()
    history = [x.copy()]
    
    for k in range(max_iter):
        x_new = np.copy(x)
        for i in range(n):
            s1 = np.dot(A[i, :i], x_new[:i])
            s2 = np.dot(A[i, i+1:], x[i+1:])
            x_new[i] = (b[i] - s1 - s2) / A[i, i]
        
        history.append(x_new.copy())
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            break
        x = x_new
    
    return x, np.array(history)


# -------------------- STREAMLIT APP --------------------
def main():
    apply_custom_css()
    st.title("🔢 Gauss–Seidel Method Solver")
    st.write("Solve a system of linear equations using the **Gauss–Seidel Iterative Method**.")

    # Matrix size input
    n = st.number_input("Enter the size of the system (n x n)", min_value=2, max_value=6, value=3)

    st.subheader("Enter Coefficient Matrix (A) and RHS Vector (b)")
    A = np.zeros((n, n))
    b = np.zeros(n)

    cols = st.columns(n + 1)
    for i in range(n):
        for j in range(n):
            A[i, j] = cols[j].number_input(f"A[{i+1},{j+1}]", value=1.0, key=f"A{i}{j}")
        b[i] = cols[-1].number_input(f"b[{i+1}]", value=1.0, key=f"b{i}")

    tol = st.number_input("Tolerance (ε)", value=1e-6, format="%.1e")
    max_iter = st.number_input("Maximum Iterations", min_value=1, value=25)

    if st.button("Solve"):
        try:
            solution, history = gauss_seidel(A, b, tol=tol, max_iter=max_iter)

            st.success("✅ Solution Found!")
            st.write("Final Solution Vector:")
            st.write(solution)

            # Iteration Table
            df = pd.DataFrame(history, columns=[f"x{i+1}" for i in range(n)])
            st.subheader("Iteration Progress")
            st.dataframe(df)

            # Plot Convergence
            fig = go.Figure()
            for i in range(n):
                fig.add_trace(go.Scatter(
                    y=df[f"x{i+1}"], mode="lines+markers", name=f"x{i+1}"
                ))
            fig.update_layout(title="Convergence of Variables", xaxis_title="Iteration", yaxis_title="Value")
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
