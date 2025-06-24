import streamlit as st
import pandas as pd
from pycaret.classification import setup as cls_setup, compare_models as cls_compare, finalize_model as cls_finalize, predict_model as cls_predict, pull as cls_pull
from pycaret.regression import setup as reg_setup, compare_models as reg_compare, finalize_model as reg_finalize, predict_model as reg_predict, pull as reg_pull

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_absolute_error


st.title("🎈 Simple Pycaret App")


password = st.sidebar.text_input("Enter Password",type="password")
if password != "ds4everyone":
    st.sidebar.error('Incorrect Password')
    st.stop()
st.sidebar.success("Access Granted") 


df = pd.read_csv("Life Expectancy Data.csv").sample(n=1000)

target = st.sidebar.selectbox("Select a target variable",df.columns)

features = st.multiselect("Select features",[c for c in df.columns if c != target],default=[c for c in df.columns if c != target] )

if not features:
    st.warning("Please select at least one feature")
    st.stop()


if st.button("Train & Evaluate"):
    model_df = df[features+[target]]
    st.dataframe(model_df.head())

    # --- ADDED THIS LINE TO HANDLE MISSING VALUES IN TARGET ---
    # Ensures no NaN values in the target column before PyCaret setup
    model_df.dropna(subset=[target], inplace=True) 
    # -----------------------------------------------------------

    with st.spinner("Training ..."):
        reg_setup(data=model_df,target=target,session_id=42,html=False)
        
        # Corrected part: Get the best model and handle cases where no model is returned
        best_model_result = reg_compare(sort="R2",n_select=1)

        if isinstance(best_model_result, list) and not best_model_result:
            st.error("No models could be trained or compared successfully by PyCaret. Please check your data and setup configurations.")
            st.stop() # Stop the app if no models are found
        elif best_model_result is None:
             st.error("PyCaret's compare_models returned None. No best model was identified. Please check your data and setup configurations.")
             st.stop()
        
        # Assign the best model to 'best'
        best = best_model_result 

        st.write("Value of 'best' before finalize_model:", best) # Keep this for debugging if needed
        model = reg_finalize(best)
        comparison_df = reg_pull()

    st.success("Training Complete!")


    st.subheader("Model Comparison")
    st.dataframe(comparison_df)


    with st.spinner("Evaluating ... "):
        pred_df = reg_predict(model,model_df)
        actual = pred_df[target]
        predicted = pred_df["Label"] if "Label" in pred_df.columns else pred_df.iloc[:, -1]

        metrics= {}

        metrics["R2"] = r2_score(actual,predicted)
        metrics["MAE"] = mean_absolute_error(actual,predicted) 

    st.success("Evaluation Done!")

    st.subheader("Metrics")

    cols = st.columns(len(metrics))
    for i, (name,val) in enumerate(metrics.items()):
        cols[i].metric(name, f"{val:4f}")
    
    st.subheader("Predictions")
    st.dataframe(pred_df.head(10))