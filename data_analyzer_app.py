import io
import re
from typing import Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Country CSV Analyzer", layout="wide")
st.title("📊 Country CSV Analyzer (DE / NL / PL)")

st.markdown(
    """
Upload CSV files containing **DE**, **NL**, or **PL** in the filename.  
The app will detect the country from the filename, prepare a dedicated analysis,  
and allow switching between countries from a dropdown list.
"""
)

# --- Helper functions --------------------------------------------------------

COUNTRY_CODES = {
    "DE": "Germany",
    "NL": "Netherlands",
    "PL": "Poland",
}

def detect_country_from_filename(name: str) -> Optional[str]:
    upper = name.upper()
    for code in COUNTRY_CODES.keys():
        if re.search(rf"(^|[^A-Z]){code}([^A-Z]|$)", upper):
            return code
    return None

def read_csv_safely(file) -> pd.DataFrame:
    content = file.read()
    file.seek(0)
    
    # For files with "DE" in name, assume tab-separated
    if "DE" in file.name.upper():
        try:
            # First try tab separator with standard encoding
            df = pd.read_csv(file, sep='\t', encoding='utf-8')
            file.seek(0)
            return df
        except:
            try:
                # Try with different encodings
                file.seek(0)
                df = pd.read_csv(file, sep='\t', encoding='latin-1')
                file.seek(0)
                return df
            except:
                # Final fallback
                file.seek(0)
                return pd.read_csv(file, sep='\t', engine="python", encoding_errors="ignore")
    
    # For other files, use the original logic
    sample = content[:2048].decode(errors="ignore")
    sep_candidates = [",", ";", "\t", "|"]
    best_sep = ","
    best_hits = 0
    for sep in sep_candidates:
        hits = sample.count(sep)
        if hits > best_hits:
            best_hits = hits
            best_sep = sep
    
    for enc in ["utf-8", "utf-8-sig", "latin-1"]:
        file.seek(0)
        try:
            df = pd.read_csv(file, sep=best_sep, encoding=enc)
            return df
        except Exception:
            continue
    
    return pd.read_csv(io.BytesIO(content), sep=best_sep, engine="python", encoding_errors="ignore")

def to_number(s):
    if pd.isna(s):
        return np.nan
    if isinstance(s, (int, float, np.number)):
        return float(s)
    s = str(s).strip().replace("€", "").replace(" ", "").replace("\xa0", "").replace(",", "").replace("%", "")
    try:
        return float(s)
    except Exception:
        return np.nan

def format_eur(x):
    if pd.isna(x):
        return ""
    return f"€{x:,.2f}"

def format_pct(x):
    if pd.isna(x):
        return ""
    return f"{x:.2f}%"

def style_expected_colors(df: pd.DataFrame):
    def color_row(row):
        val = row.get("% Expected Demand", np.nan)
        color = ""
        if pd.notna(val):
            color = "background-color: rgba(0, 170, 0, 0.15);" if val >= 0 else "background-color: rgba(220, 20, 60, 0.15);"
        return [color if c == "Expected Demand" else "" for c in df.columns]
    return df.style.apply(color_row, axis=1)

def ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    # First, clean column names
    df = df.rename(columns={c: c.strip() for c in df.columns})
    
    # Convert all numeric columns
    numeric_columns = ["Visits", "Orders", "Demand", "Expected Demand", "Demand Diff to Expected", 
                       "CVR", "AOV", "Median Order Value", "BA", "Position First Day", 
                       "Position All", "% Expected Demand", "CVR Category Avg Diff", 
                       "CVR Channel Avg Diff", "CVR Category by Channel Avg Diff"]
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = df[col].apply(to_number)
    
    # Calculate missing columns if needed
    if "% Expected Demand" not in df.columns and {"Demand", "Expected Demand"}.issubset(df.columns):
        with np.errstate(divide="ignore", invalid="ignore"):
            df["% Expected Demand"] = (df["Demand"] - df["Expected Demand"]) / df["Expected Demand"] * 100
    
    if "CVR" not in df.columns and {"Orders", "Visits"}.issubset(df.columns):
        with np.errstate(divide="ignore", invalid="ignore"):
            df["CVR"] = (df["Orders"] / df["Visits"]) * 100
    
    if {"Demand", "Expected Demand"}.issubset(df.columns) and "Demand Diff to Expected" not in df.columns:
        df["Demand Diff to Expected"] = df["Demand"] - df["Expected Demand"]
    
    if "Demand" in df.columns:
        df = df.sort_values("Demand", ascending=False).reset_index(drop=True)
    
    return df

def styled_table(df: pd.DataFrame):
    styler = style_expected_colors(df)
    currency_cols = [c for c in ["Demand", "Demand Diff to Expected", "Expected Demand", "AOV", "Median Order Value"] if c in df.columns]
    percent_cols = [c for c in ["% Expected Demand", "CVR", "BA"] if c in df.columns]
    
    if currency_cols:
        styler = styler.format({c: format_eur for c in currency_cols})
    if percent_cols:
        styler = styler.format({c: format_pct for c in percent_cols})
    
    styler = styler.set_properties(**{"text-align": "right"})
    return styler

def add_summary_row(df: pd.DataFrame) -> pd.DataFrame:
    row = {c: "" for c in df.columns}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if col in df.columns:
            row[col] = df[col].mean(skipna=True)
    
    if df.columns[0] not in ["", None]:
        row[df.columns[0]] = "Average"
    
    return pd.DataFrame([row], columns=df.columns)

def pie_not_achieved(df: pd.DataFrame):
    if "% Expected Demand" not in df.columns:
        st.info("Missing column '% Expected Demand' – cannot calculate banner achievement.")
        return
    below = (df["% Expected Demand"] < 0).sum()
    at_or_above = (df["% Expected Demand"] >= 0).sum()
    labels = ["Did not reach plan", "Reached/Exceeded plan"]
    sizes = [below, at_or_above]
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
    ax.axis("equal")
    st.pyplot(fig)

def pie_demand_vs_expected(df: pd.DataFrame):
    if not {"Demand", "Expected Demand"}.issubset(df.columns):
        st.info("Missing 'Demand' or 'Expected Demand' columns – cannot show demand vs expected share.")
        return
    total_demand = df["Demand"].sum(skipna=True)
    total_expected = df["Expected Demand"].sum(skipna=True)
    labels = ["Total Demand", "Total Expected Demand"]
    sizes = [total_demand, total_expected]
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
    ax.axis("equal")
    st.pyplot(fig)

# --- File upload & country split --------------------------------------------

uploaded_files = st.file_uploader(
    "Upload one or more CSV files", type=["csv"], accept_multiple_files=True
)

countries_data: Dict[str, list] = {"DE": [], "NL": [], "PL": []}

if uploaded_files:
    for f in uploaded_files:
        code = detect_country_from_filename(f.name)
        if code is None:
            st.warning(f"⚠️ Skipped file **{f.name}** – missing country code (DE/NL/PL) in the filename.")
            continue
        try:
            df = read_csv_safely(f)
            st.write(f"File: {f.name}")
            st.write(f"Columns detected: {list(df.columns)}")
            st.write(f"First few rows:")
            st.dataframe(df.head(3))
            
            df = ensure_required_columns(df)
            countries_data[code].append(df)
        except Exception as e:
            st.error(f"Could not load file **{f.name}**: {e}")
            st.error(f"Error details: {str(e)}")

available_codes = [code for code, lst in countries_data.items() if len(lst) > 0]
if not available_codes:
    st.info("No data uploaded yet. Upload CSV files with a country code in the filename, e.g. `banners_DE.csv`.")
    st.stop()

# Rest of the code remains the same...
