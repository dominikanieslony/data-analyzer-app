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

def manual_csv_parsing(content: str, encoding: str = 'utf-8') -> pd.DataFrame:
    """Manual fallback parsing for problematic CSV files"""
    lines = content.strip().split('\n')
    headers = lines[0].strip().split('\t')
    
    data = []
    for line in lines[1:]:
        if line.strip():
            row = line.strip().split('\t')
            data.append(row)
    
    df = pd.DataFrame(data, columns=headers)
    return df

def detect_encoding(content_bytes: bytes) -> str:
    """Detect the encoding of the file content"""
    encodings = ['utf-8-sig', 'utf-16', 'latin-1', 'iso-8859-1', 'windows-1252', 'utf-8']
    
    for encoding in encodings:
        try:
            content_bytes.decode(encoding)
            return encoding
        except UnicodeDecodeError:
            continue
    
    return 'utf-8'  # fallback

def read_csv_safely(file) -> pd.DataFrame:
    # Read the file as bytes first to detect encoding
    content_bytes = file.getvalue()
    file.seek(0)
    
    # Detect encoding
    encoding = detect_encoding(content_bytes)
    st.write(f"Detected encoding: {encoding}")
    
    # Decode content with correct encoding
    try:
        content = content_bytes.decode(encoding)
    except UnicodeDecodeError:
        # Fallback to latin-1 which rarely fails
        content = content_bytes.decode('latin-1', errors='ignore')
    
    # For files with DE in name, use tab separator with European number format
    if "DE" in file.name.upper():
        try:
            # Read with tab separator and European number formatting
            file.seek(0)
            df = pd.read_csv(
                io.BytesIO(content_bytes), 
                sep='\t', 
                thousands='.', 
                decimal=',',
                encoding=encoding
            )
            return df
        except Exception as e:
            st.error(f"Error reading DE file with pandas: {e}")
            # Fallback: manual parsing
            return manual_csv_parsing(content, encoding)
    else:
        # For other countries, use original logic
        try:
            file.seek(0)
            sample = content[:2048]
            sep_candidates = [",", ";", "\t", "|"]
            best_sep = ","
            best_hits = 0
            for sep in sep_candidates:
                hits = sample.count(sep)
                if hits > best_hits:
                    best_hits = hits
                    best_sep = sep
            
            file.seek(0)
            df = pd.read_csv(io.BytesIO(content_bytes), sep=best_sep, encoding=encoding)
            return df
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return manual_csv_parsing(content, encoding)

def to_number(s):
    if pd.isna(s):
        return np.nan
    if isinstance(s, (int, float, np.number)):
        return float(s)
    
    s = str(s).strip()
    
    # Remove currency symbols, spaces, and percentage signs
    s = s.replace("€", "").replace(" ", "").replace("\xa0", "").replace("%", "")
    
    # Handle European number format
    if "." in s and "," in s:
        # Format like 20.265,50 -> 20265.50
        s = s.replace(".", "").replace(",", ".")
    elif "." in s:
        # Thousand separator like 20.265 -> 20265
        s = s.replace(".", "")
    elif "," in s:
        # Decimal separator like 3,25 -> 3.25
        s = s.replace(",", ".")
    
    try:
        return float(s)
    except Exception:
        return np.nan

def format_eur(x):
    if pd.isna(x):
        return ""
    # Format as European style: €20.265,00
    return f"€{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def format_pct(x):
    if pd.isna(x):
        return ""
    return f"{x:.2f}%".replace(".", ",")

def format_float(x):
    if pd.isna(x):
        return ""
    return f"{x:.2f}".replace(".", ",")

def style_expected_colors(df: pd.DataFrame):
    def color_row(row):
        val = row.get("% Expected Demand", np.nan)
        color = ""
        if pd.notna(val):
            color = "background-color: rgba(0, 170, 0, 0.15);" if val >= 0 else "background-color: rgba(220, 20, 60, 0.15);"
        return [color if c == "Expected Demand" else "" for c in df.columns]
    return df.style.apply(color_row, axis=1)

def ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Clean column names
    df = df.rename(columns={c: c.strip() for c in df.columns})
    
    # Convert all potential numeric columns
    numeric_columns = [
        "Visits", "Orders", "Demand", "Expected Demand", "Demand Diff to Expected",
        "CVR", "AOV", "Median Order Value", "BA", "Position First Day", 
        "Position All", "% Expected Demand", "CVR Category Avg Diff", 
        "CVR Channel Avg Diff", "CVR Category by Channel Avg Diff"
    ]
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = df[col].apply(to_number)
    
    # Calculate missing columns
    if "% Expected Demand" not in df.columns and {"Demand", "Expected Demand"}.issubset(df.columns):
        with np.errstate(divide="ignore", invalid="ignore"):
            df["% Expected Demand"] = (df["Demand"] - df["Expected Demand"]) / df["Expected Demand"] * 100
    
    if "CVR" not in df.columns and {"Orders", "Visits"}.issubset(df.columns):
        with np.errstate(divide="ignore", invalid="ignore"):
            df["CVR"] = (df["Orders"] / df["Visits"]) * 100
    
    if {"Demand", "Expected Demand"}.issubset(df.columns) and "Demand Diff to Expected" not in df.columns:
        df["Demand Diff to Expected"] = df["Demand"] - df["Expected Demand"]
    
    # Ensure Demand Diff to Expected has correct sign based on % Expected Demand
    if "Demand Diff to Expected" in df.columns and "% Expected Demand" in df.columns:
        # If % Expected Demand is negative, make Demand Diff to Expected negative
        df["Demand Diff to Expected"] = np.where(
            df["% Expected Demand"] < 0, 
            -abs(df["Demand Diff to Expected"]), 
            abs(df["Demand Diff to Expected"])
        )
    
    # Round all numeric columns to 2 decimal places
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].round(2)
    
    if "Demand" in df.columns:
        df = df.sort_values("Demand", ascending=False).reset_index(drop=True)
    
    return df

def styled_table(df: pd.DataFrame):
    styler = style_expected_colors(df)
    
    # Currency columns (Euro format)
    currency_cols = [c for c in ["Demand", "Median Order Value", "Expected Demand", "Demand Diff to Expected", "AOV"] if c in df.columns]
    
    # Percentage columns
    percent_cols = [c for c in ["% Expected Demand", "CVR", "BA", "CVR Category Avg Diff", "CVR Channel Avg Diff", "CVR Category by Channel Avg Diff"] if c in df.columns]
    
    # Other numeric columns (2 decimal places)
    other_numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns 
                         if c not in currency_cols and c not in percent_cols]
    
    if currency_cols:
        styler = styler.format({c: format_eur for c in currency_cols})
    if percent_cols:
        styler = styler.format({c: format_pct for c in percent_cols})
    if other_numeric_cols:
        styler = styler.format({c: format_float for c in other_numeric_cols})
    
    styler = styler.set_properties(**{"text-align": "right"})
    return styler

def add_summary_row(df: pd.DataFrame) -> pd.DataFrame:
    row = {c: "" for c in df.columns}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if col in df.columns:
            row[col] = df[col].mean(skipna=True)
    
    # Round summary values to 2 decimal places
    for col in numeric_cols:
        if col in row and pd.notna(row[col]):
            row[col] = round(row[col], 2)
    
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
            st.write(f"### Processing file: {f.name}")
            
            df = read_csv_safely(f)
            st.write("**Columns detected:**", list(df.columns))
            st.write("**First 2 rows:**")
            st.dataframe(df.head(2))
            
            df = ensure_required_columns(df)
            countries_data[code].append(df)
            
        except Exception as e:
            st.error(f"Could not load file **{f.name}**: {str(e)}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")

available_codes = [code for code, lst in countries_data.items() if len(lst) > 0]
if not available_codes:
    st.info("No data uploaded yet. Upload CSV files with a country code in the filename, e.g. `banners_DE.csv`.")
    st.stop()

# Country selector
code_to_show = st.selectbox(
    "Select country for analysis",
    options=available_codes,
    format_func=lambda c: f"{COUNTRY_CODES.get(c, c)} ({c})",
)

dfs = countries_data.get(code_to_show, [])
if not dfs:
    st.warning("No data for the selected country.")
    st.stop()

df_country = pd.concat(dfs, ignore_index=True)

st.subheader(f"Data – {COUNTRY_CODES.get(code_to_show, code_to_show)} ({code_to_show})")

styler = styled_table(df_country.copy())
st.dataframe(styler, use_container_width=True, height=480)

summary_df = add_summary_row(df_country.copy())
summary_styler = styled_table(summary_df.copy())
st.dataframe(summary_styler, use_container_width=True, height=70)

st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Share of banners that did not reach the plan")
    pie_not_achieved(df_country)
with col2:
    st.subheader("Share: Demand vs Expected Demand (total)")
    pie_demand_vs_expected(df_country)

st.caption("Note: colors in **Expected Demand** reflect the sign of **% Expected Demand** (green = positive, red = negative).")
