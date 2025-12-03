import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import re
from pathlib import Path   # YANGI

import pyarrow
import pyarrow.parquet
import fastparquet

st.set_page_config(
    page_title="Book Store BI Dashboard",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent


@st.cache_data
def load_and_prepare(folder: str):

    folder_path = BASE_DIR / folder         
    orders_path = folder_path / "orders.parquet"
    users_path  = folder_path / "users.csv"
    books_path  = folder_path / "books.yaml"

    orders = pd.read_parquet(orders_path, engine="fastparquet")
    users = pd.read_csv(users_path)

    with open(books_path, "r") as f:
        books_data = yaml.safe_load(f)
    

    books_data = [{k.lstrip(':') if isinstance(k, str) else k: v for k, v in item.items()} for item in books_data]
    books = pd.json_normalize(books_data)


    def clean_unit_price(value):
        if pd.isna(value):
            return None
        text = str(value)

        if "€" in text or "EUR" in text:
            currency = "EUR"
        else:
            currency = "USD"

        cleaned = re.sub(r"[^0-9,\.]", "", text)
        cleaned = cleaned.replace(",", ".")
        if cleaned.endswith("."):
            cleaned = cleaned[:-1]

        try:
            price = float(cleaned)
        except:
            return None

        if currency == "EUR":
            price *= 1.2 
        return round(price, 2)

    if orders["unit_price"].dtype == "object":
        orders["unit_price"] = (
            orders["unit_price"].apply(clean_unit_price).astype("float64")
        )

    if "paid_price" not in orders.columns:
        orders["paid_price"] = (orders["quantity"] * orders["unit_price"]).round(2)

    orders["timestamp"] = pd.to_datetime(orders["timestamp"], errors="coerce", utc=True)
    orders["timestamp"] = orders["timestamp"].dt.tz_localize(None)
    orders["date"] = orders["timestamp"].dt.normalize()

    daily_revenue = (
        orders.groupby("date", as_index=False)["paid_price"]
              .sum()
              .rename(columns={"paid_price": "daily_revenue"})
    )

    top5_days = (
        daily_revenue.sort_values("daily_revenue", ascending=False)
                     .head(5)
                     .copy()
    )
    top5_days["date_str"] = top5_days["date"].dt.strftime("%Y-%m-%d")


    users["join_name_phone"] = (
        users["name"].str.lower().astype(str) + "|" + users["phone"].astype(str)
    )
    users["join_name_address"] = (
        users["name"].str.lower().astype(str) + "|" + users["address"].astype(str)
    )
    users["join_phone_address"] = (
        users["phone"].astype(str) + "|" + users["address"].astype(str)
    )

    users["user_hash"] = users[
        ["join_name_phone", "join_name_address", "join_phone_address"]
    ].apply(lambda row: min(row), axis=1)

    num_unique_users = users["user_hash"].nunique()


    books["author_list"] = books["author"].str.split(",")
    books["author_list"] = books["author_list"].apply(
        lambda lst: [a.strip() for a in lst]
    )
    books["author_set"] = books["author_list"].apply(
        lambda lst: tuple(sorted(lst))
    )

    num_author_sets = books["author_set"].nunique()


    orders_books = orders.merge(
        books[["id", "author_set"]],
        left_on="book_id",
        right_on="id",
        how="left",
    )

    author_sales = (
        orders_books.groupby("author_set")["quantity"]
                    .sum()
                    .reset_index(name="sold_count")
                    .sort_values("sold_count", ascending=False)
    )

    most_popular_row = author_sales.iloc[0]
    most_popular_authors = ", ".join(most_popular_row["author_set"])
    most_popular_sold_count = int(most_popular_row["sold_count"])


    orders_users = orders.merge(
        users[["id", "user_hash"]],
        left_on="user_id",
        right_on="id",
        how="left",
    )

    spending = (
        orders_users.groupby("user_hash", as_index=False)["paid_price"]
                    .sum()
                    .rename(columns={"paid_price": "total_spending"})
    )

    max_spend = spending["total_spending"].max()
    top_customers = spending[spending["total_spending"] == max_spend]
    top_hashes = top_customers["user_hash"].tolist()

    top_customer_rows = users[users["user_hash"].isin(top_hashes)]
    best_buyer_ids = sorted(top_customer_rows["id"].unique().tolist())

    metrics = {
        "daily_revenue": daily_revenue,
        "top5_days": top5_days,
        "num_unique_users": num_unique_users,
        "num_author_sets": num_author_sets,
        "most_popular_authors": most_popular_authors,
        "most_popular_sold_count": most_popular_sold_count,
        "best_buyer_ids": best_buyer_ids,
        "top_customer_rows": top_customer_rows,
        "total_revenue": float(daily_revenue["daily_revenue"].sum()),
    }

    return orders, users, books, metrics



st.title("Book Store BI Dashboard")

dataset = st.sidebar.selectbox("Select dataset", ["DATA1", "DATA2", "DATA3"])

st.sidebar.write("Current folder:", dataset)

orders, users, books, M = load_and_prepare(dataset)

daily_revenue = M["daily_revenue"]
top5_days = M["top5_days"]

tab1, tab2, tab3 = st.tabs(
    [f"📊 Revenue ({dataset})", f"👤 Users & Authors ({dataset})", f"📈 Daily Chart ({dataset})"]
)

with tab1:
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Revenue (USD)", f"{M['total_revenue']:,.2f}")
    col2.metric("Top Day Revenue (USD)", f"{top5_days['daily_revenue'].max():,.2f}")
    col3.metric("Number of Days", f"{len(daily_revenue):,}")

    st.subheader("Top 5 Days by Revenue (YYYY-MM-dd)")
    st.table(
        top5_days[["date_str", "daily_revenue"]]
        .rename(columns={"date_str": "Date", "daily_revenue": "Revenue (USD)"})
    )

with tab2:
    col1, col2 = st.columns(2)
    col1.metric("Real Unique Users", f"{M['num_unique_users']:,}")
    col2.metric("Unique Author Sets", f"{M['num_author_sets']:,}")

    st.subheader("Most Popular Author Set")
    st.write(f"**Authors:** {M['most_popular_authors']}")
    st.write(f"**Total books sold:** {M['most_popular_sold_count']}")

    st.subheader("Best Buyer (by total spending)")
    st.write(f"**Best buyer id list (aliases):** `{M['best_buyer_ids']}`")

    st.markdown("**Best buyer details:**")
    st.dataframe(
        M["top_customer_rows"][["id", "name", "address", "phone", "email"]]
        .drop_duplicates()
        .sort_values("id"),
        use_container_width=True,
    )

with tab3:
    st.subheader("Daily Revenue Over Time")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(daily_revenue["date"], daily_revenue["daily_revenue"])
    ax.set_xlabel("Date")
    ax.set_ylabel("Daily revenue (USD)")
    ax.set_title(f"Daily Revenue Over Time ({dataset})")
    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig)
@st.cache_data
def load_and_prepare(folder: str):
    base_path = f"d:/Downloads/task_2/{folder}"
    orders_path = f"{base_path}/orders.parquet"
    users_path = f"{base_path}/users.csv"
    books_path = f"{base_path}/books.yaml"    @st.cache_data
    def load_and_prepare(folder: str):
        base_path = f"d:/Downloads/task_2/{folder}"
        orders_path = f"{base_path}/orders.parquet"
        users_path = f"{base_path}/users.csv"
        books_path = f"{base_path}/books.yaml"