import html
import os
import re
from datetime import date, timedelta
from typing import Dict, List, Tuple

import pandas as pd
import psycopg2
from psycopg2 import sql
import streamlit as st
from openai import OpenAI


POSTGRES_CONFIG = {
    "POSTGRES_HOST": os.getenv("POSTGRES_HOST", "34.245.131.114"),
    "POSTGRES_PORT": os.getenv("POSTGRES_PORT", "5432"),
    "POSTGRES_DATABASE": os.getenv("POSTGRES_DATABASE", "oiai_database_dev"),
    "POSTGRES_USER": os.getenv("POSTGRES_USER", "dev"),
    "POSTGRES_PASSWORD": os.getenv("POSTGRES_PASSWORD", "*&jkahg345jk"),
}


TABLE_GROUPS: Dict[str, List[str]] = {
    "Students": [
        "public.user",
        "public.org_user",
        "public.enrollments",
        "public.user_role",
        "public.user_activity",
        "public.user_event",
        "public.user_ui_settings",
        "public.user_avatar_configuration",
        "public.daily_streak",
        "public.daily_challenge",
        "public.user_challenge",
        "public.activity_performance",
        "public.daily_activity_log",
        "public.lesson_joined",
    ],
    "Courses & Content": [
        "public.template_courses",
        "public.organization_courses",
        "public.topics",
        "public.lesson_sections",
        "public.section_contents",
        "public.section_content_embeddings",
        "public.mcqs",
        "public.mcq_options",
        "public.open_questions",
        "public.course_groups",
        "public.course_group_assignments",
        "public.course_group_topics",
        "public.course_group_templates",
    ],
    "Sessions": [
        "public.topic_session",
        "public.chapter_session",
    ],
    "Communication": [
        "public.ai_chat",
        "public.chat_sessions",
        "public.chat_history",
        "public.faqs",
        "public.documents",
    ],
    "Certificates": [
        "public.certificate_templates",
        "public.issued_certificates",
        "public.bulk_certificate_download_jobs",
    ],
    "Store & Rewards": [
        "public.item_categories",
        "public.item_catalog",
        "public.coupons",
        "public.user_coupons",
        "public.point",
    ],
    "Organization & Admin": [
        "public.org",
        "public.role",
        "public.sponsors",
        "public.student_import_jobs",
        "public.avatar_master",
    ],
    "System": [
        "public.migration",
        "public.migrations",
        "public.typeorm_metadata",
    ],
}




def get_connection():
    return psycopg2.connect(
        host=POSTGRES_CONFIG["POSTGRES_HOST"],
        port=POSTGRES_CONFIG["POSTGRES_PORT"],
        dbname=POSTGRES_CONFIG["POSTGRES_DATABASE"],
        user=POSTGRES_CONFIG["POSTGRES_USER"],
        password=POSTGRES_CONFIG["POSTGRES_PASSWORD"],
    )


def apply_ui_theme():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@400;600;700&family=Space+Grotesk:wght@400;500;600&display=swap');

        :root {
            --bg: #f5f5f2;
            --bg-accent: #efe7dd;
            --card: #ffffff;
            --ink: #1a1a1a;
            --muted: #4f5a55;
            --accent: #0b6e6b;
            --accent-strong: #05504d;
            --accent-warm: #d6841f;
            --border: #ddd6cc;
            --shadow: 0 14px 30px rgba(11, 110, 107, 0.14);
        }

        html, body, [class*="css"]  {
            font-family: 'Space Grotesk', sans-serif;
            color: var(--ink);
        }

        .stApp {
            background: radial-gradient(900px 600px at 10% -10%, #ffffff 0%, var(--bg) 55%, #efe9dc 100%);
            color: var(--ink);
        }

        [data-testid="stAppViewContainer"] * {
            color: var(--ink);
        }

        [data-testid="stSidebar"] * {
            color: #f7f4ee;
        }

        [data-testid="stSidebar"] .stTextInput input,
        [data-testid="stSidebar"] .stTextArea textarea,
        [data-testid="stSidebar"] .stSelectbox select,
        [data-testid="stSidebar"] .stNumberInput input,
        [data-testid="stSidebar"] .stDateInput input {
            color: var(--ink) !important;
            background: #ffffff !important;
        }

        h1, h2, h3, h4, h5, h6, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            font-family: 'Fraunces', serif;
            letter-spacing: -0.3px;
        }

        .stSidebar > div {
            background: linear-gradient(180deg, #123532 0%, #18413d 55%, #1f4f4a 100%);
            color: #f7f4ee;
        }

        .stSidebar .stMarkdown, .stSidebar label, .stSidebar span, .stSidebar p {
            color: #f7f4ee;
        }

        .stButton > button {
            background: var(--accent);
            color: #ffffff;
            border: none;
            border-radius: 999px;
            padding: 0.45rem 1.2rem;
            font-weight: 600;
            box-shadow: 0 8px 18px rgba(31, 122, 110, 0.2);
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }

        .stButton > button * {
            color: #ffffff !important;
        }

        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 10px 22px rgba(31, 122, 110, 0.3);
            background: var(--accent-strong);
        }

        .hero {
            background: linear-gradient(135deg, #ffffff 0%, #f6eee0 55%, #e9dcc4 100%);
            border: 1px solid var(--border);
            border-radius: 24px;
            padding: 2rem 2.2rem;
            box-shadow: var(--shadow);
            position: relative;
            overflow: hidden;
        }

        .hero::after {
            content: "";
            position: absolute;
            width: 240px;
            height: 240px;
            right: -80px;
            top: -80px;
            background: radial-gradient(circle, rgba(224, 159, 62, 0.35) 0%, rgba(224, 159, 62, 0) 70%);
        }

        .hero-title {
            font-size: 2.4rem;
            margin: 0;
        }

        .hero-subtitle {
            margin-top: 0.6rem;
            color: var(--muted);
            font-size: 1.05rem;
            max-width: 680px;
        }

        .chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.6rem;
            margin-top: 1.2rem;
        }

        .chip {
            padding: 0.35rem 0.9rem;
            border-radius: 999px;
            border: 1px solid var(--border);
            background: #fff8ee;
            font-size: 0.9rem;
            font-weight: 600;
            color: var(--accent-strong);
        }

        .section-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 1.1rem;
            margin-top: 1.5rem;
        }

        .section-card {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 1.2rem 1.3rem;
            box-shadow: 0 12px 24px rgba(27, 43, 36, 0.08);
        }

        .section-card h4 {
            margin: 0 0 0.6rem 0;
            font-size: 1.1rem;
        }

        .section-card p {
            margin: 0.3rem 0 0 0;
            color: var(--muted);
            font-size: 0.95rem;
        }

        .section-pill {
            display: inline-block;
            background: rgba(11, 110, 107, 0.14);
            color: var(--accent-strong);
            padding: 0.25rem 0.6rem;
            border-radius: 999px;
            font-size: 0.82rem;
            font-weight: 600;
        }

        .stTextInput input, .stTextArea textarea, .stSelectbox select, .stNumberInput input, .stDateInput input {
            color: var(--ink) !important;
            background: #ffffff !important;
        }

        .stTextInput label, .stTextArea label, .stSelectbox label, .stNumberInput label, .stDateInput label {
            color: var(--ink) !important;
        }

        /* Streamlit selectbox/combobox (BaseWeb) */
        .stSelectbox div[data-baseweb="select"] > div,
        .stSelectbox div[data-baseweb="select"] > div > div {
            background: #ffffff !important;
            color: var(--ink) !important;
            border-color: var(--border) !important;
        }

        .stSelectbox div[data-baseweb="select"] svg {
            fill: var(--ink) !important;
        }

        .stSelectbox div[data-baseweb="menu"] {
            background: #ffffff !important;
        }

        .stSelectbox div[data-baseweb="menu"] * {
            color: var(--ink) !important;
        }

        .stDataFrame, .stTable {
            color: var(--ink);
        }

        .profile-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }

        .profile-card {
            background: #ffffff;
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1.1rem 1.2rem;
            box-shadow: 0 10px 20px rgba(27, 43, 36, 0.08);
        }

        .profile-card h4 {
            margin: 0 0 0.8rem 0;
            font-size: 1.05rem;
        }

        .profile-item {
            display: flex;
            justify-content: space-between;
            gap: 0.8rem;
            padding: 0.35rem 0;
            border-bottom: 1px dashed #e7e0d6;
        }

        .profile-item:last-child {
            border-bottom: none;
        }

        .profile-label {
            color: var(--muted);
            font-size: 0.85rem;
            min-width: 40%;
        }

        .profile-value {
            font-weight: 600;
            text-align: right;
            color: var(--ink);
        }

        .badge {
            display: inline-block;
            padding: 0.18rem 0.5rem;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 700;
        }

        .badge-yes {
            background: rgba(11, 110, 107, 0.15);
            color: var(--accent-strong);
        }

        .badge-no {
            background: rgba(222, 90, 72, 0.18);
            color: #8b2e24;
        }

        .muted {
            color: var(--muted);
        }

        .fade-up {
            animation: fadeUp 0.5s ease-out both;
        }

        @keyframes fadeUp {
            from { opacity: 0; transform: translateY(12px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .stDataFrame {
            border-radius: 12px;
            overflow: hidden;
            border: 1px solid var(--border);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def read_table_list_from_txt(file_path: str = "all_tables.txt") -> List[str]:
    if not os.path.exists(file_path):
        return []

    tables: List[str] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("TABLE: "):
                tables.append(line.strip().split("TABLE: ", 1)[1])
    return tables


def list_tables(conn) -> List[str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT table_schema, table_name
            FROM information_schema.tables
            WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
            ORDER BY table_schema, table_name;
            """
        )
        return [f"{schema}.{table}" for schema, table in cur.fetchall()]


def split_table_name(full_name: str) -> Tuple[str, str]:
    if "." in full_name:
        schema, table = full_name.split(".", 1)
        return schema, table
    return "public", full_name


def get_columns(conn, schema: str, table: str) -> List[Dict[str, str]]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name, data_type, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s
            ORDER BY ordinal_position;
            """,
            (schema, table),
        )
        rows = cur.fetchall()

    return [
        {
            "column_name": row[0],
            "data_type": row[1],
            "is_nullable": row[2],
            "column_default": row[3],
        }
        for row in rows
    ]


def get_column_names(conn, schema: str, table: str) -> List[str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = %s AND table_name = %s
            ORDER BY ordinal_position;
            """,
            (schema, table),
        )
        return [row[0] for row in cur.fetchall()]


def table_exists(conn, schema: str, table: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT EXISTS(
                SELECT 1
                FROM information_schema.tables
                WHERE table_schema = %s AND table_name = %s
            );
            """,
            (schema, table),
        )
        return bool(cur.fetchone()[0])


def column_exists(conn, schema: str, table: str, column: str) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT EXISTS(
                SELECT 1
                FROM information_schema.columns
                WHERE table_schema = %s AND table_name = %s AND column_name = %s
            );
            """,
            (schema, table, column),
        )
        return bool(cur.fetchone()[0])


def resolve_first_existing(columns: List[str], candidates: List[str]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def get_primary_keys(conn, schema: str, table: str) -> List[str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT kcu.column_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage kcu
              ON tc.constraint_name = kcu.constraint_name
             AND tc.table_schema = kcu.table_schema
            WHERE tc.constraint_type = 'PRIMARY KEY'
              AND tc.table_schema = %s
              AND tc.table_name = %s
            ORDER BY kcu.ordinal_position;
            """,
            (schema, table),
        )
        return [row[0] for row in cur.fetchall()]


def fetch_rows(
    conn,
    schema: str,
    table: str,
    limit: int,
    offset: int,
    filter_col: str | None,
    filter_op: str | None,
    filter_val: str | None,
) -> pd.DataFrame:
    base = sql.SQL("SELECT * FROM {}.{}").format(
        sql.Identifier(schema),
        sql.Identifier(table),
    )

    where_sql = sql.SQL("")
    params: List[object] = []

    if filter_col:
        col_ident = sql.Identifier(filter_col)
        if filter_op == "contains":
            where_sql = sql.SQL(" WHERE {}::text ILIKE %s").format(col_ident)
            params.append(f"%{filter_val}%")
        elif filter_op == "equals":
            where_sql = sql.SQL(" WHERE {}::text = %s").format(col_ident)
            params.append(str(filter_val))
        elif filter_op == "is null":
            where_sql = sql.SQL(" WHERE {} IS NULL").format(col_ident)
        elif filter_op == "is not null":
            where_sql = sql.SQL(" WHERE {} IS NOT NULL").format(col_ident)

    query = base + where_sql + sql.SQL(" LIMIT %s OFFSET %s")
    params.extend([int(limit), int(offset)])

    with conn.cursor() as cur:
        cur.execute(query, params)
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]

    return pd.DataFrame(rows, columns=columns)


def count_rows(
    conn,
    schema: str,
    table: str,
    filter_col: str | None,
    filter_op: str | None,
    filter_val: str | None,
) -> int:
    base = sql.SQL("SELECT COUNT(*) FROM {}.{}").format(
        sql.Identifier(schema),
        sql.Identifier(table),
    )

    where_sql = sql.SQL("")
    params: List[object] = []

    if filter_col:
        col_ident = sql.Identifier(filter_col)
        if filter_op == "contains":
            where_sql = sql.SQL(" WHERE {}::text ILIKE %s").format(col_ident)
            params.append(f"%{filter_val}%")
        elif filter_op == "equals":
            where_sql = sql.SQL(" WHERE {}::text = %s").format(col_ident)
            params.append(str(filter_val))
        elif filter_op == "is null":
            where_sql = sql.SQL(" WHERE {} IS NULL").format(col_ident)
        elif filter_op == "is not null":
            where_sql = sql.SQL(" WHERE {} IS NOT NULL").format(col_ident)

    query = base + where_sql
    with conn.cursor() as cur:
        cur.execute(query, params)
        return int(cur.fetchone()[0])


def build_grouped_tables(all_tables: List[str]) -> Dict[str, List[str]]:
    grouped = {group: [] for group in TABLE_GROUPS}
    assigned = set()

    for group, tables in TABLE_GROUPS.items():
        for table in tables:
            if table in all_tables:
                grouped[group].append(table)
                assigned.add(table)

    remaining = sorted([t for t in all_tables if t not in assigned])
    if remaining:
        grouped["Other Tables"] = remaining

    return grouped


def display_table_name(full_name: str) -> str:
    return full_name.replace("public.", "")


def render_hero(total_tables: int, missing_tables: int, group_count: int):
    st.markdown(
        f"""
        <div class="hero fade-up">
            <h1 class="hero-title">School Management System</h1>
            <p class="hero-subtitle">
                A unified workspace for students, curriculum, sessions, certificates,
                and rewards. Built directly on your PostgreSQL tables.
            </p>
            <div class="chip-row">
                <span class="chip">{total_tables} tables connected</span>
                <span class="chip">{group_count} modules mapped</span>
                <span class="chip">{missing_tables} missing tables</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )




def format_metric_value(value, precision: int = 0) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:,.{precision}f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def format_card_value(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    if isinstance(value, float):
        return f"{value:,.1f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def format_profile_value(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return '<span class="muted">—</span>'
    if isinstance(value, bool):
        klass = "badge-yes" if value else "badge-no"
        label = "Yes" if value else "No"
        return f'<span class="badge {klass}">{label}</span>'
    return html.escape(str(value))


def build_profile_sections(user_data: Dict[str, object]) -> List[Tuple[str, List[Tuple[str, object]]]]:
    sections: List[Tuple[str, List[Tuple[str, object]]]] = []

    field_groups = [
        (
            "Account",
            [
                ("User ID", "id"),
                ("Name", "name"),
                ("Email", "email"),
                ("Grade Level", "grade_level"),
            ],
        ),
    ]

    for title, fields in field_groups:
        items = []
        for label, key in fields:
            if key in user_data:
                items.append((label, user_data.get(key)))
        if items:
            sections.append((title, items))

    return sections


def render_profile_cards(sections: List[Tuple[str, List[Tuple[str, object]]]]):
    cards: List[str] = []
    for title, items in sections:
        item_html = "".join(
            [
                (
                    f'<div class="profile-item">'
                    f'<span class="profile-label">{html.escape(label)}</span>'
                    f'<span class="profile-value">{format_profile_value(value)}</span>'
                    f"</div>"
                )
                for label, value in items
            ]
        )
        cards.append(
            f"""
            <div class="profile-card fade-up">
                <h4>{html.escape(title)}</h4>
                {item_html}
            </div>
            """
        )

    if cards:
        st.markdown(
            f'<div class="profile-grid">{"".join(cards)}</div>',
            unsafe_allow_html=True,
        )


def render_student_result_cards(df: pd.DataFrame):
    if df.empty:
        st.info("No students matched the query.")
        return

    base_cols = {"id", "name", "email", "grade_level", "created_at"}
    metric_cols = [col for col in df.columns if col not in base_cols]

    cards: List[str] = []
    for _, row in df.iterrows():
        user_id = row.get("id")
        name = row.get("name") or f"User {user_id}"
        email = row.get("email") or "No email"
        grade = row.get("grade_level") or "—"
        created_at = row.get("created_at") or "—"

        metric_items = []
        for col in metric_cols:
            metric_items.append(
                f'<div class="profile-item">'
                f'<span class="profile-label">{html.escape(col.replace("_", " ").title())}</span>'
                f'<span class="profile-value">{html.escape(format_card_value(row.get(col)))}</span>'
                f"</div>"
            )

        card_html = f"""
            <div class="profile-card fade-up">
                <h4>{html.escape(str(name))}</h4>
                <div class="profile-item">
                    <span class="profile-label">User ID</span>
                    <span class="profile-value">{html.escape(format_card_value(user_id))}</span>
                </div>
                <div class="profile-item">
                    <span class="profile-label">Email</span>
                    <span class="profile-value">{html.escape(str(email))}</span>
                </div>
                <div class="profile-item">
                    <span class="profile-label">Grade Level</span>
                    <span class="profile-value">{html.escape(str(grade))}</span>
                </div>
                <div class="profile-item">
                    <span class="profile-label">Created At</span>
                    <span class="profile-value">{html.escape(str(created_at))}</span>
                </div>
                {''.join(metric_items)}
            </div>
        """
        cards.append(card_html)

    st.markdown(
        f'<div class="profile-grid">{"".join(cards)}</div>',
        unsafe_allow_html=True,
    )


def get_llm_client(token: str) -> OpenAI | None:
    if not token:
        return None
    return OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=token,
    )


def run_llm_prompt(prompt: str, token: str) -> str:
    client = get_llm_client(token)
    if client is None:
        raise ValueError("Missing HF token.")
    completion = client.chat.completions.create(
        model="defog/llama-3-sqlcoder-8b:featherless-ai",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    message = completion.choices[0].message
    return message.content or ""


def parse_student_query(prompt: str, token: str, allowed_metrics: List[str]) -> Dict[str, object]:
    if not prompt.strip():
        raise ValueError("Missing query.")

    guidance = (
        "You are extracting filters from a natural language query about students. "
        "Return only valid JSON with this schema:\n"
        "{\"filters\": [{\"metric\": \"...\", \"op\": \">\", \"value\": 0}], "
        "\"limit\": 200, \"sort\": [{\"metric\": \"...\", \"direction\": \"desc\"}]}\n"
        "Rules:\n"
        "- metric must be one of: " + ", ".join(allowed_metrics) + "\n"
        "- op must be one of: >, >=, <, <=, =, !=\n"
        "- value must be a number\n"
        "- omit sort if not specified\n"
        "- if no filters found, return {\"filters\": []}\n"
        "Synonyms:\n"
        "- points, score -> points_earned\n"
        "- assessments, quizzes -> assessments\n"
        "- enrollments, courses -> enrollments\n"
        "- activity, events, engagement -> activity_events\n"
        "- chats, AI chats -> ai_chats\n"
        "- documents, files -> documents\n"
    )

    llm_prompt = f"{guidance}\n\nUser query: {prompt}"
    raw = run_llm_prompt(llm_prompt, token).strip()

    import json

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        raw = match.group(0)
    data = json.loads(raw)
    return data


def build_student_query_sql(
    available_metrics: Dict[str, Dict[str, object]],
    filters: List[Dict[str, object]],
    limit: int = 200,
    sort: List[Dict[str, str]] | None = None,
    include_only_active: bool = False,
) -> Tuple[str, List[object]]:
    select_cols = [
        "u.id",
        "u.name",
        "u.email",
        "u.grade_level",
        "u.created_at",
    ]

    join_sql: List[str] = []
    metric_selects: List[str] = []
    metric_refs: Dict[str, str] = {}

    table_groups: Dict[str, List[Dict[str, object]]] = {}
    for meta in available_metrics.values():
        table_groups.setdefault(meta["table"], []).append(meta)

    for idx, (table, metas) in enumerate(table_groups.items(), start=1):
        alias = f"m{idx}"
        agg_parts = []
        for meta in metas:
            agg_parts.append(f"{meta['agg']} AS {meta['alias']}")
            metric_refs[meta["metric"]] = f"COALESCE({alias}.{meta['alias']}, 0)"
            metric_selects.append(f"COALESCE({alias}.{meta['alias']}, 0) AS {meta['alias']}")

        join_sql.append(
            f"LEFT JOIN (SELECT user_id, {', '.join(agg_parts)} FROM {table} GROUP BY user_id) {alias} ON {alias}.user_id = u.id"
        )

    where_clauses = []
    params: List[object] = []
    for flt in filters:
        metric = flt.get("metric")
        op = flt.get("op")
        value = flt.get("value")
        if metric not in metric_refs:
            continue
        if op not in {">", ">=", "<", "<=", "=", "!="}:
            continue
        where_clauses.append(f"{metric_refs[metric]} {op} %s")
        params.append(value)

    sql_parts = [
        "SELECT",
        ", ".join(select_cols + metric_selects),
        "FROM public.user u",
    ]
    if join_sql:
        sql_parts.extend(join_sql)

    if include_only_active:
        where_clauses.insert(0, "u.is_deleted = FALSE")

    if where_clauses:
        sql_parts.append("WHERE " + " AND ".join(where_clauses))

    if sort:
        sort_parts = []
        for item in sort:
            metric = item.get("metric")
            direction = item.get("direction", "desc").lower()
            if metric in metric_refs and direction in {"asc", "desc"}:
                sort_parts.append(f"{metric_refs[metric]} {direction}")
        if sort_parts:
            sql_parts.append("ORDER BY " + ", ".join(sort_parts))

    sql_parts.append("LIMIT %s")
    params.append(int(limit))

    return "\n".join(sql_parts), params


def get_available_metrics(conn) -> Dict[str, Dict[str, object]]:
    metrics: Dict[str, Dict[str, object]] = {}

    if table_exists(conn, "public", "activity_performance"):
        cols = set(get_column_names(conn, "public", "activity_performance"))
        if "points_earned" in cols:
            metrics["points_earned"] = {
                "metric": "points_earned",
                "table": "public.activity_performance",
                "agg": "COALESCE(SUM(points_earned), 0)",
                "alias": "points_earned",
            }
        if "id" in cols:
            metrics["assessments"] = {
                "metric": "assessments",
                "table": "public.activity_performance",
                "agg": "COUNT(*)",
                "alias": "assessments",
            }

    if table_exists(conn, "public", "enrollments"):
        metrics["enrollments"] = {
            "metric": "enrollments",
            "table": "public.enrollments",
            "agg": "COUNT(*)",
            "alias": "enrollments",
        }

    if table_exists(conn, "public", "user_activity"):
        metrics["activity_events"] = {
            "metric": "activity_events",
            "table": "public.user_activity",
            "agg": "COUNT(*)",
            "alias": "activity_events",
        }

    if table_exists(conn, "public", "ai_chat"):
        metrics["ai_chats"] = {
            "metric": "ai_chats",
            "table": "public.ai_chat",
            "agg": "COUNT(*)",
            "alias": "ai_chats",
        }

    if table_exists(conn, "public", "documents"):
        metrics["documents"] = {
            "metric": "documents",
            "table": "public.documents",
            "agg": "COUNT(*)",
            "alias": "documents",
        }

    return metrics


def build_student_summary_prompt(
    user_data: Dict[str, object],
    metrics: List[Tuple[str, object]],
    enrollments_df: pd.DataFrame,
    assessments_df: pd.DataFrame,
    activity_df: pd.DataFrame,
    daily_df: pd.DataFrame,
) -> str:
    def df_preview(df: pd.DataFrame, limit: int = 5) -> str:
        if df.empty:
            return "None"
        return df.head(limit).to_string(index=False)

    profile_fields = [
        ("id", user_data.get("id")),
        ("name", user_data.get("name")),
        ("email", user_data.get("email")),
        ("grade_level", user_data.get("grade_level")),
        ("created_at", user_data.get("created_at")),
    ]

    metrics_lines = [f"{label}: {format_metric_value(value, 1)}" for label, value in metrics]

    prompt = (
        "You are preparing a professional student profile summary for school administrators. "
        "Write 1–2 short paragraphs (2–4 sentences total). Keep a formal, neutral tone. "
        "Prioritize engagement, assessment outcomes, and notable patterns. "
        "Do not speculate, do not repeat the same fact in different sentences, "
        "and do not mention missing data. "
        "Return only the paragraphs, no title or bullet points.\n\n"
        f"Student profile:\n{profile_fields}\n\n"
        f"Metrics:\n{metrics_lines}\n\n"
        f"Enrollments (sample):\n{df_preview(enrollments_df)}\n\n"
        f"Recent assessments (sample):\n{df_preview(assessments_df)}\n\n"
        f"Recent learning activity (sample):\n{df_preview(activity_df)}\n\n"
        f"Daily activity (sample):\n{df_preview(daily_df)}\n\n"
        "Format as plain paragraphs."
    )
    return prompt


def fetch_users(conn, search: str, limit: int) -> pd.DataFrame:
    if not table_exists(conn, "public", "user"):
        return pd.DataFrame()

    columns = get_column_names(conn, "public", "user")
    column_set = set(columns)

    select_cols = [
        col
        for col in [
            "id",
            "name",
            "email",
            "created_at",
            "grade_level",
            "phone_number",
            "city",
            "country",
            "is_email_verified",
        ]
        if col in column_set
    ]

    if "id" not in select_cols:
        select_cols.insert(0, "id")

    select_sql = sql.SQL(", ").join(sql.Identifier(col) for col in select_cols)
    where_clauses = []
    params: List[object] = []

    if "is_deleted" in column_set:
        where_clauses.append(sql.SQL("is_deleted = FALSE"))

    if search:
        search_clauses = []
        if "name" in column_set:
            search_clauses.append(sql.SQL("name ILIKE %s"))
            params.append(f"%{search}%")
        if "email" in column_set:
            search_clauses.append(sql.SQL("email ILIKE %s"))
            params.append(f"%{search}%")
        search_clauses.append(sql.SQL("CAST(id AS TEXT) ILIKE %s"))
        params.append(f"%{search}%")

        where_clauses.append(
            sql.SQL("(") + sql.SQL(" OR ").join(search_clauses) + sql.SQL(")")
        )

    query = sql.SQL("SELECT {cols} FROM {}.{}").format(
        sql.Identifier("public"),
        sql.Identifier("user"),
        cols=select_sql,
    )

    if where_clauses:
        query += sql.SQL(" WHERE ") + sql.SQL(" AND ").join(where_clauses)

    order_col = "created_at" if "created_at" in column_set else "id"
    query += sql.SQL(" ORDER BY {} DESC NULLS LAST LIMIT %s").format(
        sql.Identifier(order_col)
    )
    params.append(int(limit))

    with conn.cursor() as cur:
        cur.execute(query, params)
        rows = cur.fetchall()
        return pd.DataFrame(rows, columns=select_cols)


def fetch_user_row(conn, user_id: int) -> Dict[str, object]:
    if not table_exists(conn, "public", "user"):
        return {}

    with conn.cursor() as cur:
        cur.execute(
            "SELECT * FROM public.user WHERE id = %s",
            (user_id,),
        )
        row = cur.fetchone()
        if not row:
            return {}
        columns = [desc[0] for desc in cur.description]
        return dict(zip(columns, row))


def safe_count(conn, schema: str, table: str, where_col: str, where_val) -> int | None:
    if not table_exists(conn, schema, table):
        return None
    if not column_exists(conn, schema, table, where_col):
        return None

    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("SELECT COUNT(*) FROM {}.{} WHERE {} = %s").format(
                sql.Identifier(schema),
                sql.Identifier(table),
                sql.Identifier(where_col),
            ),
            (where_val,),
        )
        return int(cur.fetchone()[0])


def safe_sum(
    conn,
    schema: str,
    table: str,
    sum_col: str,
    where_col: str,
    where_val,
) -> float | None:
    if not table_exists(conn, schema, table):
        return None
    if not column_exists(conn, schema, table, sum_col):
        return None
    if not column_exists(conn, schema, table, where_col):
        return None

    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("SELECT COALESCE(SUM({}), 0) FROM {}.{} WHERE {} = %s").format(
                sql.Identifier(sum_col),
                sql.Identifier(schema),
                sql.Identifier(table),
                sql.Identifier(where_col),
            ),
            (where_val,),
        )
        result = cur.fetchone()[0]
        return float(result) if result is not None else None


def fetch_enrollments(conn, user_id: int) -> pd.DataFrame:
    if not table_exists(conn, "public", "enrollments"):
        return pd.DataFrame()

    enroll_cols = get_column_names(conn, "public", "enrollments")
    enroll_set = set(enroll_cols)

    select_parts: List[sql.SQL] = []
    for col in [
        "topic_id",
        "is_completed",
        "have_certificate",
        "repeat_count",
        "join_way",
        "enrolled_at",
        "created_at",
    ]:
        if col in enroll_set:
            select_parts.append(sql.SQL("e.{}").format(sql.Identifier(col)))

    topics_join = False
    if "topic_id" in enroll_set and table_exists(conn, "public", "topics"):
        topic_cols = get_column_names(conn, "public", "topics")
        topic_set = set(topic_cols)
        if "id" in topic_set:
            topics_join = True
            if "title" in topic_set:
                select_parts.append(sql.SQL("t.title AS topic_title"))
            if "subject" in topic_set:
                select_parts.append(sql.SQL("t.subject AS topic_subject"))
            if "status" in topic_set:
                select_parts.append(sql.SQL("t.status AS topic_status"))

    if not select_parts:
        select_parts.append(sql.SQL("e.user_id"))

    order_col = resolve_first_existing(
        enroll_cols,
        ["enrolled_at", "created_at", "updated_at", "id"],
    )
    if not order_col:
        order_col = "id"

    query = sql.SQL("SELECT {cols} FROM {}.{} e").format(
        sql.Identifier("public"),
        sql.Identifier("enrollments"),
        cols=sql.SQL(", ").join(select_parts),
    )

    if topics_join:
        query += sql.SQL(" LEFT JOIN {}.{} t ON t.id = e.topic_id").format(
            sql.Identifier("public"),
            sql.Identifier("topics"),
        )

    if "user_id" in enroll_set:
        query += sql.SQL(" WHERE e.user_id = %s")
    else:
        return pd.DataFrame()

    query += sql.SQL(" ORDER BY e.{} DESC NULLS LAST").format(
        sql.Identifier(order_col)
    )

    with conn.cursor() as cur:
        cur.execute(query, (user_id,))
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
        return pd.DataFrame(rows, columns=columns)


def fetch_recent_rows(
    conn,
    schema: str,
    table: str,
    columns: List[str],
    where_col: str,
    where_val,
    order_candidates: List[str],
    limit: int = 20,
) -> pd.DataFrame:
    if not table_exists(conn, schema, table):
        return pd.DataFrame()

    col_names = get_column_names(conn, schema, table)
    col_set = set(col_names)

    if where_col not in col_set:
        return pd.DataFrame()

    select_cols = [col for col in columns if col in col_set]
    if not select_cols:
        return pd.DataFrame()

    order_col = resolve_first_existing(col_names, order_candidates) or select_cols[0]

    query = sql.SQL(
        "SELECT {cols} FROM {}.{} WHERE {} = %s ORDER BY {} DESC NULLS LAST LIMIT %s"
    ).format(
        sql.Identifier(schema),
        sql.Identifier(table),
        sql.Identifier(where_col),
        sql.Identifier(order_col),
        cols=sql.SQL(", ").join(sql.Identifier(col) for col in select_cols),
    )

    with conn.cursor() as cur:
        cur.execute(query, (where_val, int(limit)))
        rows = cur.fetchall()
        return pd.DataFrame(rows, columns=select_cols)


def fetch_user_daily_activity(conn, user_id: int) -> pd.DataFrame:
    if not table_exists(conn, "public", "daily_activity_log"):
        return pd.DataFrame()

    columns = get_column_names(conn, "public", "daily_activity_log")
    column_set = set(columns)

    if "user_id" not in column_set:
        return pd.DataFrame()

    date_col = resolve_first_existing(columns, ["login", "created_at_client", "created_at"])
    if not date_col:
        return pd.DataFrame()

    select_parts = [sql.SQL("DATE({}) AS day").format(sql.Identifier(date_col))]
    if "time_spent" in column_set:
        select_parts.append(
            sql.SQL("COALESCE(SUM({}), 0) AS time_spent").format(
                sql.Identifier("time_spent")
            )
        )
    if "activities_completed" in column_set:
        select_parts.append(
            sql.SQL("COALESCE(SUM({}), 0) AS activities_completed").format(
                sql.Identifier("activities_completed")
            )
        )
    if "points_earned" in column_set:
        select_parts.append(
            sql.SQL("COALESCE(SUM({}), 0) AS points_earned").format(
                sql.Identifier("points_earned")
            )
        )

    query = sql.SQL(
        "SELECT {cols} FROM {}.{} WHERE user_id = %s GROUP BY day ORDER BY day"
    ).format(
        sql.Identifier("public"),
        sql.Identifier("daily_activity_log"),
        cols=sql.SQL(", ").join(select_parts),
    )

    with conn.cursor() as cur:
        cur.execute(query, (user_id,))
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
        return pd.DataFrame(rows, columns=columns)


def render_student_profiles(conn):
    st.header("Student Profiles")
    st.write("Search and review student activity across enrollments, learning, and chat.")

    with st.expander("LLM Access", expanded=False):
        st.text_input("HF token", type="password", key="hf_token")
        st.caption("Your token stays in this session and is used only for LLM requests.")

    search = st.text_input("Search by name, email, or ID", placeholder="Type to search...")
    limit = st.slider("Max results", min_value=10, max_value=200, value=50, step=10)

    users_df = fetch_users(conn, search, limit)
    if users_df.empty:
        st.info("No matching students found.")
        return

    users = users_df.to_dict("records")
    labels = [
        f"{user.get('name') or 'Unnamed'} ({user.get('email') or 'no-email'}) · #{user.get('id')}"
        for user in users
    ]

    selected_label = st.selectbox("Select student", labels)
    selected_index = labels.index(selected_label)
    selected_user = users[selected_index]
    user_id = int(selected_user.get("id"))

    user_data = fetch_user_row(conn, user_id)
    if not user_data:
        st.warning("Selected student could not be loaded.")
        return

    display_name = user_data.get("name") or f"User {user_id}"
    display_email = user_data.get("email") or "No email"

    st.subheader(display_name)
    st.caption(f"User ID: {user_id} · {display_email}")

    metrics = [
        ("Enrollments", safe_count(conn, "public", "enrollments", "user_id", user_id)),
        ("Activity Events", safe_count(conn, "public", "user_activity", "user_id", user_id)),
        ("Assessments", safe_count(conn, "public", "activity_performance", "user_id", user_id)),
        ("Points Earned", safe_sum(conn, "public", "activity_performance", "points_earned", "user_id", user_id)),
    ]

    metric_cols = st.columns(len(metrics))
    for col, (label, value) in zip(metric_cols, metrics):
        col.metric(label, format_metric_value(value))

    with st.expander("Summarize this student with LLM", expanded=False):
        hf_token = st.session_state.get("hf_token", "")
        if st.button("Generate summary"):
            if not hf_token:
                st.error("Please provide your HF token in the LLM Access section.")
            else:
                with st.spinner("Generating summary..."):
                    try:
                        enrollments_df = fetch_enrollments(conn, user_id)
                        assessments_df = fetch_recent_rows(
                            conn,
                            "public",
                            "activity_performance",
                            columns=[
                                "activity_type",
                                "attempts",
                                "is_right",
                                "points_earned",
                                "time_spent",
                                "submitted_at",
                            ],
                            where_col="user_id",
                            where_val=user_id,
                            order_candidates=["submitted_at", "created_at", "updated_at"],
                            limit=10,
                        )
                        activity_df = fetch_recent_rows(
                            conn,
                            "public",
                            "user_activity",
                            columns=[
                                "activity_type",
                                "activity_details",
                                "time_spent",
                                "created_at_client",
                                "created_at",
                            ],
                            where_col="user_id",
                            where_val=user_id,
                            order_candidates=["created_at_client", "created_at"],
                            limit=10,
                        )
                        daily_df = fetch_user_daily_activity(conn, user_id)

                        prompt = build_student_summary_prompt(
                            user_data=user_data,
                            metrics=metrics,
                            enrollments_df=enrollments_df,
                            assessments_df=assessments_df,
                            activity_df=activity_df,
                            daily_df=daily_df,
                        )
                        summary = run_llm_prompt(prompt, hf_token)
                        st.markdown(summary)
                    except Exception as exc:
                        st.error(f"LLM request failed: {exc}")

    with st.expander("Smart Student Finder", expanded=False):
        query_text = st.text_area(
            "Ask for students by attributes",
            placeholder="Show students with points more than 300 and assessments over 5",
        )
        max_results = st.number_input("Max results", min_value=1, max_value=1000, value=200)
        if st.button("Run student query"):
            hf_token = st.session_state.get("hf_token", "")
            if not hf_token:
                st.error("Please provide your HF token in the LLM Access section.")
            else:
                available_metrics = get_available_metrics(conn)
                if not available_metrics:
                    st.error("No metrics are available to query in this database.")
                else:
                    try:
                        parsed = parse_student_query(
                            query_text,
                            hf_token,
                            list(available_metrics.keys()),
                        )
                        filters = parsed.get("filters", [])
                        limit_value = int(parsed.get("limit", max_results))
                        sort = parsed.get("sort")
                        include_active = column_exists(conn, "public", "user", "is_deleted")
                        sql_text, params = build_student_query_sql(
                            available_metrics,
                            filters,
                            limit=limit_value,
                            sort=sort,
                            include_only_active=include_active,
                        )
                        with conn.cursor() as cur:
                            cur.execute(sql_text, params)
                            rows = cur.fetchall()
                            columns = [desc[0] for desc in cur.description]
                        result_df = pd.DataFrame(rows, columns=columns)
                        if result_df.empty:
                            st.info("No students matched the query.")
                        else:
                            render_student_result_cards(result_df)
                            with st.expander("Show as table"):
                                st.dataframe(result_df, use_container_width=True, height=420)
                    except Exception as exc:
                        st.error(f"Query failed: {exc}")

    tabs = st.tabs(["Profile", "Enrollments", "Engagement", "Chats"])

    with tabs[0]:
        sections = build_profile_sections(user_data)
        render_profile_cards(sections)

        with st.expander("All fields (raw)"):
            hide_fields = {"password"}
            profile_data = {k: v for k, v in user_data.items() if k not in hide_fields}
            profile_df = pd.DataFrame(
                {"field": list(profile_data.keys()), "value": list(profile_data.values())}
            )
            st.dataframe(profile_df, use_container_width=True, height=460)

    with tabs[1]:
        enrollments_df = fetch_enrollments(conn, user_id)
        if enrollments_df.empty:
            st.info("No enrollments found.")
        else:
            st.dataframe(enrollments_df, use_container_width=True, height=420)

    with tabs[2]:
        st.subheader("Daily Activity")
        daily_df = fetch_user_daily_activity(conn, user_id)
        if daily_df.empty:
            st.info("No daily activity log available.")
        else:
            chart_df = daily_df.set_index("day")
            st.line_chart(chart_df)

        st.subheader("Recent Assessments")
        assessments_df = fetch_recent_rows(
            conn,
            "public",
            "activity_performance",
            columns=[
                "activity_type",
                "attempts",
                "is_right",
                "points_earned",
                "time_spent",
                "submitted_at",
            ],
            where_col="user_id",
            where_val=user_id,
            order_candidates=["submitted_at", "created_at", "updated_at"],
            limit=20,
        )
        if assessments_df.empty:
            st.info("No assessment activity found.")
        else:
            st.dataframe(assessments_df, use_container_width=True, height=360)

        st.subheader("Recent Learning Activity")
        user_activity_df = fetch_recent_rows(
            conn,
            "public",
            "user_activity",
            columns=[
                "activity_type",
                "activity_details",
                "time_spent",
                "created_at_client",
                "created_at",
            ],
            where_col="user_id",
            where_val=user_id,
            order_candidates=["created_at_client", "created_at"],
            limit=20,
        )
        if user_activity_df.empty:
            st.info("No learning activity found.")
        else:
            st.dataframe(user_activity_df, use_container_width=True, height=360)

    with tabs[3]:
        st.subheader("AI Chat")
        ai_chat_df = fetch_recent_rows(
            conn,
            "public",
            "ai_chat",
            columns=["message", "ai_response", "input_method", "created_at_client", "time_spent"],
            where_col="user_id",
            where_val=user_id,
            order_candidates=["created_at_client", "created_at"],
            limit=20,
        )
        if ai_chat_df.empty:
            st.info("No AI chat history found.")
        else:
            st.dataframe(ai_chat_df, use_container_width=True, height=360)

        st.subheader("Chat Sessions")
        chat_sessions_df = fetch_recent_rows(
            conn,
            "public",
            "chat_sessions",
            columns=[
                "session_uuid",
                "organization_name",
                "selected_program_code",
                "started_at",
                "last_active_at",
            ],
            where_col="user_id",
            where_val=user_id,
            order_candidates=["started_at", "last_active_at"],
            limit=20,
        )
        if chat_sessions_df.empty:
            st.info("No chat sessions found.")
        else:
            st.dataframe(chat_sessions_df, use_container_width=True, height=360)


def prepare_engagement_signals(conn) -> List[Dict[str, object]]:
    signal_defs = [
        {
            "key": "daily_activity_log",
            "label": "Daily Activity",
            "schema": "public",
            "table": "daily_activity_log",
            "date_candidates": ["login", "created_at_client", "created_at"],
            "metric_cols": ["time_spent", "activities_completed", "points_earned"],
        },
        {
            "key": "user_activity",
            "label": "Learning Activity",
            "schema": "public",
            "table": "user_activity",
            "date_candidates": ["created_at_client", "created_at"],
            "metric_cols": ["time_spent"],
        },
        {
            "key": "activity_performance",
            "label": "Assessments",
            "schema": "public",
            "table": "activity_performance",
            "date_candidates": ["submitted_at", "created_at"],
            "metric_cols": ["time_spent", "points_earned", "attempts"],
        },
        {
            "key": "ai_chat",
            "label": "AI Chats",
            "schema": "public",
            "table": "ai_chat",
            "date_candidates": ["created_at_client", "created_at"],
            "metric_cols": ["time_spent"],
        },
        {
            "key": "chat_sessions",
            "label": "Support Sessions",
            "schema": "public",
            "table": "chat_sessions",
            "date_candidates": ["started_at", "last_active_at"],
            "metric_cols": [],
        },
    ]

    signals = []
    for signal in signal_defs:
        if not table_exists(conn, signal["schema"], signal["table"]):
            continue
        columns = get_column_names(conn, signal["schema"], signal["table"])
        date_col = resolve_first_existing(columns, signal["date_candidates"])
        if not date_col:
            continue
        metric_cols = [col for col in signal["metric_cols"] if col in columns]
        signals.append(
            {
                **signal,
                "date_col": date_col,
                "metric_cols": metric_cols,
                "columns": columns,
            }
        )
    return signals


def fetch_signal_timeseries(
    conn,
    schema: str,
    table: str,
    date_col: str,
    start_date: date,
    end_date: date,
    metric_cols: List[str],
) -> pd.DataFrame:
    columns = get_column_names(conn, schema, table)
    column_set = set(columns)

    select_parts = [
        sql.SQL("DATE({}) AS day").format(sql.Identifier(date_col)),
        sql.SQL("COUNT(*) AS events"),
    ]

    if "user_id" in column_set:
        select_parts.append(sql.SQL("COUNT(DISTINCT user_id) AS users"))

    for col in metric_cols:
        select_parts.append(
            sql.SQL("COALESCE(SUM({}), 0) AS {}").format(
                sql.Identifier(col),
                sql.Identifier(col),
            )
        )

    if "is_right" in column_set:
        select_parts.append(
            sql.SQL(
                "SUM(CASE WHEN is_right THEN 1 ELSE 0 END) AS correct_answers"
            )
        )

    end_exclusive = end_date + timedelta(days=1)

    query = sql.SQL(
        "SELECT {cols} FROM {}.{} WHERE {} >= %s AND {} < %s GROUP BY day ORDER BY day"
    ).format(
        sql.Identifier(schema),
        sql.Identifier(table),
        sql.Identifier(date_col),
        sql.Identifier(date_col),
        cols=sql.SQL(", ").join(select_parts),
    )

    with conn.cursor() as cur:
        cur.execute(query, (start_date, end_exclusive))
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
        df = pd.DataFrame(rows, columns=columns)

    if not df.empty and "correct_answers" in df.columns and "events" in df.columns:
        df["accuracy"] = df["correct_answers"] / df["events"].replace(0, pd.NA)

    return df


def fetch_table_date_bounds(
    conn,
    schema: str,
    table: str,
    date_col: str,
) -> Tuple[date | None, date | None]:
    if not table_exists(conn, schema, table):
        return None, None
    if not column_exists(conn, schema, table, date_col):
        return None, None

    query = sql.SQL(
        "SELECT MIN(DATE({})), MAX(DATE({})) FROM {}.{} WHERE {} IS NOT NULL"
    ).format(
        sql.Identifier(date_col),
        sql.Identifier(date_col),
        sql.Identifier(schema),
        sql.Identifier(table),
        sql.Identifier(date_col),
    )

    with conn.cursor() as cur:
        cur.execute(query)
        result = cur.fetchone()
        if not result:
            return None, None
        return result[0], result[1]


def resolve_engagement_date_range(
    conn, signals: List[Dict[str, object]]
) -> Tuple[date, date, date | None, date | None]:
    data_min = None
    data_max = None

    for signal in signals:
        min_date, max_date = fetch_table_date_bounds(
            conn,
            signal["schema"],
            signal["table"],
            signal["date_col"],
        )
        if min_date and (data_min is None or min_date < data_min):
            data_min = min_date
        if max_date and (data_max is None or max_date > data_max):
            data_max = max_date

    today = date.today()
    if data_max is None:
        return today - timedelta(days=30), today, None, None

    end_date = data_max
    start_date = end_date - timedelta(days=30)
    if data_min and start_date < data_min:
        start_date = data_min

    return start_date, end_date, data_min, data_max


def render_engagement_insights(conn):
    st.header("Engagement Insights")
    st.write("Trend activity signals across learning, assessments, and chat.")

    signals = prepare_engagement_signals(conn)
    if not signals:
        st.info("No engagement tables available.")
        return

    default_start, default_end, data_min, data_max = resolve_engagement_date_range(
        conn, signals
    )

    if data_min and data_max:
        st.caption(f"Data available from {data_min} to {data_max}.")

    date_input_kwargs = {"value": (default_start, default_end)}
    if data_min:
        date_input_kwargs["min_value"] = data_min
    if data_max:
        date_input_kwargs["max_value"] = data_max

    date_range = st.date_input("Date range", **date_input_kwargs)
    if isinstance(date_range, tuple):
        start_date, end_date = date_range
    else:
        start_date = date_range
        end_date = date_range

    if start_date > end_date:
        st.error("Start date must be before end date.")
        return

    tabs = st.tabs([signal["label"] for signal in signals])

    for tab, signal in zip(tabs, signals):
        with tab:
            df = fetch_signal_timeseries(
                conn,
                signal["schema"],
                signal["table"],
                signal["date_col"],
                start_date,
                end_date,
                signal["metric_cols"],
            )

            if df.empty:
                st.info("No data for the selected range.")
                continue

            total_events = df["events"].sum() if "events" in df.columns else None
            avg_users = (
                df["users"].mean() if "users" in df.columns else None
            )
            total_time = (
                df["time_spent"].sum() if "time_spent" in df.columns else None
            )

            metric_row = st.columns(3)
            metric_row[0].metric("Total Events", format_metric_value(total_events))
            metric_row[1].metric("Avg Daily Users", format_metric_value(avg_users, 1))
            metric_row[2].metric("Total Time Spent", format_metric_value(total_time, 1))

            chart_cols = [
                col
                for col in [
                    "events",
                    "users",
                    "time_spent",
                    "activities_completed",
                    "points_earned",
                    "attempts",
                    "accuracy",
                ]
                if col in df.columns
            ]

            if chart_cols:
                st.line_chart(df.set_index("day")[chart_cols])

            with st.expander("Raw data"):
                st.dataframe(df, use_container_width=True, height=320)


def render_table_view(conn, full_table_name: str):
    schema, table = split_table_name(full_table_name)
    widget_prefix = f"{schema}_{table}"

    st.subheader(f"{schema}.{table}")

    columns = get_columns(conn, schema, table)
    primary_keys = set(get_primary_keys(conn, schema, table))

    meta_rows = []
    for col in columns:
        meta_rows.append(
            {
                "column": col["column_name"],
                "type": col["data_type"],
                "nullable": col["is_nullable"],
                "default": col["column_default"],
                "primary_key": "yes" if col["column_name"] in primary_keys else "",
            }
        )

    with st.expander("Columns", expanded=False):
        st.dataframe(pd.DataFrame(meta_rows), use_container_width=True, height=280)

    col_names = [col["column_name"] for col in columns]
    filter_col = st.selectbox(
        "Filter column",
        ["(none)"] + col_names,
        key=f"{widget_prefix}_filter_col",
    )

    filter_col_value = None
    filter_op_value = None

    if filter_col != "(none)":
        filter_op_value = st.selectbox(
            "Operator",
            ["contains", "equals", "is null", "is not null"],
            key=f"{widget_prefix}_filter_op",
        )
        if filter_op_value in {"contains", "equals"}:
            filter_col_value = st.text_input(
                "Filter value",
                key=f"{widget_prefix}_filter_val",
            )

    limit = st.number_input(
        "Row limit",
        min_value=1,
        max_value=10000,
        value=500,
        step=50,
        key=f"{widget_prefix}_limit",
    )
    offset = st.number_input(
        "Row offset",
        min_value=0,
        value=0,
        step=100,
        key=f"{widget_prefix}_offset",
    )

    auto_load = st.checkbox("Auto load data", value=True, key=f"{widget_prefix}_auto")
    load_clicked = st.button("Load data", key=f"{widget_prefix}_load")

    if auto_load or load_clicked:
        try:
            df = fetch_rows(
                conn,
                schema,
                table,
                limit=int(limit),
                offset=int(offset),
                filter_col=filter_col if filter_col != "(none)" else None,
                filter_op=filter_op_value,
                filter_val=filter_col_value or "",
            )
            st.dataframe(df, use_container_width=True, height=520)
            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download CSV",
                data=csv_data,
                file_name=f"{schema}.{table}.csv",
                mime="text/csv",
                key=f"{widget_prefix}_download",
            )
        except Exception as exc:
            st.error(f"Failed to load data: {exc}")

    if st.checkbox("Show row count", value=False, key=f"{widget_prefix}_count"):
        try:
            total = count_rows(
                conn,
                schema,
                table,
                filter_col=filter_col if filter_col != "(none)" else None,
                filter_op=filter_op_value,
                filter_val=filter_col_value or "",
            )
            st.info(f"Total rows: {total}")
        except Exception as exc:
            st.error(f"Failed to count rows: {exc}")


def render_group_section(conn, title: str, tables: List[str]):
    st.header(title)
    if not tables:
        st.info("No tables mapped to this section.")
        return

    selected = st.selectbox(
        "Select a table",
        tables,
        format_func=display_table_name,
        key=f"select_{title}",
    )
    render_table_view(conn, selected)


def main():
    st.set_page_config(
        page_title="School Management System",
        page_icon="🎓",
        layout="wide",
    )

    apply_ui_theme()

    st.sidebar.markdown(
        """
        <div style="padding: 1.4rem 1.2rem 0.6rem;">
            <div style="font-family: 'Fraunces', serif; font-size: 1.4rem;">School Console</div>
            <div style="font-size: 0.9rem; color: #d9d2c3;">PostgreSQL-backed operations</div>
        </div>
        <div style="height: 1px; background: rgba(255,255,255,0.2); margin: 0.6rem 0 0.8rem;"></div>
        """,
        unsafe_allow_html=True,
    )

    table_list_from_txt = read_table_list_from_txt("all_tables.txt")

    try:
        with get_connection() as conn:
            table_list_from_db = list_tables(conn)
    except Exception as exc:
        st.error(f"Database connection failed: {exc}")
        if table_list_from_txt:
            st.warning(
                "Found tables in all_tables.txt, but the app needs a live database "
                "connection to browse rows."
            )
        st.stop()

    expected_tables = table_list_from_txt or table_list_from_db
    missing_tables = sorted([t for t in expected_tables if t not in table_list_from_db])

    if missing_tables:
        st.warning(
            "Some tables listed in all_tables.txt were not found in the database: "
            + ", ".join(missing_tables)
        )

    grouped_tables = build_grouped_tables(table_list_from_db)

    sections = (
        ["Dashboard", "Student Profiles", "Engagement Insights"]
        + list(grouped_tables.keys())
        + ["Table Explorer"]
    )
    section = st.sidebar.radio("Section", sections)

    with get_connection() as conn:
        if section == "Dashboard":
            render_hero(
                total_tables=len(table_list_from_db),
                missing_tables=len(missing_tables),
                group_count=len(grouped_tables),
            )

            st.markdown("## Snapshot")
            st.write(
                "Review the core activity signals and jump into a module using the sidebar. "
                "Every table remains available in the Table Explorer."
            )

            load_metrics = st.button("Load metrics")
            if load_metrics:
                metrics = [
                    ("Students", "public.user"),
                    ("Enrollments", "public.enrollments"),
                    ("Courses", "public.template_courses"),
                    ("Organizations", "public.org"),
                    ("Certificates", "public.issued_certificates"),
                ]
                cols = st.columns(len(metrics))
                for idx, (label, table_name) in enumerate(metrics):
                    schema, table = split_table_name(table_name)
                    try:
                        total = count_rows(conn, schema, table, None, None, None)
                        cols[idx].metric(label, f"{total:,}")
                    except Exception as exc:
                        cols[idx].metric(label, "N/A")
                        cols[idx].caption(str(exc))

            with st.expander("All Tables", expanded=False):
                st.dataframe(
                    pd.DataFrame({"table": table_list_from_db}),
                    use_container_width=True,
                    height=320,
                )
        elif section == "Student Profiles":
            render_student_profiles(conn)
        elif section == "Engagement Insights":
            render_engagement_insights(conn)
        elif section == "Table Explorer":
            st.header("Table Explorer")
            selected = st.selectbox(
                "Select a table",
                table_list_from_db,
                format_func=display_table_name,
                key="explorer_table",
            )
            render_table_view(conn, selected)
        else:
            render_group_section(conn, section, grouped_tables.get(section, []))


if __name__ == "__main__":
    main()
