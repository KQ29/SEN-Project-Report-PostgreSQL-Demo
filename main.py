import os
import re
from typing import Optional, Tuple, Dict, List, Set

import pandas as pd
import streamlit as st
import psycopg2
import psycopg2.extras

from openai import OpenAI

# ==========================================================
# HF Inference Providers (OpenAI-compatible router)
# ==========================================================
HF_OPENAI_BASE_URL = "https://router.huggingface.co/v1"

# ==========================================================
# Views (Витрины)
# ==========================================================
V_TEACHER_DATA_SQL = r"""
CREATE OR REPLACE VIEW public.v_teacher_data AS
SELECT
  'lesson'::text AS record_type,

  u.id AS student_id,
  u.name AS student_name,
  u.grade_level AS grade_level_text,
  NULLIF(regexp_replace(COALESCE(u.grade_level,''), '\D', '', 'g'), '')::int AS grade_level_num,

  t.id AS topic_id,
  t.title AS topic_title,
  t.subject AS topic_subject,

  COALESCE(lj.started_at, lj.created_at)::timestamp AS event_time,

  lj.started_at::timestamp AS started_at,
  lj.updated_at::timestamp AS updated_at,

  lj.total_time_spent_seconds::double precision AS lesson_time_spent_seconds,
  lj.cur_progress::double precision            AS lesson_cur_progress,
  lj.max_progress::double precision            AS lesson_max_progress,

  CASE
    WHEN lj.max_progress IS NOT NULL AND lj.max_progress > 0
    THEN (100.0 * (lj.cur_progress / lj.max_progress))::double precision
    ELSE NULL
  END AS lesson_completion_pct,

  -- activity fields (NULL for lesson rows)
  NULL::timestamp         AS submitted_at,
  NULL::text              AS activity_type,
  NULL::boolean           AS is_right,
  NULL::double precision  AS activity_points_earned,
  NULL::double precision  AS activity_time_spent,

  -- topic_session fields (NULL for lesson rows)
  NULL::double precision AS topic_completion_percent,
  NULL::double precision AS topic_total_time_spent

FROM public.lesson_joined lj
LEFT JOIN public.topics t ON t.id = lj.topic_id
LEFT JOIN public."user" u ON u.id = lj.user_id

UNION ALL

SELECT
  'activity'::text AS record_type,

  u.id AS student_id,
  u.name AS student_name,
  u.grade_level AS grade_level_text,
  NULLIF(regexp_replace(COALESCE(u.grade_level,''), '\D', '', 'g'), '')::int AS grade_level_num,

  t.id AS topic_id,
  t.title AS topic_title,
  t.subject AS topic_subject,

  ap.submitted_at::timestamp AS event_time,

  -- lesson fields (NULL for activity rows)
  NULL::timestamp        AS started_at,
  NULL::timestamp        AS updated_at,
  NULL::double precision AS lesson_time_spent_seconds,
  NULL::double precision AS lesson_cur_progress,
  NULL::double precision AS lesson_max_progress,
  NULL::double precision AS lesson_completion_pct,

  -- activity fields
  ap.submitted_at::timestamp            AS submitted_at,
  ap.activity_type                      AS activity_type,
  ap.is_right                           AS is_right,
  ap.points_earned::double precision    AS activity_points_earned,
  ap.time_spent::double precision       AS activity_time_spent,

  -- topic_session fields (if exists)
  ts.completion_percent::double precision AS topic_completion_percent,
  ts.total_time_spent::double precision   AS topic_total_time_spent

FROM public.activity_performance ap
LEFT JOIN public.chapter_session cs ON cs.id = ap.chapter_session_id
LEFT JOIN public.topic_session ts   ON ts.id = cs.topic_session_id
LEFT JOIN public.enrollments e      ON e.id  = ts.enrollment_id
LEFT JOIN public.topics t           ON t.id  = e.topic_id
LEFT JOIN public."user" u           ON u.id  = ap.user_id
;
""".strip()

V_TEACHER_TOPIC_SESSIONS_SQL = r"""
CREATE OR REPLACE VIEW public.v_teacher_topic_sessions AS
SELECT
  ts.id AS topic_session_id,
  ts.local_id AS topic_session_local_id,

  ts.user_id AS student_id,
  u.name AS student_name,
  u.grade_level AS grade_level_text,
  NULLIF(regexp_replace(COALESCE(u.grade_level,''), '\D', '', 'g'), '')::int AS grade_level_num,

  ts.enrollment_id,
  e.topic_id,
  t.title AS topic_title,
  t.subject AS topic_subject,

  ts.started_at::timestamp   AS started_at,
  ts.completed_at::timestamp AS completed_at,
  ts.total_time_spent::double precision AS total_time_spent,
  ts.completion_percent::double precision AS completion_percent,

  ts.created_at::timestamp AS created_at,
  ts.updated_at::timestamp AS updated_at

FROM public.topic_session ts
LEFT JOIN public.enrollments e ON e.id = ts.enrollment_id
LEFT JOIN public.topics t      ON t.id = e.topic_id
LEFT JOIN public."user" u      ON u.id = ts.user_id
;
""".strip()

PROJECT_OVERVIEW_MD = """
## Teacher Analytics (Natural Language → SQL)

- Teacher asks questions in English
- Deterministic out-of-scope guard blocks advice/how-to questions (hardcoded exception)
- LLM generates SQL (read-only, SQL-only output)
- SQL runs on PostgreSQL
- Results show as table / chart

### Rules
✅ Must use Views (Витрины): public.v_teacher_data / public.v_teacher_topic_sessions  
✅ Auto-repair (1–2 tries) + rollback  
✅ LLM handles time phrases using vta.event_time  
✅ No JSON output (SQL only)  
✅ Schema-aware validation blocks hallucinated view columns (e.g., vta.class)  
"""

# ==========================================================
# Hardcoded out-of-scope guard (requested exception)
# ==========================================================
DATA_KEYWORDS = {
    "student", "students", "learner", "learners",
    "grade", "year", "class",
    "topic", "topics", "subject", "subjects",
    "lesson", "lessons",
    "activity", "activities",
    "enrollment", "enrollments", "session", "sessions",
    "time", "spent", "duration", "total", "average", "avg", "sum",
    "completion", "complete", "progress", "trend", "over time",
    "accuracy", "correct", "wrong", "right", "points", "score",
    "submitted", "submission", "attempt", "attempts",
    "this month", "last month", "this year", "last year",
    "week", "weekly", "month", "monthly", "yearly",
    "top", "bottom", "rank",
}

ADVICE_TRIGGERS = {
    "how to", "how do i", "tips", "advice", "motivate", "motivation",
    "strategy", "strategies", "improve teaching", "teach better",
    "ride a bike", "bike", "life hack", "opinion", "essay"
}

ANALYTICS_TRIGGERS = {
    "top", "bottom", "rank", "average", "avg", "sum", "total",
    "time spent", "duration", "completion", "progress", "accuracy",
    "over time", "trend", "per student", "by student", "by topic", "by subject"
}

def is_data_question(question: str) -> bool:
    q = (question or "").strip().lower()
    if not q:
        return False

    # strong advice/how-to signals -> out-of-scope unless it also clearly asks analytics
    if any(t in q for t in ADVICE_TRIGGERS) and not any(t in q for t in ANALYTICS_TRIGGERS):
        return False

    # If it contains a 4-digit year or ISO date, likely analytics.
    if re.search(r"\b(19|20)\d{2}\b", q):
        return True
    if re.search(r"\b\d{4}-\d{2}-\d{2}\b", q):
        return True

    # grade/year/class patterns
    if re.search(r"\b(year|grade|class)\s+\d{1,2}\b", q):
        return True

    # keyword match
    return any(kw in q for kw in DATA_KEYWORDS)

def out_of_scope_message() -> str:
    return (
        "This dashboard only answers questions about learning data "
        "(students, lessons, activities, topics, progress, time spent, points, etc.).\n\n"
        "Examples:\n"
        "- Top 10 students by total lesson time spent last month\n"
        "- Show Year 4 students\n"
        "- Show Year 11 students"
    )

# ==========================================================
# Safety (read-only SQL) — hard boundary
# ==========================================================
DISALLOWED_SQL_KEYWORDS = [
    "insert", "update", "delete", "drop", "alter", "truncate", "create", "grant", "revoke",
    "copy", "vacuum", "analyze", "refresh", "execute", "call", "do", "set", "pg_sleep",
    "lock", "comment", "security", "cluster", "reindex",
]
DISALLOWED_TOKENS = [
    "pg_catalog",
    "information_schema",
    "pg_read_file",
    "pg_ls_dir",
    "lo_export",
    "lo_import",
    "dblink",
]

def strip_code_fences(text: str) -> str:
    s = (text or "").strip()
    s = re.sub(r"^```sql\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"^```\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()

def extract_first_sql(text_out: str) -> str:
    s = strip_code_fences(text_out)
    m = re.search(r"\b(with|select)\b", s, flags=re.IGNORECASE)
    if m:
        s = s[m.start():]
    s = s.strip()
    s = re.sub(r";+\s*$", "", s)
    return s.strip()

def _extract_last_limit_value(sql: str) -> Optional[int]:
    matches = list(re.finditer(r"\blimit\s+(\d+)\b", sql or "", flags=re.IGNORECASE))
    if not matches:
        return None
    try:
        return int(matches[-1].group(1))
    except Exception:
        return None

def _mentions_views(sql: str) -> bool:
    low = (sql or "").lower()
    return ("v_teacher_data" in low) or ("v_teacher_topic_sessions" in low)

def _has_select_star(sql: str) -> bool:
    low = (sql or "").lower()
    if re.search(r"\bselect\s+\*\b", low):
        return True
    if re.search(r"\bselect\s+[a-zA-Z_][\w]*\.\*\b", low):
        return True
    return False

def parse_schema_hint(schema_hint: str) -> Dict[str, Set[str]]:
    """
    Parses fetch_schema_hint() output into:
      { "v_teacher_data": {col,...}, "v_teacher_topic_sessions": {col,...} }
    """
    out: Dict[str, Set[str]] = {"v_teacher_data": set(), "v_teacher_topic_sessions": set()}
    for line in (schema_hint or "").splitlines():
        line = line.strip()
        if not line.startswith("public.v_teacher_data:") and not line.startswith("public.v_teacher_topic_sessions:"):
            continue
        try:
            left, cols_part = line.split(":", 1)
            table = left.replace("public.", "").strip()
            for chunk in cols_part.split(","):
                chunk = chunk.strip()
                if not chunk:
                    continue
                col = chunk.split(" ", 1)[0].strip()  # "col (type)" -> "col"
                if col and table in out:
                    out[table].add(col)
        except Exception:
            continue
    return out

def extract_view_aliases(sql: str) -> Dict[str, Set[str]]:
    """
    Finds aliases used for the two views.
    Example: FROM public.v_teacher_data vta -> alias vta
             JOIN public.v_teacher_topic_sessions ts -> alias ts
    """
    aliases: Dict[str, Set[str]] = {"v_teacher_data": set(), "v_teacher_topic_sessions": set()}
    s = (sql or "")

    # capture alias after FROM/JOIN <view> [AS] <alias>
    patterns = [
        (r"\bfrom\s+public\.v_teacher_data\s+(?:as\s+)?([a-zA-Z_][\w]*)\b", "v_teacher_data"),
        (r"\bjoin\s+public\.v_teacher_data\s+(?:as\s+)?([a-zA-Z_][\w]*)\b", "v_teacher_data"),
        (r"\bfrom\s+public\.v_teacher_topic_sessions\s+(?:as\s+)?([a-zA-Z_][\w]*)\b", "v_teacher_topic_sessions"),
        (r"\bjoin\s+public\.v_teacher_topic_sessions\s+(?:as\s+)?([a-zA-Z_][\w]*)\b", "v_teacher_topic_sessions"),
    ]
    for pat, key in patterns:
        for m in re.finditer(pat, s, flags=re.IGNORECASE):
            aliases[key].add(m.group(1))
    return aliases

def validate_view_columns(sql: str, schema_cols: Dict[str, Set[str]]) -> Tuple[bool, str]:
    """
    Validates any <alias>.<column> where <alias> is an alias of one of the views.
    This blocks hallucinated columns like vta.class.
    """
    s = sql or ""
    aliases = extract_view_aliases(s)

    vta_like = aliases["v_teacher_data"]
    vtts_like = aliases["v_teacher_topic_sessions"]

    # If query uses view but with no alias, we can't reliably validate unqualified columns.
    # We still allow execution; DB errors will be repaired.
    v_teacher_data_cols = schema_cols.get("v_teacher_data", set())
    vtts_cols = schema_cols.get("v_teacher_topic_sessions", set())

    # scan all qualified references: alias.column
    for alias, col in re.findall(r"\b([a-zA-Z_][\w]*)\.([a-zA-Z_][\w]*)\b", s):
        if alias in vta_like:
            if v_teacher_data_cols and col not in v_teacher_data_cols:
                return False, f"Unknown column for v_teacher_data alias '{alias}': {col}"
        if alias in vtts_like:
            if vtts_cols and col not in vtts_cols:
                return False, f"Unknown column for v_teacher_topic_sessions alias '{alias}': {col}"

    return True, "OK"

def is_safe_sql(core_sql: str, *, max_rows: int, schema_cols: Dict[str, Set[str]]) -> Tuple[bool, str]:
    s = (core_sql or "").strip()
    if not s:
        return False, "Empty SQL."
    if ";" in s:
        return False, "Multiple statements / semicolons are not allowed."

    low = s.lower().strip()
    if not (low.startswith("select") or low.startswith("with")):
        return False, "Only SELECT/WITH queries are allowed."

    for kw in DISALLOWED_SQL_KEYWORDS:
        if re.search(rf"\b{re.escape(kw)}\b", low):
            return False, f"Disallowed keyword detected: {kw}"

    for tok in DISALLOWED_TOKENS:
        if tok.lower() in low:
            return False, f"Disallowed reference detected: {tok}"

    if re.search(r"\bselect\b[\s\S]*\binto\b", low):
        return False, "SELECT INTO is not allowed."
    if re.search(r"\bfor\s+update\b|\bfor\s+share\b", low):
        return False, "FOR UPDATE/SHARE is not allowed."

    if _has_select_star(s):
        return False, "SELECT * is not allowed. Select explicit columns."

    lim = _extract_last_limit_value(s)
    if lim is None:
        return False, f"Missing LIMIT. Add LIMIT <= {int(max_rows)}."
    if lim > int(max_rows):
        return False, f"LIMIT {lim} exceeds max allowed {int(max_rows)}."

    if not _mentions_views(s):
        return False, "Query must use v_teacher_data and/or v_teacher_topic_sessions (Витрины)."

    ok, msg = validate_view_columns(s, schema_cols)
    if not ok:
        return False, msg

    return True, "OK"

# ==========================================================
# DB helpers
# ==========================================================
def get_conn(host: str, port: int, dbname: str, user: str, password: str, sslmode: str):
    return psycopg2.connect(
        host=host,
        port=port,
        dbname=dbname,
        user=user,
        password=password,
        sslmode=sslmode if sslmode else None,
    )

def run_query_psycopg2(conn, sql: str) -> pd.DataFrame:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(sql)
        rows = cur.fetchall()
    return pd.DataFrame(rows)

def setup_views(conn) -> None:
    try:
        with conn.cursor() as cur:
            cur.execute(V_TEACHER_DATA_SQL)
            cur.execute(V_TEACHER_TOPIC_SESSIONS_SQL)
        conn.commit()
    except Exception:
        conn.rollback()
        raise

def fetch_schema_hint(conn) -> str:
    tables = [
        "v_teacher_data",
        "v_teacher_topic_sessions",
        "lesson_joined",
        "activity_performance",
        "chapter_session",
        "topic_session",
        "enrollments",
        "topics",
        "user",
    ]

    sql = """
    SELECT table_name, column_name, data_type
    FROM information_schema.columns
    WHERE table_schema = 'public'
      AND table_name = ANY(%s)
    ORDER BY table_name, ordinal_position;
    """
    out: Dict[str, List[str]] = {t: [] for t in tables}
    with conn.cursor() as cur:
        cur.execute(sql, (tables,))
        for table_name, column_name, data_type in cur.fetchall():
            if table_name in out:
                out[table_name].append(f"{column_name} ({data_type})")

    lines = []
    lines.append("Preferred views (use these first):")
    lines.append(" - public.v_teacher_data")
    lines.append(" - public.v_teacher_topic_sessions")
    lines.append("")
    lines.append("Schema (public):")
    for t in tables:
        cols = out.get(t, [])
        if not cols:
            continue
        if t == "user":
            lines.append('public."user": ' + ", ".join(cols))
        else:
            lines.append(f"public.{t}: " + ", ".join(cols))
    lines.append("")
    lines.append("Notes:")
    lines.append("- v_teacher_data has unified columns for lessons+activities.")
    lines.append("- event_time is timestamp (no timezone).")
    lines.append('- If referencing the user table, prefer: public."user"')
    return "\n".join(lines).strip()

# ==========================================================
# LLM (SQL only; NO JSON)
# ==========================================================
def nl_to_sql_llm(
    question: str,
    hf_token: str,
    model_id: str,
    schema_hint: str,
    *,
    max_rows: int,
) -> str:
    client = OpenAI(base_url=HF_OPENAI_BASE_URL, api_key=hf_token)

    system = (
        "You generate exactly ONE PostgreSQL query.\n"
        "Hard rules:\n"
        "- Output ONLY SQL (no markdown, no explanation).\n"
        "- Only SELECT or WITH.\n"
        "- No semicolons.\n"
        "- Avoid pg_catalog and information_schema.\n"
        f"- ALWAYS include LIMIT <= {int(max_rows)}.\n"
        "- Do NOT use SELECT *.\n"
        "\n"
        "Views (Витрины) rule:\n"
        "- Use public.v_teacher_data and/or public.v_teacher_topic_sessions.\n"
        "- Prefer the views even if raw tables exist.\n"
        "\n"
        "Correctness rules:\n"
        "- Do not invent columns. Use ONLY columns present in the schema hint.\n"
        "- v_teacher_data is event-level. If the user asks to list/show students,\n"
        "  return ONE row per student (use DISTINCT on student_id/student_name or GROUP BY).\n"
        "- For time filters, use vta.event_time (alias vta if you use it).\n"
        "\n"
        "If you truly cannot answer from the schema, return a message row:\n"
        f"SELECT 'Cannot answer from schema'::text AS message LIMIT 1\n"
    )

    user = f"""
Teacher question:
{question}

Max rows policy:
- LIMIT <= {int(max_rows)}

Schema hint:
{schema_hint}
""".strip()

    resp = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
        top_p=1,
        max_tokens=800,
    )
    return extract_first_sql(resp.choices[0].message.content or "")

def repair_sql_llm(
    question: str,
    bad_sql: str,
    error_msg: str,
    hf_token: str,
    model_id: str,
    schema_hint: str,
    *,
    max_rows: int,
) -> str:
    client = OpenAI(base_url=HF_OPENAI_BASE_URL, api_key=hf_token)

    system = (
        "You fix PostgreSQL SQL.\n"
        "Return exactly ONE corrected SQL query.\n"
        "Hard rules:\n"
        "- Output ONLY SQL (no markdown, no explanation).\n"
        "- Only SELECT or WITH.\n"
        "- No semicolons.\n"
        "- Avoid pg_catalog and information_schema.\n"
        f"- ALWAYS include LIMIT <= {int(max_rows)}.\n"
        "- Do NOT use SELECT *.\n"
        "\n"
        "Views (Витрины) rule:\n"
        "- Use public.v_teacher_data and/or public.v_teacher_topic_sessions.\n"
        "- Prefer the views over raw tables.\n"
        "\n"
        "Correctness rules:\n"
        "- Do not invent columns. Use ONLY columns present in the schema hint.\n"
        "- If listing/showing students, deduplicate to ONE row per student (DISTINCT or GROUP BY).\n"
        "- For time filters, use vta.event_time (alias vta if you use it).\n"
    )

    user = f"""
Original teacher question:
{question}

Max rows policy:
- LIMIT <= {int(max_rows)}

Schema hint:
{schema_hint}

Bad SQL:
{bad_sql}

Error / constraint:
{error_msg}

Fix it.
""".strip()

    resp = client.chat.completions.create(
        model=model_id,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
        top_p=1,
        max_tokens=800,
    )
    return extract_first_sql(resp.choices[0].message.content or "")

# ==========================================================
# Generate + Repair (hardcoded out-of-scope guard + SQL-only)
# ==========================================================
def generate_sql_with_repairs(
    question: str,
    hf_token: str,
    model_id: str,
    schema_hint: str,
    *,
    max_rows: int,
    max_repairs: int = 2,
) -> Tuple[Optional[str], List[Dict[str, str]], str]:
    attempts: List[Dict[str, str]] = []
    schema_cols = parse_schema_hint(schema_hint)

    # deterministic out-of-scope guard (your requested exception)
    if not is_data_question(question):
        attempts.append({"stage": "guard_block", "sql": "", "msg": "Out-of-scope (deterministic guard)."})
        return None, attempts, out_of_scope_message()

    if not hf_token:
        attempts.append({"stage": "no_token", "sql": "", "msg": "HF token missing."})
        return None, attempts, "HF token is required."

    current_sql = nl_to_sql_llm(
        question=question,
        hf_token=hf_token,
        model_id=model_id,
        schema_hint=schema_hint,
        max_rows=max_rows,
    )
    attempts.append({"stage": "initial", "sql": current_sql, "msg": "generated"})

    for i in range(max_repairs + 1):
        ok, msg = is_safe_sql(current_sql, max_rows=max_rows, schema_cols=schema_cols)
        if ok:
            return current_sql, attempts, ""

        attempts.append({"stage": f"blocked_{i}", "sql": current_sql, "msg": msg})
        if i >= max_repairs:
            return None, attempts, f"SQL blocked and could not be repaired: {msg}"

        current_sql = repair_sql_llm(
            question=question,
            bad_sql=current_sql,
            error_msg=f"Safety block: {msg}",
            hf_token=hf_token,
            model_id=model_id,
            schema_hint=schema_hint,
            max_rows=max_rows,
        )
        attempts.append({"stage": f"repair_{i+1}", "sql": current_sql, "msg": "repaired after block"})

    return None, attempts, "SQL generation failed."

def execute_with_auto_repair(
    conn,
    question: str,
    sql: str,
    hf_token: str,
    model_id: str,
    schema_hint: str,
    *,
    max_rows: int,
    max_repairs: int = 2,
) -> Tuple[Optional[pd.DataFrame], str, List[Dict[str, str]]]:
    attempts: List[Dict[str, str]] = []
    current_sql = sql
    schema_cols = parse_schema_hint(schema_hint)

    for i in range(max_repairs + 1):
        try:
            df = run_query_psycopg2(conn, current_sql)
            attempts.append({"stage": f"exec_ok_{i}", "sql": current_sql, "msg": "execution ok"})
            return df, current_sql, attempts
        except Exception as e:
            err = str(e)
            attempts.append({"stage": f"exec_fail_{i}", "sql": current_sql, "msg": err})

            try:
                conn.rollback()
            except Exception:
                pass

            if not hf_token or i >= max_repairs:
                return None, current_sql, attempts

            repaired = repair_sql_llm(
                question=question,
                bad_sql=current_sql,
                error_msg=f"PostgreSQL error: {err}",
                hf_token=hf_token,
                model_id=model_id,
                schema_hint=schema_hint,
                max_rows=max_rows,
            )

            ok, msg = is_safe_sql(repaired, max_rows=max_rows, schema_cols=schema_cols)
            if not ok:
                attempts.append({"stage": f"repair_blocked_{i+1}", "sql": repaired, "msg": msg})
                return None, current_sql, attempts

            current_sql = repaired

    return None, current_sql, attempts

# ==========================================================
# Visualization
# ==========================================================
def auto_chart(df: pd.DataFrame):
    if df is None or df.empty:
        return

    date_cols = [c for c in df.columns if any(k in c.lower() for k in ["date", "time", "submitted", "week", "month", "event", "start"])]
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    if date_cols and num_cols:
        x = date_cols[0]
        preferred = [c for c in num_cols if any(k in c.lower() for k in ["avg", "sum", "total", "pct", "percent", "points", "time"])]
        y = preferred[0] if preferred else num_cols[0]

        tmp = df.copy()
        tmp[x] = pd.to_datetime(tmp[x], errors="coerce")
        tmp = tmp.dropna(subset=[x])
        if not tmp.empty:
            st.line_chart(tmp.set_index(x)[[y]])
            return

    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    if cat_cols and num_cols:
        x = cat_cols[0]
        preferred = [c for c in num_cols if any(k in c.lower() for k in ["avg", "sum", "total", "pct", "percent", "points", "time"])]
        y = preferred[0] if preferred else num_cols[0]
        tmp = df[[x, y]].dropna()
        if len(tmp) <= 50:
            st.bar_chart(tmp.set_index(x)[[y]])

# ==========================================================
# Streamlit UI
# ==========================================================
def main():
    st.set_page_config(page_title="Teacher Analytics (NL → SQL)", layout="wide")
    st.title("Teacher Analytics Dashboard (Natural Language → SQL)")
    st.markdown(PROJECT_OVERVIEW_MD)

    st.sidebar.header("PostgreSQL Connection")
    host = st.sidebar.text_input("Host", value=os.getenv("PGHOST", "localhost"))
    port = int(st.sidebar.number_input("Port", value=int(os.getenv("PGPORT", "5432")), step=1))
    dbname = st.sidebar.text_input("Database", value=os.getenv("PGDATABASE", "postgres"))
    dbuser = st.sidebar.text_input("User", value=os.getenv("PGUSER", "postgres"))
    dbpass = st.sidebar.text_input("Password", value=os.getenv("PGPASSWORD", ""), type="password")
    sslmode = st.sidebar.selectbox("SSL Mode", ["", "disable", "require"], index=0)

    st.sidebar.header("HF Model (SQL-only)")
    hf_token = st.sidebar.text_input("HF Token", value=os.getenv("HF_TOKEN", ""), type="password")
    model_id = st.sidebar.text_input(
        "Model",
        value=os.getenv("HF_LLM_MODEL", "defog/llama-3-sqlcoder-8b:featherless-ai"),
    )

    st.sidebar.header("Policy")
    max_rows = int(
        st.sidebar.number_input(
            "Max rows (LIMIT)",
            min_value=10,
            max_value=5000,
            value=int(os.getenv("MAX_ROWS", "200")),
            step=10,
        )
    )

    st.sidebar.header("Views (Витрины)")
    st.sidebar.caption("Creates/updates: public.v_teacher_data and public.v_teacher_topic_sessions")
    setup_btn = st.sidebar.button("Setup / Update Views", type="secondary")
    show_views = st.sidebar.checkbox("Show view SQL", value=False)

    question = st.text_input("Question", placeholder="e.g., Top 10 students by time spent last month")

    colA, colB, colC = st.columns([1, 1, 2])
    gen_btn = colA.button("Generate SQL", type="primary")
    run_btn = colB.button("Run SQL")
    debug = colC.checkbox("Show debug", value=False)

    if "final_sql" not in st.session_state:
        st.session_state.final_sql = ""
    if "schema_hint" not in st.session_state:
        st.session_state.schema_hint = ""
    if "attempts" not in st.session_state:
        st.session_state.attempts = []
    if "exec_attempts" not in st.session_state:
        st.session_state.exec_attempts = []
    if "guard_message" not in st.session_state:
        st.session_state.guard_message = ""

    def with_conn(fn):
        conn = None
        try:
            conn = get_conn(host, port, dbname, dbuser, dbpass, sslmode)
            return fn(conn)
        finally:
            if conn is not None:
                conn.close()

    if show_views:
        st.subheader("View SQL: public.v_teacher_data")
        st.code(V_TEACHER_DATA_SQL, language="sql")
        st.subheader("View SQL: public.v_teacher_topic_sessions")
        st.code(V_TEACHER_TOPIC_SESSIONS_SQL, language="sql")

    if setup_btn:
        try:
            def _do(conn):
                setup_views(conn)
                st.session_state.schema_hint = fetch_schema_hint(conn)
            with_conn(_do)
            st.success("Views created/updated successfully.")
        except Exception as e:
            st.error(f"View setup failed: {e}")

    if not st.session_state.schema_hint and host and dbname and dbuser:
        try:
            def _load(conn):
                st.session_state.schema_hint = fetch_schema_hint(conn)
            with_conn(_load)
        except Exception:
            st.session_state.schema_hint = (
                "Schema hint unavailable (could not introspect). "
                "Prefer using public.v_teacher_data / public.v_teacher_topic_sessions."
            )

    if gen_btn:
        st.session_state.exec_attempts = []
        st.session_state.attempts = []
        st.session_state.final_sql = ""
        st.session_state.guard_message = ""

        if not question.strip():
            st.error("Type a question first.")
        else:
            with st.spinner("Generating SQL..."):
                sql, attempts, guard_msg = generate_sql_with_repairs(
                    question=question,
                    hf_token=hf_token,
                    model_id=model_id,
                    schema_hint=st.session_state.schema_hint,
                    max_rows=max_rows,
                    max_repairs=2,
                )
                st.session_state.attempts = attempts
                st.session_state.guard_message = guard_msg

                if guard_msg:
                    st.info(guard_msg)
                elif sql is None:
                    st.error("SQL generation failed (blocked or could not be repaired).")
                else:
                    st.session_state.final_sql = sql
                    st.success("SQL generated (safe).")

    if st.session_state.final_sql:
        st.subheader("Generated SQL (to execute)")
        st.code(st.session_state.final_sql, language="sql")

    if run_btn:
        st.session_state.exec_attempts = []

        if st.session_state.guard_message:
            st.warning("Out-of-scope. No query executed.")
        elif not st.session_state.final_sql:
            st.error("Generate SQL first.")
        else:
            with st.spinner("Running query..."):
                def _exec(conn):
                    df, final_sql, exec_attempts = execute_with_auto_repair(
                        conn=conn,
                        question=question,
                        sql=st.session_state.final_sql,
                        hf_token=hf_token,
                        model_id=model_id,
                        schema_hint=st.session_state.schema_hint,
                        max_rows=max_rows,
                        max_repairs=2,
                    )
                    st.session_state.exec_attempts = exec_attempts
                    st.session_state.final_sql = final_sql
                    return df

                try:
                    df = with_conn(_exec)
                    if df is None:
                        st.error("Query failed (and could not be repaired).")
                    else:
                        st.subheader("Result")
                        st.dataframe(df, use_container_width=True)
                        st.subheader("Chart (auto)")
                        auto_chart(df)
                except Exception as e:
                    st.error(f"Query failed: {e}")

    if debug:
        st.subheader("Debug: Out-of-scope decision")
        st.write({"is_data_question": is_data_question(question or "")})

        st.subheader("Debug: Schema hint sent to LLM")
        st.code(st.session_state.schema_hint or "", language="text")

        if st.session_state.attempts:
            st.subheader("Debug: Generation attempts")
            for i, a in enumerate(st.session_state.attempts):
                st.markdown(f"**{i+1}. {a['stage']}** — {a['msg']}")
                if a.get("sql"):
                    st.code(a["sql"], language="sql")

        if st.session_state.exec_attempts:
            st.subheader("Debug: Execution/repair attempts")
            for i, a in enumerate(st.session_state.exec_attempts):
                st.markdown(f"**{i+1}. {a['stage']}** — {a['msg']}")
                st.code(a["sql"], language="sql")

if __name__ == "__main__":
    main()
