import os, time, math, json, threading
from datetime import date, datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd
import streamlit as st

BASE = "https://api.jquants.com/v2"

# Free plan: at most 5 API requests per minute. Shared limiter across threads.
_API_LOCK = threading.Lock()
_LAST_REQUEST_AT = 0.0

st.set_page_config(page_title="割安株AI", page_icon="📊", layout="wide")

st.markdown("""
<style>
.block-container {max-width: 1100px; padding-top: 1rem; padding-left: .8rem; padding-right: .8rem;}
[data-testid="stMetricValue"] {font-size: 1.15rem;}
.small {font-size:.85rem;color:#666;}
@media (max-width: 700px) {
  .block-container {padding-left:.55rem;padding-right:.55rem;}
  h1 {font-size:1.7rem !important;}
}
</style>
""", unsafe_allow_html=True)


def secret(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name, "")
        if value:
            return str(value)
    except Exception:
        pass
    return os.environ.get(name, default)


JQ_KEY = secret("JQUANTS_API_KEY")
OPENAI_KEY = secret("OPENAI_API_KEY")
OPENAI_MODEL = secret("OPENAI_MODEL", "gpt-5")


class JQuantsError(RuntimeError):
    pass


@st.cache_data(ttl=3600, show_spinner=False)
def jq_get(path: str, params: dict | None = None) -> list[dict]:
    if not JQ_KEY:
        raise JQuantsError("JQUANTS_API_KEY が設定されていません。")
    url = BASE + path
    p = dict(params or {})
    rows: list[dict] = []
    for _ in range(100):
        global _LAST_REQUEST_AT
        with _API_LOCK:
            elapsed = time.monotonic() - _LAST_REQUEST_AT
            if _LAST_REQUEST_AT and elapsed < 12.5:
                time.sleep(12.5 - elapsed)
            _LAST_REQUEST_AT = time.monotonic()
            r = requests.get(url, params=p, headers={"x-api-key": JQ_KEY}, timeout=45)
        if r.status_code == 429:
            raise JQuantsError("J-Quants APIのレート制限(429)です。少し待ってから再実行してください。")
        if r.status_code in (401, 403):
            raise JQuantsError("J-Quants APIキーまたは契約プランを確認してください。")
        if not r.ok:
            try:
                detail = r.json()
            except Exception:
                detail = r.text[:300]
            raise JQuantsError(f"J-Quants APIエラー {r.status_code}: {detail}")
        data = r.json()
        rows.extend(data.get("data") or data.get("Data") or [])
        token = data.get("pagination_key") or data.get("PaginationKey")
        if not token:
            break
        p["pagination_key"] = token
    return rows


def num(x):
    return pd.to_numeric(x, errors="coerce")


def normalize_code(s):
    return str(s).replace(".0", "")[:4]


def normalize_master(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    x = df.copy()
    aliases = {
        "CoName": "CompanyName", "CompanyName": "CompanyName",
        "S33Name": "Sector", "Sector33CodeName": "Sector",
        "MktName": "Market", "MarketCodeName": "Market",
    }
    for src, dst in aliases.items():
        if src in x.columns and dst not in x.columns:
            x[dst] = x[src]
    if "Code" in x.columns:
        x["Code"] = x["Code"].map(normalize_code)
    return x


@st.cache_data(ttl=86400, show_spinner=False)
def load_master() -> pd.DataFrame:
    return normalize_master(pd.DataFrame(jq_get("/equities/master")))


def latest_valuation() -> pd.DataFrame:
    # /equities/valuation supports date/code filters. Try recent calendar days
    # until a trading-day snapshot is found.
    # Free-plan data may be delayed. Use the last day confirmed by the
    # user's API error, rather than requesting today's out-of-range data.
    # Update JQUANTS_LAST_AVAILABLE_DATE in Streamlit Secrets if the
    # subscription's accessible end date changes (YYYY-MM-DD).
    cutoff_text = secret("JQUANTS_LAST_AVAILABLE_DATE", "2026-07-16")
    try:
        cutoff = date.fromisoformat(cutoff_text)
    except ValueError:
        raise JQuantsError("JQUANTS_LAST_AVAILABLE_DATE は YYYY-MM-DD 形式で設定してください。")
    last_day = min(date.today(), cutoff)
    for days_back in range(0, 15):
        d = last_day - timedelta(days=days_back)
        try:
            rows = jq_get("/equities/valuation", {"date": d.strftime("%Y%m%d")})
        except JQuantsError as exc:
            if "subscription covers" in str(exc).lower():
                raise JQuantsError(
                    "指定した日付がJ-Quants契約の取得可能期間外です。"
                    "Streamlit Secrets の JQUANTS_LAST_AVAILABLE_DATE を確認してください。"
                ) from exc
            raise
        if rows:
            x = pd.DataFrame(rows)
            x["Code"] = x["Code"].map(normalize_code)
            return x
    raise JQuantsError("契約期間内のバリュエーションデータを取得できませんでした。")


@st.cache_data(ttl=21600, show_spinner=False)
def load_valuation_snapshot() -> pd.DataFrame:
    return latest_valuation()


@st.cache_data(ttl=21600, show_spinner=False)
def load_financial_summary(code: str) -> pd.DataFrame:
    rows = jq_get("/fins/summary", {"code": code})
    x = pd.DataFrame(rows)
    if x.empty:
        return x
    x["Code"] = x["Code"].map(normalize_code) if "Code" in x else code
    for c in ["DiscDate", "DisclosedDate", "CurPerEn", "CurFYEn", "CurrentPeriodEndDate", "CurrentFiscalYearEndDate"]:
        if c in x.columns:
            x[c] = pd.to_datetime(x[c], errors="coerce")
    return x


@st.cache_data(ttl=21600, show_spinner=False)
def load_financial_details(code: str) -> pd.DataFrame:
    rows = jq_get("/fins/details", {"code": code})
    return pd.DataFrame(rows)


def choose_fy_rows(fin: pd.DataFrame, years: int) -> pd.DataFrame:
    if fin.empty:
        return fin
    x = fin.copy()
    period_col = next((c for c in ["CurPerType", "TypeOfCurrentPeriod"] if c in x.columns), None)
    if period_col:
        x = x[x[period_col].astype(str).str.upper().eq("FY")]
    end_col = next((c for c in ["CurFYEn", "CurrentFiscalYearEndDate", "CurPerEn", "CurrentPeriodEndDate"] if c in x.columns), None)
    disc_col = next((c for c in ["DiscDate", "DisclosedDate"] if c in x.columns), None)
    if end_col:
        x["FYEnd"] = pd.to_datetime(x[end_col], errors="coerce")
    elif disc_col:
        x["FYEnd"] = pd.to_datetime(x[disc_col], errors="coerce")
    else:
        x["FYEnd"] = pd.NaT
    if disc_col:
        x["DiscDate2"] = pd.to_datetime(x[disc_col], errors="coerce")
    else:
        x["DiscDate2"] = pd.NaT
    # Keep the latest disclosure for each fiscal year-end to avoid forecast revisions.
    if "FYEnd" in x:
        x = x.sort_values(["FYEnd", "DiscDate2"]).drop_duplicates("FYEnd", keep="last")
    return x.sort_values("FYEnd").tail(years + 1).reset_index(drop=True)


def calc_financial_features(fin: pd.DataFrame, details: pd.DataFrame | None, years: int) -> dict:
    out = {"ROE_5Y_Avg": math.nan, "ROE_5Y_Std": math.nan, "ROE_Obs": 0,
           "Avg_CFO_2Y": math.nan, "EquityRatio": math.nan, "NetCash": math.nan,
           "NetCashConfirmed": False, "ROE_Trend": math.nan}
    if fin.empty:
        return out
    f = choose_fy_rows(fin, years)
    if f.empty:
        return out

    def col(*names):
        for n in names:
            if n in f.columns:
                return num(f[n])
        return pd.Series(index=f.index, dtype=float)

    profit = col("Profit", "NP", "NetIncome")
    equity = col("Equity", "Eq", "NetAssets")
    assets = col("TotalAssets", "TA", "Assets")
    cfo = col("CashFlowsFromOperatingActivities", "CFO", "OperatingCashFlow")
    cash = col("CashAndEquivalents", "CashEq", "CashAndCashEquivalents")

    # ROE = net income attributable to owners / average beginning and ending equity.
    roe_vals = []
    for i in range(len(f)):
        if pd.isna(profit.iloc[i]) or pd.isna(equity.iloc[i]):
            continue
        if i > 0 and pd.notna(equity.iloc[i-1]):
            avg_eq = (equity.iloc[i] + equity.iloc[i-1]) / 2
        else:
            avg_eq = equity.iloc[i]
        if avg_eq and not pd.isna(avg_eq):
            roe_vals.append(float(profit.iloc[i] / avg_eq * 100))
    if roe_vals:
        s = pd.Series(roe_vals)
        out["ROE_5Y_Avg"] = s.tail(years).mean()
        out["ROE_5Y_Std"] = s.tail(years).std(ddof=0) if len(s.tail(years)) > 1 else 0.0
        out["ROE_Obs"] = int(len(s.tail(years)))
        if len(s) >= 3:
            out["ROE_Trend"] = float(s.tail(min(years, len(s))).iloc[-1] - s.tail(min(years, len(s))).iloc[0])

    if cfo.notna().sum() >= 1:
        out["Avg_CFO_2Y"] = cfo.dropna().tail(2).mean()
    if assets.notna().sum() and equity.notna().sum():
        a, e = assets.iloc[-1], equity.iloc[-1]
        if pd.notna(a) and a != 0 and pd.notna(e):
            out["EquityRatio"] = float(e / a)
    if cash.notna().sum():
        out["Cash"] = float(cash.dropna().iloc[-1])

    # Premium-only details: try to identify interest-bearing debt from the nested FS map.
    debt = math.nan
    if details is not None and not details.empty and "FS" in details.columns:
        latest = details.iloc[-1]["FS"]
        if isinstance(latest, dict):
            debt_terms = [
                "interest-bearing", "borrowings", "loans payable", "bonds payable",
                "short-term borrowings", "long-term borrowings", "current portion of long-term debt",
            ]
            total = 0.0
            found = False
            for k, v in latest.items():
                kl = str(k).lower()
                if any(t in kl for t in debt_terms) and "lease" not in kl:
                    vv = pd.to_numeric(v, errors="coerce")
                    if pd.notna(vv):
                        total += float(vv)
                        found = True
            if found:
                debt = total
    if pd.notna(out.get("Cash", math.nan)) and pd.notna(debt):
        out["NetCash"] = out["Cash"] - debt
        out["NetCashConfirmed"] = True
    return out


def sector_excluded(master: pd.DataFrame) -> pd.Series:
    sector = master.get("Sector", pd.Series(index=master.index, dtype=str)).fillna("").astype(str)
    company = master.get("CompanyName", pd.Series(index=master.index, dtype=str)).fillna("").astype(str)
    bad = ["銀行", "証券", "商品先物", "保険", "その他金融", "不動産"]
    is_bad_sector = sector.apply(lambda s: any(x in s for x in bad))
    name_bad = company.str.contains("ETF|REIT|投資法人|インフラファンド", case=False, regex=True, na=False)
    return is_bad_sector | name_bad


def score_one(vrow, features, ratio_limit, strict_net_cash):
    roe = float(vrow["ROE_pct"])
    per = float(vrow["PER"])
    fair = roe * 2
    ratio = per / fair if fair > 0 else math.inf
    score = 0.0

    # 45: valuation. Lower PER / fair PER is better.
    under = max(0.0, min(1.0, 1.0 - ratio))
    score += under * 45

    avg_roe = features.get("ROE_5Y_Avg", math.nan)
    std_roe = features.get("ROE_5Y_Std", math.nan)
    trend = features.get("ROE_Trend", math.nan)
    if pd.notna(avg_roe) and avg_roe > 0:
        gap = abs(roe - avg_roe) / abs(avg_roe)
        stability = 1.0 / (1.0 + max(0.0, std_roe if pd.notna(std_roe) else 99.0))
        trend_bonus = 1.0 if pd.isna(trend) else max(0.0, min(1.0, (trend + 10) / 20))
        roe_score = (0.55 * (1 - min(1, gap)) + 0.30 * stability + 0.15 * trend_bonus) * 20
    else:
        roe_score = 0.0
    score += roe_score

    mcap = float(vrow["MktCap"]) if pd.notna(vrow.get("MktCap")) else math.nan
    cfo = features.get("Avg_CFO_2Y", math.nan)
    cfo_yield = cfo / mcap if pd.notna(cfo) and pd.notna(mcap) and mcap > 0 else math.nan
    cfo_score = max(0.0, min(0.20, cfo_yield)) / 0.20 * 20 if pd.notna(cfo_yield) else 0.0
    score += cfo_score

    eqr = features.get("EquityRatio", math.nan)
    eq_score = max(0.0, min(0.50, eqr)) / 0.50 * 15 if pd.notna(eqr) else 0.0
    score += eq_score

    netcash = features.get("NetCash", math.nan)
    net_ok = pd.notna(netcash) and netcash > 0
    if strict_net_cash and not net_ok:
        return None
    if not (roe > 0 and per > 0 and ratio <= ratio_limit):
        return None

    result = dict(vrow)
    result.update(features)
    result.update({"Fair_PER": fair, "PER_Fair_Ratio": ratio, "CFO_to_MktCap": cfo_yield,
                   "UndervaluationScore": under*45, "ROESustainabilityScore": roe_score,
                   "CFOYieldScore": cfo_score, "EquityRatioScore": eq_score, "TotalScore": score,
                   "NetCashOK": net_ok})
    return result


def heuristic_reason(row: pd.Series) -> str:
    reasons = []
    ratio = row.get("PER_Fair_Ratio", math.nan)
    if pd.notna(ratio) and ratio <= 0.35:
        reasons.append("理論PERに対してかなり低い")
    elif pd.notna(ratio) and ratio <= 0.50:
        reasons.append("理論PERに対して割安")
    avg = row.get("ROE_5Y_Avg", math.nan)
    roe = row.get("ROE_pct", math.nan)
    if pd.notna(avg) and pd.notna(roe):
        if roe >= avg * 0.9:
            reasons.append("ROEが過去平均から大きく崩れていない")
        else:
            reasons.append("現在ROEが5年平均を下回るため要確認")
    cfo = row.get("CFO_to_MktCap", math.nan)
    if pd.notna(cfo) and cfo >= 0.05:
        reasons.append("営業CF利回りが高い")
    eq = row.get("EquityRatio", math.nan)
    if pd.notna(eq) and eq >= 0.50:
        reasons.append("自己資本比率50%以上")
    if row.get("NetCashOK"):
        reasons.append("ネットキャッシュがプラス")
    return "、".join(reasons) if reasons else "数値条件を満たすが、財務データの追加確認が必要"


def openai_analysis(row: pd.Series) -> str:
    if not OPENAI_KEY:
        return ""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_KEY)
        payload = {
            "銘柄コード": row.get("Code"), "会社": row.get("CompanyName"),
            "PER": row.get("PER"), "ROE": row.get("ROE_pct"),
            "理論PER": row.get("Fair_PER"), "PER/理論PER": row.get("PER_Fair_Ratio"),
            "ROE5年平均": row.get("ROE_5Y_Avg"), "ROE5年標準偏差": row.get("ROE_5Y_Std"),
            "ROEトレンド": row.get("ROE_Trend"), "営業CF/時価総額": row.get("CFO_to_MktCap"),
            "自己資本比率": row.get("EquityRatio"), "ネットキャッシュ": row.get("NetCash"),
        }
        prompt = f"""あなたは日本株のファンダメンタル分析担当です。以下の数値だけを根拠に、なぜこの銘柄が割安に見えるのかを分析してください。\n\n{json.dumps(payload, ensure_ascii=False, default=str)}\n\n次の順で日本語で簡潔に回答:\n1. 割安の根拠\n2. 割安が正当化される可能性\n3. ROE持続性の懸念\n4. 財務安全性\n5. 追加確認すべき決算・開示情報\n断定的な投資推奨はしない。数値にない事実は推測と明記する。"""
        resp = client.responses.create(model=OPENAI_MODEL, input=prompt)
        return resp.output_text
    except Exception as e:
        return f"AI分析を実行できませんでした: {e}"


def build_results(master, val, years, ratio_limit, financial_limit, strict_net_cash, use_details):
    v = val.copy()
    for c in ["PER", "ROE", "MktCap"]:
        if c in v:
            v[c] = num(v[c])
    required = ["Code", "PER", "ROE"]
    missing = [c for c in required if c not in v.columns]
    if missing:
        raise JQuantsError("株価指標APIの項目が想定と異なります: " + ", ".join(missing))
    v = v.dropna(subset=required).copy()
    v = v[(v["PER"] > 0) & (v["ROE"] > 0)]
    v["ROE_pct"] = v["ROE"] * 100
    v["Fair_PER"] = v["ROE_pct"] * 2
    v["PER_Fair_Ratio"] = v["PER"] / v["Fair_PER"]

    # First-stage valuation screen before expensive financial API calls.
    pool = v[v["PER_Fair_Ratio"] <= ratio_limit].sort_values("PER_Fair_Ratio").head(financial_limit).copy()
    if pool.empty:
        return pd.DataFrame(), 0, 0

    allowed = set(master["Code"].astype(str)) if "Code" in master else set()
    pool = pool[pool["Code"].isin(allowed)]

    records = []
    failures = 0
    progress = st.progress(0, text="財務データを取得しています…")
    codes = pool["Code"].tolist()
    # Conservative concurrency. Summary/details have a 60 req/min endpoint cap.
    workers = min(5, max(1, len(codes)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(load_financial_summary, c): c for c in codes}
        done = 0
        for fut in as_completed(futures):
            code = futures[fut]
            done += 1
            progress.progress(done/len(codes), text=f"財務データ {done}/{len(codes)}: {code}")
            try:
                fin = fut.result()
                details = None
                if use_details:
                    try:
                        details = load_financial_details(code)
                    except Exception:
                        details = None
                features = calc_financial_features(fin, details, years)
                vr = pool[pool["Code"] == code].iloc[0].to_dict()
                row = score_one(vr, features, ratio_limit, strict_net_cash)
                if row is not None:
                    row["CompanyName"] = master.loc[master["Code"] == code, "CompanyName"].iloc[0] if "CompanyName" in master.columns and not master.loc[master["Code"] == code].empty else code
                    row["Sector"] = master.loc[master["Code"] == code, "Sector"].iloc[0] if "Sector" in master.columns and not master.loc[master["Code"] == code].empty else ""
                    records.append(row)
            except Exception:
                failures += 1
    progress.empty()
    result = pd.DataFrame(records)
    if result.empty:
        return result, len(pool), failures
    result["HeuristicReason"] = result.apply(heuristic_reason, axis=1)
    return result.sort_values("TotalScore", ascending=False).reset_index(drop=True), len(pool), failures


st.title("割安株AI")
st.caption("ROE × 2 = 理論PER。無料プラン対応（データは約12週間遅延）。数値スクリーニング → AI二次分析")
st.info("無料プランではデータが遅延し、ROEの5年履歴とネットキャッシュの厳密判定はできません。取得可能な期間の情報を使った参考スクリーニングです。")

with st.sidebar:
    st.header("設定")
    st.write("J-Quants API: " + ("接続設定済み" if JQ_KEY else "未設定"))
    years = st.slider("ROE履歴", 3, 5, 5)
    ratio_limit = st.slider("割安判定：実PER / 理論PER", 0.20, 0.80, 0.50, 0.05)
    financial_limit = st.slider("財務分析する上位候補数", 5, 50, 10, 5, help="無料プランは毎分5回までのため、まず10社で試してください。")
    strict_net_cash = st.checkbox("ネットキャッシュ > 0 を必須", value=False, help="無料プランでは有利子負債の詳細が確認できないため、ONにすると候補が0件になる可能性があります。")
    use_details = False
    st.caption("無料プラン対応：Premium専用の財務詳細APIは呼び出しません。")
    min_score = st.slider("最低スコア", 0, 100, 50)
    run = st.button("スクリーニング実行", type="primary", use_container_width=True)

if not JQ_KEY:
    st.error("JQUANTS_API_KEY が未設定です。StreamlitのSecretsに設定してください。")
    st.stop()

if run or "result" not in st.session_state:
    try:
        with st.spinner("銘柄一覧と取得可能なバリュエーションを取得しています…"):
            master = load_master()
            master = master.loc[~sector_excluded(master)].copy()
            val = load_valuation_snapshot()
            val["Code"] = val["Code"].map(normalize_code)
            val = val[val["Code"].isin(set(master["Code"]))]
        with st.spinner("割安候補を精査しています…"):
            result, pool_n, failures = build_results(master, val, years, ratio_limit, financial_limit, strict_net_cash, use_details)
            result = result[result["TotalScore"] >= min_score].copy() if not result.empty else result
            st.session_state["result"] = result
            st.session_state["pool_n"] = pool_n
            st.session_state["failures"] = failures
            st.session_state["updated"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    except Exception as e:
        st.error(str(e))
        st.caption("取得期間外の場合はJ-Quantsの契約期間をご確認ください。無料プランのデータは約12週間遅延します。")
        st.stop()

r = st.session_state.get("result", pd.DataFrame())
st.caption(f"実行: {st.session_state.get('updated','-')} / 基準日: {secret('JQUANTS_LAST_AVAILABLE_DATE', '2026-07-16')}以前 / 財務分析対象: {st.session_state.get('pool_n',0)}社 / 最終候補: {len(r)}社")
if st.session_state.get("failures", 0):
    st.caption(f"取得失敗: {st.session_state['failures']}社（API制限・プラン不足等の可能性）")

if r.empty:
    st.warning("条件に合う銘柄がありません。割安比率・最低スコア・ネットキャッシュ条件を調整してください。")
    st.stop()

cols = st.columns(4)
cols[0].metric("候補数", len(r))
cols[1].metric("PER/理論PER 中央値", f"{r['PER_Fair_Ratio'].median():.2f}")
cols[2].metric("ROE 中央値", f"{r['ROE_pct'].median():.1f}%")
cols[3].metric("総合スコア 中央値", f"{r['TotalScore'].median():.1f}")

show = pd.DataFrame({
    "コード": r["Code"],
    "会社": r["CompanyName"],
    "PER": r["PER"].round(1),
    "ROE": r["ROE_pct"].round(1).astype(str) + "%",
    "理論PER": r["Fair_PER"].round(1),
    "割安比率": (r["PER_Fair_Ratio"]*100).round(1).astype(str) + "%",
    "ネットキャッシュ": r["NetCash"].round(0),
    "営業CF/時価総額": (r["CFO_to_MktCap"]*100).round(1).astype(str) + "%",
    "自己資本比率": (r["EquityRatio"]*100).round(1).astype(str) + "%",
    "総合スコア": r["TotalScore"].round(1),
})
st.dataframe(show, use_container_width=True, hide_index=True)

st.download_button("候補一覧CSV", r.to_csv(index=False).encode("utf-8-sig"), "waribiki_candidates.csv", "text/csv", use_container_width=True)

st.subheader("銘柄詳細")
code = st.selectbox("銘柄", r["Code"].tolist())
x = r[r["Code"] == code].iloc[0]

c1, c2, c3, c4 = st.columns(4)
c1.metric("PER", f"{x['PER']:.1f}x")
c2.metric("ROE", f"{x['ROE_pct']:.1f}%")
c3.metric("理論PER", f"{x['Fair_PER']:.1f}x")
c4.metric("割安比率", f"{x['PER_Fair_Ratio']*100:.1f}%")

st.markdown("#### 数値判定")
st.write(f"**{x['CompanyName']}**：実PERは理論PERの **{x['PER_Fair_Ratio']*100:.1f}%**。")
st.write(f"ROE 5年平均 **{x.get('ROE_5Y_Avg', math.nan):.1f}%** / 現在ROE **{x['ROE_pct']:.1f}%** / ROE観測数 **{int(x.get('ROE_Obs',0))}年**")
st.write(f"営業CF 2期平均/時価総額 **{x.get('CFO_to_MktCap', math.nan)*100:.1f}%** / 自己資本比率 **{x.get('EquityRatio', math.nan)*100:.1f}%** / ネットキャッシュ **{x.get('NetCash', math.nan):,.0f}**")
st.write("**一次判定:** " + x.get("HeuristicReason", ""))

st.markdown("#### AI二次分析")
if OPENAI_KEY:
    if st.button("この銘柄をAI分析", use_container_width=True):
        with st.spinner("AIが割安理由を分析しています…"):
            st.write(openai_analysis(x))
else:
    st.info("OPENAI_API_KEYをSecretsに追加すると、ここで『なぜ安いのか／安さが正当か』の二次分析を実行できます。未設定でも数値スクリーニングは動作します。")

st.caption("重要：このアプリはスクリーニング補助です。ROE×2は独自の仮定であり、理論価格を保証するものではありません。")
