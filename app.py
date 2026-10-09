import os
import math
import time
import threading
import io
import zipfile
import re
import html
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests
import streamlit as st

BASE = 'https://api.jquants.com/v2'
_LOCK = threading.Lock()
_LAST_CALL = 0.0

st.set_page_config(page_title='割安株AI', page_icon='📊', layout='wide')
st.markdown("""<style>
/* Bright, friendly mobile-first theme. Financial calculations unchanged. */
:root{color-scheme:light}
.stApp,[data-testid="stAppViewContainer"]{background:linear-gradient(155deg,#fff9f2 0%,#f5fcf8 50%,#f5f7ff 100%);color:#26364b}
[data-testid="stSidebar"],[data-testid="stSidebarContent"]{background:#f1faf6}
.block-container{max-width:1050px;padding:1.4rem 1rem 4rem}
h1,h2,h3{color:#26364b!important;letter-spacing:-.025em}
h1{font-weight:850!important}h2,h3{font-weight:750!important}
p,li,label{color:#34465b}
[data-testid="stMetric"]{background:#fff;border:1px solid #e4eceb;border-radius:20px;padding:14px;box-shadow:0 5px 18px rgba(42,77,65,.055)}
[data-testid="stMetricValue"]{font-size:1.55rem;color:#168b71;font-weight:800}
[data-testid="stMetricLabel"]{color:#5b6b7b}
.stButton>button[kind="primary"],button[kind="primary"]{background:#24c59b;color:#102d28;border:none;border-radius:16px;font-weight:800;min-height:48px;box-shadow:0 4px 0 #159575}
.stButton>button[kind="primary"]:hover,button[kind="primary"]:hover{background:#3fdbb0;color:#102d28}
.stButton>button:not([kind="primary"]),[data-testid="stDownloadButton"] button{border-radius:14px;border:1px solid #b9e7d8;background:#fff;color:#167c67;font-weight:650}
[data-testid="stDataFrame"],[data-testid="stExpander"]{border-radius:18px;overflow:hidden}
[data-testid="stExpander"]{background:#fff;border:1px solid #e5eeeb}
[data-testid="stAlert"]{border-radius:17px}
[data-testid="stTextInput"] input,[data-testid="stNumberInput"] input{border-radius:12px}
a{color:#0b9876!important}
.roe-row{display:flex;align-items:center;gap:12px;background:#fff;border:1px solid #e1eee9;border-radius:15px;padding:12px 14px;margin:8px 0;box-shadow:0 3px 12px rgba(33,90,72,.04)}
.roe-year{min-width:56px;color:#5e6f7d;font-size:.94rem;font-weight:650}
.roe-track{height:11px;flex:1;background:#e4f3ed;border-radius:10px;overflow:hidden}
.roe-fill{height:100%;border-radius:10px;background:#27c69e}
.roe-fill.negative{background:#f17d8a}
.roe-value{min-width:76px;text-align:right;color:#128b70;font-size:1.2rem;font-weight:800;font-variant-numeric:tabular-nums}
.roe-value.negative{color:#d9506b}
.cf-value{font-size:clamp(11px,2.6vw,15px);font-weight:750;color:#128b70;white-space:nowrap;text-align:right;min-width:125px;font-variant-numeric:tabular-nums}
.cf-value.negative{color:#d9506b}
.welcome-card{background:linear-gradient(120deg,#ddfff1,#e6f3ff 70%,#fff0dd);border:1px solid #d7eee4;border-radius:24px;padding:20px 22px;margin:0 0 20px;box-shadow:0 8px 24px rgba(37,124,96,.06)}
.welcome-kicker{font-size:.8rem;font-weight:800;color:#198c71;letter-spacing:.08em}
.welcome-card h1{margin:.25rem 0 .35rem;font-size:2.1rem!important;color:#1a6155!important}
.welcome-card p{margin:0;color:#516777;font-size:.97rem;line-height:1.6}
.pill-note{display:inline-block;background:#e7f9f1;color:#137b61;border-radius:999px;padding:5px 11px;font-size:.8rem;font-weight:700;margin:8px 5px 0 0}
@media(max-width:640px){.block-container{padding:.85rem .72rem 4rem}h1{font-size:1.8rem!important}h2{font-size:1.4rem!important}[data-testid="stMetricValue"]{font-size:1.22rem!important}.roe-row{gap:8px;padding:10px}.roe-value{font-size:1.05rem;min-width:67px}.cf-value{min-width:115px;font-size:11px}.welcome-card{padding:16px;border-radius:19px}.welcome-card h1{font-size:1.65rem!important}.welcome-card p{font-size:.86rem}}
</style>""", unsafe_allow_html=True)


def japanese_large_number(value):
    """円単位の金額を日本語の位取りで表示する。"""
    try:
        number = float(value)
        if not math.isfinite(number):
            return '—'
    except (TypeError, ValueError):
        return '—'
    sign = '−' if number < 0 else ''
    integer = int(round(abs(number)))
    if integer == 0:
        return '0円'
    parts = []
    for unit_value, unit_label in ((10**12, '兆'), (10**8, '億'), (10**4, '万'), (10**3, '千')):
        amount, integer = divmod(integer, unit_value)
        if amount:
            parts.append(f'{amount:,}{unit_label}')
    if integer:
        parts.append(f'{integer:,}')
    return sign + ''.join(parts) + '円'


def secret(name, default=''):
    try:
        value = st.secrets.get(name)
        if value is not None and str(value).strip():
            return str(value)
    except Exception:
        pass
    return os.getenv(name, default)


API_KEY = secret('JQUANTS_API_KEY')
EDINET_KEY = secret('EDINET_API_KEY')
EDINETDB_KEY = secret('EDINETDB_API_KEY')
OPENAI_KEY = secret('OPENAI_API_KEY')
OPENAI_MODEL = secret('OPENAI_MODEL', 'gpt-5')


class JQuantsError(RuntimeError):
    pass


@st.cache_data(ttl=3600, show_spinner=False)
def jq_get(path, params=None):
    global _LAST_CALL
    if not API_KEY:
        raise JQuantsError('JQUANTS_API_KEY が未設定です。')
    rows = []
    p = dict(params or {})
    for _ in range(100):
        with _LOCK:
            wait = 12.5 - (time.monotonic() - _LAST_CALL)
            if _LAST_CALL and wait > 0:
                time.sleep(wait)
            _LAST_CALL = time.monotonic()
            try:
                response = requests.get(BASE + path, params=p, headers={'x-api-key': API_KEY}, timeout=45)
            except requests.RequestException as exc:
                raise JQuantsError(f'通信エラー: {exc}') from exc
        if response.status_code == 429:
            raise JQuantsError('API回数制限です。1分ほど待って再実行してください。')
        if response.status_code in (401, 403):
            raise JQuantsError('APIキーまたは契約プランを確認してください。')
        if not response.ok:
            raise JQuantsError(f'J-Quants APIエラー {response.status_code}: {response.text[:250]}')
        payload = response.json()
        rows.extend(payload.get('data') or payload.get('Data') or [])
        token = payload.get('pagination_key') or payload.get('PaginationKey')
        if not token:
            return rows
        p['pagination_key'] = token
    raise JQuantsError('APIのページ数が上限を超えました。')


def number(value):
    return pd.to_numeric(value, errors='coerce')


def numeric_column(frame, *names):
    for name in names:
        if name in frame.columns:
            return number(frame[name])
    return pd.Series(float('nan'), index=frame.index, dtype='float64')


def code4(value):
    text = str(value).strip()
    if text.endswith('.0'):
        text = text[:-2]
    return text[:4]


@st.cache_data(ttl=86400, show_spinner=False)
def load_master():
    frame = pd.DataFrame(jq_get('/equities/master'))
    if frame.empty or 'Code' not in frame:
        raise JQuantsError('銘柄一覧を取得できませんでした。')
    for old, new in [('CoName', 'CompanyName'), ('S33Name', 'Sector'), ('Sector33CodeName', 'Sector')]:
        if old in frame and new not in frame:
            frame[new] = frame[old]
    frame['Code'] = frame['Code'].map(code4)
    return frame


@st.cache_data(ttl=21600, show_spinner=False)
def load_valuation():
    cutoff_text = secret('JQUANTS_LAST_AVAILABLE_DATE', '2026-07-16')
    try:
        cutoff = date.fromisoformat(cutoff_text)
    except ValueError as exc:
        raise JQuantsError('JQUANTS_LAST_AVAILABLE_DATE は YYYY-MM-DD 形式にしてください。') from exc
    last_day = min(date.today(), cutoff)
    for days_back in range(15):
        day = last_day - timedelta(days=days_back)
        rows = jq_get('/equities/valuation', {'date': day.strftime('%Y%m%d')})
        if rows:
            frame = pd.DataFrame(rows)
            frame['Code'] = frame['Code'].map(code4)
            return frame, day.isoformat()
    raise JQuantsError('契約期間内の株価指標データが見つかりませんでした。')


@st.cache_data(ttl=21600, show_spinner=False)
def load_financials(code):
    return pd.DataFrame(jq_get('/fins/summary', {'code': code}))


def annual_rows(frame, years):
    if frame.empty:
        return frame
    frame = frame.copy()
    period = next((c for c in ('CurPerType', 'TypeOfCurrentPeriod') if c in frame), None)
    if period:
        frame = frame.loc[frame[period].astype(str).str.upper().eq('FY')].copy()
    if frame.empty:
        return frame
    end_col = next((c for c in ('CurFYEn', 'CurrentFiscalYearEndDate', 'CurPerEn', 'CurrentPeriodEndDate') if c in frame), None)
    disclosed = next((c for c in ('DiscDate', 'DisclosedDate') if c in frame), None)
    if end_col is None:
        return frame.iloc[0:0]
    frame['_year_end'] = pd.to_datetime(frame[end_col], errors='coerce')
    frame['_disclosed'] = pd.to_datetime(frame[disclosed], errors='coerce') if disclosed else pd.NaT
    frame = frame.dropna(subset=['_year_end'])
    frame = frame.sort_values(['_year_end', '_disclosed']).drop_duplicates('_year_end', keep='last')
    return frame.tail(years + 1).reset_index(drop=True)


# EDINETPORTAL is an independent third-party service, not the FSA's official API.
# fiscalYear is camelCase in the observed live API response (11 records with roe).
# Only historical financial fields are used; valuation stays with J-Quants.
EDINETPORTAL_BASE = 'https://edinetportal.kazuma-45a.workers.dev'


@st.cache_data(ttl=86400, show_spinner=False)
def load_edinetportal_financials(code):
    url = f'{EDINETPORTAL_BASE}/v1/companies/{code}/financials'
    response = requests.get(url, timeout=20, headers={'Accept': 'application/json', 'User-Agent': 'waribiki-ai/1.2'})
    response.raise_for_status()
    return extract_annual_records(response.json())


def extract_annual_records(payload):
    """Accept arrays and documented/common wrapper shapes; never invent values."""
    if isinstance(payload, list):
        return [v for v in payload if isinstance(v, dict)]
    if isinstance(payload, dict):
        for key in ('financials', 'data', 'results', 'history', 'items', 'annual', 'years'):
            val = payload.get(key)
            if isinstance(val, list):
                return [v for v in val if isinstance(v, dict)]
            if isinstance(val, dict):
                nested = extract_annual_records(val)
                if nested:
                    return nested
    return []


@st.cache_data(ttl=86400, show_spinner=False)
def edinetdb_lookup(code):
    """EDINET DB search is optional and requires the user's own API key."""
    url = 'https://edinetdb.jp/v1/search'
    response = requests.get(url, params={'q': code}, headers={'X-API-Key': EDINETDB_KEY}, timeout=15)
    response.raise_for_status()
    data = response.json()
    entries = extract_annual_records(data)
    for entry in entries:
        stock = str(entry.get('securities_code') or entry.get('stock_code') or entry.get('ticker') or '').strip()
        if stock[:4] == code[:4]:
            value = entry.get('edinet_code') or entry.get('code')
            if value and str(value).startswith('E'):
                return str(value)
    return None


@st.cache_data(ttl=86400, show_spinner=False)
def load_edinetdb_financials(code):
    edinet_code = edinetdb_lookup(code)
    if not edinet_code:
        return [], '証券コードからEDINETコードを特定できません'
    response = requests.get(
        f'https://edinetdb.jp/v1/companies/{edinet_code}/financials',
        params={'years': 6, 'period': 'annual'},
        headers={'X-API-Key': EDINETDB_KEY, 'Accept': 'application/json'}, timeout=20)
    response.raise_for_status()
    return extract_annual_records(response.json()), edinet_code


@st.cache_data(ttl=21600, show_spinner=False)
def load_latest_earnings(stock_code):
    """Latest disclosed earnings for one selected company; not bulk fetched."""
    if not EDINETDB_KEY:
        return None, 'EDINETDB_API_KEYが未設定です'
    try:
        edinet_code = edinetdb_lookup(stock_code)
        if not edinet_code:
            return None, '証券コードと企業の対応を確認できません'
        response = requests.get(
            f'https://edinetdb.jp/v1/companies/{edinet_code}/earnings',
            params={'limit': 12},
            headers={'X-API-Key': EDINETDB_KEY, 'Accept': 'application/json'},
            timeout=20)
        response.raise_for_status()
        payload = response.json()
        data = payload.get('data', payload) if isinstance(payload, dict) else payload
        if isinstance(data, dict):
            rows = data.get('earnings', [])
        elif isinstance(data, list):
            rows = data
        else:
            rows = []
        rows = [r for r in rows if isinstance(r, dict) and r.get('disclosure_date')]
        rows.sort(key=lambda r: (str(r.get('disclosure_date', '')), bool(r.get('is_correction'))), reverse=True)
        return (rows[0] if rows else None), ('' if rows else '決算短信のデータがありません')
    except (requests.RequestException, ValueError, TypeError) as exc:
        return None, f'決算速報の取得に失敗しました（{type(exc).__name__}）'


def safe_numeric(value):
    try:
        n = float(value)
        return n if math.isfinite(n) else None
    except (TypeError, ValueError):
        return None


def field_value(row, *names):
    # Only explicitly identified fields are read. No unit guessing or zero filling.
    for name in names:
        if name in row and row[name] is not None:
            value = pd.to_numeric(row[name], errors='coerce')
            if pd.notna(value) and math.isfinite(float(value)):
                return float(value)
    return math.nan


def portal_features(rows, years, cutoff):
    out = dict(ROE_Avg=math.nan, ROE_Std=math.nan, ROE_Obs=0,
               ROE_Trend=math.nan, CFO_Avg_Yen=math.nan, CFO_Obs=0,
               CFO_Positive_Years=0, CFO_Volatility=math.nan,
               EquityRatio=math.nan, NetCash=math.nan, NetCashConfirmed=False,
               FinancialSource='EDINETPORTAL', FinancialYears=0, ROE_Series=[], CFO_Series=[], CFO_Source='EDINETPORTAL')
    observations = []
    equity_by_year = {}
    net_by_year = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        # Exclude financial statements published after the J-Quants valuation date.
        disclosed = next((row.get(k) for k in ('submit_date', 'submitted_at', 'filing_date', 'disclosure_date', 'filed_at', 'filingDate', 'disclosureDate', 'submittedAt') if row.get(k)), None)
        if disclosed:
            parsed = pd.to_datetime(disclosed, errors='coerce', utc=True)
            if pd.notna(parsed) and parsed.date() > cutoff:
                continue
        fiscal = next((row.get(k) for k in ('fiscalYear', 'fiscal_year', 'year', 'fy', 'period_end', 'fiscal_year_end', 'fiscalYearEnd') if row.get(k) is not None), None)
        if fiscal is None:
            continue
        try:
            year = int(str(fiscal).upper().replace('FY', '')[:4])
        except (TypeError, ValueError):
            continue
        if year > cutoff.year:
            continue
        # Without a verified publication date, use only fiscal years that ended
        # before the valuation year. This avoids relying on current-year data.
        if disclosed is None and year >= cutoff.year:
            continue
        roe = field_value(row, 'roe', 'return_on_equity', 'roe_percent', 'roe_pct')
        equity_by_year[year] = field_value(row, 'shareholders_equity', 'equity_attributable_to_owners_of_parent', 'equity', 'shareholdersEquity')
        net_by_year[year] = field_value(row, 'net_income_attributable_to_owners_of_parent', 'net_income', 'profit_attributable_to_owners_of_parent', 'netIncome')
        eq_ratio = field_value(row, 'equity_ratio', 'equity_ratio_percent', 'equity_to_asset_ratio', 'equityRatio')
        # EDINETPORTAL documents percentage ratios as 0..100 (not 0..1).
        if pd.notna(roe) and not (-200 <= roe <= 500):
            roe = math.nan
        if pd.notna(eq_ratio):
            eq_ratio = eq_ratio / 100 if 0 <= eq_ratio <= 100 else math.nan
        cfo = field_value(row, 'operating_cash_flow', 'cash_flow_from_operations', 'cash_flows_from_operating_activities', 'cash_flow_operating', 'operatingCashFlow', 'cashFlowFromOperatingActivities', 'cashFlowsFromOperatingActivities', 'cashflowFromOperations', 'cashFlowOperating')
        observations.append((year, roe, eq_ratio, cfo))
    # Where ROE is absent, derive it only with consecutive-year comparable equity.
    derived = []
    for yr, roe, eq_ratio, cfo in observations:
        if pd.isna(roe) and yr - 1 in equity_by_year:
            cur, prev, profit = equity_by_year.get(yr), equity_by_year.get(yr - 1), net_by_year.get(yr)
            if all(pd.notna(v) for v in (cur, prev, profit)) and cur > 0 and prev > 0:
                roe = profit / ((cur + prev) / 2) * 100
                if not (-200 <= roe <= 500):
                    roe = math.nan
        derived.append((yr, roe, eq_ratio, cfo))
    observations = derived
    observations.sort(key=lambda x: x[0])
    dedup = {r[0]: r for r in observations}
    recent = [dedup[y] for y in sorted(dedup)][-years:]
    out['FinancialYears'] = len(recent)
    out['ROE_Series'] = [{'年度': r[0], 'ROE (%)': float(r[1])} for r in recent if pd.notna(r[1])]
    out['CFO_Series'] = [{'年度': r[0], '営業CF (元データ)': float(r[3])} for r in recent if pd.notna(r[3])]
    roes = [r[1] for r in recent if pd.notna(r[1])]
    if roes:
        out['ROE_Avg'] = float(pd.Series(roes).mean())
        out['ROE_Std'] = float(pd.Series(roes).std(ddof=0)) if len(roes) >= 2 else math.nan
        out['ROE_Obs'] = len(roes)
        if len(roes) >= 2:
            out['ROE_Trend'] = roes[-1] - roes[0]
    ratios = [r[2] for r in recent if pd.notna(r[2])]
    if ratios:
        out['EquityRatio'] = ratios[-1]
    # EDINETPORTAL amounts must be verified against API metadata before using
    # them for yields: different financial APIs may return yen or million yen.
    # CFO history is still useful for sign/consistency, regardless of unit.
    cfos = [r[3] for r in recent if pd.notna(r[3])]
    if cfos:
        out['CFO_Obs'] = len(cfos)
        out['CFO_Positive_Years'] = sum(c > 0 for c in cfos)
        if len(cfos) >= 2 and sum(cfos) / len(cfos) > 0:
            out['CFO_Volatility'] = abs(cfos[-1] - cfos[-2]) / max(abs((cfos[-1] + cfos[-2]) / 2), 1)
    return out


# EDINET official API v2: locate annual reports near J-Quants annual disclosure dates.
# Never use unaudited / unmatched documents as financial evidence.
EDINET_BASE = 'https://api.edinet-fsa.go.jp/api/v2'


@st.cache_data(ttl=86400 * 7, show_spinner=False)
def edinet_daily_list(day):
    if not EDINET_KEY:
        return []
    response = requests.get(f'{EDINET_BASE}/documents.json',
                            params={'date': day, 'type': 2, 'Subscription-Key': EDINET_KEY},
                            timeout=22)
    response.raise_for_status()
    data = response.json()
    return data.get('results', []) if isinstance(data, dict) else []


def edinet_reports(code, jq_frame, years, cutoff):
    """Search around fiscal-year end, not earnings announcement date.

    EDINET lists are indexed by SUBMISSION date. Annual reports generally arrive
    weeks after fiscal year end; FY disclosures are not the annual report.
    Only reports submitted by the J-Quants valuation cutoff are eligible.
    """
    annual = annual_rows(jq_frame, years + 1)
    if annual.empty or '_year_end' not in annual:
        return [], '検索不可：J-Quantsの決算期末日がありません'
    ends = sorted(set(x.date() for x in annual['_year_end'].dropna()
                      if x.date() <= cutoff), reverse=True)
    if not ends:
        return [], '検索不可：基準日以前の決算期末日なし'
    # Search latest two fiscal periods; comparative contexts may contain 5 years.
    # Bounded requests, caching and early exit avoid hundreds of calls per stock.
    days = []
    for fiscal_end in ends[:2]:
        for offset in range(45, 126):
            d = fiscal_end + timedelta(days=offset)
            if d <= cutoff and d.weekday() < 5:
                days.append(d.isoformat())
    days = sorted(set(days), reverse=True)
    found = {}
    errors = []
    checked = 0
    for day in days:
        try:
            docs = edinet_daily_list(day)
            checked += 1
        except (requests.RequestException, ValueError, KeyError) as exc:
            errors.append(f'{day}: {type(exc).__name__}: {str(exc)[:75]}')
            if len(errors) >= 5 and not found:
                break
            continue
        for doc in docs:
            sec = str(doc.get('secCode') or '').strip()[:4]
            if sec != str(code)[:4] or str(doc.get('docTypeCode')) != '120':
                continue
            if str(doc.get('xbrlFlag')) != '1':
                continue
            doc_id = doc.get('docID')
            if doc_id:
                found[doc_id] = {'id': doc_id, 'date': day,
                                 'name': doc.get('filerName', ''),
                                 'periodEnd': doc.get('periodEnd', '')}
        if len(found) >= 2:
            break
    reports = sorted(found.values(), key=lambda r: r['date'], reverse=True)[:years]
    if reports:
        return reports, f'報告書{len(reports)}件・検索{checked}日'
    if errors:
        return [], f'APIエラー（検索{checked}日）：{errors[0]}'
    return [], f'報告書なし（検索{checked}日、期末日基準、証券コード{code}）'


@st.cache_data(ttl=86400 * 30, show_spinner=False)
def edinet_xbrl_values(doc_id):
    """Parse official EDINET CSV: multiple comparative years from one annual report.

    Only primary consolidated contexts and explicit yen monetary units are accepted.
    """
    response = requests.get(f'{EDINET_BASE}/documents/{doc_id}',
                            params={'type': 5, 'Subscription-Key': EDINET_KEY}, timeout=90)
    response.raise_for_status()
    if not response.content.startswith(b'PK'):
        raise ValueError('EDINET CSV ZIPを取得できませんでした')
    targets = {
        'ProfitLossAttributableToOwnersOfParent': 'profit',
        'ProfitLossAttributableToOwnersOfParentIFRS': 'profit',
        'NetIncome': 'profit',
        'ProfitLoss': 'profit',
        'Equity': 'equity', 'EquityAttributableToOwnersOfParent': 'equity',
        # NetAssets includes minority interests and is not equivalent to shareholders' equity.
        # Do not substitute it for consolidated equity in ROE calculations.

        'Assets': 'assets',
        'CashFlowsFromOperatingActivities': 'cfo',
        'NetCashProvidedByUsedInOperatingActivities': 'cfo',
    }
    result = {}
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        names = [n for n in archive.namelist() if n.lower().endswith('.csv')
                 and 'xbrl_to_csv/' in n.lower() and 'audit' not in n.lower()]
        for name in names:
            try:
                df = pd.read_csv(io.BytesIO(archive.read(name)), sep='\t', encoding='utf-16',
                                 dtype=str, on_bad_lines='skip')
            except (UnicodeError, pd.errors.ParserError, ValueError):
                continue
            if df.shape[1] < 4:
                continue
            cols = list(df.columns)
            def pick(*parts):
                return next((c for c in cols if any(part in str(c).lower() for part in parts)), None)
            element_col = pick('要素id', '要素ｉｄ', 'element id', 'elementid') or cols[0]
            context_col = pick('コンテキストid', 'コンテキストｉｄ', 'context id', 'contextid') or cols[1]
            value_col = pick('値', 'value') or cols[2]
            unit_col = pick('単位id', '単位ｉｄ', 'unit id', 'unitid')
            for _, r in df.iterrows():
                tag = str(r[element_col]).split(':')[-1]
                metric = targets.get(tag)
                if not metric:
                    continue
                ctx = str(r[context_col])
                if ('NonConsolidated' in ctx or 'Member' in ctx or 'Segment' in ctx
                        or 'Restated' in ctx):
                    continue
                match = re.match(r'^(CurrentYear|Prior([1-5])Year)(Duration|Instant)$', ctx)
                if not match:
                    continue
                kind = match.group(3)
                if kind != ('Duration' if metric in ('profit', 'cfo') else 'Instant'):
                    continue
                # Never accept amounts with unknown units: ratios and scaled values
                # must not be mistaken for yen.
                if not unit_col:
                    continue
                unit = str(r[unit_col]).strip().lower()
                if unit not in ('jpy', 'iso4217:jpy', 'yen', '円'):
                    continue
                raw = str(r[value_col]).replace(',', '').strip()
                try:
                    value = float(raw)
                except ValueError:
                    continue
                if not math.isfinite(value):
                    continue
                offset = int(match.group(2) or 0)
                result.setdefault(offset, {}).setdefault(metric, value)
    return result


def edinet_features(reports, years, cutoff):
    """Use year offsets within each report; never mix duration with instant facts."""
    history = {}
    for doc in sorted(reports, key=lambda r: r['date']):
        submitted = date.fromisoformat(doc['date'])
        if submitted > cutoff:
            continue
        try:
            values = edinet_xbrl_values(doc['id'])
        except (requests.RequestException, ValueError, zipfile.BadZipFile):
            continue
        # Report fiscal end is usually before submission; prefer metadata if supplied.
        fiscal_end = doc.get('periodEnd')
        try:
            end = date.fromisoformat(str(fiscal_end)[:10]) if fiscal_end else None
        except ValueError:
            end = None
        if end is None or end > submitted:
            # Without a verified fiscal year end, offsets cannot be assigned safely.
            continue
        for offset, metrics in values.items():
            yr = end.year - offset
            if yr > cutoff.year or yr < cutoff.year - 12:
                continue
            history.setdefault(yr, {}).update(metrics)
    keys = sorted(history)[-(years + 1):]
    out = dict(ROE_Avg=math.nan, ROE_Std=math.nan, ROE_Obs=0,
               ROE_Trend=math.nan, CFO_Avg_Yen=math.nan, CFO_Obs=0,
               CFO_Positive_Years=0, CFO_Volatility=math.nan,
               EquityRatio=math.nan, NetCash=math.nan, NetCashConfirmed=False,
               FinancialSource='EDINET（金融庁・CSV）', FinancialYears=len(keys),
               EDINET_Documents=len(reports), HistoryYears=', '.join(map(str, keys)))
    roes = []
    for prev, curr in zip(keys, keys[1:]):
        if curr != prev + 1:
            continue
        old, new = history[prev], history[curr]
        e0, e1, profit = old.get('equity'), new.get('equity'), new.get('profit')
        if all(v is not None for v in (e0, e1, profit)) and e0 + e1 > 0:
            roes.append(profit / ((e0 + e1) / 2) * 100)
    roes = roes[-years:]
    if roes:
        out['ROE_Obs'] = len(roes)
        out['ROE_Avg'] = float(pd.Series(roes).mean())
        if len(roes) > 1:
            out['ROE_Std'] = float(pd.Series(roes).std(ddof=0))
            out['ROE_Trend'] = roes[-1] - roes[0]
    cfos = [history[y]['cfo'] for y in keys[-years:] if 'cfo' in history[y]]
    if cfos:
        out['CFO_Obs'] = len(cfos)
        out['CFO_Positive_Years'] = sum(v > 0 for v in cfos)
        if len(cfos) >= 2 and sum(cfos[-2:]) > 0:
            out['CFO_Volatility'] = abs(cfos[-1] - cfos[-2]) / max(abs(sum(cfos[-2:]) / 2), 1)
    if keys:
        last = history[keys[-1]]
        if last.get('equity') is not None and last.get('assets', 0) > 0:
            ratio = last['equity'] / last['assets']
            if 0 <= ratio <= 1:
                out['EquityRatio'] = ratio
    return out


def financial_features(frame, years):
    result = dict(ROE_Avg=math.nan, ROE_Std=math.nan, ROE_Obs=0,
                  ROE_Trend=math.nan, CFO_Avg_Yen=math.nan, CFO_Obs=0,
                  CFO_Positive_Years=0, CFO_Volatility=math.nan,
                  EquityRatio=math.nan, NetCash=math.nan, NetCashConfirmed=False,
                  FinancialSource='J-Quants', FinancialYears=0, ROE_Series=[], CFO_Series=[], CFO_Source='J-Quants')
    annual = annual_rows(frame, years)
    if annual.empty:
        return result
    profit = numeric_column(annual, 'NP', 'Profit', 'NetIncome')
    equity = numeric_column(annual, 'Eq', 'Equity')
    assets = numeric_column(annual, 'TA', 'TotalAssets')
    ratio = numeric_column(annual, 'EqAR', 'EquityToAssetRatio')
    cfo = numeric_column(annual, 'CFO', 'CashFlowsFromOperatingActivities', 'OperatingCashFlow')
    result['FinancialYears'] = len(annual)
    roe_values = []
    for index in range(1, len(annual)):
        p, e0, e1 = profit.iloc[index], equity.iloc[index - 1], equity.iloc[index]
        if pd.notna(p) and pd.notna(e0) and pd.notna(e1):
            average_equity = (e0 + e1) / 2
            if average_equity > 0:
                value = float(p / average_equity * 100)
                if math.isfinite(value):
                    roe_values.append(value)
    result['ROE_Series'] = [{'年度': int(annual.iloc[i]['_year_end'].year), 'ROE (%)': float(profit.iloc[i] / ((equity.iloc[i-1] + equity.iloc[i])/2) * 100)} for i in range(1, len(annual)) if pd.notna(profit.iloc[i]) and pd.notna(equity.iloc[i-1]) and pd.notna(equity.iloc[i]) and (equity.iloc[i-1] + equity.iloc[i]) > 0][-years:]
    if roe_values:
        values = pd.Series(roe_values[-years:])
        result['ROE_Avg'] = float(values.mean())
        result['ROE_Std'] = float(values.std(ddof=0)) if len(values) >= 2 else math.nan
        result['ROE_Obs'] = len(values)
        if len(values) >= 2:
            result['ROE_Trend'] = float(values.iloc[-1] - values.iloc[0])
    valid_cfo = cfo.dropna().tail(2)
    result['CFO_Series'] = [{'年度': int(annual.iloc[i]['_year_end'].year), '営業CF (元データ)': float(cfo.iloc[i])} for i in range(len(annual)) if pd.notna(cfo.iloc[i])][-years:]
    if len(valid_cfo):
        result['CFO_Avg_Yen'] = float(valid_cfo.mean())
        result['CFO_Obs'] = len(valid_cfo)
        result['CFO_Positive_Years'] = int((valid_cfo > 0).sum())
        if len(valid_cfo) == 2 and valid_cfo.mean() > 0:
            result['CFO_Volatility'] = float(abs(valid_cfo.iloc[-1] - valid_cfo.iloc[0]) / valid_cfo.mean())
    valid_ratio = ratio.dropna()
    if len(valid_ratio) and 0 <= valid_ratio.iloc[-1] <= 1:
        result['EquityRatio'] = float(valid_ratio.iloc[-1])
    elif len(assets) and pd.notna(assets.iloc[-1]) and assets.iloc[-1] > 0 and pd.notna(equity.iloc[-1]):
        calculated = float(equity.iloc[-1] / assets.iloc[-1])
        if 0 <= calculated <= 1:
            result['EquityRatio'] = calculated
    return result


def merge_history(jq, portal):
    if portal['ROE_Obs'] >= 2 and portal['ROE_Obs'] > jq['ROE_Obs']:
        # Use a coherent history from one source, never average incompatible series.
        merged = dict(portal)
        # CFO yields must use J-Quants yen values and J-Quants market cap.
        merged['CFO_Avg_Yen'] = jq['CFO_Avg_Yen']
        # Prefer the longer, consistent portal CFO history for trend and sign checks.
        # Retain J-Quants yen-denominated CFO for the market-cap yield.
        if portal['CFO_Obs'] >= jq['CFO_Obs'] and portal['CFO_Obs'] > 0:
            merged['CFO_Source'] = portal.get('FinancialSource', 'EDINETPORTAL')
        else:
            merged['CFO_Obs'] = jq['CFO_Obs']
            merged['CFO_Positive_Years'] = jq['CFO_Positive_Years']
            merged['CFO_Volatility'] = jq['CFO_Volatility']
            merged['CFO_Series'] = jq.get('CFO_Series', [])
            merged['CFO_Source'] = 'J-Quants'
        return merged
    return jq


def excluded(frame):
    sector = frame.get('Sector', pd.Series('', index=frame.index)).fillna('').astype(str)
    company = frame.get('CompanyName', pd.Series('', index=frame.index)).fillna('').astype(str)
    return (sector.str.contains('銀行|証券|商品先物|保険|その他金融|不動産', regex=True)
            | company.str.contains('ETF|REIT|投資法人|インフラファンド', case=False, regex=True))


def score_row(row, features, ratio_limit):
    per, roe = row['PER'], row['ROE_pct']
    # Historical-profitability-adjusted reference PER, not a theoretical fair value.
    # Benchmark 12x is a transparent screening assumption, not an observed market multiple.
    hist = features.get('ROE_Avg', math.nan)
    obs = int(features.get('ROE_Obs', 0))
    stable = features.get('ROE_Std', math.nan)
    normalized = min(max(hist, 0), 20) if obs >= 3 and pd.notna(hist) else min(max(roe, 0), 12)
    quality = 0.70 + 0.035 * normalized  # ROE 0% => 0.70; ROE 20% => 1.40
    if obs < 3:
        quality *= 0.75
    elif pd.notna(stable):
        quality *= max(0.70, 1 - max(0, stable - 5) / 50)
    fair = max(5.0, min(20.0, 12.0 * quality))
    roe_for_valuation = normalized
    ratio = per / fair
    if not (0 < per and 0 < roe and ratio <= ratio_limit):
        return None

    obs = features['ROE_Obs']
    historical = features['ROE_Avg']
    # Sparse history lowers confidence in the apparent discount.
    history_factor = 1.0 if obs >= 3 else (0.75 if obs == 2 else 0.45)
    undervaluation = 40 * max(0, min(1, 1 - ratio)) * history_factor

    roe_score = 0.0
    if pd.notna(historical) and historical > 0 and obs >= 2:
        gap = min(1.0, abs(roe - historical) / max(abs(historical), 1))
        std = features['ROE_Std']
        stability = 1 / (1 + max(0, std) / 15) if pd.notna(std) else 0
        trend = features['ROE_Trend']
        trend_bonus = max(0, min(1, (trend + 10) / 20)) if pd.notna(trend) else 0
        roe_score = 25 * (0.55 * (1 - gap) + 0.30 * stability + 0.15 * trend_bonus)
        if obs == 2:
            roe_score *= 0.75

    market_cap_million_yen = row.get('MktCap', math.nan)
    cfo = features['CFO_Avg_Yen']
    # J-Quants valuation market capitalization: million JPY; CFO: JPY.
    cfo_yield = (cfo / (market_cap_million_yen * 1_000_000)
                 if pd.notna(cfo) and pd.notna(market_cap_million_yen) and market_cap_million_yen > 0
                 else math.nan)
    if pd.notna(cfo_yield) and (not math.isfinite(cfo_yield) or abs(cfo_yield) > 2):
        cfo_yield = math.nan
    # Two positive years are needed for full CFO credit; volatile CFO is discounted.
    cfo_score = 0.0
    if pd.notna(cfo_yield) and cfo_yield > 0:
        cfo_score = 20 * min(.20, cfo_yield) / .20
        if features['CFO_Obs'] < 2:
            cfo_score *= .25
        elif features['CFO_Positive_Years'] < 2:
            cfo_score *= .25
        else:
            volatility = features['CFO_Volatility']
            if pd.notna(volatility):
                cfo_score *= max(.25, 1 - min(1.0, volatility) * .75)

    eqr = features['EquityRatio']
    eq_score = 15 * max(0, min(.50, eqr)) / .50 if pd.notna(eqr) else 0
    output = dict(row)
    output.update(features)
    output.update(Fair_PER=fair, ROE_For_Valuation=roe_for_valuation,
                  PER_Fair_Ratio=ratio, CFO_to_MktCap=cfo_yield,
                  UndervaluationScore=undervaluation, ROESustainabilityScore=roe_score,
                  CFOYieldScore=cfo_score, EquityRatioScore=eq_score,
                  TotalScore=undervaluation + roe_score + cfo_score + eq_score)
    return output


def build_results(master, valuation, years, ratio_limit, limit, use_portal, use_edinet, use_db, valuation_day):
    v = valuation.copy()
    for field in ('PER', 'ROE', 'MktCap'):
        if field in v:
            v[field] = number(v[field])
    missing = [field for field in ('Code', 'PER', 'ROE') if field not in v]
    if missing:
        raise JQuantsError('株価指標の必要項目がありません: ' + ', '.join(missing))
    v = v.dropna(subset=['Code', 'PER', 'ROE'])
    v = v[(v['PER'] > 0) & (v['ROE'] > 0)].copy()
    v['ROE_pct'] = v['ROE'] * 100
    # Preselection is intentionally broad: the final reference PER needs historical data.
    v['Fair_PER'] = 20.0
    v['PER_Fair_Ratio'] = v['PER'] / v['Fair_PER']
    v = v[v['Code'].isin(set(master['Code']))]
    pool = v[v['PER_Fair_Ratio'] <= ratio_limit].sort_values('PER_Fair_Ratio').head(limit)
    if pool.empty:
        return pd.DataFrame(), 0, 0
    lookup = master.drop_duplicates('Code').set_index('Code')
    rows, failures = [], 0
    progress = st.progress(0, text='財務データ取得中')
    with ThreadPoolExecutor(max_workers=min(3, len(pool))) as executor:
        jobs = {executor.submit(load_financials, str(r['Code'])): r for _, r in pool.iterrows()}
        for done, job in enumerate(as_completed(jobs), start=1):
            item = jobs[job]
            progress.progress(done / len(pool), text=f'財務データ {done}/{len(pool)}')
            try:
                jq_frame = job.result()
                features = financial_features(jq_frame, years)
                features['EDINET_Status'] = '未実行'
                if use_edinet and EDINET_KEY:
                    try:
                        reports, edinet_error = edinet_reports(str(item['Code']), jq_frame, years + 1, date.fromisoformat(valuation_day))
                        features['EDINET_Status'] = edinet_error or f'報告書 {len(reports)} 件'
                        if reports:
                            official = edinet_features(reports, years, date.fromisoformat(valuation_day))
                            if official['ROE_Obs'] > features['ROE_Obs']:
                                status = features['EDINET_Status']
                                features = merge_history(features, official)
                                features['EDINET_Status'] = status
                    except (requests.RequestException, ValueError, TypeError, KeyError) as exc:
                        features['EDINET_Status'] = '取得失敗: ' + str(exc)[:75]
                features['Portal_Status'] = '無効'
                features['DB_Status'] = '未設定' if not EDINETDB_KEY else '無効'
                if use_portal:
                    try:
                        portal_rows = load_edinetportal_financials(str(item['Code']))
                        portal = portal_features(portal_rows, years, date.fromisoformat(valuation_day))
                        features['Portal_Status'] = f'取得 {len(portal_rows)} 行 / ROE {portal["ROE_Obs"]} 年'
                        if portal['ROE_Obs'] > features['ROE_Obs']:
                            features = merge_history(features, portal)
                            features['Portal_Status'] = f'取得 {len(portal_rows)} 行 / ROE {portal["ROE_Obs"]} 年'
                    except Exception as exc:
                        features['Portal_Status'] = f'{type(exc).__name__}: {str(exc)[:95]}'
                if use_db and EDINETDB_KEY and features['ROE_Obs'] < years:
                    try:
                        db_rows, db_code = load_edinetdb_financials(str(item['Code']))
                        db_features = portal_features(db_rows, years, date.fromisoformat(valuation_day))
                        db_features['FinancialSource'] = 'EDINET DB'
                        status = f'{db_code}: {len(db_rows)} 行 / ROE {db_features["ROE_Obs"]} 年'
                        if db_features['ROE_Obs'] > features['ROE_Obs']:
                            old_portal = features['Portal_Status']
                            old_edinet = features['EDINET_Status']
                            features = merge_history(features, db_features)
                            features['Portal_Status'] = old_portal
                            features['EDINET_Status'] = old_edinet
                        features['DB_Status'] = status
                    except Exception as exc:
                        features['DB_Status'] = f'{type(exc).__name__}: {str(exc)[:95]}'
                row = score_row(item.to_dict(), features, ratio_limit)
                if row:
                    code = row['Code']
                    row['CompanyName'] = lookup.at[code, 'CompanyName'] if 'CompanyName' in lookup else code
                    row['Sector'] = lookup.at[code, 'Sector'] if 'Sector' in lookup else ''
                    rows.append(row)
            except Exception:
                failures += 1
    progress.empty()
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values('TotalScore', ascending=False).reset_index(drop=True)
    return result, len(pool), failures


def percent(value):
    return f'{value * 100:.1f}%' if pd.notna(value) else '未取得'


def yen_percent(value):
    return f'{value:.1f}%' if pd.notna(value) else '未取得'


def reason(row):
    notes = []
    if row['PER_Fair_Ratio'] <= .5:
        notes.append('独自の参考PERを下回る')
    if row['ROE_Obs'] >= 2 and pd.notna(row['ROE_Avg']):
        notes.append('ROE履歴を確認済み' if row['ROE_pct'] >= row['ROE_Avg'] * .9 else '過去平均よりROE低下')
    if row['CFO_Obs'] >= 2 and row['CFO_Positive_Years'] == 2 and pd.notna(row['CFO_to_MktCap']) and row['CFO_to_MktCap'] >= .05:
        notes.append('営業CF利回り5%以上')
    if pd.notna(row['EquityRatio']) and row['EquityRatio'] >= .5:
        notes.append('自己資本比率50%以上')
    if row['ROE_Obs'] < 2:
        notes.append('ROE履歴不足・参考評価')
    if row['ROE_pct'] > 30:
        notes.append('高ROEを30%に制限して評価')
    if row['CFO_Obs'] < 2 or row['CFO_Positive_Years'] < 2:
        notes.append('営業CFの継続性未確認')
    notes.append('ネットキャッシュ未判定')
    return ' / '.join(notes)


def quality_assessment(item, years):
    """既存の100点スコアを4要素に分解。未取得項目は満点扱いしない。"""
    sections = [
        ('割安度', 'UndervaluationScore', 40),
        ('収益性・安定性', 'ROESustainabilityScore', 25),
        ('キャッシュ創出力', 'CFOYieldScore', 20),
        ('財務健全性', 'EquityRatioScore', 15),
    ]
    strengths, cautions = [], []
    if item['PER_Fair_Ratio'] <= 0.4:
        strengths.append('実PERが独自の参考PERより低い水準です')
    if item['ROE_Obs'] >= 3 and pd.notna(item['ROE_Avg']) and item['ROE_Avg'] >= 8:
        strengths.append('複数年のROE平均が8%以上です')
    if item['CFO_Obs'] >= 3 and item['CFO_Positive_Years'] >= 3:
        strengths.append('営業CFの黒字を複数年確認できています')
    if pd.notna(item['EquityRatio']) and item['EquityRatio'] >= 0.5:
        strengths.append('自己資本比率が50%以上です')
    if item['ROE_Obs'] < years:
        cautions.append(f"ROE履歴は{int(item['ROE_Obs'])}/{years}年分です")
    if item['CFO_Obs'] < years:
        cautions.append(f"営業CF履歴は{int(item['CFO_Obs'])}/{years}年分です")
    if item['CFO_Positive_Years'] < min(3, int(item['CFO_Obs'])):
        cautions.append('営業CFがマイナスの年度があります')
    if pd.isna(item['EquityRatio']):
        cautions.append('自己資本比率を取得できていません')
    if not strengths:
        strengths.append('取得済み指標から総合スコアを算出しています')
    if not cautions:
        cautions.append('株価や業績は変動するため、直近の決算も確認してください')
    return sections, strengths[:3], cautions[:3]


def date_freshness_message(valuation_day):
    try:
        age = (date.today() - date.fromisoformat(valuation_day)).days
    except (ValueError, TypeError):
        return '株価指標の基準日を確認できません'
    if age > 30:
        return f'株価指標は{age}日前のデータです。現在のPERや株価とは異なる可能性があります。'
    return f'株価指標は{age}日前のデータです。'


st.markdown("""<div class="welcome-card">
<div class="welcome-kicker">🌱 はじめての株分析を、もっと身近に</div>
<h1>📈 割安株AI</h1>
<p>気になる日本株を、わかりやすい点数とグラフでチェック。<br>決算情報と過去5年の財務データを、スマホでも見やすくまとめます。</p>
<span class="pill-note">💚 100点満点の評価</span><span class="pill-note">📅 株価の基準日を表示</span><span class="pill-note">📊 5年の財務推移</span>
</div>""", unsafe_allow_html=True)
st.caption('※ 点数や参考PERは独自の比較指標です。投資成果や目標株価を保証するものではありません。')
with st.sidebar:
    st.header('🔎 銘柄を探す')
    st.caption('条件を変更して候補を絞り込めます。')
    years = st.slider('ROE履歴の最大年数', 3, 5, 5)
    ratio_limit = st.slider('割安判定（実PER / 参考PER）', .20, .80, .50, .05)
    limit = st.slider('財務分析する上位候補数', 5, 50, 10, 5)
    strict_net_cash = st.checkbox('ネットキャッシュ > 0 を必須', value=False)
    min_score = st.slider('最低スコア', 0, 100, 0)
    with st.expander('詳細設定（データ取得）', expanded=False):
        st.caption('通常は変更不要です。')
        use_edinet = st.checkbox('EDINET公式CSVで補完（低速）', value=False)
        use_portal = st.checkbox('長期財務データを取得', value=True)
        use_db = st.checkbox('追加データで不足履歴を補完', value=bool(EDINETDB_KEY), disabled=not bool(EDINETDB_KEY))
    run = st.button('🔍 割安株を探す', type='primary', use_container_width=True)

if not API_KEY:
    st.error('Streamlit Secrets に JQUANTS_API_KEY を設定してください。')
    st.stop()
if strict_net_cash:
    st.error('無料プランでは有利子負債を確実に取得できないため、ネットキャッシュ必須条件は実行できません。チェックを外してください。')
    st.stop()

settings = (years, ratio_limit, limit, min_score, use_portal, use_edinet, use_db)
if run or 'result' not in st.session_state or st.session_state.get('settings') != settings:
    try:
        with st.spinner('銘柄一覧・株価指標を取得中…'):
            master = load_master()
            master = master.loc[~excluded(master)].copy()
            valuation, valuation_day = load_valuation()
        with st.spinner('割安候補を確認中…'):
            result, analyzed, failures = build_results(master, valuation, years, ratio_limit, limit, use_portal, use_edinet, use_db, valuation_day)
            if not result.empty:
                result = result[result['TotalScore'] >= min_score].copy()
        st.session_state.update(result=result, analyzed=analyzed, failures=failures,
                                valuation_day=valuation_day, settings=settings,
                                updated=datetime.now().strftime('%Y-%m-%d %H:%M'))
    except Exception as exc:
        st.error(str(exc))
        st.stop()

result = st.session_state['result']
valuation_day_display = st.session_state['valuation_day']
st.info(f'**PER・時価総額などの株価指標の基準日：{valuation_day_display}**\n\n{date_freshness_message(valuation_day_display)}')
st.caption(f"分析実行：{st.session_state['updated']} / 財務分析：{st.session_state['analyzed']}社 / 候補：{len(result)}社")
if st.session_state.get('failures'):
    st.warning(f"財務データ取得に失敗した銘柄: {st.session_state['failures']}社")
if result.empty:
    st.info('条件に合う銘柄がありません。最低スコアや割安判定を調整してください。')
    st.stop()

m1, m2, m3, m4 = st.columns(4)
m1.metric('候補数', len(result))
m2.metric('PER/参考PER 中央値', f"{result['PER_Fair_Ratio'].median():.2f}")
m3.metric('ROE 中央値', f"{result['ROE_pct'].median():.1f}%")
m4.metric('スコア中央値', f"{result['TotalScore'].median():.1f}")

table = pd.DataFrame({
    'コード': result['Code'], '会社': result['CompanyName'],
    'PER（株価基準日共通）': result['PER'].round(1), 'ROE': result['ROE_pct'].map(yen_percent),
    '参考PER': result['Fair_PER'].round(1),
    '割安比率': result['PER_Fair_Ratio'].map(percent),
    '営業CF利回り': result['CFO_to_MktCap'].map(percent),
    '自己資本比率': result['EquityRatio'].map(percent),
    'ROE観測年数': result['ROE_Obs'], '5年ROE充足': result['ROE_Obs'].map(lambda n: '取得済' if n >= years else f'不足（{n}/{years}）'),
    'ネットキャッシュ': '未判定', '総合スコア': result['TotalScore'].round(1),
})
st.caption(f'一覧のPERはすべて {valuation_day_display} 時点の株価指標です。財務指標は別の決算期に基づく場合があります。')
st.dataframe(table, use_container_width=True, hide_index=True)
st.download_button('候補一覧CSV', result.to_csv(index=False).encode('utf-8-sig'),
                   'waribiki_candidates.csv', 'text/csv', use_container_width=True)

st.subheader('銘柄詳細')
selected = st.selectbox('銘柄', result['Code'].tolist(), format_func=lambda c: f"{c} {result.loc[result['Code'] == c, 'CompanyName'].iloc[0]}")
item = result.loc[result['Code'] == selected].iloc[0]
c1, c2, c3, c4 = st.columns(4)
c1.metric(f'PER（{valuation_day_display}）', f"{item['PER']:.1f}倍")
c2.metric('ROE', f"{item['ROE_pct']:.1f}%")
c3.metric('参考PER', f"{item['Fair_PER']:.1f}倍")
c4.metric('割安比率', percent(item['PER_Fair_Ratio']))
st.caption(f'実PER・時価総額：{valuation_day_display} 時点の株価指標 ／ ROE・営業CF：取得済み決算データ（各年度）')
# Manual latest-price scenario: Yahoo! is linked, never scraped.
st.subheader('最新の決算速報（自動取得）')
latest_earnings, earnings_error = load_latest_earnings(str(selected).strip()[:4])
latest_eps = None
latest_eps_date = None
if latest_earnings:
    disclosure = str(latest_earnings.get('disclosure_date') or '不明')[:10]
    quarter = str(latest_earnings.get('quarter') or '').upper()
    fiscal_end = str(latest_earnings.get('fiscal_year_end') or '未取得')[:10]
    latest_eps = safe_numeric(latest_earnings.get('eps'))
    latest_eps_date = disclosure
    st.success(f'決算速報：{disclosure} 開示 ／ 対象決算期：{fiscal_end} ／ 区分：{quarter or "未取得"}')
    e1, e2 = st.columns(2)
    e1.metric('短信のEPS（実績）', f'{latest_eps:,.2f}円' if latest_eps is not None else '未取得')
    profit = safe_numeric(latest_earnings.get('net_income'))
    e2.metric('短信の純利益', japanese_large_number(profit * 1_000_000) if profit is not None else '未取得')
    st.caption('短信の利益などの金額は百万円単位のため円換算しています。EPSは円/株です。四半期EPSは通期EPSではありません。')
    source_pdf = str(latest_earnings.get('pdf_url') or '')
    if source_pdf.startswith('https://'):
        st.link_button('決算短信の原本を確認 ↗', source_pdf)
    if quarter not in ('FY', 'Q4', '4', 'FULL_YEAR'):
        st.warning('四半期・累計決算のEPSです。この数値をそのまま年間PERの分母には使用しません。')
    elif latest_eps is None or latest_eps <= 0:
        st.warning('通期のEPSが未取得または0以下のため、この速報からPERを計算できません。')
    else:
        st.info('通期実績EPSを取得しました。株式分割・併合や株価の株数基準が一致する場合に限り、下の入力株価からPERを試算できます。')
else:
    st.info('最新決算速報：' + earnings_error + '。従来の財務履歴は引き続き利用できます。')

st.subheader('最新株価で再評価（手入力）')
st.caption('Yahoo!ファイナンス等で終値を確認し、下に入力してください。株価は自動取得しません。最新の通期実績EPSがある場合のみ、EPSを使った参考PERを計算します。')
code_for_yahoo = str(selected).strip()[:4]
st.link_button('Yahoo!ファイナンスで株価を確認 ↗',
               f'https://finance.yahoo.co.jp/quote/{code_for_yahoo}.T',
               use_container_width=True)

@st.cache_data(ttl=21600, show_spinner=False)
def historical_close(code, asof):
    # Request only the already-licensed historical period; no Yahoo scraping.
    day = date.fromisoformat(asof)
    for back in range(6):
        target = (day - timedelta(days=back)).strftime('%Y%m%d')
        try:
            records = jq_get('/equities/bars/daily', {'code': code, 'date': target})
        except (JQuantsError, requests.RequestException, ValueError):
            return None
        for record in records:
            for key in ('AdjC', 'C', 'Close', 'AdjustmentClose', 'AdjustmentClosePrice'):
                try:
                    value = float(record.get(key))
                    if math.isfinite(value) and value > 0:
                        return value
                except (ValueError, TypeError):
                    pass
    return None

old_close = historical_close(code_for_yahoo, valuation_day_display)
st.caption(f'過去のPERの株価基準日：{valuation_day_display} ／ 過去の終値：' +
           (f'{old_close:,.1f}円（自動取得）' if old_close else '未取得（入力してください）'))
with st.form('price_update_' + code_for_yahoo):
    baseline = st.number_input(f'過去の終値（{valuation_day_display}、円）',
                               min_value=0.0, value=float(old_close or 0),
                               step=1.0, format='%.2f',
                               help='過去のPERと同じ基準日の株価。株式分割があった場合は比較可能な株価に調整してください。')
    recent_price = st.number_input('確認した新しい株価（円）', min_value=0.0,
                                   value=0.0, step=1.0, format='%.2f')
    observed_date = st.date_input('その株価の確認日', value=date.today(), max_value=date.today())
    apply_price = st.form_submit_button('この株価でPER・評価を再計算', type='primary', use_container_width=True)
if apply_price:
    if baseline <= 0 or recent_price <= 0:
        st.warning('過去の終値と新しい株価を両方入力してください。')
    elif observed_date < date.fromisoformat(valuation_day_display):
        st.warning('確認日は過去のPER基準日以降にしてください。')
    else:
        st.session_state['manual_price_' + code_for_yahoo] = (baseline, recent_price, observed_date.isoformat())

manual = st.session_state.get('manual_price_' + code_for_yahoo)
if manual:
    base_p, new_p, new_date = manual
    # The old PER is rescaled by price only. EPS is held constant, not refreshed.
    use_fy_eps = bool(latest_earnings and str(latest_earnings.get('quarter') or '').upper() in ('FY', 'Q4', '4', 'FULL_YEAR') and latest_eps is not None and latest_eps > 0 and latest_eps_date and latest_eps_date <= new_date)
    # Prefer an explicitly identified full-year actual EPS. Otherwise use the historical price-only estimate.
    new_per = (new_p / latest_eps) if use_fy_eps else (float(item['PER']) * new_p / base_p)
    new_ratio = new_per / float(item['Fair_PER'])
    obs = int(item['ROE_Obs'])
    history_factor = 1.0 if obs >= 3 else (0.75 if obs == 2 else 0.45)
    new_discount_score = 40 * max(0, min(1, 1 - new_ratio)) * history_factor
    new_score = (new_discount_score + float(item['ROESustainabilityScore'])
                 + float(item['CFOYieldScore']) + float(item['EquityRatioScore']))
    st.success(f'入力株価：{new_p:,.2f}円（{new_date}）')
    p1, p2 = st.columns(2)
    p1.metric('最新通期実績EPSによる参考PER' if use_fy_eps else '株価だけ更新した参考PER', f'{new_per:.2f}倍',
              delta=f'{new_per-float(item["PER"]):+.2f}倍（過去比）')
    p2.metric('参考・再計算スコア', f'{new_score:.0f}/100点')
    st.write(f'参考PERに対する比率：**{new_ratio:.1%}**')
    if new_ratio > ratio_limit:
        st.warning('新しい株価では、当初の割安判定条件を満たしません。')
    st.warning(('通期実績EPSは決算速報から取得しましたが、株式分割・併合と株価の株数基準は自動照合していません。営業CF利回り・時価総額・その他財務項目は更新していないため総合評価は参考値です。' if use_fy_eps else '最新の通期EPSを利用できなかったため、過去PERを株価比で換算した参考値です。決算・株式分割・時価総額・営業CF利回りは更新していません。'))
else:
    st.info('以下の総合評価は、過去の株価基準日による評価です。最新株価での試算は上の入力欄から行えます。')

st.subheader('🌟 この銘柄の評価')
sections, strengths, cautions = quality_assessment(item, years)
st.metric(f'過去基準日の総合評価（{valuation_day_display}・100点満点）', f"{item['TotalScore']:.0f}点")
st.caption('独自のスクリーニング評価です。投資成果や将来の株価を予測するものではありません。')
for title, key, maximum in sections:
    raw = item.get(key, 0)
    score = max(0.0, min(float(raw), maximum)) if pd.notna(raw) else 0.0
    st.write(f'**{title}：{score:.0f} / {maximum}点**')
    st.progress(min(1.0, score / maximum))
st.markdown('**良い点**')
for message in strengths:
    st.write('・' + message)
st.markdown('**注意点・データの不足**')
for message in cautions:
    st.write('・' + message)
with st.expander('評価に使ったデータの確認', expanded=False):
    st.write(f"ROE：{int(item['ROE_Obs'])}/{years}年分")
    st.write(f"営業CF：{int(item['CFO_Obs'])}/{years}年分（黒字 {int(item['CFO_Positive_Years'])}年）")
    st.write(f"株価指標基準日：{valuation_day_display}")
    st.caption('ROE・営業CFの各年度は下の財務推移で確認できます。データソースの詳細は診断欄にまとめています。')

st.write(f"**過去ROE平均: {yen_percent(item['ROE_Avg'])}**（{int(item['ROE_Obs'])}/{years}年分）")
st.write(f"営業CF利回り: {percent(item['CFO_to_MktCap'])}（CF観測 {int(item['CFO_Obs'])} 年、うち黒字 {int(item['CFO_Positive_Years'])} 年）")
st.caption('参考PERは基準12倍を長期ROE・変動性・観測年数で調整した独自指標です。適正PERや目標株価ではありません。')
if item['CFO_Obs'] < 2 or item['CFO_Positive_Years'] < 2:
    st.caption('営業CFが2年連続プラスと確認できないため、営業CFスコアを減点しています。')
st.write(f"自己資本比率: {percent(item['EquityRatio'])} / ネットキャッシュ: 未判定")
if item['ROE_Obs'] < years:
    st.caption('指定年数分のROE履歴は取得できていません。履歴の平均値を長期平均とみなさないでください。')

# Read-only diagnosis: only fetches the selected ticker, on explicit request.
st.subheader('📊 過去の財務推移')
roe_series = item.get('ROE_Series', [])
if isinstance(roe_series, list) and roe_series:
    roe_df = pd.DataFrame(roe_series).drop_duplicates('年度', keep='last').sort_values('年度').tail(years)
    st.markdown('**ROEの5年推移**')
    st.caption('年ごとの数値を大きく表示しています。赤色はマイナスROEです。')
    valid_roe = pd.to_numeric(roe_df['ROE (%)'], errors='coerce').dropna()
    max_abs_roe = max(10.0, float(valid_roe.abs().max())) if not valid_roe.empty else 10.0
    for _, rec in roe_df.iterrows():
        v = pd.to_numeric(rec['ROE (%)'], errors='coerce')
        if pd.isna(v):
            continue
        yr = html.escape(str(int(rec['年度'])))
        neg = ' negative' if float(v) < 0 else ''
        width = max(2.0, min(100.0, abs(float(v)) / max_abs_roe * 100.0))
        st.markdown(f'<div class="roe-row"><span class="roe-year">{yr}年</span>'
                    f'<div class="roe-track"><div class="roe-fill{neg}" style="width:{width:.1f}%"></div></div>'
                    f'<span class="roe-value{neg}">{float(v):+.1f}%</span></div>', unsafe_allow_html=True)
    with st.expander('折れ線グラフと数値表を見る', expanded=False):
        st.line_chart(roe_df.set_index('年度')['ROE (%)'], color='#85F3C5', height=240)
        st.dataframe(roe_df, hide_index=True, use_container_width=True)
else:
    st.info('ROEの年度別推移を表示できません。')
cfo_series = item.get('CFO_Series', [])
if isinstance(cfo_series, list) and cfo_series:
    cfo_df = pd.DataFrame(cfo_series).drop_duplicates('年度', keep='last').sort_values('年度').tail(years)
    st.markdown('#### 営業CFの5年推移')
    st.caption('営業キャッシュフローを円単位で、億・万・千の位取りで表示します。')
    # The amount is treated as yen; validate provider units against filings before general release.
    cf_numeric = pd.to_numeric(cfo_df['営業CF (元データ)'], errors='coerce')
    cf_max = max(1.0, float(cf_numeric.abs().max())) if cf_numeric.notna().any() else 1.0
    for _, cf_rec in cfo_df.iterrows():
        cf_val = pd.to_numeric(cf_rec['営業CF (元データ)'], errors='coerce')
        if pd.isna(cf_val):
            continue
        cf_year = html.escape(str(int(cf_rec['年度'])))
        cf_negative = ' negative' if float(cf_val) < 0 else ''
        cf_width = max(2.0, min(100.0, abs(float(cf_val)) / cf_max * 100.0))
        cf_formatted = japanese_large_number(cf_val)
        st.markdown(
            f'<div class="roe-row"><span class="roe-year">{cf_year}年</span>'
            f'<div class="roe-track"><div class="roe-fill{cf_negative}" style="width:{cf_width:.1f}%"></div></div>'
            f'<span class="cf-value{cf_negative}">{cf_formatted}</span></div>',
            unsafe_allow_html=True,
        )
    with st.expander('営業CFの数値表を見る', expanded=False):
        cf_display = cfo_df.copy()
        cf_display['営業CF（円）'] = cf_display['営業CF (元データ)'].map(japanese_large_number)
        st.dataframe(cf_display[['年度', '営業CF（円）']], hide_index=True, use_container_width=True)
    if item.get('CFO_Source') != 'J-Quants':
        st.caption('営業CFは円単位として表示しています。銘柄ごとのAPI値と決算原本の照合は未完了です。EDINETPORTAL由来の営業CFは利回り計算に使用していません。')
else:
    st.info('営業CFの年度別推移は未取得です。')

with st.expander('データの出典・取得状況・詳細診断（任意）', expanded=False):
    st.caption('データ提供元：J-Quants、EDINETPORTAL。追加設定時はEDINET DB・金融庁EDINETも使用します。各サービスの利用条件に従ってください。')
    st.write('ROE履歴の出典:', item.get('FinancialSource', '不明'))
    st.write('営業CF履歴の出典:', item.get('CFO_Source', '不明'))
    st.caption('EDINETPORTAL: ' + str(item.get('Portal_Status', '未実行')))
    st.caption('EDINET DB: ' + str(item.get('DB_Status', '未実行')))
    st.caption('EDINET公式: ' + str(item.get('EDINET_Status', '未実行')))
    st.markdown('**取得した年度・項目名・数値を確認する**')
    st.caption('選択中の銘柄だけを再取得し、返されたJSONの構造とROE計算に必要な項目を確認します。APIキーは表示しません。')
    if st.button('この銘柄の財務データを診断', key=f'portal_diagnose_{selected}'):
        try:
            # Use the existing cached API call; avoid extra calls for all tickers.
            diagnostic_rows = load_edinetportal_financials(str(selected))
            st.write(f'取得した年度別データ: {len(diagnostic_rows)} 行')
            if not diagnostic_rows:
                st.warning('財務データが0行です。APIの返却形式か対象銘柄を確認してください。')
            else:
                field_groups = {
                    '年度候補': ('fiscalYear', 'fiscal_year', 'year', 'fy', 'period_end', 'fiscal_year_end', 'fiscalYearEnd'),
                    '開示日候補': ('submit_date', 'submitted_at', 'filing_date', 'disclosure_date', 'filed_at', 'filingDate', 'disclosureDate', 'submittedAt'),
                    'ROE候補': ('roe', 'return_on_equity', 'roe_percent', 'roe_pct', 'returnOnEquity'),
                    '純利益候補': ('net_income_attributable_to_owners_of_parent', 'net_income', 'profit_attributable_to_owners_of_parent', 'netIncome'),
                    '自己資本候補': ('shareholders_equity', 'equity_attributable_to_owners_of_parent', 'equity', 'shareholdersEquity'),
                    '営業CF候補': ('operating_cash_flow', 'cash_flow_from_operations', 'cash_flows_from_operating_activities', 'cash_flow_operating', 'operatingCashFlow', 'cashFlowFromOperatingActivities', 'cashFlowsFromOperatingActivities'),
                }
                all_keys = sorted({str(k) for r in diagnostic_rows for k in r.keys()})
                st.write('**APIから返された項目名（全行の和集合）**')
                st.code(', '.join(all_keys) or '(項目なし)', language=None)
                st.write('**現在のコードが認識できる項目**')
                check = []
                for group, aliases in field_groups.items():
                    matched = [k for k in aliases if k in all_keys]
                    check.append({'分類': group, '一致した項目': ', '.join(matched) if matched else '一致なし'})
                st.dataframe(pd.DataFrame(check), hide_index=True, use_container_width=True)
                st.write('**各行の主要項目（元データを変更せず表示）**')
                preview_keys = list(dict.fromkeys(k for aliases in field_groups.values() for k in aliases if k in all_keys))
                # Show additional actual keys to diagnose mismatches.
                preview_keys += [k for k in all_keys if k not in preview_keys][:18]
                preview = [{k: str(r.get(k, ''))[:160] for k in preview_keys} for r in diagnostic_rows[:30]]
                st.dataframe(pd.DataFrame(preview), hide_index=True, use_container_width=True)
                st.write('**1行目のJSON構造（最大12,000文字）**')
                import json
                st.code(json.dumps(diagnostic_rows[0], ensure_ascii=False, indent=2, default=str)[:12000], language='json')
                st.download_button('診断用JSONを保存', json.dumps(diagnostic_rows, ensure_ascii=False, indent=2, default=str).encode('utf-8'), file_name=f'portal_diagnostic_{selected}.json', mime='application/json', key=f'portal_json_{selected}')
                diagnosed = portal_features(diagnostic_rows, years, date.fromisoformat(st.session_state['valuation_day']))
                st.info(f'修正後ロジックの判定: 財務年度 {diagnosed["FinancialYears"]} 年 / ROE {diagnosed["ROE_Obs"]} 年。項目名が「一致なし」の場合はマッピング修正が必要です。')
        except Exception as exc:
            st.error(f'診断中の取得エラー: {type(exc).__name__}: {str(exc)[:350]}')

st.subheader('AIによる補足分析（任意）')
if not OPENAI_KEY:
    st.caption('OPENAI_API_KEY が未設定のため、AI分析は無効です。数値スクリーニングは利用できます。')
elif st.button('選択銘柄をAI分析'):
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_KEY)
        fields = ['Code', 'CompanyName', 'PER', 'ROE_pct', 'Fair_PER', 'PER_Fair_Ratio',
                  'ROE_Avg', 'ROE_Obs', 'CFO_to_MktCap', 'EquityRatio']
        data = {k: (None if pd.isna(item.get(k)) else str(item.get(k))) for k in fields}
        prompt = ('以下の数値だけを使い、割安の理由、割安が正当化されるリスク、'
                  'ROEの持続性、追加確認事項を日本語で簡潔に説明してください。'
                  '未取得の項目を推測で補わず、投資を断定的に推奨しないでください。\n' + str(data))
        response = client.responses.create(model=OPENAI_MODEL, input=prompt)
        st.write(response.output_text)
    except Exception as exc:
        st.error(f'AI分析に失敗しました: {exc}')
