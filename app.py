import os
import math
import time
import threading
import io
import zipfile
import re
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
st.markdown('''<style>.block-container{max-width:1100px;padding:1rem .7rem} [data-testid="stMetricValue"]{font-size:1.15rem}</style>''', unsafe_allow_html=True)


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
               FinancialSource='EDINETPORTAL', FinancialYears=0)
    observations = []
    equity_by_year = {}
    net_by_year = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        # Exclude financial statements published after the J-Quants valuation date.
        disclosed = next((row.get(k) for k in ('submit_date', 'submitted_at', 'filing_date', 'disclosure_date', 'filed_at') if row.get(k)), None)
        if disclosed:
            parsed = pd.to_datetime(disclosed, errors='coerce', utc=True)
            if pd.notna(parsed) and parsed.date() > cutoff:
                continue
        fiscal = next((row.get(k) for k in ('fiscal_year', 'year', 'fy', 'period_end', 'fiscal_year_end') if row.get(k) is not None), None)
        if fiscal is None:
            continue
        try:
            year = int(str(fiscal).upper().replace('FY', '')[:4])
        except (TypeError, ValueError):
            continue
        if year > cutoff.year:
            continue
        # If filing date is absent, exclude current fiscal year to avoid look-ahead.
        if disclosed is None and year >= cutoff.year:
            continue
        roe = field_value(row, 'roe', 'return_on_equity', 'roe_percent', 'roe_pct')
        equity_by_year[year] = field_value(row, 'shareholders_equity', 'equity_attributable_to_owners_of_parent', 'equity')
        net_by_year[year] = field_value(row, 'net_income_attributable_to_owners_of_parent', 'net_income', 'profit_attributable_to_owners_of_parent')
        eq_ratio = field_value(row, 'equity_ratio', 'equity_ratio_percent', 'equity_to_asset_ratio')
        # EDINETPORTAL documents percentage ratios as 0..100 (not 0..1).
        if pd.notna(roe) and not (-200 <= roe <= 500):
            roe = math.nan
        if pd.notna(eq_ratio):
            eq_ratio = eq_ratio / 100 if 0 <= eq_ratio <= 100 else math.nan
        cfo = field_value(row, 'operating_cash_flow', 'cash_flow_from_operations', 'cash_flows_from_operating_activities', 'cash_flow_operating')
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
                  FinancialSource='J-Quants', FinancialYears=0)
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
    if roe_values:
        values = pd.Series(roe_values[-years:])
        result['ROE_Avg'] = float(values.mean())
        result['ROE_Std'] = float(values.std(ddof=0)) if len(values) >= 2 else math.nan
        result['ROE_Obs'] = len(values)
        if len(values) >= 2:
            result['ROE_Trend'] = float(values.iloc[-1] - values.iloc[0])
    valid_cfo = cfo.dropna().tail(2)
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
        if jq['CFO_Obs'] >= 2:
            merged['CFO_Obs'] = jq['CFO_Obs']
            merged['CFO_Positive_Years'] = jq['CFO_Positive_Years']
            merged['CFO_Volatility'] = jq['CFO_Volatility']
        return merged
    return jq


def excluded(frame):
    sector = frame.get('Sector', pd.Series('', index=frame.index)).fillna('').astype(str)
    company = frame.get('CompanyName', pd.Series('', index=frame.index)).fillna('').astype(str)
    return (sector.str.contains('銀行|証券|商品先物|保険|その他金融|不動産', regex=True)
            | company.str.contains('ETF|REIT|投資法人|インフラファンド', case=False, regex=True))


def score_row(row, features, ratio_limit):
    per, roe = row['PER'], row['ROE_pct']
    # A one-off extremely high ROE must not imply an unlimited fair PER.
    # 30% is a conservative screening cap, not a forecast of fair valuation.
    roe_for_valuation = min(30.0, max(0.0, roe))
    fair = roe_for_valuation * 2
    ratio = per / fair if fair > 0 else math.inf
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
    v['Fair_PER'] = v['ROE_pct'].clip(upper=30) * 2
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
        notes.append('理論PERに対して割安')
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


st.title('割安株AI')
st.caption('J-Quants V2 ＋ EDINETPORTAL ＋ EDINET DB（任意）｜参考スクリーニング')
st.warning('財務履歴はEDINETPORTALを優先し、不足時は任意のEDINET DBを使用します。取得状況を銘柄別に表示します。公式EDINETの広範囲日付検索は初期状態で無効です。5年分の取得は保証されません。理論PERは独自の参考値であり適正株価ではありません。')
with st.sidebar:
    st.header('設定')
    st.write('J-Quants API: ' + ('設定済み' if API_KEY else '未設定'))
    years = st.slider('ROE履歴の最大年数', 3, 5, 5)
    ratio_limit = st.slider('割安判定（実PER / 理論PER）', .20, .80, .50, .05)
    limit = st.slider('財務分析する上位候補数', 5, 50, 10, 5)
    strict_net_cash = st.checkbox('ネットキャッシュ > 0 を必須', value=False)
    min_score = st.slider('最低スコア', 0, 100, 0)
    st.write('EDINET API: ' + ('設定済み' if EDINET_KEY else '未設定'))
    use_edinet = st.checkbox('EDINET公式CSVで過去年度を補完（低速・詳細検証用）', value=False)
    use_portal = st.checkbox('EDINETPORTALの長期財務履歴を取得（APIキー不要）', value=True)
    st.write('EDINET DB API: ' + ('設定済み' if EDINETDB_KEY else '未設定（任意）'))
    use_db = st.checkbox('EDINET DBで不足履歴を補完（任意・利用回数制限あり）', value=bool(EDINETDB_KEY), disabled=not bool(EDINETDB_KEY))
    run = st.button('スクリーニング実行', type='primary', use_container_width=True)

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
st.caption(f"実行日時: {st.session_state['updated']} / 株価指標基準日: {st.session_state['valuation_day']} / 財務分析: {st.session_state['analyzed']}社 / 候補: {len(result)}社")
if st.session_state.get('failures'):
    st.warning(f"財務データ取得に失敗した銘柄: {st.session_state['failures']}社")
if result.empty:
    st.info('条件に合う銘柄がありません。最低スコアや割安判定を調整してください。')
    st.stop()

m1, m2, m3, m4 = st.columns(4)
m1.metric('候補数', len(result))
m2.metric('PER/理論PER 中央値', f"{result['PER_Fair_Ratio'].median():.2f}")
m3.metric('ROE 中央値', f"{result['ROE_pct'].median():.1f}%")
m4.metric('スコア中央値', f"{result['TotalScore'].median():.1f}")

table = pd.DataFrame({
    'コード': result['Code'], '会社': result['CompanyName'],
    'PER': result['PER'].round(1), 'ROE': result['ROE_pct'].map(yen_percent),
    '理論PER': result['Fair_PER'].round(1),
    '割安比率': result['PER_Fair_Ratio'].map(percent),
    '営業CF利回り': result['CFO_to_MktCap'].map(percent),
    '自己資本比率': result['EquityRatio'].map(percent),
    'ROE観測年数': result['ROE_Obs'], '5年ROE充足': result['ROE_Obs'].map(lambda n: '取得済' if n >= years else f'不足（{n}/{years}）'), '財務履歴出典': result['FinancialSource'],
    'EDINETPORTAL': result.get('Portal_Status', pd.Series('未実行', index=result.index)),
    'EDINET DB': result.get('DB_Status', pd.Series('未実行', index=result.index)),
    'EDINET公式': result.get('EDINET_Status', pd.Series('未実行', index=result.index)),
    'ネットキャッシュ': '未判定', '総合スコア': result['TotalScore'].round(1),
})
st.dataframe(table, use_container_width=True, hide_index=True)
st.download_button('候補一覧CSV', result.to_csv(index=False).encode('utf-8-sig'),
                   'waribiki_candidates.csv', 'text/csv', use_container_width=True)

st.subheader('銘柄詳細')
selected = st.selectbox('銘柄', result['Code'].tolist(), format_func=lambda c: f"{c} {result.loc[result['Code'] == c, 'CompanyName'].iloc[0]}")
item = result.loc[result['Code'] == selected].iloc[0]
c1, c2, c3, c4 = st.columns(4)
c1.metric('PER', f"{item['PER']:.1f}倍")
c2.metric('ROE', f"{item['ROE_pct']:.1f}%")
c3.metric('理論PER', f"{item['Fair_PER']:.1f}倍")
c4.metric('割安比率', percent(item['PER_Fair_Ratio']))
st.write(f"**{item['CompanyName']}**：{reason(item)}")
st.write(f"ROE履歴平均: {yen_percent(item['ROE_Avg'])}（観測 {int(item['ROE_Obs'])} 年、最大 {years} 年／出典: {item['FinancialSource']}）")
st.caption('EDINETPORTAL: ' + str(item.get('Portal_Status', '未実行')))
st.caption('EDINET DB: ' + str(item.get('DB_Status', '未実行')))
st.caption('EDINET公式: ' + str(item.get('EDINET_Status', '未実行')))
if str(item.get('EDINET_Status', '')).startswith('APIエラー'):
    st.warning('EDINET APIでエラーが発生しています。SecretsのEDINET_API_KEYと通信状況を確認してください。')
st.write(f"営業CF利回り: {percent(item['CFO_to_MktCap'])}（CF観測 {int(item['CFO_Obs'])} 年、うち黒字 {int(item['CFO_Positive_Years'])} 年）")
if item['ROE_pct'] > 30:
    st.caption('直近ROEが30%を超えるため、参考理論PERの計算では30%を上限としています。')
if item['CFO_Obs'] < 2 or item['CFO_Positive_Years'] < 2:
    st.caption('営業CFが2年連続プラスと確認できないため、営業CFスコアを減点しています。')
st.write(f"自己資本比率: {percent(item['EquityRatio'])} / ネットキャッシュ: 未判定")
if item['ROE_Obs'] < years:
    st.caption('指定年数分のROE履歴は取得できていません。履歴の平均値を長期平均とみなさないでください。')

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
