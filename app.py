import os
import math
import time
import threading
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


def financial_features(frame, years):
    result = dict(ROE_Avg=math.nan, ROE_Std=math.nan, ROE_Obs=0,
                  ROE_Trend=math.nan, CFO_Avg_Yen=math.nan, CFO_Obs=0, CFO_Positive_Years=0, CFO_Volatility=math.nan,
                  EquityRatio=math.nan, NetCash=math.nan, NetCashConfirmed=False)
    annual = annual_rows(frame, years)
    if annual.empty:
        return result
    profit = numeric_column(annual, 'NP', 'Profit', 'NetIncome')
    equity = numeric_column(annual, 'Eq', 'Equity')
    assets = numeric_column(annual, 'TA', 'TotalAssets')
    ratio = numeric_column(annual, 'EqAR', 'EquityToAssetRatio')
    cfo = numeric_column(annual, 'CFO', 'CashFlowsFromOperatingActivities', 'OperatingCashFlow')
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
    # Prefer the issuer's reported equity-to-assets ratio over Eq / TA.
    valid_ratio = ratio.dropna()
    if len(valid_ratio) and 0 <= valid_ratio.iloc[-1] <= 1:
        result['EquityRatio'] = float(valid_ratio.iloc[-1])
    elif len(assets) and pd.notna(assets.iloc[-1]) and assets.iloc[-1] > 0 and pd.notna(equity.iloc[-1]):
        calculated = float(equity.iloc[-1] / assets.iloc[-1])
        if 0 <= calculated <= 1:
            result['EquityRatio'] = calculated
    # Free-plan financial summaries do not reliably expose total interest-bearing debt.
    # Therefore net cash remains unconfirmed instead of inventing a value.
    return result


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


def build_results(master, valuation, years, ratio_limit, limit):
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
    with ThreadPoolExecutor(max_workers=min(5, len(pool))) as executor:
        jobs = {executor.submit(load_financials, str(r['Code'])): r for _, r in pool.iterrows()}
        for done, job in enumerate(as_completed(jobs), start=1):
            item = jobs[job]
            progress.progress(done / len(pool), text=f'財務データ {done}/{len(pool)}')
            try:
                features = financial_features(job.result(), years)
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
st.caption('J-Quants V2 無料プラン対応｜min(ROE, 30%) × 2 = 参考理論PER')
st.warning('独自の参考指標であり適正株価ではありません。ROEは評価上30%で上限を設け、履歴不足と営業CFの変動を減点します。無料プランのデータ遅延・履歴不足に注意。ネットキャッシュは未判定です。')
with st.sidebar:
    st.header('設定')
    st.write('J-Quants API: ' + ('設定済み' if API_KEY else '未設定'))
    years = st.slider('ROE履歴の最大年数', 3, 5, 5)
    ratio_limit = st.slider('割安判定（実PER / 理論PER）', .20, .80, .50, .05)
    limit = st.slider('財務分析する上位候補数', 5, 50, 10, 5)
    strict_net_cash = st.checkbox('ネットキャッシュ > 0 を必須', value=False)
    min_score = st.slider('最低スコア', 0, 100, 50)
    run = st.button('スクリーニング実行', type='primary', use_container_width=True)

if not API_KEY:
    st.error('Streamlit Secrets に JQUANTS_API_KEY を設定してください。')
    st.stop()
if strict_net_cash:
    st.error('無料プランでは有利子負債を確実に取得できないため、ネットキャッシュ必須条件は実行できません。チェックを外してください。')
    st.stop()

settings = (years, ratio_limit, limit, min_score)
if run or 'result' not in st.session_state or st.session_state.get('settings') != settings:
    try:
        with st.spinner('銘柄一覧・株価指標を取得中…'):
            master = load_master()
            master = master.loc[~excluded(master)].copy()
            valuation, valuation_day = load_valuation()
        with st.spinner('割安候補を確認中…'):
            result, analyzed, failures = build_results(master, valuation, years, ratio_limit, limit)
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
    'ROE観測年数': result['ROE_Obs'],
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
st.write(f"ROE履歴平均: {yen_percent(item['ROE_Avg'])}（観測 {int(item['ROE_Obs'])} 年、最大 {years} 年）")
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
