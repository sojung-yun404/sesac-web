"""Sparkify 이탈 방어 대시보드용 합성 데이터 생성.

원본 mini 데이터셋(225명)은 세그먼트 분석에 표본이 부족하므로,
노트북 EDA에서 확인된 통계적 구조를 보존한 채 5,000명 규모로 확장한다.

원본: https://www.kaggle.com/code/chriskue/sparkify-user-churn-prediction

출력:
  data/churn-summary.json  KPI / 일별 추이 / 드라이버 구간 / 세그먼트 / 모델 성능
  data/churn-users.json    사용자 단위 피처 (배열 기반 + 컬럼 사전)

표준 라이브러리만 사용. 시드 고정으로 재현 가능.
"""

import json
import math
import random
from datetime import date, timedelta
from pathlib import Path

SEED = 42
N_USERS = 5000

# 원본 노트북 In[15] 기준 관측 기간
WINDOW_START = date(2018, 10, 1)
WINDOW_END = date(2018, 12, 3)
WINDOW_DAYS = (WINDOW_END - WINDOW_START).days + 1  # 64일

# 원본 registration min/max
REG_MIN = date(2018, 3, 18)
REG_MAX = date(2018, 11, 26)

TARGET_CHURN_RATE = 0.231  # 52/225

# 이탈 드라이버 가중치. 노트북 GBT 피처 중요도 순서를 따른다:
#   가입 기간 > Thumbs Down > 친구 수
# 절대 크기는 잠재 위험이 충분히 양극화되도록 잡는다. 분산이 작으면 모든
# 사용자의 p가 기저율 근처에 몰려 어떤 모델도 세그먼트를 분리할 수 없다.
W_TENURE = 3.90    # 재적 기간 (가장 강한 드라이버)
W_CONTENT = 2.33   # 추천 품질 불만
W_DORMANT = 2.03   # 청취 강도 저하
W_SOCIAL = 1.88    # 소셜 연결 부족
W_ADVERT = 1.73    # 광고 피로 (무료 사용자만)

# 예측 스코어에 더할 노이즈. 실제 모델은 잠재 위험을 완벽히 복원하지 못한다.
MODEL_NOISE = 1.20

# 재적 기간 기반 유형의 절대 경계. 상대 z-score만 쓰면 정착한 사용자까지
# "온보딩 실패"로 분류되어 CRM 메시지가 어긋난다.
ONBOARDING_MAX_DAYS = 30    # 미만: 온보딩 실패
TENURE_TYPE_MAX_DAYS = 90   # 30~90: 정착 실패 / 이상: 재적 기간은 유형에서 제외
PAID_RATE = 0.36           # 노트북 In[24]: 유료가 소수
MONTHLY_FEE = 10900        # 원 (PRD §5 F6 가정)

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data"

STATES = [
    "CA", "TX", "NY", "FL", "PA", "IL", "OH", "GA", "NC", "MI",
    "NJ", "VA", "WA", "AZ", "MA", "TN", "IN", "MO", "MD", "WI",
    "CO", "MN", "SC", "AL", "LA", "KY", "OR", "OK", "CT", "IA",
]
OPERATING_SYSTEMS = ["Windows", "Mac", "iPhone", "Linux", "iPad"]
OS_WEIGHTS = [0.50, 0.28, 0.11, 0.07, 0.04]

RISK_TYPES = {
    "onboarding": {
        "label": "온보딩 실패",
        "action": "첫 플레이리스트 만들기 가이드 · D+3/D+7 온보딩 저니",
    },
    "early_tenure": {
        "label": "정착 실패",
        "action": "습관 형성 유도 · 주간 개인화 믹스 · 청취 리마인더",
    },
    "content": {
        "label": "콘텐츠 불만",
        "action": "취향 재설정 요청 · 큐레이션 재추천 · 신규 장르 제안",
    },
    "isolated": {
        "label": "고립형",
        "action": "친구 초대 인센티브 · 공유 플레이리스트 유도",
    },
    "dormant": {
        "label": "저활성",
        "action": "리인게이지먼트 푸시 · 개인화 위클리 믹스",
    },
    "ad_fatigue": {
        "label": "광고 피로",
        "action": "유료 체험 프로모션 · 광고 없는 주말 티저",
    },
}


def sigmoid(x):
    if x < -60:
        return 0.0
    if x > 60:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def percentile(sorted_values, q):
    """0~1 분위수. sorted_values는 정렬된 리스트."""
    if not sorted_values:
        return 0.0
    idx = q * (len(sorted_values) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_values[lo]
    frac = idx - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def build_base_users(rng):
    """기본 속성과 잠재 위험 로짓을 생성한다.

    행동 피처는 잠재 위험 z에 조건부로 만들어, 이탈자가 재적 기간이 짧고
    청취량·소셜 활동이 적다는 노트북 EDA 결과를 재현한다.
    """
    users = []

    for uid in range(1, N_USERS + 1):
        # --- 기본 속성 -------------------------------------------------
        reg_span = (REG_MAX - REG_MIN).days
        # 최근 가입자가 더 많은 성장 곡선 (제곱 분포)
        reg_offset = int(reg_span * (rng.random() ** 0.65))
        registration = REG_MIN + timedelta(days=reg_offset)

        # 관측 종료 시점 기준 재적 일수
        days_member_full = (WINDOW_END - registration).days
        if days_member_full < 3:
            registration = WINDOW_END - timedelta(days=3)
            days_member_full = 3

        is_male = 1 if rng.random() < 0.52 else 0
        is_paid = 1 if rng.random() < PAID_RATE else 0
        os_name = rng.choices(OPERATING_SYSTEMS, weights=OS_WEIGHTS)[0]
        state = rng.choice(STATES)

        # --- 잠재 위험 로짓 z ------------------------------------------
        # 노트북 GBT 피처 중요도 상위: days_member > thumbs_down > num_friend
        # 각 성향은 먼저 잠재값으로 뽑고, 그 잠재값에서 행동 피처를 만든다.

        # 재적 기간이 짧을수록 위험 (가장 강한 드라이버)
        tenure_risk = clamp((70 - min(days_member_full, 200)) / 70, -2.0, 1.0)

        # 개인 성향 잠재변수
        taste_mismatch = rng.gauss(0, 1)      # 추천 품질 불만 성향
        sociality = rng.gauss(0, 1)           # 소셜 활동 성향
        intensity = rng.gauss(0, 1)           # 청취 강도 성향
        ad_sensitivity = rng.gauss(0, 1)      # 광고 민감도

        z = (
            -1.55
            + W_TENURE * tenure_risk
            + W_CONTENT * taste_mismatch
            - W_SOCIAL * sociality
            - W_DORMANT * intensity
            + (W_ADVERT * ad_sensitivity if not is_paid else 0.0)
            + 0.42 * is_male
            + rng.gauss(0, 0.68)
        )

        users.append({
            "uid": uid,
            "registration": registration,
            "days_member_full": days_member_full,
            "is_male": is_male,
            "is_paid": is_paid,
            "os": os_name,
            "state": state,
            "z": z,
            "taste_mismatch": taste_mismatch,
            "sociality": sociality,
            "intensity": intensity,
            "ad_sensitivity": ad_sensitivity,
        })

    return users


def calibrate_intercept(users, target_rate):
    """전체 이탈률이 목표치가 되도록 로짓 절편을 이분 탐색으로 보정."""
    lo, hi = -6.0, 6.0
    for _ in range(60):
        mid = (lo + hi) / 2
        rate = sum(sigmoid(u["z"] + mid) for u in users) / len(users)
        if rate > target_rate:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def assign_outcomes(users, rng, shift):
    """이탈/다운그레이드 여부와 그 시점을 확정한다."""
    for u in users:
        p = sigmoid(u["z"] + shift)
        u["p_true"] = p
        u["churn"] = 1 if rng.random() < p else 0

        # 다운그레이드는 유료 사용자만. 여성이 더 높다 (노트북 In[23])
        if u["is_paid"]:
            p_down = sigmoid(u["z"] + shift - 0.75 + (0.30 if not u["is_male"] else 0.0))
            u["downgrade"] = 1 if rng.random() < p_down else 0
        else:
            u["downgrade"] = 0

        # 이탈 시점: 관측 기간 내 균등하되 후반부에 약간 몰림
        if u["churn"]:
            offset = int(WINDOW_DAYS * (rng.random() ** 0.85))
            u["churn_day"] = min(offset, WINDOW_DAYS - 1)
        else:
            u["churn_day"] = None

        if u["downgrade"]:
            limit = u["churn_day"] if u["churn_day"] is not None else WINDOW_DAYS - 1
            u["downgrade_day"] = rng.randint(0, max(limit, 0))
        else:
            u["downgrade_day"] = None


def build_behavior(users, rng):
    """잠재 성향에서 관측 가능한 행동 피처를 생성한다."""
    for u in users:
        # 관측 기간 중 실제로 활동한 일수
        active_from = max(0, (u["registration"] - WINDOW_START).days)
        active_to = u["churn_day"] if u["churn_day"] is not None else WINDOW_DAYS - 1
        active_days = max(1, active_to - active_from + 1)
        u["active_from"] = active_from
        u["active_to"] = active_to

        # 세션 빈도: 청취 강도가 높을수록 잦음
        base_freq = clamp(0.42 + 0.20 * u["intensity"] - 0.10 * max(u["z"], 0), 0.04, 0.95)
        u["session_freq"] = base_freq

        num_sessions = max(1, int(round(active_days * base_freq * rng.uniform(0.75, 1.25))))
        u["num_sessions"] = num_sessions

        # 세션당 곡 수: 원본 av_song_session 중앙값 ~50~90 수준
        av_song_session = max(3.0, rng.gauss(62 + 22 * u["intensity"], 26))
        u["av_song_session"] = av_song_session

        num_songs = int(round(num_sessions * av_song_session))
        u["num_songs"] = num_songs

        # 곡 길이 평균 249초 (원본 length describe)
        u["sum_listened"] = round(num_songs * rng.gauss(249, 12), 1)

        # 세션 지속시간 (시간)
        u["dur_session"] = round(clamp(av_song_session * 249 / 3600 * rng.uniform(0.8, 1.6), 0.05, 12), 2)

        # 아티스트 수: 곡 수에 로그 비례 (반복 청취 때문)
        u["num_artists"] = int(round(min(num_songs, num_songs ** 0.86 * rng.uniform(0.85, 1.15))))

        # 소셜/큐레이션 행동 — 곡 수 대비 비율로 생성
        up_rate = clamp(rng.gauss(0.052 - 0.020 * u["taste_mismatch"], 0.012), 0.002, 0.20)
        down_rate = clamp(rng.gauss(0.011 + 0.011 * u["taste_mismatch"], 0.004), 0.0002, 0.09)
        pl_rate = clamp(rng.gauss(0.028 + 0.010 * u["sociality"], 0.008), 0.001, 0.12)
        fr_rate = clamp(rng.gauss(0.017 + 0.011 * u["sociality"], 0.006), 0.0, 0.09)

        u["num_thumbs_up"] = int(round(num_songs * up_rate))
        u["num_thumbs_down"] = int(round(num_songs * down_rate))
        u["num_playlist"] = int(round(num_songs * pl_rate))
        u["num_friend"] = int(round(num_songs * fr_rate))

        # 광고 노출: 무료 사용자만
        if u["is_paid"] and not u["downgrade"]:
            u["num_advert"] = 0
        else:
            ad_rate = clamp(rng.gauss(0.075 + 0.022 * u["ad_sensitivity"], 0.018), 0.005, 0.25)
            u["num_advert"] = int(round(num_songs * ad_rate))

        # 파생 비율 (진단·유형 분류에 사용)
        denom = max(num_songs, 1)
        u["down_rate"] = u["num_thumbs_down"] / denom
        u["ad_rate"] = u["num_advert"] / denom
        u["days_member"] = u["days_member_full"]


def score_users(users, rng):
    """예측 모델 스코어를 시뮬레이션한다.

    실제 모델은 완벽하지 않으므로 잠재 로짓에 노이즈를 더한다.
    노트북의 F1 0.97은 누수가 의심되므로(PRD §6 참조) 현실적인 수준을 목표로 한다.
    """
    for u in users:
        noisy = u["z"] * 0.92 + rng.gauss(0, MODEL_NOISE)
        u["risk_score"] = round(sigmoid(noisy), 4)


def evaluate(users, threshold):
    tp = sum(1 for u in users if u["churn"] == 1 and u["risk_score"] >= threshold)
    fp = sum(1 for u in users if u["churn"] == 0 and u["risk_score"] >= threshold)
    fn = sum(1 for u in users if u["churn"] == 1 and u["risk_score"] < threshold)
    tn = sum(1 for u in users if u["churn"] == 0 and u["risk_score"] < threshold)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(users)
    return {
        "threshold": round(threshold, 3),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "accuracy": round(accuracy, 3),
    }


def pick_best_threshold(users):
    best = None
    t = 0.10
    while t <= 0.90:
        m = evaluate(users, t)
        if best is None or m["f1"] > best["f1"]:
            best = m
        t += 0.01
    return best


def _standardize(users, key_fn):
    """관측 지표를 z-score로 표준화한다. (평균 0, 표준편차 1)"""
    vals = [key_fn(u) for u in users]
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / n
    sd = math.sqrt(var) if var > 0 else 1.0
    return {id(u): (key_fn(u) - mean) / sd for u in users}


def assign_risk_types(users):
    """위험 기여도가 가장 큰 요인으로 위험 유형을 하나만 배정한다.

    각 드라이버를 관측 지표에서 z-score로 표준화한 뒤, 모델 계수 크기를
    가중치로 곱해 '이 사용자의 위험을 무엇이 끌어올렸는가'를 분해한다.
    임계값 기반 if-else 대신 이 방식을 쓰는 이유: 임계값 방식은 조건 순서에
    따라 유형이 갈려, 고위험 사용자가 먼저 걸린 유형에 쏠리고 나머지 유형에는
    저위험 사용자만 남는 편향이 생긴다.
    """
    # 위험을 높이는 방향이 +가 되도록 부호를 맞춘다
    tenure_z = _standardize(users, lambda u: -min(u["days_member"], 200))
    content_z = _standardize(users, lambda u: u["down_rate"])
    social_z = _standardize(users, lambda u: -(u["num_friend"] + u["num_playlist"]) / max(u["num_songs"], 1))
    dormant_z = _standardize(users, lambda u: -u["session_freq"])

    free_users = [u for u in users if not u["is_paid"]]
    ad_z = _standardize(free_users, lambda u: u["ad_rate"]) if free_users else {}

    # 가중치는 z 생성 시 사용한 계수와 동일 (드라이버 중요도)
    for u in users:
        contrib = {
            "content": W_CONTENT * content_z[id(u)],
            "isolated": W_SOCIAL * social_z[id(u)],
            "dormant": W_DORMANT * dormant_z[id(u)],
        }
        if not u["is_paid"]:
            contrib["ad_fatigue"] = W_ADVERT * ad_z[id(u)]

        # 재적 기간은 상대 z-score만으로 배정하면 90일차 사용자도 "재적이 짧은 편"이라
        # 이 유형에 걸려, 마케터가 정착한 사용자에게 온보딩 메시지를 보내게 된다.
        # 상대 기여도로 순위는 매기되 유형 배정은 절대 기간으로 막는다.
        if u["days_member"] < TENURE_TYPE_MAX_DAYS:
            key = "onboarding" if u["days_member"] < ONBOARDING_MAX_DAYS else "early_tenure"
            contrib[key] = W_TENURE * tenure_z[id(u)]

        u["risk_type"] = max(contrib.items(), key=lambda kv: kv[1])[0]
        u["risk_contrib"] = {k: round(v, 3) for k, v in contrib.items()}


def bucket_stats(users, key_fn, buckets):
    """구간별 이탈률.

    buckets는 (라벨, 범위설명, 판별함수) 리스트.
    라벨은 차트 축에 그려지므로 짧게 유지하고, 구체적인 경계값은 range로 분리해
    툴팁과 표에서만 보여준다 (긴 라벨은 축 영역을 넘어 잘린다).
    """
    rows = []
    for label, rng, pred in buckets:
        members = [u for u in users if pred(key_fn(u), u)]
        n = len(members)
        churned = sum(u["churn"] for u in members)
        rows.append({
            "label": label,
            "range": rng,
            "users": n,
            "churned": churned,
            "rate": round(churned / n, 4) if n else 0.0,
            "low_sample": n < 100,
        })
    return rows


def quartile_buckets(users, key_fn, fmt):
    values = sorted(key_fn(u) for u in users)
    q1 = percentile(values, 0.25)
    q2 = percentile(values, 0.50)
    q3 = percentile(values, 0.75)
    return [
        ("하위 25%", f"{fmt(values[0])} ~ {fmt(q1)}", lambda v, u, hi=q1: v <= hi),
        ("25~50%", f"{fmt(q1)} ~ {fmt(q2)}", lambda v, u, lo=q1, hi=q2: lo < v <= hi),
        ("50~75%", f"{fmt(q2)} ~ {fmt(q3)}", lambda v, u, lo=q2, hi=q3: lo < v <= hi),
        ("상위 25%", f"{fmt(q3)} 이상", lambda v, u, lo=q3: v > lo),
    ]


def build_daily_series(users):
    """일별 신규 해지 / 다운그레이드 / 활성 사용자."""
    churn_by_day = [0] * WINDOW_DAYS
    down_by_day = [0] * WINDOW_DAYS
    active_by_day = [0.0] * WINDOW_DAYS

    for u in users:
        if u["churn_day"] is not None:
            churn_by_day[u["churn_day"]] += 1
        if u["downgrade_day"] is not None:
            down_by_day[u["downgrade_day"]] += 1

        for d in range(u["active_from"], u["active_to"] + 1):
            # 주말 가중 (금~일 활동 증가)
            weekday = (WINDOW_START + timedelta(days=d)).weekday()
            season = 1.18 if weekday >= 4 else 0.94
            active_by_day[d] += u["session_freq"] * season

    def moving_avg(series, window=7):
        out = []
        for i in range(len(series)):
            lo = max(0, i - window + 1)
            chunk = series[lo:i + 1]
            out.append(round(sum(chunk) / len(chunk), 2))
        return out

    return {
        "dates": [(WINDOW_START + timedelta(days=d)).isoformat() for d in range(WINDOW_DAYS)],
        "churn": churn_by_day,
        "downgrade": down_by_day,
        "active": [int(round(a)) for a in active_by_day],
        "churn_ma7": moving_avg(churn_by_day),
        "downgrade_ma7": moving_avg(down_by_day),
    }


def build_summary(users, model):
    total = len(users)
    churned = sum(u["churn"] for u in users)
    paid = [u for u in users if u["is_paid"]]
    downgraded = sum(u["downgrade"] for u in users)

    # 잔존 사용자 = 아직 해지하지 않은 사용자 (캠페인 대상 모집단)
    retained = [u for u in users if u["churn"] == 0]
    high_risk = [u for u in users if u["risk_score"] >= model["threshold"] and u["churn"] == 0]
    high_risk_paid = [u for u in high_risk if u["is_paid"]]

    # 직전 기간 대비: 관측 기간을 반으로 나눠 비교
    half = WINDOW_DAYS // 2
    recent_churn = sum(1 for u in users if u["churn_day"] is not None and u["churn_day"] >= half)
    prev_churn = sum(1 for u in users if u["churn_day"] is not None and u["churn_day"] < half)

    kpis = {
        "total_users": total,
        "hard_churn_users": churned,
        "hard_churn_rate": round(churned / total, 4),
        "paid_users": len(paid),
        "soft_churn_users": downgraded,
        "soft_churn_rate": round(downgraded / len(paid), 4) if paid else 0.0,
        "high_risk_users": len(high_risk),
        "high_risk_paid_users": len(high_risk_paid),
        "at_risk_mrr": len(high_risk_paid) * MONTHLY_FEE,
        "recent_churn": recent_churn,
        "prev_churn": prev_churn,
        "churn_delta": round((recent_churn - prev_churn) / prev_churn, 4) if prev_churn else 0.0,
        "retained_users": len(retained),
    }

    drivers = [
        {
            "key": "tenure",
            "title": "가입 기간",
            "note": "이탈이 어느 시점에 몰리는지 → 온보딩 개입 시점을 정한다",
            "rows": bucket_stats(users, lambda u: u["days_member"], [
                ("~30일", "가입 30일 미만", lambda v, u: v < 30),
                ("30~60일", "가입 30~60일", lambda v, u: 30 <= v < 60),
                ("60~90일", "가입 60~90일", lambda v, u: 60 <= v < 90),
                ("90일~", "가입 90일 이상", lambda v, u: v >= 90),
            ]),
        },
        {
            "key": "thumbs_down",
            "title": "Thumbs Down 비율",
            "note": "추천 품질 불만이 이탈로 이어지는지 확인한다",
            "rows": bucket_stats(
                users, lambda u: u["down_rate"],
                quartile_buckets(users, lambda u: u["down_rate"], lambda v: f"{v * 100:.1f}%"),
            ),
        },
        {
            "key": "friends",
            "title": "친구 수",
            "note": "소셜 그래프가 잔존에 기여하는지 확인한다",
            "rows": bucket_stats(users, lambda u: u["num_friend"], [
                ("0명", "친구 없음", lambda v, u: v == 0),
                ("1~5명", "친구 1~5명", lambda v, u: 1 <= v <= 5),
                ("6~15명", "친구 6~15명", lambda v, u: 6 <= v <= 15),
                ("16명~", "친구 16명 이상", lambda v, u: v >= 16),
            ]),
        },
        {
            "key": "session_freq",
            "title": "세션 빈도",
            "note": "사용 빈도 저하가 이탈 선행지표인지 확인한다",
            "rows": bucket_stats(
                users, lambda u: u["session_freq"],
                quartile_buckets(users, lambda u: u["session_freq"], lambda v: f"{v:.2f}"),
            ),
        },
        {
            "key": "advert",
            "title": "광고 노출 (무료 사용자)",
            "note": "광고 피로가 이탈 요인인지 확인한다",
            "rows": bucket_stats(
                [u for u in users if not u["is_paid"]],
                lambda u: u["ad_rate"],
                quartile_buckets(
                    [u for u in users if not u["is_paid"]],
                    lambda u: u["ad_rate"], lambda v: f"{v * 100:.1f}%",
                ),
            ),
        },
    ]

    # 위험 유형별 분포.
    # 세 가지 비율을 구분해서 싣는다. 하나로 뭉치면 해석이 어긋난다.
    #   expected_churn_rate : 캠페인 대상(고위험 잔존)의 평균 예측 확률
    #                         — 앞으로 이만큼 빠져나갈 것으로 본다 (ROI 계산 입력)
    #   flagged_hit_rate    : 이 유형에서 고위험으로 분류된 전체 중 실제 이탈 비율
    #                         — 과거 스냅샷에서 모델이 맞았는지 검증하는 값
    #   segment_churn_rate  : 이 유형 전체의 이탈률 — 기저율 비교용 맥락
    segments = []
    for key, meta in RISK_TYPES.items():
        targets = [u for u in high_risk if u["risk_type"] == key]
        flagged = [u for u in users
                   if u["risk_type"] == key and u["risk_score"] >= model["threshold"]]
        all_of_type = [u for u in users if u["risk_type"] == key]

        segments.append({
            "key": key,
            "label": meta["label"],
            "action": meta["action"],
            "target_users": len(targets),
            "paid_users": sum(u["is_paid"] for u in targets),
            "expected_churn_rate": round(
                sum(u["risk_score"] for u in targets) / len(targets), 4) if targets else 0.0,
            "flagged_hit_rate": round(
                sum(u["churn"] for u in flagged) / len(flagged), 4) if flagged else 0.0,
            "segment_users": len(all_of_type),
            "segment_churn_rate": round(
                sum(u["churn"] for u in all_of_type) / len(all_of_type), 4) if all_of_type else 0.0,
        })
    segments.sort(key=lambda s: s["target_users"], reverse=True)

    # 가입 코호트별 이탈률
    cohorts = {}
    for u in users:
        key = u["registration"].strftime("%Y-%m")
        c = cohorts.setdefault(key, {"month": key, "users": 0, "churned": 0})
        c["users"] += 1
        c["churned"] += u["churn"]
    cohort_rows = sorted(cohorts.values(), key=lambda c: c["month"])
    for c in cohort_rows:
        c["rate"] = round(c["churned"] / c["users"], 4)

    return {
        "meta": {
            "generated_for": "Sparkify 이탈 방어 대시보드",
            "source": "https://www.kaggle.com/code/chriskue/sparkify-user-churn-prediction",
            "synthetic": True,
            "seed": SEED,
            "window_start": WINDOW_START.isoformat(),
            "window_end": WINDOW_END.isoformat(),
            "window_days": WINDOW_DAYS,
            "monthly_fee": MONTHLY_FEE,
        },
        "kpis": kpis,
        "daily": build_daily_series(users),
        "drivers": drivers,
        "segments": segments,
        "cohorts": cohort_rows,
        "model": model,
        "risk_types": {k: v for k, v in RISK_TYPES.items()},
    }


USER_COLUMNS = [
    "uid", "risk_score", "risk_type", "is_paid", "is_male", "churn", "downgrade",
    "days_member", "num_sessions", "num_songs", "num_thumbs_up", "num_thumbs_down",
    "num_playlist", "num_friend", "num_advert", "session_freq", "state", "os",
]


def build_users_payload(users):
    state_dict = sorted({u["state"] for u in users})
    os_dict = sorted({u["os"] for u in users})
    type_dict = list(RISK_TYPES.keys())

    state_idx = {s: i for i, s in enumerate(state_dict)}
    os_idx = {s: i for i, s in enumerate(os_dict)}
    type_idx = {s: i for i, s in enumerate(type_dict)}

    rows = []
    for u in sorted(users, key=lambda x: -x["risk_score"]):
        rows.append([
            u["uid"],
            u["risk_score"],
            type_idx[u["risk_type"]],
            u["is_paid"],
            u["is_male"],
            u["churn"],
            u["downgrade"],
            u["days_member"],
            u["num_sessions"],
            u["num_songs"],
            u["num_thumbs_up"],
            u["num_thumbs_down"],
            u["num_playlist"],
            u["num_friend"],
            u["num_advert"],
            round(u["session_freq"], 3),
            state_idx[u["state"]],
            os_idx[u["os"]],
        ])

    return {
        "columns": USER_COLUMNS,
        "dicts": {"state": state_dict, "os": os_dict, "risk_type": type_dict},
        "rows": rows,
    }


def main():
    rng = random.Random(SEED)

    users = build_base_users(rng)
    shift = calibrate_intercept(users, TARGET_CHURN_RATE)
    assign_outcomes(users, rng, shift)
    build_behavior(users, rng)
    score_users(users, rng)
    assign_risk_types(users)

    model = pick_best_threshold(users)
    model["note"] = (
        "원본 노트북의 GBT F1 0.97은 누수가 의심되어(PRD §6) "
        "예측 시점을 고정한 현실적 성능을 가정했다."
    )
    model["baseline_notebook_f1"] = 0.97

    summary = build_summary(users, model)
    payload = build_users_payload(users)

    OUT_DIR.mkdir(exist_ok=True)
    summary_path = OUT_DIR / "churn-summary.json"
    users_path = OUT_DIR / "churn-users.json"

    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )
    users_path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )

    k = summary["kpis"]
    print(f"users            : {k['total_users']:,}")
    print(f"hard churn       : {k['hard_churn_users']:,} ({k['hard_churn_rate']:.1%})  목표 {TARGET_CHURN_RATE:.1%}")
    print(f"soft churn       : {k['soft_churn_users']:,} ({k['soft_churn_rate']:.1%} of paid)")
    print(f"high risk (잔존) : {k['high_risk_users']:,}  (유료 {k['high_risk_paid_users']:,})")
    print(f"at-risk MRR      : {k['at_risk_mrr']:,}원")
    print(f"model            : threshold={model['threshold']} "
          f"P={model['precision']} R={model['recall']} F1={model['f1']} Acc={model['accuracy']}")
    print()
    print(f"{'유형':<10} {'타겟':>6} {'예측이탈':>8} {'적중률':>7} {'유형전체':>8} {'유형이탈률':>9}")
    for s in summary["segments"]:
        print(f"  {s['label']:<8} {s['target_users']:>5,} "
              f"{s['expected_churn_rate']:>9.1%} {s['flagged_hit_rate']:>7.1%} "
              f"{s['segment_users']:>8,} {s['segment_churn_rate']:>9.1%}")
    print()
    print(f"{summary_path.name:<22} {summary_path.stat().st_size / 1024:>8.1f} KB")
    print(f"{users_path.name:<22} {users_path.stat().st_size / 1024:>8.1f} KB")


if __name__ == "__main__":
    main()
