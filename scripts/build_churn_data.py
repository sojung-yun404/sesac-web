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

# 비율 지표(Thumbs Down·광고 노출)의 분모 하한. 이보다 적게 들은 사용자는
# 비율이 0에 붙어버려 "추천이 만족스러움"과 "거의 안 들음"이 구분되지 않는다.
MIN_SONGS_FOR_RATIO = 50

# 리텐션 감쇠 시상수(주). 가입 직후 접속 확률이 바닥값으로 내려앉는 속도.
RETENTION_TAU_WEEKS = 2.5

# 이탈 위험률의 평균 재적일수. 창 안에서 가입한 사용자의 해지 시점을
# 가입 후 경과일 기준 지수분포로 뽑을 때 쓴다.
CHURN_MEAN_TENURE_DAYS = 24
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

# 위험 유형별 캠페인 정의.
#   channel : 실제로 무엇을 보내는가
#   cost    : 1인당 원가(원). 푸시·인앱은 발송비 수준, 인센티브·체험은 실비.
#             실제 집행가로 바꿔 쓰라는 의미의 기본값이며 화면에서 수정 가능하다.
RISK_TYPES = {
    "onboarding": {
        "label": "온보딩 실패",
        "action": "첫 플레이리스트 만들기 가이드 · D+3/D+7 온보딩 저니",
        "channel": "앱 푸시 + 인앱 가이드",
        "cost": 500,
    },
    "early_tenure": {
        "label": "정착 실패",
        "action": "습관 형성 유도 · 주간 개인화 믹스 · 청취 리마인더",
        "channel": "주간 푸시 + 이메일",
        "cost": 500,
    },
    "content": {
        "label": "콘텐츠 불만",
        "action": "취향 재설정 요청 · 큐레이션 재추천 · 신규 장르 제안",
        "channel": "인앱 설문 + 재추천 배너",
        "cost": 800,
    },
    "isolated": {
        "label": "고립형",
        "action": "친구 초대 인센티브 · 공유 플레이리스트 유도",
        "channel": "초대 리워드 (양방 지급)",
        "cost": 2000,
    },
    "dormant": {
        "label": "저활성",
        "action": "리인게이지먼트 푸시 · 개인화 위클리 믹스",
        "channel": "앱 푸시 + 이메일",
        "cost": 500,
    },
    "ad_fatigue": {
        "label": "광고 피로",
        "action": "유료 체험 프로모션 · 광고 없는 주말 티저",
        "channel": "유료 1개월 무료 체험",
        "cost": 10900,
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
            "tenure_risk": tenure_risk,
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

        # 이탈 시점은 위험률(hazard)을 따른다. 가입 직후가 가장 위험하고
        # 시간이 지날수록 잦아든다 — 원본 노트북의 "이탈자 절반이 50일 안에
        # 이탈한다"는 관찰과 같은 구조다.
        #
        # 시점을 창 안에서 임의로 뽑으면 가입일 제약과 겹쳐 후반부에 근거 없이
        # 몰리고, "해지 3배 급증" 같은 가짜 신호가 KPI에 뜬다.
        reg_offset = (u["registration"] - WINDOW_START).days   # 음수면 창 이전 가입
        first_day = max(0, reg_offset)
        last_day = WINDOW_DAYS - 1

        if u["churn"]:
            if reg_offset >= 0:
                # 창 안에서 가입 → 가입 후 경과일을 지수분포로 뽑는다
                tenure_at_churn = rng.expovariate(1.0 / CHURN_MEAN_TENURE_DAYS)
                day = reg_offset + int(tenure_at_churn)
                # 창을 넘어가면 아직 관측되지 않은 것이므로 창 안으로 되돌린다
                u["churn_day"] = min(day, last_day) if day >= first_day else first_day
            else:
                # 창 이전 가입자는 이미 초기 위험 구간을 넘긴 생존자다.
                # 남은 위험률은 완만하므로 창 안에서 균등하게 본다.
                u["churn_day"] = rng.randint(first_day, last_day)
        else:
            u["churn_day"] = None

        if u["downgrade"]:
            limit = u["churn_day"] if u["churn_day"] is not None else WINDOW_DAYS - 1
            u["downgrade_day"] = rng.randint(first_day, limit) if limit >= first_day else first_day
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

        # 세션 빈도의 기준선도 가입 시점을 뺀 내재 위험으로 잡는다.
        # 여기에 재적 기간이 섞이면 늦게 가입했다는 이유만으로 기준선이 낮아져,
        # 감쇠를 분리해 둔 의미가 사라진다. 가입 시점의 영향은 아래 decay가 맡는다.
        z_intrinsic = u["z"] - W_TENURE * u["tenure_risk"]
        base_freq = clamp(0.42 + 0.20 * u["intensity"] - 0.10 * max(z_intrinsic, 0), 0.04, 0.95)
        u["session_freq"] = base_freq

        # 접속일을 실제로 추출한다. 빈도의 합만으로는 DAU밖에 못 만들고,
        # WAU/MAU는 "기간 내 고유 사용자"라 접속일 자체가 있어야 계산된다.
        # 비트마스크로 담아 롤링 윈도우를 빠르게 훑는다.
        # 접속 확률은 가입 후 경과 주차에 따라 감쇠한다.
        # 이걸 넣지 않으면 위험도가 '관측 종료일 기준 재적일수'에만 걸려,
        # 늦게 가입한 사용자는 가입 직후부터 영구히 활동이 낮게 나온다.
        # 그러면 코호트 표에서 최근 코호트가 W1부터 무너지는 것처럼 보이는데,
        # 이는 획득 품질 신호가 아니라 생성 방식의 부산물이다.
        # 바닥값은 '가입 시점을 뺀' 내재 위험으로만 정한다.
        # 재적 기간까지 넣으면 늦게 가입했다는 이유만으로 감쇠가 빨라져,
        # 코호트를 같은 주차끼리 비교해도 최근 코호트가 나쁘게 보인다.
        # 리텐션 곡선의 모양은 취향·소셜·청취 강도가 결정하고,
        # 가입 시점은 '지금 며칠 됐는가'로만 위험에 반영되는 게 맞다.
        floor = clamp(0.78 - 0.55 * sigmoid(z_intrinsic), 0.12, 0.88)
        reg_day = (u["registration"] - WINDOW_START).days   # 음수면 창 이전 가입

        mask = 0
        hits = 0
        for d in range(active_from, active_to + 1):
            weekday = (WINDOW_START + timedelta(days=d)).weekday()
            season = 1.18 if weekday >= 4 else 0.94   # 금~일 활동 증가
            weeks_since = max(0, (d - reg_day)) / 7.0
            decay = floor + (1 - floor) * math.exp(-weeks_since / RETENTION_TAU_WEEKS)
            if rng.random() < clamp(base_freq * season * decay, 0.01, 0.98):
                mask |= (1 << d)
                hits += 1
        # 관측 창 안에서 가입했다면 가입일은 접속한 것으로 본다.
        # 가입 행위 자체가 사용이고, 코호트 표의 W0가 100%가 되는 표준 규약과도 맞다.
        # 창 이전 가입자에게 적용하면 창 첫날에 3천여 명이 몰려 DAU가 튄다.
        if u["registration"] >= WINDOW_START and not (mask & (1 << active_from)):
            mask |= (1 << active_from)
            hits += 1
        elif hits == 0:   # 최소 하루는 접속한 것으로 둔다
            mask |= (1 << active_from)
            hits = 1
        u["active_mask"] = mask
        u["active_day_count"] = hits

        num_sessions = max(1, int(round(hits * rng.uniform(0.9, 1.5))))
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
    """일별 신규 해지 / 다운그레이드."""
    churn_by_day = [0] * WINDOW_DAYS
    down_by_day = [0] * WINDOW_DAYS

    for u in users:
        if u["churn_day"] is not None:
            churn_by_day[u["churn_day"]] += 1
        if u["downgrade_day"] is not None:
            down_by_day[u["downgrade_day"]] += 1

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
        "churn_ma7": moving_avg(churn_by_day),
        "downgrade_ma7": moving_avg(down_by_day),
    }


def build_cohort_matrix(users):
    """주간 리텐션 코호트 (삼각 행렬).

    행 = 가입 주, 열 = 가입 후 경과 주차, 셀 = 그 주에 한 번이라도 접속한 비율.

    가입 월별 이탈률을 그냥 나열하면 최근 코호트일수록 재적 기간이 짧아 이탈률이
    높게 나온다 — 획득 품질이 나빠진 것처럼 보이지만 실제로는 관측 기간의 차이다.
    코호트를 '가입 후 같은 주차'끼리 비교하면 그 편향이 사라진다. 그게 이 표의 요점.

    관측 창(64일) 안에서 가입한 사용자만 대상으로 한다. 창 이전 가입자는
    초기 몇 주의 활동 기록이 아예 없어 W0·W1을 계산할 수 없다.
    """
    full_weeks = WINDOW_DAYS // 7   # 온전히 관측된 주 수 (부분 주는 값이 낮게 나와 제외)

    cohorts = {}
    for u in users:
        if u["registration"] < WINDOW_START:
            continue
        reg_day = (u["registration"] - WINDOW_START).days
        wk = reg_day // 7
        if wk >= full_weeks:
            continue
        cohorts.setdefault(wk, []).append(u)

    rows = []
    for wk in sorted(cohorts):
        members = cohorts[wk]
        size = len(members)
        cells = []
        for k in range(full_weeks - wk):
            lo = (wk + k) * 7
            hi = lo + 6
            wmask = ((1 << (hi + 1)) - 1) ^ ((1 << lo) - 1)
            active = sum(1 for u in members if u["active_mask"] & wmask)
            cells.append({
                "week": k,
                "rate": round(active / size, 4),
                "users": active,
            })
        start = WINDOW_START + timedelta(days=wk * 7)
        rows.append({
            "cohort": start.isoformat(),
            "label": start.strftime("%m/%d"),
            "size": size,
            "low_sample": size < 100,
            "cells": cells,
        })

    # 주차별 평균 리텐션 (코호트 크기 가중) — 표 하단 요약행
    avg = []
    for k in range(full_weeks):
        num = sum(c["cells"][k]["users"] for c in rows if len(c["cells"]) > k)
        den = sum(c["size"] for c in rows if len(c["cells"]) > k)
        if den:
            avg.append({"week": k, "rate": round(num / den, 4), "cohorts": sum(1 for c in rows if len(c["cells"]) > k)})
    return {
        "weeks": full_weeks,
        "rows": rows,
        "average": avg,
        "note": "관측 창 안에서 가입한 사용자만. 부분 주는 값이 낮게 나오므로 제외했다.",
    }


def build_engagement_series(users):
    """DAU / WAU / MAU와 고착도(DAU÷MAU).

    셋 다 '해당 기간에 한 번이라도 접속한 고유 사용자'다. 창이 30일이므로
    앞쪽 29일은 창이 덜 찬 상태라 값이 인위적으로 낮게 나온다.
    그 구간을 그대로 그리면 "MAU가 급증하는 중"으로 오독되므로,
    세 지표가 모두 온전해지는 시점부터만 내보낸다.
    """
    masks = [u["active_mask"] for u in users]
    start = 29  # MAU 창(30일)이 처음으로 가득 차는 날

    def window_mask(d, span):
        lo = max(0, d - span + 1)
        return ((1 << (d + 1)) - 1) ^ ((1 << lo) - 1)

    dau, wau, mau, sticky = [], [], [], []
    for d in range(start, WINDOW_DAYS):
        wd, ww, wm = window_mask(d, 1), window_mask(d, 7), window_mask(d, 30)
        a = sum(1 for m in masks if m & wd)
        w = sum(1 for m in masks if m & ww)
        mo = sum(1 for m in masks if m & wm)
        dau.append(a)
        wau.append(w)
        mau.append(mo)
        sticky.append(round(a / mo, 4) if mo else 0.0)

    return {
        "dates": [(WINDOW_START + timedelta(days=d)).isoformat()
                  for d in range(start, WINDOW_DAYS)],
        "dau": dau,
        "wau": wau,
        "mau": mau,
        "stickiness": sticky,
        "note": "DAU/WAU/MAU는 각각 1일·7일·30일 창의 고유 접속자 수. "
                "30일 창이 가득 차는 날부터 표시한다.",
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

    # 전·후반기 비교는 반드시 '비율'로 한다.
    # 원시 해지 건수를 비교하면 모수 증가를 보정하지 못해 급증으로 오독된다.
    # 특히 신규 가입이 계속 들어오는 구간에서는 후반기 건수가 구조적으로 크다.
    half = WINDOW_DAYS // 2

    def half_window(lo, hi):
        return ((1 << (hi + 1)) - 1) ^ ((1 << lo) - 1)

    w1, w2 = half_window(0, half - 1), half_window(half, WINDOW_DAYS - 1)

    def period_rate(wmask, lo, hi):
        # 분모: 그 기간에 한 번이라도 활동한 사용자 (= 이탈할 수 있었던 모수)
        base = sum(1 for u in users if u["active_mask"] & wmask)
        churned = sum(1 for u in users
                      if u["churn_day"] is not None and lo <= u["churn_day"] <= hi)
        return base, churned, (churned / base if base else 0.0)

    prev_base, prev_churn, prev_rate = period_rate(w1, 0, half - 1)
    recent_base, recent_churn, recent_rate = period_rate(w2, half, WINDOW_DAYS - 1)

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
        "recent_churn_rate": round(recent_rate, 4),
        "prev_churn_rate": round(prev_rate, 4),
        "recent_base": recent_base,
        "prev_base": prev_base,
        "churn_delta": round((recent_rate - prev_rate) / prev_rate, 4) if prev_rate else 0.0,
        "retained_users": len(retained),
    }

    # 비율 지표(Thumbs Down·광고 노출)는 분모가 작으면 값이 튄다.
    ratio_pop = [u for u in users if u["num_songs"] >= MIN_SONGS_FOR_RATIO]
    ratio_pop_free = [u for u in ratio_pop if not u["is_paid"]]

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
            # 곡 수가 적으면 Thumbs Down 비율이 0에 수렴해 하위 구간에 몰린다.
            # 그런 사용자는 '추천이 만족스러워서'가 아니라 '거의 안 들어서' 0이고,
            # 저활성 때문에 이탈률이 높아 구간별 비교가 U자로 왜곡된다.
            # 비율이 의미를 갖는 최소 청취량 이상만 대상으로 한다.
            "key": "thumbs_down",
            "title": "Thumbs Down 비율",
            "note": f"추천 품질 불만이 이탈로 이어지는지 확인한다 "
                    f"(비율이 의미 있는 {MIN_SONGS_FOR_RATIO}곡 이상 청취자만)",
            "rows": bucket_stats(
                ratio_pop, lambda u: u["down_rate"],
                quartile_buckets(ratio_pop, lambda u: u["down_rate"], lambda v: f"{v * 100:.1f}%"),
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
            "note": f"광고 피로가 이탈 요인인지 확인한다 "
                    f"({MIN_SONGS_FOR_RATIO}곡 이상 청취한 무료 사용자만)",
            "rows": bucket_stats(
                ratio_pop_free, lambda u: u["ad_rate"],
                quartile_buckets(ratio_pop_free, lambda u: u["ad_rate"],
                                 lambda v: f"{v * 100:.1f}%"),
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
            "channel": meta["channel"],
            "cost": meta["cost"],
            "target_users": len(targets),
            "paid_users": sum(u["is_paid"] for u in targets),
            # ROI 계산용: 대상자의 예측 이탈 확률 합 = 기대 이탈 인원
            "expected_churners": round(sum(u["risk_score"] for u in targets), 1),
            "expected_churners_paid": round(
                sum(u["risk_score"] for u in targets if u["is_paid"]), 1),
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
        "engagement": build_engagement_series(users),
        "cohort_matrix": build_cohort_matrix(users),
        "baseline_churn_rate": kpis["hard_churn_rate"],
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
