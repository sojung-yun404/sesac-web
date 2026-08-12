"""서울/경기 아파트 실거래 데이터를 대시보드용 JSON(regions.json, trades-*.json)으로 변환.

파이썬 표준 라이브러리만 사용한다 (pandas 금지, analyze.py와 동일한 기조).
출력 스키마는 docs/DATA_CONTRACT.md 를 그대로 따른다.

입력 소스는 두 가지이고, 있으면 API 쪽을 우선한다 (건축년도 정보가 있어서 더 낫다):
  - seoul-apt-latest.csv        : 기존 서울 원본 CSV (건축년도 없음 -> build_year=0 고정)
  - data/raw/molit-{seoul,gyeonggi}.csv : scripts/fetch_molit.py 로 수집한 API 원본
                                            (건축년도 포함, 서울/경기 둘 다 지원)

구조:
  - parse_seoul_csv() : 서울 CSV(기존 경로) -> 정규화 레코드(Record) 리스트
  - parse_api_csv()   : fetch_molit.py 결과 CSV(서울/경기 공용 포맷) -> Record 리스트
  - build_regions()     : 정규화 레코드 -> regions.json 의 "regions" 배열
  - build_trades_json() : 정규화 레코드 -> trades-*.json 전체 구조
  뒤 둘은 시도(sido)에 상관없이 Record 리스트만 받으므로, 어느 파서를 거쳤든
  같은 Record 형태로만 넘어오면 그대로 재사용된다.
"""
import csv
import json
import statistics
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import NamedTuple, Optional

BASE_DIR = Path(__file__).resolve().parent.parent
SEOUL_CSV = BASE_DIR / "seoul-apt-latest.csv"
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"

# scripts/fetch_molit.py 가 만드는 API 원본. 있으면 이쪽을 우선 사용한다
# (건축년도가 있어서 build_year 필터가 실제로 동작함). 서울 CSV와 달리
# 아직 발급받은 API 키로 수집하기 전에는 존재하지 않을 수 있다 -> 없으면
# 기존처럼 SEOUL_CSV 로 폴백한다 (경기는 API 경로 외에 대체 소스가 없으므로
# 파일이 없으면 그냥 데이터셋에서 빠진다).
SEOUL_API_CSV = RAW_DIR / "molit-seoul.csv"
GYEONGGI_API_CSV = RAW_DIR / "molit-gyeonggi.csv"

# docs/DATA_CONTRACT.md 의 평형 버킷 정의와 한 글자도 다르지 않게 맞춘다.
BUCKETS = [
    {"key": "u59", "label": "59㎡ 이하", "min": 0, "max": 60},
    {"key": "s60", "label": "60~80㎡", "min": 60, "max": 80},
    {"key": "s80", "label": "84㎡대 (국평)", "min": 80, "max": 90},
    {"key": "s90", "label": "90~135㎡", "min": 90, "max": 135},
    {"key": "o135", "label": "135㎡ 초과", "min": 135, "max": None},
]


class Record(NamedTuple):
    """전처리 전 구간의 정규화된 거래 1건. 시도(csv)가 달라져도 이 형태로만 맞추면
    build_regions()/build_trades_json() 을 그대로 재사용할 수 있다."""
    sido: str
    region: str      # 구/시
    dong: str
    complex: str
    month: str        # "YYYY-MM"
    area_x10: int      # 전용면적 x10 (정수)
    floor: int
    build_year: int   # 정보 없으면 0
    price: int         # 만원


# ---------------------------------------------------------------------------
# 파싱: 원본 CSV -> 정규화 레코드
# ---------------------------------------------------------------------------

def parse_int(value, default=0):
    """방어적 int 파싱. 실패하면 default 반환 (default=None 이면 실패를 알리는 용도)."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_area_x10(value):
    """"84.8" -> 848. 실패하면 None."""
    try:
        return round(float(value) * 10)
    except (TypeError, ValueError):
        return None


def parse_seoul_csv(csv_path):
    """서울 실거래 CSV를 읽어 매매 건만 Record 리스트로 반환한다.

    이 CSV에는 건축년도 컬럼이 없으므로 build_year 는 계약서 규정대로 항상 0.
    """
    records = []
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["deal_type"] != "매매":
                continue

            area_x10 = parse_area_x10(row["area_m2"])
            price = parse_int(row["price"], default=None)
            if area_x10 is None or price is None:
                continue  # 면적/가격이 깨진 행은 방어적으로 제외

            floor = parse_int(row["floor"], default=0)

            records.append(Record(
                sido="서울",
                region=row["gu"],
                dong=row["dong"],
                complex=row["complex"],
                month=row["contract_ym"],
                area_x10=area_x10,
                floor=floor,
                build_year=0,  # 서울 CSV에 건축년도 컬럼 없음 -> "정보 없음"
                price=price,
            ))
    return records


def parse_api_csv(csv_path, sido):
    """fetch_molit.py 가 만든 원본 CSV -> Record 리스트.

    이 CSV는 이미 매매 거래만, 해제된 거래(cdealType=O)는 제외한 상태로 저장돼
    있다 (fetch_molit.py의 item_to_row 참고) -> 여기서 다시 걸러낼 필요 없음.
    건축년도(build_year)가 원본에 들어있으므로 0으로 고정하지 않고 그대로 쓴다.
    """
    records = []
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            area_x10 = parse_area_x10(row["area_m2"])
            price = parse_int(row["price"], default=None)
            if area_x10 is None or price is None:
                continue  # 면적/가격이 깨진 행은 방어적으로 제외

            records.append(Record(
                sido=sido,
                region=row["region"],
                dong=row["dong"],
                complex=row["complex"],
                month=row["contract_ym"],
                area_x10=area_x10,
                floor=parse_int(row["floor"], default=0),
                build_year=parse_int(row["build_year"], default=0),
                price=price,
            ))
    return records


# ---------------------------------------------------------------------------
# 집계: 정규화 레코드 -> regions.json
# ---------------------------------------------------------------------------

def compute_months(records):
    """전체 레코드에서 등장하는 월(YYYY-MM)을 오래된 순으로 정렬해 반환.
    YYYY-MM 형식은 문자열 정렬 = 시간 순 정렬이라 별도 파싱이 필요 없다."""
    return sorted({r.month for r in records})


def bucket_key_for(area_x10):
    """area_x10(전용면적 x10, 정수)이 속하는 버킷 key. 실수 나눗셈 없이 x10 단위
    그대로 비교해 부동소수점 오차를 피한다."""
    for b in BUCKETS:
        if area_x10 < b["min"] * 10:
            continue
        if b["max"] is not None and area_x10 >= b["max"] * 10:
            continue
        return b["key"]
    return None


def median_round(prices):
    """계약서 규정: statistics.median 결과를 반올림해 정수로 맞춘다."""
    return round(statistics.median(prices))


def summarize(prices):
    """[중위가, 최저가, 최고가, 거래건수]"""
    return [median_round(prices), min(prices), max(prices), len(prices)]


def compute_bucket_stats(bucket_records, months):
    prices_all = [r.price for r in bucket_records]
    stats = {"all": summarize(prices_all)}

    recent6 = set(months[-6:])
    recent3 = set(months[-3:])
    r6_prices = [r.price for r in bucket_records if r.month in recent6]
    r3_prices = [r.price for r in bucket_records if r.month in recent3]
    # 최근 구간에 거래가 아예 없을 수도 있으므로(버킷 자체는 존재해도) 그때는 키를 생략한다.
    if r6_prices:
        stats["r6"] = summarize(r6_prices)
    if r3_prices:
        stats["r3"] = summarize(r3_prices)

    by_month = defaultdict(list)
    for r in bucket_records:
        by_month[r.month].append(r.price)
    stats["monthly"] = [
        [median_round(by_month[m]), len(by_month[m])] if by_month.get(m) else None
        for m in months
    ]

    return stats


def build_regions(sido_records, months):
    """sido_records: [(sido, records), ...] -> regions.json 의 "regions" 배열.

    경기 데이터가 추가되면 [("서울", seoul_records), ("경기", gyeonggi_records)]
    형태로 넘기면 그대로 합쳐진 배열이 나온다.
    """
    regions = []
    for sido, records in sido_records:
        by_region = defaultdict(list)
        for r in records:
            by_region[r.region].append(r)

        for region_name in sorted(by_region):
            by_bucket = defaultdict(list)
            for r in by_region[region_name]:
                key = bucket_key_for(r.area_x10)
                if key:
                    by_bucket[key].append(r)

            stats = {}
            for b in BUCKETS:
                bucket_records = by_bucket.get(b["key"])
                if bucket_records:  # 거래 없는 버킷은 키 자체를 생략
                    stats[b["key"]] = compute_bucket_stats(bucket_records, months)

            regions.append({"sido": sido, "name": region_name, "stats": stats})
    return regions


# ---------------------------------------------------------------------------
# 직렬화: 정규화 레코드 -> trades-*.json
# ---------------------------------------------------------------------------

def build_trades_json(sido, records, months):
    """문자열(구/동/단지명)을 사전으로 분리하고, 거래는 인덱스 배열로 압축한다."""
    regions_sorted = sorted({r.region for r in records})
    dongs_sorted = sorted({r.dong for r in records})
    complexes_sorted = sorted({r.complex for r in records})

    region_idx = {name: i for i, name in enumerate(regions_sorted)}
    dong_idx = {name: i for i, name in enumerate(dongs_sorted)}
    complex_idx = {name: i for i, name in enumerate(complexes_sorted)}
    month_idx = {name: i for i, name in enumerate(months)}

    trades = [
        [
            region_idx[r.region],
            dong_idx[r.dong],
            complex_idx[r.complex],
            month_idx[r.month],
            r.area_x10,
            r.floor,
            r.build_year,
            r.price,
        ]
        for r in records
    ]

    return {
        "meta": {"sido": sido, "months": months},
        "regions": regions_sorted,
        "dongs": dongs_sorted,
        "complexes": complexes_sorted,
        "fields": ["r", "d", "c", "m", "a", "f", "y", "p"],
        "trades": trades,
    }


# ---------------------------------------------------------------------------
# 출력
# ---------------------------------------------------------------------------

def write_json(path, obj):
    # 용량 우선: 공백 없이, 한글은 escape 하지 않는다.
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, separators=(",", ":"))


def main():
    DATA_DIR.mkdir(exist_ok=True)

    # 서울: API 원본이 있으면 그걸 우선 사용 (건축년도 포함) -> 없으면 기존 CSV로 폴백.
    # 기존 동작(=API 원본이 없는 지금)을 절대 깨면 안 되므로, 이 분기가 핵심이다.
    if SEOUL_API_CSV.exists():
        seoul_records = parse_api_csv(SEOUL_API_CSV, "서울")
        seoul_source = f"API ({SEOUL_API_CSV.name})"
    else:
        seoul_records = parse_seoul_csv(SEOUL_CSV)
        seoul_source = f"CSV ({SEOUL_CSV.name})"

    datasets = [
        ("서울", "trades-seoul.json", seoul_records),
    ]

    # 경기: 대체 소스가 없으므로 API 원본이 있을 때만 데이터셋에 추가한다.
    if GYEONGGI_API_CSV.exists():
        gyeonggi_records = parse_api_csv(GYEONGGI_API_CSV, "경기")
        datasets.append(("경기", "trades-gyeonggi.json", gyeonggi_records))
        print(f"[소스] 경기: API ({GYEONGGI_API_CSV.name})")

    print(f"[소스] 서울: {seoul_source}")

    all_records = [r for _, _, recs in datasets for r in recs]
    months = compute_months(all_records)

    regions_json = {
        "meta": {
            "generated": date.today().isoformat(),
            "months": months,
            "buckets": BUCKETS,
            "source": "국토교통부 실거래가 공개시스템",
            "datasets": [
                {"sido": sido, "file": file, "trades": len(recs)}
                for sido, file, recs in datasets
            ],
        },
        "regions": build_regions([(sido, recs) for sido, _, recs in datasets], months),
    }
    write_json(DATA_DIR / "regions.json", regions_json)

    for sido, file, recs in datasets:
        write_json(DATA_DIR / file, build_trades_json(sido, recs, months))

    print(f"[완료] regions.json: 지역 {len(regions_json['regions'])}개, 월 {len(months)}개")
    for sido, file, recs in datasets:
        print(f"[완료] {file}: 거래 {len(recs)}건")


if __name__ == "__main__":
    main()
