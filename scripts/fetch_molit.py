"""국토교통부 아파트 매매 실거래가 API(RTMSDataSvcAptTradeDev)로 서울/경기 데이터를 수집.

파이썬 표준 라이브러리만 사용한다 (requests 금지, build_data.py/analyze.py와 동일한 기조).

수집 결과는 원시 CSV로 저장한다:
  - data/raw/molit-seoul.csv
  - data/raw/molit-gyeonggi.csv
build_data.py 는 이 파일이 있으면(있는 지역만) API 경로를 우선 사용하고,
없으면 기존처럼 seoul-apt-latest.csv 를 읽는다. 자세한 내용은 build_data.py 상단 참고.

사용법:
  python3 scripts/fetch_molit.py --self-test   # XML 파서 단위 테스트 (키 불필요)
  python3 scripts/fetch_molit.py --dry-run     # 호출 계획만 출력 (키 불필요)
  python3 scripts/fetch_molit.py               # 실제 수집 (키 필요, MOLIT_API_KEY)

인증키:
  1) 환경변수 MOLIT_API_KEY
  2) 저장소 루트 .env 파일의 MOLIT_API_KEY=... 줄
  둘 다 없으면 발급 안내를 출력하고 종료한다. 키는 어떤 경우에도 로그/에러 메시지에
  절대 출력하지 않는다 (URL을 출력할 일이 있으면 항상 mask_service_key()를 거친다).
"""
import argparse
import csv
import os
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import NamedTuple

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
ENV_PATH = BASE_DIR / ".env"

ENDPOINT = "http://apis.data.go.kr/1613000/RTMSDataSvcAptTradeDev/getRTMSDataSvcAptTradeDev"

# 개발계정 일일 호출 한도(10,000건)를 지키기 위한 여유값. 필요하면 조정.
NUM_OF_ROWS = 1000
REQUEST_TIMEOUT = 10  # 초
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2.0  # 1차 실패 후 2초, 2차 4초, 3차 8초 대기
DELAY_RANGE = (0.1, 0.3)  # 호출 사이 딜레이(초) — 서버 부하/차단 방지

# 수집 대상 기간: 2025-07 ~ 2026-06 (12개월). 기존 서울 CSV와 동일한 범위로 맞춘다.
COLLECT_START = (2025, 7)
COLLECT_END = (2026, 6)


# ---------------------------------------------------------------------------
# 지역코드(LAWD_CD)
# ---------------------------------------------------------------------------

class RegionCode(NamedTuple):
    code: str    # 법정동코드 앞 5자리 (시군구 코드)
    sido: str    # "서울" | "경기"
    name: str    # 사람이 읽는 지역명. 경기 일반구는 "시명 + 구명" 형태로 명확히 구분한다.


# 서울 25개 자치구. 코드 출처: 행정표준코드관리시스템(법정동코드) 기준 시군구코드.
SEOUL_REGIONS = [
    RegionCode("11110", "서울", "종로구"),
    RegionCode("11140", "서울", "중구"),
    RegionCode("11170", "서울", "용산구"),
    RegionCode("11200", "서울", "성동구"),
    RegionCode("11215", "서울", "광진구"),
    RegionCode("11230", "서울", "동대문구"),
    RegionCode("11260", "서울", "중랑구"),
    RegionCode("11290", "서울", "성북구"),
    RegionCode("11305", "서울", "강북구"),
    RegionCode("11320", "서울", "도봉구"),
    RegionCode("11350", "서울", "노원구"),
    RegionCode("11380", "서울", "은평구"),
    RegionCode("11410", "서울", "서대문구"),
    RegionCode("11440", "서울", "마포구"),
    RegionCode("11470", "서울", "양천구"),
    RegionCode("11500", "서울", "강서구"),
    RegionCode("11530", "서울", "구로구"),
    RegionCode("11545", "서울", "금천구"),
    RegionCode("11560", "서울", "영등포구"),
    RegionCode("11590", "서울", "동작구"),
    RegionCode("11620", "서울", "관악구"),
    RegionCode("11650", "서울", "서초구"),
    RegionCode("11680", "서울", "강남구"),
    RegionCode("11710", "서울", "송파구"),
    RegionCode("11740", "서울", "강동구"),
]

# 경기도 31개 시/군. 단, 아래 8개 시는 일반구(자치권 없는 행정구)로 나뉘어 있고
# 부동산 실거래가 API의 LAWD_CD는 "시" 전체 코드가 아니라 "구" 단위 코드를 써야
# 실제 법정동을 커버한다 (시 전체를 아우르는 상위 코드는 별도로 존재하지 않거나,
# 있어도 이 API에서 구별 법정동 자료를 반환하지 않는다):
#   수원시(4) 성남시(3) 안양시(2) 부천시(3) 안산시(2) 고양시(3) 용인시(3) 화성시(4)
# 그래서 실제 LAWD_CD 개수는 31개보다 많은 47개다. 지역명은 "시명 구명"으로 붙여
# (예: "수원시 영통구") 다른 시의 동명 구("장안구" 등)와 헷갈리지 않게 한다.
#
# 코드 출처: 행정표준코드관리시스템 기준 시군구코드(2026-08 기준, 부천시 3구
# 재설치(2024-01-01)·화성시 4구 신설 반영). 지어낸 값이 아니라 실제 조회 결과이며,
# 특히 부천시/화성시 구코드는 비교적 최근에 바뀐 값이라 실행 전 최신 여부를
# code.go.kr 등에서 한 번 더 확인하는 것을 권장한다.
GYEONGGI_REGIONS = [
    RegionCode("41111", "경기", "수원시 장안구"),
    RegionCode("41113", "경기", "수원시 권선구"),
    RegionCode("41115", "경기", "수원시 팔달구"),
    RegionCode("41117", "경기", "수원시 영통구"),
    RegionCode("41131", "경기", "성남시 수정구"),
    RegionCode("41133", "경기", "성남시 중원구"),
    RegionCode("41135", "경기", "성남시 분당구"),
    RegionCode("41150", "경기", "의정부시"),
    RegionCode("41171", "경기", "안양시 만안구"),
    RegionCode("41173", "경기", "안양시 동안구"),
    RegionCode("41192", "경기", "부천시 원미구"),
    RegionCode("41194", "경기", "부천시 소사구"),
    RegionCode("41196", "경기", "부천시 오정구"),
    RegionCode("41210", "경기", "광명시"),
    RegionCode("41220", "경기", "평택시"),
    RegionCode("41250", "경기", "동두천시"),
    RegionCode("41271", "경기", "안산시 상록구"),
    RegionCode("41273", "경기", "안산시 단원구"),
    RegionCode("41281", "경기", "고양시 덕양구"),
    RegionCode("41285", "경기", "고양시 일산동구"),
    RegionCode("41287", "경기", "고양시 일산서구"),
    RegionCode("41290", "경기", "과천시"),
    RegionCode("41310", "경기", "구리시"),
    RegionCode("41360", "경기", "남양주시"),
    RegionCode("41370", "경기", "오산시"),
    RegionCode("41390", "경기", "시흥시"),
    RegionCode("41410", "경기", "군포시"),
    RegionCode("41430", "경기", "의왕시"),
    RegionCode("41450", "경기", "하남시"),
    RegionCode("41461", "경기", "용인시 처인구"),
    RegionCode("41463", "경기", "용인시 기흥구"),
    RegionCode("41465", "경기", "용인시 수지구"),
    RegionCode("41480", "경기", "파주시"),
    RegionCode("41500", "경기", "이천시"),
    RegionCode("41550", "경기", "안성시"),
    RegionCode("41570", "경기", "김포시"),
    RegionCode("41591", "경기", "화성시 만세구"),
    RegionCode("41593", "경기", "화성시 효행구"),
    RegionCode("41595", "경기", "화성시 병점구"),
    RegionCode("41597", "경기", "화성시 동탄구"),
    RegionCode("41610", "경기", "광주시"),
    RegionCode("41630", "경기", "양주시"),
    RegionCode("41650", "경기", "포천시"),
    RegionCode("41670", "경기", "여주시"),
    RegionCode("41800", "경기", "연천군"),
    RegionCode("41820", "경기", "가평군"),
    RegionCode("41830", "경기", "양평군"),
]

ALL_REGIONS = SEOUL_REGIONS + GYEONGGI_REGIONS

SEOUL_EXPECTED_COUNT = 25
GYEONGGI_SI_GUN_COUNT = 31  # 행정구역상 "시/군" 개수 (일반구 분리 전)


def generate_months(start=COLLECT_START, end=COLLECT_END):
    """(년, 월) 튜플 사이의 모든 "YYYY-MM" 문자열을 오래된 순으로 반환."""
    months = []
    y, m = start
    ey, em = end
    while (y, m) <= (ey, em):
        months.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months


MONTHS = generate_months()


# ---------------------------------------------------------------------------
# 인증키
# ---------------------------------------------------------------------------

class ApiKeyMissing(Exception):
    pass


def _read_env_file(path):
    """.env 파일을 최소한으로 파싱해 dict로 반환. KEY=VALUE 형태만 지원."""
    values = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        values[key] = value
    return values


def get_raw_api_key():
    """환경변수 -> .env 순으로 원본 인증키 문자열을 찾는다. 없으면 ApiKeyMissing."""
    key = os.environ.get("MOLIT_API_KEY")
    if key:
        return key.strip()

    env_values = _read_env_file(ENV_PATH)
    key = env_values.get("MOLIT_API_KEY")
    if key:
        return key.strip()

    raise ApiKeyMissing()


def print_api_key_guide():
    # 키 발급 경로 안내. 여기에는 실제 키 값이 절대 들어가지 않는다.
    print(
        "MOLIT_API_KEY 를 찾을 수 없습니다.\n"
        "\n"
        "발급 방법:\n"
        "  1) https://www.data.go.kr 회원가입 후 로그인\n"
        "  2) '국토교통부_아파트 매매 실거래자료' 검색 -> 활용신청 (개발계정, 보통 즉시 승인)\n"
        "  3) 마이페이지 > 데이터활용 > 개발계정 상세보기에서 '일반 인증키(Decoding)' 복사\n"
        "  4) 아래 둘 중 하나로 저장:\n"
        "       - 환경변수: export MOLIT_API_KEY='발급받은키'\n"
        "       - 저장소 루트 .env 파일에 한 줄 추가: MOLIT_API_KEY=발급받은키\n"
        "         (.env 는 이미 .gitignore에 등록되어 있어 커밋되지 않습니다)\n"
        "\n"
        "키가 준비되면 다시 실행하세요: python3 scripts/fetch_molit.py",
        file=sys.stderr,
    )


def build_service_key_param(raw_key):
    """서비스키를 쿼리스트링에 넣을 형태로 변환.

    공공데이터포털은 Encoding키/Decoding키 두 형태를 준다.
    - Decoding키(원문)를 받은 게 기본 전제 -> 우리가 직접 quote() 해서 인코딩한다.
    - 이미 인코딩된 키(%2B 등 퍼센트 인코딩이 포함된 형태)를 넣어도 이중 인코딩되지
      않도록, '%'가 포함돼 있으면 이미 인코딩된 것으로 간주하고 그대로 사용한다.
    """
    if "%" in raw_key:
        return raw_key  # 이미 인코딩된 키로 간주 (이중 인코딩 방지)
    return urllib.parse.quote(raw_key, safe="")


def mask_service_key(url):
    """로그/출력용: serviceKey 값을 *** 로 가린 URL을 반환."""
    return re.sub(r"(serviceKey=)[^&]*", r"\1***", url)


# ---------------------------------------------------------------------------
# URL 빌드 & HTTP 호출
# ---------------------------------------------------------------------------

def build_url(service_key_param, region_code, month, page_no):
    deal_ymd = month.replace("-", "")
    # serviceKey는 이미 인코딩을 마쳤으므로 urlencode()로 다시 인코딩하지 않는다
    # (이중 인코딩되면 인증에 실패한다). 나머지 파라미터만 urlencode 한다.
    other = urllib.parse.urlencode({
        "LAWD_CD": region_code,
        "DEAL_YMD": deal_ymd,
        "pageNo": page_no,
        "numOfRows": NUM_OF_ROWS,
    })
    return f"{ENDPOINT}?serviceKey={service_key_param}&{other}"


# 문서에는 성공 코드가 "00"으로 적혀 있으나 실제 운영 API는 "000"을 돌려준다.
SUCCESS_CODES = ("00", "000")


class MolitApiError(Exception):
    """API가 HTTP 200을 반환했지만 resultCode가 정상이 아닌 경우."""

    def __init__(self, result_code, result_msg):
        self.result_code = result_code
        self.result_msg = result_msg
        super().__init__(f"resultCode={result_code} resultMsg={result_msg}")


# 이 문자열들이 resultMsg에 보이면 재시도해도 소용없는 치명적 오류로 보고 즉시 중단한다
# (키 자체가 잘못됐거나 하루 호출 한도를 넘긴 경우 등).
FATAL_ERROR_HINTS = (
    "SERVICE_KEY", "서비스키", "등록되지 않은", "LIMITED_NUMBER", "요청건수",
    "일일", "제한횟수", "NODATA_ERROR" ,
)


def fetch_page(service_key_param, region_code, month, page_no):
    """한 페이지를 호출해 (items, total_count) 를 반환. 실패 시 예외 발생."""
    url = build_url(service_key_param, region_code, month, page_no)
    try:
        with urllib.request.urlopen(url, timeout=REQUEST_TIMEOUT) as resp:
            body = resp.read()
    except urllib.error.URLError as e:
        # 에러 메시지에도 키가 들어가지 않도록 마스킹된 URL만 언급한다.
        raise RuntimeError(f"네트워크 오류 ({mask_service_key(url)}): {e}") from e

    items, total_count, result_code, result_msg = parse_response(body)
    # 성공 코드는 문서상 "00" 이지만 실제 운영 API는 "000" 을 돌려준다. 둘 다 허용한다.
    if result_code not in SUCCESS_CODES:
        raise MolitApiError(result_code, result_msg)
    return items, total_count


def fetch_page_with_retry(service_key_param, region_code, month, page_no):
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fetch_page(service_key_param, region_code, month, page_no)
        except MolitApiError as e:
            if any(hint in (e.result_msg or "") for hint in FATAL_ERROR_HINTS):
                raise  # 재시도 무의미 -> 상위로 전파해서 전체 수집을 중단시킨다
            last_err = e
        except (RuntimeError, ET.ParseError) as e:
            last_err = e

        if attempt < MAX_RETRIES:
            wait = RETRY_BACKOFF_BASE ** attempt
            print(f"  [재시도 {attempt}/{MAX_RETRIES}] {last_err} -> {wait:.0f}초 후 재시도")
            time.sleep(wait)
    raise last_err


# ---------------------------------------------------------------------------
# XML 파싱
# ---------------------------------------------------------------------------

def parse_response(xml_bytes):
    """API 응답 XML -> (items: list[dict], total_count: int, result_code: str, result_msg: str).

    items의 각 원소는 {태그명: 텍스트} 형태의 얕은 dict (아이템 안에 하위 엘리먼트가
    없다는 이 API의 실제 응답 구조를 그대로 반영).
    """
    root = ET.fromstring(xml_bytes)

    header = root.find("./header")
    result_code = header.findtext("resultCode", default="").strip() if header is not None else ""
    result_msg = header.findtext("resultMsg", default="").strip() if header is not None else ""

    body = root.find("./body")
    total_count = 0
    items = []
    if body is not None:
        total_count = parse_int(body.findtext("totalCount", default="0"))
        for item_el in body.findall("./items/item"):
            item = {}
            for child in item_el:
                item[child.tag] = (child.text or "").strip()
            items.append(item)

    return items, total_count, result_code, result_msg


def parse_int(value, default=0):
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def parse_amount(value):
    """"138,000" -> 138000. 실패하면 None."""
    try:
        return int(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def item_to_row(item, sido, region_name):
    """API item(dict) -> 원시 CSV 한 행(dict), 또는 제외 대상이면 None.

    - 해제된 거래(cdealType == 'O')는 여기서 이미 제외한다 (DATA_CONTRACT: 해제된
      거래는 전처리 단계에서 제외). build_data.py 쪽에서 다시 걸러낼 필요 없음.
    """
    if item.get("cdealType", "").strip().upper() == "O":
        return None

    price = parse_amount(item.get("dealAmount"))
    if price is None:
        return None

    year = item.get("dealYear", "").strip()
    month = item.get("dealMonth", "").strip().zfill(2)
    day = item.get("dealDay", "").strip().zfill(2)

    complex_name = item.get("aptNm", "").strip()
    dong = item.get("umdNm", "").strip()
    if not complex_name or not dong or not year or not month:
        return None

    return {
        "sido": sido,
        "region": region_name,
        "dong": dong,
        "complex": complex_name,
        "contract_ym": f"{year}-{month}",
        "contract_date": f"{year}-{month}-{day}",
        "area_m2": item.get("excluUseAr", "").strip(),
        "floor": item.get("floor", "0").strip() or "0",
        "build_year": item.get("buildYear", "0").strip() or "0",
        "price": str(price),
    }


RAW_CSV_FIELDS = [
    "sido", "region", "dong", "complex", "contract_ym", "contract_date",
    "area_m2", "floor", "build_year", "price",
]


# ---------------------------------------------------------------------------
# 자체 테스트 (--self-test): 실제 API 응답 형태를 흉내낸 XML로 파서 검증
# ---------------------------------------------------------------------------

SAMPLE_XML = """<?xml version="1.0" encoding="UTF-8"?>
<response>
    <header>
        <resultCode>00</resultCode>
        <resultMsg>OK</resultMsg>
    </header>
    <body>
        <items>
            <item>
                <aptNm>래미안길음센터피스</aptNm>
                <umdNm>길음동</umdNm>
                <dealYear>2025</dealYear>
                <dealMonth>7</dealMonth>
                <dealDay>15</dealDay>
                <dealAmount>  138,000</dealAmount>
                <excluUseAr>84.89</excluUseAr>
                <floor>12</floor>
                <buildYear>2019</buildYear>
                <cdealType></cdealType>
                <sggCd>11290</sggCd>
            </item>
            <item>
                <aptNm>해제된아파트</aptNm>
                <umdNm>정릉동</umdNm>
                <dealYear>2025</dealYear>
                <dealMonth>7</dealMonth>
                <dealDay>3</dealDay>
                <dealAmount>50,000</dealAmount>
                <excluUseAr>59.9</excluUseAr>
                <floor>3</floor>
                <buildYear>2005</buildYear>
                <cdealType>O</cdealType>
                <sggCd>11290</sggCd>
            </item>
        </items>
        <numOfRows>1000</numOfRows>
        <pageNo>1</pageNo>
        <totalCount>2</totalCount>
    </body>
</response>
"""

SAMPLE_XML_ERROR = """<?xml version="1.0" encoding="UTF-8"?>
<response>
    <header>
        <resultCode>30</resultCode>
        <resultMsg>SERVICE_KEY_IS_NOT_REGISTERED_ERROR</resultMsg>
    </header>
</response>
"""


def run_self_test():
    failures = []

    def check(label, cond):
        status = "OK" if cond else "FAIL"
        print(f"  [{status}] {label}")
        if not cond:
            failures.append(label)

    print("[self-test] 정상 응답 파싱")
    items, total_count, result_code, result_msg = parse_response(SAMPLE_XML.encode("utf-8"))
    check("resultCode == '00'", result_code == "00")
    check("totalCount == 2", total_count == 2)
    check("items 2건 파싱", len(items) == 2)
    check("dealAmount 콤마/공백 포함 필드 접근 가능", items[0]["dealAmount"].strip() == "138,000")

    # 실제 운영 API는 resultCode를 "000"으로 준다. 문서상 "00"만 성공으로 처리하면
    # 모든 호출이 실패하므로, 두 형태 모두 성공으로 인정되는지 반드시 확인한다.
    print("[self-test] 성공 코드 '00'/'000' 양쪽 인정")
    check("'00' 은 성공", "00" in SUCCESS_CODES)
    check("'000' 은 성공", "000" in SUCCESS_CODES)
    check("빈 문자열은 실패로 취급", "" not in SUCCESS_CODES)
    check("실제 오류코드는 실패로 취급", "30" not in SUCCESS_CODES)

    print("[self-test] item_to_row 변환 + 해제 거래 필터링")
    row0 = item_to_row(items[0], "서울", "성북구")
    row1 = item_to_row(items[1], "서울", "성북구")
    check("정상 거래는 dict 반환", row0 is not None)
    check("해제 거래(cdealType=O)는 None", row1 is None)
    if row0 is not None:
        check("price 콤마 제거 후 int 파싱 (138000)", row0["price"] == "138000")
        check("contract_ym 조합 (2025-07)", row0["contract_ym"] == "2025-07")
        check("region_name 그대로 반영", row0["region"] == "성북구")
        check("build_year 보존 (2019)", row0["build_year"] == "2019")

    print("[self-test] 에러 응답 파싱")
    _, _, err_code, err_msg = parse_response(SAMPLE_XML_ERROR.encode("utf-8"))
    check("resultCode == '30' (정상 아님)", err_code == "30")
    check("resultMsg에 SERVICE_KEY 포함", "SERVICE_KEY" in err_msg)

    print("[self-test] 키 인코딩 방어 로직")
    check("Decoding키는 quote() 적용됨", build_service_key_param("a+b/c") == urllib.parse.quote("a+b/c", safe=""))
    check("이미 인코딩된 키는 그대로 통과", build_service_key_param("a%2Bb") == "a%2Bb")

    print("[self-test] serviceKey 마스킹")
    sample_url = build_url("SECRET123", "11290", "2025-07", 1)
    masked = mask_service_key(sample_url)
    check("마스킹된 URL에 원본 키 없음", "SECRET123" not in masked)
    check("마스킹된 URL에 *** 포함", "serviceKey=***" in masked)

    print()
    if failures:
        print(f"[self-test] 실패 {len(failures)}건: {failures}")
        return False
    print("[self-test] 전체 통과")
    return True


# ---------------------------------------------------------------------------
# dry-run: 실제 호출 없이 계획만 출력
# ---------------------------------------------------------------------------

def run_dry_run():
    seoul_n = len(SEOUL_REGIONS)
    gg_n = len(GYEONGGI_REGIONS)

    print("[dry-run] 지역코드 검증")
    print(f"  서울: {seoul_n}개 (기대값 {SEOUL_EXPECTED_COUNT}) "
          f"{'OK' if seoul_n == SEOUL_EXPECTED_COUNT else 'MISMATCH'}")
    print(f"  경기: {gg_n}개 LAWD_CD (행정구역상 시/군 {GYEONGGI_SI_GUN_COUNT}개 중 "
          f"8개 시가 일반구로 분리되어 있어 코드 수는 더 많다)")
    print()

    total_pairs = len(ALL_REGIONS) * len(MONTHS)
    print(f"[dry-run] 수집 대상: 지역 {len(ALL_REGIONS)}개 x 월 {len(MONTHS)}개 "
          f"({MONTHS[0]} ~ {MONTHS[-1]}) = {total_pairs}건 (지역,월) 조합")
    print(f"  최소 예상 호출 수(페이지당 1회 가정): {total_pairs}회")
    print(f"  (지역/월별 거래가 {NUM_OF_ROWS}건을 넘으면 추가 페이지 호출이 붙는다)")

    avg_delay = sum(DELAY_RANGE) / 2
    est_seconds = total_pairs * avg_delay
    print(f"  호출 간 딜레이 평균 {avg_delay:.2f}초 기준 예상 소요시간: "
          f"약 {est_seconds / 60:.1f}분 (페이지 추가 호출 제외, 최소 추정치)")
    print()

    masked_key = "***"
    print("[dry-run] 호출 URL 예시 (앞 3개 + 마지막 1개, 키는 마스킹)")
    sample_indices = list(range(min(3, total_pairs)))
    if total_pairs > 3:
        sample_indices.append(total_pairs - 1)
    pairs = [(r, m) for r in ALL_REGIONS for m in MONTHS]
    for idx in sample_indices:
        region, month = pairs[idx]
        url = build_url(masked_key, region.code, month, 1)
        print(f"  [{idx + 1}/{total_pairs}] {region.sido} {region.name} {month}")
        print(f"    {url}")

    print()
    print(f"[dry-run] 출력 파일: {RAW_DIR / 'molit-seoul.csv'}, {RAW_DIR / 'molit-gyeonggi.csv'}")
    print("[dry-run] 키가 없어도 위 계획까지만 확인 가능합니다. "
          "실제 수집은 MOLIT_API_KEY 설정 후 옵션 없이 실행하세요.")


# ---------------------------------------------------------------------------
# 실제 수집
# ---------------------------------------------------------------------------

def load_done_pairs(csv_path):
    """이미 저장된 (지역코드, 월) 조합 집합을 raw CSV에서 복원한다 (resume용).

    거래가 0건이었던 (지역,월)은 CSV에 행이 남지 않으므로 이 방식으로는 감지되지
    않는다 -> 그런 조합은 재수집되지만, 0건 재확인은 비용이 작고 최신 데이터 오류를
    막아주므로 오히려 안전한 쪽으로 둔다.
    """
    done = set()
    if not csv_path.exists():
        return done
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            done.add((row["region"], row["contract_ym"]))
    return done


def run_collect():
    try:
        raw_key = get_raw_api_key()
    except ApiKeyMissing:
        print_api_key_guide()
        sys.exit(1)

    service_key_param = build_service_key_param(raw_key)

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    seoul_csv = RAW_DIR / "molit-seoul.csv"
    gyeonggi_csv = RAW_DIR / "molit-gyeonggi.csv"

    # resume: region "이름" 기준으로 이미 받은 (지역,월) 판별 (서울/경기 파일 합쳐서 확인)
    done_pairs = set()
    for path in (seoul_csv, gyeonggi_csv):
        with_names = load_done_pairs(path)
        done_pairs |= with_names

    def open_writer(path):
        is_new = not path.exists() or path.stat().st_size == 0
        f = open(path, "a", encoding="utf-8", newline="")
        writer = csv.DictWriter(f, fieldnames=RAW_CSV_FIELDS)
        if is_new:
            writer.writeheader()
            f.flush()
        return f, writer

    seoul_f, seoul_writer = open_writer(seoul_csv)
    gg_f, gg_writer = open_writer(gyeonggi_csv)
    writers = {"서울": (seoul_f, seoul_writer), "경기": (gg_f, gg_writer)}

    total_pairs = len(ALL_REGIONS) * len(MONTHS)
    pair_no = 0
    grand_total_saved = 0

    try:
        for region in ALL_REGIONS:
            for month in MONTHS:
                pair_no += 1
                if (region.name, month) in done_pairs:
                    print(f"[{pair_no}/{total_pairs}] {region.sido} {region.name} {month} "
                          f"-> 이미 수집됨, 건너뜀")
                    continue

                print(f"[{pair_no}/{total_pairs}] {region.sido} {region.name} {month} 수집 중...")
                page_no = 1
                collected = 0
                saved_rows = []
                while True:
                    items, total_count = fetch_page_with_retry(
                        service_key_param, region.code, month, page_no)
                    for item in items:
                        row = item_to_row(item, region.sido, region.name)
                        if row is not None:
                            saved_rows.append(row)
                    collected += len(items)
                    time.sleep(random.uniform(*DELAY_RANGE))

                    if collected >= total_count or not items:
                        break
                    page_no += 1

                f, writer = writers[region.sido]
                for row in saved_rows:
                    writer.writerow(row)
                f.flush()
                grand_total_saved += len(saved_rows)

                print(f"  -> {len(saved_rows)}건 저장 (해제 거래 제외), "
                      f"누적 저장 {grand_total_saved}건")
    finally:
        seoul_f.close()
        gg_f.close()

    print(f"\n[완료] 총 {grand_total_saved}건 저장")
    print(f"  {seoul_csv}")
    print(f"  {gyeonggi_csv}")
    print("다음: python3 scripts/build_data.py 로 regions.json / trades-*.json 생성")


# ---------------------------------------------------------------------------
# 진입점
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                         help="실제 호출 없이 계획만 출력 (키 불필요)")
    parser.add_argument("--self-test", action="store_true",
                         help="XML 파서 단위 테스트 (키 불필요)")
    args = parser.parse_args()

    if args.self_test:
        ok = run_self_test()
        sys.exit(0 if ok else 1)

    if args.dry_run:
        run_dry_run()
        return

    run_collect()


if __name__ == "__main__":
    main()
