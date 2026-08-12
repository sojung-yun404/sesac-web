import csv
from collections import defaultdict

CSV_PATH = "seoul-apt-latest.csv"
TARGET_GU = "성북구"


AREA84_MIN = 80.0
AREA84_MAX = 90.0


def load_all_rows():
    with open(CSV_PATH, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def load_rows():
    return [row for row in load_all_rows() if row["gu"] == TARGET_GU]


def is_area84(row):
    return AREA84_MIN <= float(row["area_m2"]) < AREA84_MAX


def print_by_deal_type(rows):
    # 매매는 price, 전세/월세는 deposit(보증금)을 대표 금액으로 사용한다.
    sums = defaultdict(int)
    rent_sums = defaultdict(int)
    counts = defaultdict(int)

    for row in rows:
        deal_type = row["deal_type"]
        amount = row["price"] if deal_type == "매매" else row["deposit"]
        if amount == "":
            continue
        sums[deal_type] += int(amount)
        counts[deal_type] += 1
        if deal_type == "월세" and row["monthly_rent"] != "":
            rent_sums[deal_type] += int(row["monthly_rent"])

    print(f"[{TARGET_GU} 거래유형별 평균가 & 건수]")
    for deal_type in sorted(counts, key=lambda k: -counts[k]):
        avg_manwon = sums[deal_type] / counts[deal_type]
        avg_eok = round(avg_manwon / 10000, 2)
        label = "평균 매매가" if deal_type == "매매" else "평균 보증금"
        line = f"- {deal_type}: {label} {avg_eok}억 원, {counts[deal_type]}건"
        if deal_type == "월세":
            avg_rent = round(rent_sums[deal_type] / counts[deal_type])
            line += f", 평균 월세 {avg_rent}만 원"
        print(line)


def print_top5_sale(rows):
    sales = [row for row in rows if row["deal_type"] == "매매"]

    # 아파트(complex)별 최고가 거래 1건만 남긴다.
    best_by_complex = {}
    for row in sales:
        price = int(row["price"])
        current = best_by_complex.get(row["complex"])
        if current is None or price > int(current["price"]):
            best_by_complex[row["complex"]] = row

    top5 = sorted(best_by_complex.values(), key=lambda r: int(r["price"]), reverse=True)[:5]

    print(f"\n[{TARGET_GU} 최고가 매매 아파트 Top 5 (아파트별 중복 제거)]")
    for i, row in enumerate(top5, start=1):
        price_eok = round(int(row["price"]) / 10000, 2)
        print(f"{i}. {row['complex']} ({row['dong']}) - {price_eok}억 원, "
              f"{row['area_m2']}㎡, {row['floor']}층, {row['contract_date']}")


def print_area84_target_gu(rows):
    sales = [row for row in rows if row["deal_type"] == "매매" and is_area84(row)]
    if not sales:
        print(f"\n[{TARGET_GU} 전용 84㎡대 매매]\n데이터가 없습니다.")
        return

    avg_eok = round(sum(int(r["price"]) for r in sales) / len(sales) / 10000, 2)
    print(f"\n[{TARGET_GU} 전용 84㎡대 매매 평균가]")
    print(f"- 평균 {avg_eok}억 원, {len(sales)}건")


def print_area84_by_gu(all_rows):
    sums = defaultdict(int)
    counts = defaultdict(int)

    for row in all_rows:
        if row["deal_type"] == "매매" and is_area84(row):
            sums[row["gu"]] += int(row["price"])
            counts[row["gu"]] += 1

    averages = {
        gu: sums[gu] / counts[gu] for gu in counts
    }
    ranked = sorted(averages.items(), key=lambda kv: kv[1])

    print(f"\n[서울 25개 구 전용 84㎡대 매매 평균가 최저 Top 5]")
    for i, (gu, avg_manwon) in enumerate(ranked[:5], start=1):
        avg_eok = round(avg_manwon / 10000, 2)
        print(f"{i}. {gu}: {avg_eok}억 원 ({counts[gu]}건)")

    print(f"\n[서울 25개 구 전용 84㎡대 매매 평균가 최고 Top 5]")
    for i, (gu, avg_manwon) in enumerate(ranked[-1:-6:-1], start=1):
        avg_eok = round(avg_manwon / 10000, 2)
        print(f"{i}. {gu}: {avg_eok}억 원 ({counts[gu]}건)")


def main():
    all_rows = load_all_rows()
    rows = [row for row in all_rows if row["gu"] == TARGET_GU]
    if not rows:
        print(f"{TARGET_GU} 데이터가 없습니다.")
        return

    print_by_deal_type(rows)
    print_top5_sale(rows)
    print_area84_target_gu(rows)
    print_area84_by_gu(all_rows)


if __name__ == "__main__":
    main()
