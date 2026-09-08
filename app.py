import requests
import xml.etree.ElementTree as ET

# 🔑 새 발급받은 인증키
NEW_SERVICE_KEY = "92ac0b865a4117f6158886b90c8ad86908d7d67bead9c9166eb0fef68cddd2da"

def test_new_key_and_endpoints():
    print("=" * 80)
    print("🔑 새 인증키 기반 KRA API 엔드포인트 수신 검증 중...")
    print("=" * 80)

    # 파라미터 기본값 (2024년 9월 1일 서울 1경주 테스트)
    base_params = {
        "serviceKey": NEW_SERVICE_KEY,
        "pageNo": "1",
        "numOfRows": "10",
        "meet": "1",
        "rc_date": "20240901",
        "rc_no": "1"
    }

    # API299 및 현행 주요 엔드포인트 후보군
    targets = [
        ("API299 경주결과 (타입A)", "http://apis.data.go.kr/B551015/API299/raceResultList_1"),
        ("API299 경주결과 (타입B)", "http://apis.data.go.kr/B551015/API299/getRaceResultList"),
        ("API299 경주결과 (타입C)", "http://apis.data.go.kr/B551015/API299"),
        ("경주마명단 (racehorselist)", "http://apis.data.go.kr/B551015/racehorselist/getracehorselist"),
        ("주로/날씨 (API189_1)", "http://apis.data.go.kr/B551015/API189_1/trackConditionList")
    ]

    for label, url in targets:
        print(f"\n▶ [{label}] 호출 시도: {url}")
        try:
            # requests가 serviceKey를 이중 인코딩하지 않도록 안전하게 전달
            res = requests.get(url, params=base_params, timeout=5)
            print(f"  • HTTP Status: {res.status_code}")
            
            if res.status_code == 200:
                root = ET.fromstring(res.content)
                res_code = root.findtext(".//resultCode")
                res_msg = root.findtext(".//resultMsg")
                items = root.findall(".//item")
                
                print(f"  • API 결과: [{res_code}] {res_msg} (수신데이터 {len(items)}건)")
                if res_code in ["00", "0", "NORMAL_SERVICE", "OK"]:
                    print(f"  🎉 [성공!] 유효한 엔드포인트 및 인증키 정상 작동 확인!")
                    print(f"     👉 성공 URL: {url}")
            else:
                print(f"  🔴 에러 원문: {res.text[:150]}")
        except Exception as e:
            print(f"  💥 연결 실패: {e}")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    test_new_key_and_endpoints()
