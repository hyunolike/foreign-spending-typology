"""report.html -> 제출용 PDF (A4).

한글은 시스템 나눔폰트를 사용한다. Ubuntu: apt-get install -y fonts-nanum
렌더러는 Chromium(Playwright). 차트 이미지는 outputs/figures 를 참조하므로
`src/run_all.py`와 `src/run_external.py`를 먼저 실행해야 한다.
"""
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT_DIR = ROOT / "outputs" / "report"
OUT = OUT_DIR / "외국인소비_잠재관광상권_아이디어요약서.pdf"

# 이 환경의 Playwright 번들 버전과 브라우저 버전이 달라 실행 파일을 직접 지정한다.
CHROME_CANDIDATES = [
    "/opt/pw-browsers/chromium-1194/chrome-linux/chrome",
    "/usr/bin/chromium",
    "/usr/bin/google-chrome",
]

# 공모전 공식 양식의 푸터: 좌측 "AI금융빅데이터플랫폼", 우측 쪽번호.
# 주최 측 로고는 저장소에 두지 않고 data/ 아래에서 읽어 data URI로 끼워 넣는다.
LOGO = ROOT / "data" / "template_assets" / "bc_logo.png"


def _footer() -> str:
    mark = ""
    if LOGO.exists():
        import base64

        b64 = base64.b64encode(LOGO.read_bytes()).decode()
        mark = (f'<img src="data:image/png;base64,{b64}" '
                'style="height:9px;vertical-align:-1px;margin-right:5px">')
    return (
        '<div style="width:100%;font-size:7.5pt;color:#8a8880;'
        'font-family:NanumGothic,sans-serif;padding:0 15mm;">'
        f'<span style="float:left">{mark}AI금융빅데이터플랫폼</span>'
        '<span style="float:right"><span class="pageNumber"></span>'
        ' / <span class="totalPages"></span></span></div>'
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    exe = next((c for c in CHROME_CANDIDATES if Path(c).exists()), None)

    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=exe, args=["--no-sandbox"])
        page = browser.new_page()
        page.goto((HERE / "report.html").as_uri(), wait_until="networkidle")
        page.pdf(
            path=str(OUT), format="A4", print_background=True,
            margin={"top": "16mm", "bottom": "14mm", "left": "15mm", "right": "15mm"},
            display_header_footer=True,
            header_template="<div></div>", footer_template=_footer(),
        )
        browser.close()
    print(f"{OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
