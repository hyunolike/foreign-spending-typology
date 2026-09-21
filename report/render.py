"""제출용 PDF 2종을 만든다 (A4).

- 아이디어 요약서: 공모전 공식 양식(2. 아이디어 요약서.doc)의 틀을 따른다
- 분석 결과물: 분석 과정·검증 절차·차트 전체를 담은 별첨 문서

한글은 시스템 나눔폰트를 쓴다. Ubuntu: apt-get install -y fonts-nanum
차트 이미지는 outputs/figures 를 참조하므로 `src/run_all.py`와
`src/run_external.py`를 먼저 실행해야 한다.
"""
from __future__ import annotations

import base64
from pathlib import Path

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT_DIR = ROOT / "outputs" / "report"

# 이 환경의 Playwright 번들 버전과 브라우저 버전이 달라 실행 파일을 직접 지정한다.
CHROME_CANDIDATES = [
    "/opt/pw-browsers/chromium-1194/chrome-linux/chrome",
    "/usr/bin/chromium",
    "/usr/bin/google-chrome",
]

# 공식 양식 푸터에 들어가는 주최 측 로고. 저장소에 두지 않고 data/ 아래에서 읽는다.
LOGO = ROOT / "data" / "template_assets" / "bc_logo.png"

DOCS = [
    ("summary.html", "01_아이디어요약서_외국인소비_잠재관광상권_장현호.pdf"),
    ("report.html", "02_분석결과물_외국인소비_잠재관광상권_장현호.pdf"),
]

MARGIN = {"top": "16mm", "bottom": "14mm", "left": "15mm", "right": "15mm"}


def _footer() -> str:
    mark = ""
    if LOGO.exists():
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
    footer = _footer()

    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=exe, args=["--no-sandbox"])
        page = browser.new_page()
        for src, name in DOCS:
            out = OUT_DIR / name
            page.goto((HERE / src).as_uri(), wait_until="networkidle")
            page.pdf(
                path=str(out), format="A4", print_background=True, margin=MARGIN,
                display_header_footer=True,
                header_template="<div></div>", footer_template=footer,
            )
            print(f"{out.name}  ({out.stat().st_size / 1024:.0f} KB)")
        browser.close()


if __name__ == "__main__":
    main()
