import json
import re
from typing import Literal
import requests


def scrape_core_portfolio(
    portf_name: Literal["equity100", "core-balanced", "core-growth", "core-defensive"],
):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:137.0) Gecko/20100101 Firefox/137.0",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Referer": "https://www.syfe.com/core",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "same-origin",
        "Priority": "u=0, i",
        "Pragma": "no-cache",
        "Cache-Control": "no-cache",
    }

    response = requests.get(
        f"https://www.syfe.com/core/{portf_name}",
        # cookies=cookies,
        headers=headers,
    )

    return json.loads(
        re.search(r"portfolioDataStringified\s\=\s\`(.+)\`", response.text)
        .group(1)
        .replace("&quot;", '"')
    )
