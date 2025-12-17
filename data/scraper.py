import time
import random
import re
import requests
import pandas as pd
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

BASE_URL = "https://www.shl.com"
CATALOG_URL = "https://www.shl.com/products/product-catalog/"
OUTPUT_FILE = "data/shl_assessments.csv"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"
}

TEST_TYPE_MAP = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}

def create_session():
    session = requests.Session()
    retry_strategy = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.headers.update(HEADERS)
    return session

session = create_session()

def fetch_catalog_page(start_index):
    params = {"type": 1, "start": start_index}
    response = session.get(CATALOG_URL, params=params, timeout=20)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")

def extract_catalog_assessments(soup):
    results = []
    header = soup.find(string=lambda t: t and 'Individual Test Solutions' in t)

    if header:
        container = header.find_parent("table") or header.find_next_sibling()
        search_scope = container if container else soup
    else:
        search_scope = soup
        
    for a in search_scope.select("a[href*='/view/']"):
        name = a.get_text(strip=True)
        href = a.get("href")
        if name and href:
            results.append({
                "assessment_name": name,
                "assessment_url": BASE_URL + href if href.startswith('/') else href
            })
    return results

def extract_test_type(soup):
    text = soup.get_text(" ", strip=True)
    match = re.search(r"Test Type:\s*([A-Z\s]+).*?Remote Testing", text, re.IGNORECASE | re.DOTALL)
    
    if not match:
        match = re.search(r"Test Type:\s*([A-Z\s]+)", text, re.IGNORECASE | re.DOTALL)

    if match:
        code_string = match.group(1).upper()
        codes = re.findall(r"[A-Z]", code_string)
        valid_types = sorted({TEST_TYPE_MAP[c] for c in codes if c in TEST_TYPE_MAP})
        return ", ".join(valid_types)
    return "N/A"

def extract_description(soup):
    label = soup.find(string=lambda t: t and t.strip() == "Description")
    if label:
        sibling = label.next_sibling
        while sibling and not sibling.get_text(strip=True):
            sibling = sibling.next_sibling
        if sibling:
            raw_text = sibling.get_text(" ", strip=True)
            return re.split(r"(Job levels|Languages|Assessment length)", raw_text, 1, re.IGNORECASE)[0].strip()

    for p in soup.find_all(["p", "div"]):
        text = p.get_text(" ", strip=True)
        if text.lower().startswith("description"):
            clean = re.sub(r"^description[:\s-]*", "", text, flags=re.IGNORECASE)
            return re.split(r"(Job levels|Languages|Assessment length)", clean, 1, re.IGNORECASE)[0].strip()
    return "No description available"

def extract_duration(soup):
    text = soup.get_text(" ", strip=True)
    match = re.search(r"Approximate Completion Time in minutes\s*=\s*(\d+)", text, re.IGNORECASE)
    if not match:
        match = re.search(r"(\d+)\s*min", text, re.IGNORECASE)
    return int(match.group(1)) if match else None

def enrich_assessment(row):
    try:
        res = session.get(row["assessment_url"], timeout=20)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")

        row["description"] = extract_description(soup)
        row["duration_minutes"] = extract_duration(soup)
        row["test_type"] = extract_test_type(soup)
    except Exception as e:
        row["description"] = f"Error: {str(e)}"
        row["duration_minutes"] = None
        row["test_type"] = "Unknown"
    return row

def main():
    print(f"Starting SHL Scraper...")
    all_links = []
    current_start = 0

    with tqdm(desc="Fetching Catalog", unit="page") as pbar:
        while True:
            try:
                soup = fetch_catalog_page(current_start)
                batch = extract_catalog_assessments(soup)
                if not batch: break
                
                all_links.extend(batch)
                current_start += 12
                pbar.update(1)
                time.sleep(random.uniform(2, 4))
            except Exception as e:
                print(f"\nStopped at index {current_start}: {e}")
                break

    catalog_df = pd.DataFrame(all_links).drop_duplicates("assessment_url").reset_index(drop=True)
    print(f"Found {len(catalog_df)} unique individual assessments.")

    final_data = []
    for _, row in tqdm(catalog_df.iterrows(), total=len(catalog_df), desc="Extracting Details", unit="item"):
        final_data.append(enrich_assessment(row.to_dict()))
        time.sleep(random.uniform(4, 7))

    df = pd.DataFrame(final_data)
    df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
    print(f"Task Complete! Data saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()