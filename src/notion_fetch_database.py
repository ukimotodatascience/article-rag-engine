import os
import requests
import re
from datetime import datetime
from dotenv import load_dotenv

# .envファイルから環境変数を読み込む
load_dotenv()

NOTION_TOKEN = os.environ["NOTION_TOKEN"]
DATABASE_ID = os.environ["NOTION_DATABASE_ID"]

HEADERS = {
    "Authorization": f"Bearer {NOTION_TOKEN}",
    "Notion-Version": "2022-06-28",
    "Content-Type": "application/json"
}

def query_database(database_id: str) -> list[dict]:
    """データベースを検索し、中に入っている全ページの情報を取得する"""
    url = f"https://api.notion.com/v1/databases/{database_id}/query"
    results = []
    
    # 一度に最大100件ずつ取得
    payload = {
        "page_size": 100,
        "filter": {
            "or": [
                {
                    "property": "媒体",
                    "formula": {
                        "string": {
                            "equals": "Qiita"
                        }
                    }
                },
                {
                    "property": "媒体",
                    "formula": {
                        "string": {
                            "equals": "Zenn"
                        }
                    }
                }
            ]
        }
    }
    
    while True:
        res = requests.post(url, headers=HEADERS, json=payload)
        if res.status_code != 200:
            print(f"API Error ({res.status_code}): {res.text}")
        res.raise_for_status()
        data = res.json()
        
        results.extend(data["results"])
        
        # さらに続きがある場合は cursor を更新して再度リクエスト
        if not data.get("has_more"):
            break
            
        payload["start_cursor"] = data["next_cursor"]
        
    return results

def get_block_children(block_id: str) -> list[dict]:
    """指定したブロック（ページ）の子ブロック（テキストなど）を取得する"""
    url = f"https://api.notion.com/v1/blocks/{block_id}/children"
    results = []
    params = {"page_size": 100}
    
    while True:
        res = requests.get(url, headers=HEADERS, params=params)
        res.raise_for_status()
        data = res.json()
        
        results.extend(data["results"])
        
        if not data.get("has_more"):
            break
            
        params["start_cursor"] = data["next_cursor"]
        
    return results

def extract_plain_text_from_block(block: dict) -> str:
    """ブロック情報からプレーンテキストを抽出する"""
    block_type = block["type"]
    content = block.get(block_type, {})
    rich_text = content.get("rich_text", [])
    if isinstance(rich_text, list):
        return "".join(t["plain_text"] for t in rich_text)
    return ""

def get_page_title(page: dict) -> str:
    """ページのプロパティからタイトル（記事名）を抽出する"""
    properties = page.get("properties", {})
    # title タイプのプロパティを探す
    for prop_name, prop_data in properties.items():
        if prop_data["type"] == "title":
            title_array = prop_data.get("title", [])
            if title_array:
                return "".join([t["plain_text"] for t in title_array])
    return "Untitled"

def get_page_url(page: dict) -> str:
    """ページのプロパティからURLを抽出する"""
    properties = page.get("properties", {})
    # URL タイプのプロパティを探す (名前が "URL" または "url" であることを想定)
    for name in ["URL", "url"]:
        url_prop = properties.get(name, {})
        if url_prop.get("type") == "url":
            return url_prop.get("url", "") or ""
    
    # URLプロパティが見つからない場合、デバッグ用にプロパティ名を出力（最初の1回のみなど検討）
    return ""

def sanitize_filename(name: str) -> str:
    """Windowsのファイル名に使えない記号をアンダースコアに置換する"""
    return re.sub(r'[\\/*?:"<>|]', "_", name)

if __name__ == "__main__":
    if not DATABASE_ID:
        print("Error: NOTION_DATABASE_ID is not set in .env")
        exit(1)
        
    pages = query_database(DATABASE_ID)

    # 保存用の data フォルダを作成
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)

    for page in pages:
        page_id = page["id"]
        title = get_page_title(page)
        url = get_page_url(page)
        safe_title = sanitize_filename(title)
        
        output_filename = os.path.join(output_dir, f"{safe_title}.md")
        
        # 差分チェック：ローカルファイルの更新日時とNotionの最終更新日時を比較
        if os.path.exists(output_filename):
            local_mtime = os.path.getmtime(output_filename)
            notion_time_str = page.get("last_edited_time", "").replace("Z", "+00:00")
            if notion_time_str:
                notion_mtime = datetime.fromisoformat(notion_time_str).timestamp()
                
                # Notionの更新がローカルファイルの更新より古ければスキップ
                if notion_mtime <= local_mtime:
                    print(f"スキップ: {title}")
                    continue

        try:
            blocks = get_block_children(page_id)
            
            with open(output_filename, "w", encoding="utf-8") as f:
                # メタデータとしてURLのみを保存（Page IDは削除）
                if url:
                    f.write(f"<!-- URL: {url} -->\n")
                f.write(f"# {title}\n\n")
                
                for block in blocks:
                    text = extract_plain_text_from_block(block)
                    if text:
                        f.write(text + "\n\n")
            
            # 出力はタイトルとURLのみ
            print(f"{title}: {url}")
        except Exception as e:
            print(f"エラー: {title} ({e})")
