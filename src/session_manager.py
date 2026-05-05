import os
import json
import uuid
from datetime import datetime
from pathlib import Path
import config

class SessionManager:
    def __init__(self, sessions_dir: str = config.SESSIONS_DIR):
        self.sessions_dir = Path(sessions_dir)
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def create_session(self, name: str = "New Chat") -> str:
        """新しいセッションを作成し、IDを返します"""
        session_id = str(uuid.uuid4())
        self.save_session(session_id, [], name)
        return session_id

    def save_session(self, session_id: str, messages: list, name: str = None, last_retrieved: list = None):
        """セッションをファイルに保存します"""
        file_path = self.sessions_dir / f"{session_id}.json"
        
        # 既存のデータを読み込んで名前を保持する
        existing_data = {}
        if file_path.exists():
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
            except:
                pass

        session_name = name or existing_data.get("name", "New Chat")
        
        # 最初のユーザーメッセージがあれば、それを名前にする（名前がデフォルトの場合）
        if session_name == "New Chat" and messages:
            for msg in messages:
                if msg["role"] == "user":
                    session_name = msg["content"][:30] + ("..." if len(msg["content"]) > 30 else "")
                    break

        data = {
            "id": session_id,
            "name": session_name,
            "messages": messages,
            "last_retrieved": last_retrieved or [],
            "updated_at": datetime.now().isoformat()
        }
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_session(self, session_id: str) -> dict:
        """セッションをファイルから読み込みます"""
        file_path = self.sessions_dir / f"{session_id}.json"
        if not file_path.exists():
            return None
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading session {session_id}: {e}")
            return None

    def list_sessions(self) -> list:
        """保存されている全セッションの一覧を取得します"""
        sessions = []
        for file_path in self.sessions_dir.glob("*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    sessions.append({
                        "id": data["id"],
                        "name": data["name"],
                        "updated_at": data["updated_at"]
                    })
            except:
                continue
        
        # 更新日時で降順ソート
        return sorted(sessions, key=lambda x: x["updated_at"], reverse=True)

    def delete_session(self, session_id: str):
        """セッションを削除します"""
        file_path = self.sessions_dir / f"{session_id}.json"
        if file_path.exists():
            file_path.unlink()
