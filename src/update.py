import subprocess
import sys
import os
import time

def run_step(name, script_name):
    """
    指定されたスクリプトを実行し、結果を報告する
    """
    print(f"\n{'='*60}")
    print(f"🚀 ステップ: {name}")
    print(f"📜 実行ファイル: {script_name}")
    print(f"{'='*60}\n")
    
    script_path = os.path.join("src", script_name)
    
    if not os.path.exists(script_path):
        # src/ ディレクトリ内で実行されている場合を考慮
        if os.path.exists(script_name):
            script_path = script_name
        else:
            print(f"❌ エラー: {script_path} が見つかりません。")
            return False

    start_time = time.time()
    
    try:
        # sys.executable を使用して現在の Python 環境で実行
        process = subprocess.run(
            [sys.executable, script_path],
            check=True,
            text=True
        )
        
        duration = time.time() - start_time
        print(f"\n✅ {name} が完了しました。 (所要時間: {duration:.2f}秒)")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {name} の実行中にエラーが発生しました (終了コード: {e.returncode})")
        return False
    except Exception as e:
        print(f"\n❌ 予期しないエラーが発生しました: {e}")
        return False

def main():
    print("\n" + "✨" * 20)
    print("  RAG エンジン更新システム")
    print("  Notionデータの同期とインデックス更新を開始します")
    print("✨" * 20)
    
    total_start_time = time.time()
    
    # 1. Notionからデータを取得
    if not run_step("Notionデータ取得", "notion_fetch_database.py"):
        print("\n⚠️ Notionからのデータ取得に失敗したため、インデックス更新を中止します。")
        sys.exit(1)
        
    # 2. ベクトルインデックスを更新
    if not run_step("インデックス更新", "indexer.py"):
        print("\n⚠️ インデックスの更新に失敗しました。")
        sys.exit(1)
        
    total_duration = time.time() - total_start_time
    print(f"\n{'*'*60}")
    print(f"🎉 すべての更新プロセスが正常に完了しました！")
    print(f"📊 総所要時間: {total_duration:.2f}秒")
    print(f"{'*'*60}\n")

if __name__ == "__main__":
    main()
