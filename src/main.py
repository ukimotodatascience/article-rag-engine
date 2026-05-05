import sys
import logging

# 余計なログ出力を抑える
logging.basicConfig(level=logging.WARNING)

from retriever import HybridRetriever
from generator import LocalLLMGenerator
import config

def main():
    print("RAGシステムを初期化しています...")
    try:
        # コンポーネントの初期化
        retriever = HybridRetriever()
        generator = LocalLLMGenerator() # デフォルトモデルを使用
        
        print("\n✅ システムの準備が完了しました！(終了するには 'exit' または 'quit' と入力してください)")
        print("=" * 60)
        
        while True:
            try:
                # ユーザーからの入力
                query = input("\n👤 質問を入力してください: ").strip()
                
                if not query:
                    continue
                if query.lower() in ['exit', 'quit']:
                    print("終了します。お疲れ様でした！")
                    break
                    
                # 1. 検索フェーズ
                # retriever.search()内のプリント出力を抑制するため、必要な情報だけ表示
                results = retriever.search(query, limit=config.RETRIEVER_LIMIT)
                
                if not results:
                    print("関連するドキュメントが見つかりませんでした。")
                    continue
                    
                # 2. 生成フェーズ
                generator.generate(query, results)
                
            except KeyboardInterrupt:
                print("\n終了します。")
                break
            except Exception as e:
                print(f"\n❌ エラーが発生しました: {e}")
    except Exception as e:
         print(f"初期化中にエラーが発生しました: {e}")

if __name__ == "__main__":
    main()
