import apsw, sqlite_vec, jieba, json
from collections import Counter

DB_PATH = "/Users/observer/observerAI/real_rag.db"
CACHE_PATH = "/Users/observer/observerAI/df_cache.json"

def build_cache():
    print("[雷达] 开始扫描全库，重构词频映射...")
    con = apsw.Connection(DB_PATH)
    con.enableloadextension(True); sqlite_vec.load(con); con.enableloadextension(False)
    cur = con.cursor()

    rows = list(cur.execute("SELECT id, text FROM chunks"))
    total_chunks = len(rows)
    con.close()

    if total_chunks == 0:
        return

    df_counter = Counter()
    from collections import defaultdict
    inverted_index = defaultdict(list)
    for row in rows:
        cid, text = row[0], row[1]
        words = set(w for w in jieba.lcut(text) if len(w) >= 2)
        for w in words:
            df_counter[w] += 1
            if len(w) == 2:
                inverted_index[w].append(cid)

    df_dict = {word: count / total_chunks for word, count in df_counter.items()}
    df_dict["__total_chunks__"] = total_chunks
    df_dict["__inverted_index__"] = inverted_index
    
    with open(CACHE_PATH, 'w', encoding='utf-8') as f:
        json.dump(df_dict, f, ensure_ascii=False, indent=2)

    print(f"[雷达] 扫描完成！共分析 {total_chunks} 块语料，水印已更新。")

if __name__ == "__main__":
    build_cache()
