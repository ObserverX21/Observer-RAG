import apsw
import sqlite_vec
import json
import os

# 加载极速词频缓存
DF_CACHE_PATH = "/Users/observer/observerAI/df_cache.json"
DF_CACHE = {}
if os.path.exists(DF_CACHE_PATH):
    try:
        with open(DF_CACHE_PATH, 'r', encoding='utf-8') as f:
            DF_CACHE = json.load(f)
    except Exception as e:
        print(f"[!] 词频缓存读取失败(可能后台正在重写): {e}")
        DF_CACHE = {}
import struct
import jieba
from sentence_transformers import SentenceTransformer, CrossEncoder
from FlagEmbedding import FlagReranker

# ============ 可调参数区 ============
DB_PATH   = "/Users/observer/observerAI/real_rag.db"
BGE_PATH  = "/Users/observer/observerAI/models/bge-m3"
JINA_PATH = "/Users/observer/observerAI/models/jina-reranker"

W_FTS      = 1.0
W_VEC     = 1.0
RRF_K     = 60
RECALL_N  = 15
TOP_N     = 7
GUARANTEE_VEC = 3    # 向量头部强制保底进候选
DF_THRESHOLD  = 0.30  # 小库(285块)泛洪阈值：doc_freq>30% 即剔除
USE_RERANK    = False  # 定案:reranker在此语料失效,用裸向量
RERANK_MODEL  = "bge"   # 用bge-reranker
TRUNCATE_N    = 200     # 送rerank前截断字数，None=不截断
BGE_RERANK_PATH = "/Users/observer/observerAI/models/bge-reranker"
# ====================================

# 通用停用词（语言层面的，不含任何领域/项目专有词）
STOPWORDS = set([
    "怎么", "他的", "我的", "为什么", "这个", "那个", "什么", "就是",
    "然后", "可以", "我们", "你们", "他们", "自己", "还是", "一个",
    "现在", "因为", "所以", "但是", "如果", "这样", "一下", "已经",
    "没有", "知道", "觉得", "时候", "东西", "事情", "问题",
    # 结构性角色标签，非语义词，每块必带，必须硬踢
    "观测者", "大鲸鱼",
])

def rrf_merge(list_fts, list_vec, k=RRF_K, w_fts=W_FTS, w_vec=W_VEC):
    scores = {}
    for rank, cid in enumerate(list_fts):
        scores[cid] = scores.get(cid, 0) + w_fts * (1.0 / (k + rank + 1))
    for rank, cid in enumerate(list_vec):
        scores[cid] = scores.get(cid, 0) + w_vec * (1.0 / (k + rank + 1))
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def get_total_chunks(cur):
    return list(cur.execute("SELECT count(*) FROM chunks"))[0][0]

def doc_freq(cur, word, total):
    """直接读取内存 JSON，纳秒级极速降噪"""
    return DF_CACHE.get(word, 0.0)

def build_fts_query(cur, query, total):
    words = jieba.lcut(query)
    candidates = [w for w in words if len(w) >= 2 and w not in STOPWORDS and w.strip()]
    kept = []
    dropped = []
    for w in candidates:
        df = doc_freq(cur, w, total)
        if df >= DF_THRESHOLD:
            dropped.append((w, round(df, 2)))
        else:
            kept.append(w)
    print(f"[DF降噪] 保留: {kept} | 剔除(高频泛洪): {dropped}")
    if not kept:
        return None
    return " OR ".join([f"\"{w}\"" for w in kept])

def retrieve(query, model=None, rerank="__auto__"):

    import re
    topic_filter = None
    match = re.search(r'\[主题:\s*([^\]]+)\]', query)
    if match:
        topic_filter = match.group(1).strip()
        query = query.replace(match.group(0), "").strip()
        print(f"[主题过滤] 已锁定主题: {topic_filter}")
    """静默检索：返回 [(cid, text), ...]，供 ask.py 复用。逻辑与 search() 一致。"""
    if model is None:
        model = SentenceTransformer(BGE_PATH, device="mps")
    if rerank == "__auto__":
        if USE_RERANK:
            if RERANK_MODEL == "jina":
                rerank = FlagReranker(JINA_PATH, use_fp16=True, device="mps", trust_remote_code=True)
            elif RERANK_MODEL == "bge":
                rerank = CrossEncoder(BGE_RERANK_PATH, device="mps")
            else:
                raise ValueError(f"未知 RERANK_MODEL: {RERANK_MODEL}")
        else:
            rerank = None

    con = apsw.Connection(DB_PATH)
    con.enableloadextension(True)
    sqlite_vec.load(con)
    con.enableloadextension(False)
    cur = con.cursor()

    total = get_total_chunks(cur)
    fts_q = build_fts_query(cur, query, total)
    fts_ids = []
    if fts_q:
        fts_rows = list(cur.execute(
            "SELECT rowid FROM fts_chunks WHERE fts_chunks MATCH ? ORDER BY rank LIMIT ?",
            (fts_q, RECALL_N)))
        fts_ids = [r[0] for r in fts_rows]
    if not fts_ids and fts_q:
        kw = [w for w in fts_q.split(" OR ") if w.strip()]
        seen = []
        inv_idx = DF_CACHE.get("__inverted_index__", {})
        for w in kw:
            if len(w) == 2 and w in inv_idx:
                for cid in inv_idx[w]:
                    if cid not in seen:
                        seen.append(cid)
                    if len(seen) >= RECALL_N:
                        break
            elif len(w) > 2:
                rows = list(cur.execute("SELECT id FROM chunks WHERE text LIKE ? LIMIT ?", (f"%{w}%", RECALL_N)))
                for r in rows:
                    if r[0] not in seen:
                        seen.append(r[0])
            if len(seen) >= RECALL_N:
                break
        fts_ids = seen[:RECALL_N]

    qvec = model.encode([query], normalize_embeddings=True)[0]
    qblob = struct.pack("1024f", *qvec.tolist())
    vec_rows = list(cur.execute(
        "SELECT chunk_id, distance FROM vec_chunks WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
        (qblob, RECALL_N)))
    vec_ids = [r[0] for r in vec_rows]

    merged = rrf_merge(fts_ids, vec_ids)
    rrf_top = [cid for cid, _ in merged]
    guaranteed = vec_ids[:GUARANTEE_VEC]
    final_candidates = []
    for cid in guaranteed + rrf_top:
        if cid not in final_candidates:
            final_candidates.append(cid)
    top15 = final_candidates[:RECALL_N]

    # === 强制保底 (Numpy 内存计算版) ===
    if topic_filter:
        import numpy as np
        sub_ids = [r[0] for r in cur.execute("SELECT id FROM chunks WHERE source LIKE ?", (f"%{topic_filter}%",))]
        if sub_ids:
            placeholders_sub = ",".join("?" * len(sub_ids))
            # 把该标签下的所有向量拉进内存
            vec_rows = list(cur.execute(f"SELECT chunk_id, embedding FROM vec_chunks WHERE chunk_id IN ({placeholders_sub})", sub_ids))
            if vec_rows:
                valid_ids = [row[0] for row in vec_rows]
                # 转换为 numpy 矩阵 (float32)
                matrix = np.array([np.frombuffer(row[1], dtype=np.float32) for row in vec_rows])
                # 矩阵乘法（点积即余弦相似度，因为入库时已经归一化）
                scores = np.dot(matrix, qvec)
                # 找到分数最高的那个
                anchor_id = valid_ids[np.argmax(scores)]
                
                print(f"[保底] 已强制锁定主题内最优块 id={anchor_id}")
                if anchor_id in top15:
                    top15.remove(anchor_id)
                top15.insert(0, anchor_id)
                top15 = top15[:RECALL_N]
                print(f"[候选] Top15 名单: {top15}")

    if not top15:
        con.close()
        return []

    placeholders = ",".join("?" * len(top15))
    text_map = {}
    for cid, text, source in cur.execute(f"SELECT id, text, source FROM chunks WHERE id IN ({placeholders})", top15):
        text_map[cid] = f"[来源: {source}] {text}"
    con.close()

    cand_ids = [cid for cid in top15 if cid in text_map]
    if USE_RERANK:
        pairs = [[query, (text_map[cid][:TRUNCATE_N] if TRUNCATE_N else text_map[cid])] for cid in cand_ids]
        if RERANK_MODEL == "bge":
            scores = rerank.predict(pairs)
        else:
            scores = rerank.compute_score(pairs)
        ranked = sorted(zip(cand_ids, scores), key=lambda x: x[1], reverse=True)
    else:
        ranked = [(cid, 0.0) for cid in cand_ids]

    return [(cid, text_map[cid]) for cid, _ in ranked[:TOP_N]]


def search(query):
    print(f"\n{'='*50}\nQuery: {query}\n{'='*50}")

    model  = SentenceTransformer(BGE_PATH, device="mps")
    if USE_RERANK:
        if RERANK_MODEL == "jina":
            rerank = FlagReranker(JINA_PATH, use_fp16=True, device="mps", trust_remote_code=True)
        elif RERANK_MODEL == "bge":
            rerank = CrossEncoder(BGE_RERANK_PATH, device="mps")
        else:
            raise ValueError(f"未知 RERANK_MODEL: {RERANK_MODEL}")
    else:
        rerank = None

    con = apsw.Connection(DB_PATH)
    con.enableloadextension(True)
    sqlite_vec.load(con)
    con.enableloadextension(False)
    cur = con.cursor()

    total = get_total_chunks(cur)
    print(f"[库] 总块数: {total}")

    # --- 路一：FTS5（动态 DF 降噪）---
    fts_q = build_fts_query(cur, query, total)
    print(f"[FTS5]   查询串: {fts_q}")
    fts_ids = []
    if fts_q:
        fts_rows = list(cur.execute(
            "SELECT rowid FROM fts_chunks WHERE fts_chunks MATCH ? ORDER BY rank LIMIT ?",
            (fts_q, RECALL_N)
        ))
        fts_ids = [r[0] for r in fts_rows]
    # trigram 对 <3 字词失效 → MATCH 空则用 LIKE 兜底关键词召回
    if not fts_ids and fts_q:
        kw = [w for w in fts_q.split(" OR ") if w.strip()]
        seen = []
        inv_idx = DF_CACHE.get("__inverted_index__", {})
        for w in kw:
            if len(w) == 2 and w in inv_idx:
                for cid in inv_idx[w]:
                    if cid not in seen:
                        seen.append(cid)
                    if len(seen) >= RECALL_N:
                        break
            elif len(w) > 2:
                rows = list(cur.execute("SELECT id FROM chunks WHERE text LIKE ? LIMIT ?", (f"%{w}%", RECALL_N)))
                for r in rows:
                    if r[0] not in seen:
                        seen.append(r[0])
            if len(seen) >= RECALL_N:
                break
        fts_ids = seen[:RECALL_N]
        print(f"[FTS5]   MATCH空→LIKE兜底 {len(fts_ids)} 条: {fts_ids}")
    print(f"[FTS5]   召回 {len(fts_ids)} 条: {fts_ids}")

    # --- 路二：向量 ---
    qvec = model.encode([query], normalize_embeddings=True)[0]
    qblob = struct.pack("1024f", *qvec.tolist())
    vec_rows = list(cur.execute(
        "SELECT chunk_id, distance FROM vec_chunks WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
        (qblob, RECALL_N)
    ))
    vec_ids = [r[0] for r in vec_rows]
    print(f"[Vector] 召回 {len(vec_ids)} 条: {vec_ids}")
    print(f"[Vector] 距离: {[round(r[1],4) for r in vec_rows]}")

    # --- RRF ---
    merged = rrf_merge(fts_ids, vec_ids)
    rrf_top = [cid for cid, _ in merged]

    # --- 第二刀：向量头强制保底 ---
    guaranteed = vec_ids[:GUARANTEE_VEC]
    final_candidates = []
    for cid in guaranteed + rrf_top:
        if cid not in final_candidates:
            final_candidates.append(cid)
    top15 = final_candidates[:RECALL_N]
    print(f"[保底]   向量头{GUARANTEE_VEC}: {guaranteed}")
    print(f"[RRF]    融合排序: {rrf_top}")
    print(f"[候选]   最终 Top{len(top15)}: {top15}")

    if not top15:
        print("两路都没召回，结束。")
        return

    placeholders = ",".join("?" * len(top15))
    text_map = {}
    for cid, text in cur.execute(f"SELECT id, text FROM chunks WHERE id IN ({placeholders})", top15):
        text_map[cid] = text
    con.close()

    cand_ids = [cid for cid in top15 if cid in text_map]
    if USE_RERANK:
        pairs = [[query, (text_map[cid][:TRUNCATE_N] if TRUNCATE_N else text_map[cid])] for cid in cand_ids]
        if RERANK_MODEL == "bge":
            scores = rerank.predict(pairs)
        else:
            scores = rerank.compute_score(pairs)
        ranked = sorted(zip(cand_ids, scores), key=lambda x: x[1], reverse=True)
    else:
        # 旁路 jina：直接按向量召回顺序排（text_map 顺序即向量序）
        ranked = [(cid, 0.0) for cid in cand_ids]

    print(f"\n{'='*50}\n[Jina 精排] 最终 Top {TOP_N}\n{'='*50}")
    for cid, s in ranked[:TOP_N]:
        print(f"\nscore={s:.4f} | chunk_id={cid}")
        print(f"{text_map[cid][:150]}")
        print("-" * 50)

if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "观测者怎么吐槽他的实验搭档"
    search(q)
