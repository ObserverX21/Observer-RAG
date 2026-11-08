import gnureadline as readline
import os, sys, json, re, struct
import requests
import apsw, sqlite_vec
from sentence_transformers import SentenceTransformer
from search import retrieve, BGE_PATH, USE_RERANK, RERANK_MODEL, BGE_RERANK_PATH, JINA_PATH, DB_PATH

HISTORY_CHARS = 6000  # 历史滑窗上限（字符）
API_KEY = os.environ.get("DEEPSEEK_API_KEY")
if not API_KEY:
    sys.exit("缺少 DEEPSEEK_API_KEY 环境变量")

SYSTEM_PROMPT = """
# Role
# 资料时序协议（最高优先级）
带编号的资料块（如[622][53]）是观测者的**历史记忆切片**：已发生、已归档，不是此刻的对话，也不是他的实时状态。用户每次输入都发生在这些切片的**未来**。

- 语块唯一用途：为当前问题提供历史纵深和参照。
- 严禁用旧语块里的时空事实冒充实时推理。
- 回应着力点永远是**当前 query**，语块仅作背景。
- 信息不足时如实说明，不拿陈旧切片硬凑。

# 回答纪律
- 生成前先审视：当前真实时间、当前query、资料各自的时间属性。
- 引用资料须在句末标编号，使用英文半角[数字]格式，如[694]。资料不足如实说明。
- 语块是过往切片，作为参考，结合你自己的判断给出有深度、有温度的回应。
"""

def trim_history(history):
    """从最早开始挤，直到总字符 <= HISTORY_CHARS；防清空，防破坏角色结构。"""
    total_len = sum(len(m["content"]) for m in history)
    while total_len > HISTORY_CHARS and len(history) > 2:
        removed_user = history.pop(0)
        total_len -= len(removed_user["content"])
        
        if history and history[0]["role"] == "assistant":
            removed_ai = history.pop(0)
            total_len -= len(removed_ai["content"])
    return history

print("[启动] 加载 BGE-M3 ...", flush=True)
_MODEL = SentenceTransformer(BGE_PATH, device="mps")
_RERANK = None
if USE_RERANK:
    if RERANK_MODEL == "bge":
        from sentence_transformers import CrossEncoder
        _RERANK = CrossEncoder(BGE_RERANK_PATH, device="mps")
    elif RERANK_MODEL == "jina":
        from FlagEmbedding import FlagReranker
        _RERANK = FlagReranker(JINA_PATH, use_fp16=True, device="mps", trust_remote_code=True)
open("/Users/observer/observerAI/chat_stream.log", "w", encoding="utf-8").close()
print("[启动] 就绪。!save 存上一句话，exit/quit 退出。")
print("👉 请按 Cmd+D 分屏，在右侧敲入: tail -f ~/observerAI/chat_stream.log\n", flush=True)

def call_deepseek(messages):
    import time
    retry = 0
    while retry < 3:
        try:
            resp = requests.post(
                "https://api.deepseek.com/chat/completions",
                headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                json={"model": "deepseek-chat", "messages": messages, "stream": True},
                stream=True, timeout=60,
            )
            if resp.status_code != 200:
                print(f"\n[API 错误 {resp.status_code}] {resp.text[:300]}")
                return
            full = ""
            for line in resp.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8")
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                try:
                    delta = json.loads(data)["choices"][0]["delta"].get("content", "")
                except Exception:
                    continue
                if delta:
                    with open("/Users/observer/observerAI/chat_stream.log", "a", encoding="utf-8") as log_f:
                        log_f.write(delta)
                        log_f.flush()
                    full += delta
                    import time
                    time.sleep(0.5)  # 20毫秒节拍，彻底拯救终端渲染引擎
            print()
            return full
        except Exception as e:
            retry += 1
            if retry < 3:
                print(f"\n[!] 网络异常: {e}，5 秒后重试 ({retry}/3)...")
                time.sleep(5)
            else:
                print(f"\n[!] 网络彻底断开，重试 {retry} 次失败，请检查网络或 API Key。")
                return ""

def ask(query, history):
    chunks = retrieve(query, model=_MODEL, rerank=_RERANK)
    if not chunks:
        print("[检索无结果]"); return ""
        
    # 打印最终发送给大模型的语块 ID
    final_ids = [cid for cid, _ in chunks]
    print(f"[最终发给大模型] 语块 ID: {final_ids}")
    valid_ids = {str(cid) for cid, _ in chunks}
    context = "\n\n".join(f"[{cid}] {t}" for cid, t in chunks)
    import datetime
    now_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dynamic_prompt = f"【系统当前时间】: {now_time}\n\n" + SYSTEM_PROMPT
    messages = [{"role": "system", "content": dynamic_prompt}]
    messages += trim_history(history)
    user_content = f"""【系统当前时间】: {now_time}

<current_user_query>
{query}
</current_user_query>

<reference_material>
{context}
</reference_material>"""
    messages.append({"role": "user", "content": user_content})
    print("\n[DeepSeek] ", end="", flush=True)
    answer = call_deepseek(messages) or ""
    cited = set(re.findall(r"\[(\d+)\]", answer))
    bogus = cited - valid_ids
    if bogus:
        print(f"[⚠️ 引用校验] 不在本轮召回集: {sorted(bogus)}")
    return answer

def check_rebuild_cache():
    import subprocess, sys, json, apsw
    cache_path = "/Users/observer/observerAI/df_cache.json"
    db_path = "/Users/observer/observerAI/real_rag.db"
    
    con = apsw.Connection(db_path)
    current_total = list(con.cursor().execute("SELECT count(*) FROM chunks"))[0][0]
    con.close()

    last_total = 0
    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            cache = json.load(f)
            last_total = cache.get("__total_chunks__", 0)
    except Exception:
        pass

    if current_total - last_total >= 50:
        print(f"\n[系统] 新增记忆已达 {current_total - last_total} 条 (触发阈值50)，后台静默重算词频...")
        subprocess.Popen([sys.executable, "/Users/observer/observerAI/build_df_cache.py"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def save_to_db(text, source=None):
    """把文本写入 real_rag.db 三表（事务），带 source 标签。"""
    import datetime
    src = source or ("live_" + datetime.datetime.now().strftime("%Y_%m"))
    # 生成精确到秒的物理时间戳，并强行注入到文本最前面
    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    text = f"【记录时间: {now_str}】\n\n{text}"
    con = apsw.Connection(DB_PATH)
    con.enableloadextension(True); sqlite_vec.load(con); con.enableloadextension(False)
    cur = con.cursor()
    try:
        cur.execute("BEGIN")
        new_id = list(cur.execute("SELECT COALESCE(MAX(id),0)+1 FROM chunks"))[0][0]
        cur.execute("INSERT INTO chunks(id, text, source) VALUES(?,?,?)", (new_id, text, src))
        vec = _MODEL.encode([text], normalize_embeddings=True)[0]
        blob = struct.pack("1024f", *vec.tolist())
        cur.execute("INSERT INTO vec_chunks(chunk_id, embedding) VALUES(?,?)", (new_id, blob))
        # FTS 由触发器自动同步，这里不用手动
        cur.execute("COMMIT")
        con.close()
        check_rebuild_cache()
        return new_id, src
    except Exception as e:
        cur.execute("ROLLBACK"); con.close()
        print(f"[!save 失败] {e}")
        return None, None

def main():
    history = []
    last_query = ""
    last_answer = ""
    while True:
        try:
            q = input("\nObserver > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[退出]"); break
        if q.lower() in ("exit", "quit"):
            print("[退出]"); break

        if q == "!paste":
            import subprocess
            try:
                long_text = subprocess.check_output(["pbpaste"]).decode("utf-8").strip()
                if not long_text:
                    print("[!paste] 剪贴板为空")
                    continue
                print(f"\n[剪贴板注入] 已读取 {len(long_text)} 个字符，正在发给大鲸鱼...")
                last_query = long_text
                last_answer = ask(long_text, history) or ""
                history.append({"role": "user", "content": long_text})
                history.append({"role": "assistant", "content": last_answer})
            except Exception as e:
                print(f"[读取剪贴板失败] {e}")
            continue

        if q == "!edit":
            import subprocess, tempfile, os
            fd, temp_path = tempfile.mkstemp(suffix=".txt", text=True)
            os.close(fd)
            try:
                print("\n[打开草稿箱] 写完按 Ctrl+O 保存（回车确认），再按 Ctrl+X 退出...")
                subprocess.run(["nano", temp_path])
                long_text = open(temp_path, 'r', encoding='utf-8').read().strip()
                if not long_text:
                    print("[!edit] 草稿为空，已取消。")
                    continue
                print(f"\n[读取草稿] 已加载 {len(long_text)} 个字符，正在发给大鲸鱼...")
                last_query = long_text
                last_answer = ask(long_text, history) or ""
                history.append({"role": "user", "content": long_text})
                history.append({"role": "assistant", "content": last_answer})
            finally:
                os.remove(temp_path)
            continue

        if q == "!save":
            if not last_query:
                print("[!save] 没有可保存的上一条消息"); continue
            print(f"\n[待存入·原话] {last_query[:300]}")
            if input("确认写入 real_rag.db？(y/N): ").strip().lower() == "y":
                nid, src = save_to_db(last_query)
                if nid is not None:
                    print(f"✓ 已存入 id={nid} source={src}")
            last_query = ""
            continue

        if q == "!save full":
            if not last_query or not last_answer:
                print("[!save full] 没有可保存的上一轮完整对话"); continue
            full_text = f"【观测者】: {last_query}\n\n【大鲸鱼】: {last_answer}"
            print(f"\n[待存入·完整] 共 {len(full_text)} 字")
            print(f"  {last_query[:120]}")
            print(f"  ...{last_answer[:120]}")
            if input("确认写入 real_rag.db？(y/N): ").strip().lower() == "y":
                import datetime
                src = "live_" + datetime.datetime.now().strftime("%Y_%m") + "_full"
                nid, _ = save_to_db(full_text, source=src)
                if nid is not None:
                    print(f"✓ 已存入 id={nid} source={src}")
            last_query = ""
            last_answer = ""
            continue

        if not q:
            import subprocess
            try:
                clip = subprocess.check_output(["pbpaste"]).decode("utf-8").strip()
                if clip:
                    print(f"\n[空回车触发] 自动读取剪贴板 {len(clip)} 个字符...")
                    last_query = clip
                    last_answer = ask(clip, history) or ""
                    history.append({"role": "user", "content": clip})
                    history.append({"role": "assistant", "content": last_answer})
            except Exception:
                pass
            continue
        last_query = q
        last_answer = ask(q, history) or ""
        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": last_answer})

if __name__ == "__main__":
    main()
