import json
import numpy as np
import faiss
import re
import traceback
from config import Config  # 导入配置文件
from openai import OpenAI

def clean_text(text):
    """清理文本中的非法字符，控制文本长度"""
    if not text:
        return ""
    # 移除控制字符，保留换行和制表符
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    # 移除重复的空白字符
    text = re.sub(r'\s+', ' ', text)
    # 确保文本长度在合理范围内
    return text.strip()

# 向量化查询 - 通用函数，被多处使用
def vectorize_query(query, model_name=Config.model_name, batch_size=Config.batch_size) -> np.ndarray:
    """向量化文本查询，返回嵌入向量，改进错误处理和批处理"""
    embedding_client = OpenAI(
        api_key=Config.api_key,
        base_url=Config.base_url
    )
    
    if not query:
        print("警告: 传入向量化的查询为空")
        return np.array([])
        
    if isinstance(query, str):
        query = [query]
    
    # 验证所有查询文本，确保它们符合API要求
    valid_queries = []
    for q in query:
        if not q or not isinstance(q, str):
            print(f"警告: 跳过无效查询: {type(q)}")
            continue
            
        # 清理文本并检查长度
        clean_q = clean_text(q)
        if not clean_q:
            print("警告: 清理后的查询文本为空")
            continue
            
        # 检查长度是否在API限制范围内
        if len(clean_q) > 8000:
            print(f"警告: 查询文本过长 ({len(clean_q)} 字符)，截断至 8000 字符")
            clean_q = clean_q[:8000]
        
        valid_queries.append(clean_q)
    
    if not valid_queries:
        print("错误: 所有查询都无效，无法进行向量化")
        return np.array([])
    
    # 分批处理有效查询
    all_vectors = []
    for i in range(0, len(valid_queries), batch_size):
        batch = valid_queries[i:i + batch_size]
        try:
            # 记录批次信息便于调试
            print(f"正在向量化批次 {i//batch_size + 1}/{(len(valid_queries)-1)//batch_size + 1}, "
                  f"包含 {len(batch)} 个文本，第一个文本长度: {len(batch[0][:50])}...")
                  
            completion = embedding_client.embeddings.create(
                model=model_name,
                input=batch,
                dimensions=Config.dimensions,
                encoding_format="float"
            )
            vectors = [embedding.embedding for embedding in completion.data]
            all_vectors.extend(vectors)
            print(f"批次 {i//batch_size + 1} 向量化成功，获得 {len(vectors)} 个向量")
        except Exception as e:
            print(f"向量化批次 {i//batch_size + 1} 失败：{str(e)}")
            print(f"问题批次中的第一个文本: {batch[0][:100]}...")
            traceback.print_exc()
            # 如果是第一批就失败，直接返回空数组
            if i == 0:
                return np.array([])
            # 否则返回已处理的向量
            break
    
    # 检查是否获得了任何向量
    if not all_vectors:
        print("错误: 向量化过程没有产生任何向量")
        return np.array([])
        
    return np.array(all_vectors)

def vector_search(query, index_path, metadata_path, limit=5, normalize=True):
    """
    基本向量搜索函数（优化版）
    - query: 用户查询字符串
    - index_path: 已构建的FAISS索引路径
    - metadata_path: 元数据文件路径
    - limit: 返回Top-N结果
    - normalize: 是否执行L2归一化（推荐在使用IP索引时开启）
    """

    # ====== 1️⃣ 文本向量化 ======
    query_vector = vectorize_query(query)
    if query_vector is None or len(query_vector) == 0:
        print("警告: 查询向量为空")
        return []

    query_vector = np.array(query_vector, dtype=np.float32).reshape(1, -1)

    # ====== 2️⃣ 可选归一化 ======
    if normalize:
        faiss.normalize_L2(query_vector)

    # ====== 3️⃣ 加载索引 ======
    try:
        index = faiss.read_index(index_path)
    except Exception as e:
        print(f"警告: 无法加载索引文件 {index_path}: {e}")
        return []

    # ====== 4️⃣ 加载元数据 ======
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    except UnicodeDecodeError:
        print(f"警告: {metadata_path} 编码错误，使用UTF-8忽略非法字符重新加载")
        with open(metadata_path, 'rb') as f:
            content = f.read().decode('utf-8', errors='ignore')
            metadata = json.loads(content)
    except Exception as e:
        print(f"警告: 加载元数据失败: {e}")
        return []

    # ====== 5️⃣ 向量搜索 ======
    try:
        scores, indices = index.search(query_vector, limit)
    except Exception as e:
        print(f"警告: 搜索失败: {e}")
        traceback.print_exc()
        return []

    # ====== 6️⃣ 构建结果 ======
    results = []
    for idx, score in zip(indices[0], scores[0]):
        if 0 <= idx < len(metadata):
            item = metadata[idx].copy()
            item["score"] = float(score)
            results.append(item)

    # ====== 7️⃣ 按相似度排序并返回 ======
    results.sort(key=lambda x: x["score"], reverse=True)
    return results
