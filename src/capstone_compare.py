import re
import sys

from dotenv import load_dotenv
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_community.document_loaders import TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# TODO 1 (14.2): 문제 정의와 문서 범위
# 반도체 FAB 공정 운영 절차서(SOP)와 장비 알람 대응 매뉴얼을 근거로,
# 오퍼레이터·신입 공정 엔지니어의 공정 조건·알람 조치·로트 처리 질문에 답하는 RAG.
# 문서에 근거가 없으면 "제공된 문서에서 확인할 수 없습니다."라고 답한다.
# 상세 정의는 results/design.md 참고.
# 실험 조건: python src/capstone_compare.py [문서 경로] [Top-K]  (기본: data/fab.txt, k=2)
DOC_PATH = sys.argv[1] if len(sys.argv) > 1 else "data/fab.txt"
K = int(sys.argv[2]) if len(sys.argv) > 2 else 2
print(f"실험 조건: 문서={DOC_PATH}, Top-K={K}")

# TODO 2 (14.3): Baseline RAG
docs = TextLoader(DOC_PATH, encoding="utf-8").load()
chunks = RecursiveCharacterTextSplitter(
    chunk_size=120,
    chunk_overlap=20,
).split_documents(docs)

print("=== 문서 로드 확인 ===")
print("Document 수:", len(docs))
print("Chunk 수:", len(chunks))
print("Chunk[0] metadata:", chunks[0].metadata)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = FAISS.from_documents(chunks, embeddings)

baseline = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": K},
)

# TODO 3 (14.4): 고정 질문셋 10개 (유형별 2개)
# 복합 질문은 expected_keyword를 리스트로 두고 모두 포함되어야 HIT.
test_cases = [
    {"type": "정답 명확", "question": "CMP 연마 패드는 언제 교체하나요?", "expected_keyword": "500매"},
    {"type": "정답 명확", "question": "Hold 된 로트를 Release 하려면 누구의 승인이 필요한가요?", "expected_keyword": "엔지니어의 승인"},
    {"type": "표현 다름", "question": "세정 약품은 얼마나 자주 갈아줘야 해?", "expected_keyword": "24시간마다"},
    {"type": "표현 다름", "question": "같은 공정에서 다시 작업하는 건 몇 번까지 돼?", "expected_keyword": "최대 2회"},
    {"type": "키워드", "question": "ETCH-ALM-2047 알람 조치 방법은?", "expected_keyword": "진공 펌프"},
    {"type": "키워드", "question": "CMP-ALM-0322 발생 시 무엇을 해야 하나요?", "expected_keyword": "슬러리 필터"},
    {"type": "복합", "question": "CVD-ALM-1103 발생 시 1차 조치와, Hold 된 로트의 Release 승인자를 함께 알려줘",
     "expected_keyword": ["즉시 공정을 중단", "엔지니어의 승인"]},
    {"type": "복합", "question": "포토, 식각, CVD 장비의 PM 주기를 각각 알려줘",
     "expected_keyword": ["포토 장비의 정기 점검", "식각 장비의 정기 점검", "CVD 장비의 정기 점검"]},
    {"type": "문서에 없음", "question": "이번 분기 웨이퍼 수율은 얼마인가요?", "expected_keyword": None},
    {"type": "문서에 없음", "question": "EUV 노광 장비의 소스 파워 설정값은?", "expected_keyword": None},
]


# TODO 4 (14.5, 14.8): Retrieval 평가
def inspect(retriever, question: str):
    return retriever.invoke(question)


def hit(retrieved, expected_keyword) -> int:
    joined = "\n".join(doc.page_content for doc in retrieved)
    keywords = expected_keyword if isinstance(expected_keyword, list) else [expected_keyword]
    return 1 if all(keyword in joined for keyword in keywords) else 0


def reciprocal_rank(retrieved, expected_keyword) -> float:
    # 정답 키워드가 (복합 질문은 모두) 처음 갖춰지는 순위 n에 대해 1/n, 없으면 0
    for n in range(1, len(retrieved) + 1):
        if hit(retrieved[:n], expected_keyword):
            return 1 / n
    return 0.0


def evaluate(name: str, retriever) -> tuple[float, float, float]:
    scores = []
    ranks = []
    doc_counts = []

    print(f"\n=== {name} ===")
    for i, case in enumerate(test_cases, start=1):
        retrieved = inspect(retriever, case["question"])
        expected_keyword = case["expected_keyword"]

        print(f"\nQ{i} [{case['type']}] {case['question']}")

        if expected_keyword is None:
            print("판정: Retrieval Hit Rate 평가 제외 (문서에 없는 질문)")
        else:
            score = hit(retrieved, expected_keyword)
            rr = reciprocal_rank(retrieved, expected_keyword)
            scores.append(score)
            ranks.append(rr)
            print("판정:", "HIT" if score else "MISS", f"(RR={round(rr, 3)})")

        doc_counts.append(len(retrieved))

        print("검색 문서 수:", len(retrieved))
        for j, doc in enumerate(retrieved, start=1):
            print(f"  [{j}] {doc.page_content.replace(chr(10), ' / ')}")

    hit_rate = sum(scores) / len(scores)
    mrr = sum(ranks) / len(ranks)
    avg_docs = sum(doc_counts) / len(doc_counts)
    print(f"\n{name} Hit Rate: {sum(scores)}/{len(scores)} = {round(hit_rate, 3)}")
    print(f"{name} MRR: {round(mrr, 3)} / 평균 검색 문서 수: {round(avg_docs, 1)}")
    return hit_rate, mrr, avg_docs


# TODO 5 (14.6): 개선 전략
# 개선 1) Hybrid: 알람 코드처럼 정확한 토큰이 중요한 질문을 BM25로 보완 (step10 EnsembleRetriever)
def tokenize(text: str) -> list[str]:
    # ponytail: 공백·기호 기준 토큰화라 한국어 조사 분리는 안 됨. 필요하면 형태소 분석기(Kiwi)로 교체
    return re.findall(r"[\w\-]+", text)


bm25 = BM25Retriever.from_documents(chunks, preprocess_func=tokenize)
bm25.k = K
ensemble = EnsembleRetriever(
    retrievers=[baseline, bm25],
    weights=[0.5, 0.5],
)
# Baseline과 공정 비교를 위해 RRF 결합 결과도 Top-K로 자릅니다.
hybrid = ensemble | RunnableLambda(lambda found: found[:K])

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# 개선 2) Query Decomposition: 복합 질문을 하위 질문으로 나눠 각각 Hybrid 검색 (step11 decompose_query)
def decompose_query(question: str) -> list[str]:
    prompt = f"""
다음 질문을 문서 검색에 적합한 독립적인 하위 질문으로 나누세요.
이미 단일 질문이면 원래 질문 한 줄만 출력하세요.
각 줄에 질문 하나만 출력하세요.
번호나 설명은 붙이지 마세요.

[질문]
{question}
"""
    result = llm.invoke(prompt).content.strip()
    return [
        line.strip("- ").strip()
        for line in result.splitlines()
        if line.strip()
    ]


def decomposed_search(question: str):
    merged = []
    seen = set()
    for sub_question in decompose_query(question):
        for doc in hybrid.invoke(sub_question):
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                merged.append(doc)
    return merged


improved = RunnableLambda(decomposed_search)


# 개선 3) Reranker: Hybrid 후보는 찾지만 Top-K 컷에서 정답이 잘리는 문제 보완 (step10 rerank)
def relevance_score(question: str, document: str) -> int:
    prompt = f"""
다음 질문과 문서의 관련성을 0~100 사이 정수 하나로 평가하세요.
설명하지 말고 숫자만 출력하세요.

[질문]
{question}

[문서]
{document}
"""
    response = llm.invoke(prompt).content.strip()
    match = re.search(r"\d+", response)
    if not match:
        return 0
    return min(100, max(0, int(match.group())))


def rerank(question: str, docs, top_n: int = K):
    scored = [(relevance_score(question, doc.page_content), doc) for doc in docs]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_n]]


def decomposed_rerank_search(question: str):
    merged = []
    seen = set()
    for sub_question in decompose_query(question):
        # Top-K로 자르기 전의 Hybrid 후보 전체(Dense K + BM25 K)를 재정렬
        for doc in rerank(sub_question, ensemble.invoke(sub_question)):
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                merged.append(doc)
    return merged


reranked = RunnableLambda(decomposed_rerank_search)

# TODO 6 (14.7, 14.8): 동일 질문셋으로 Baseline과 개선 Pipeline 비교
results = {
    "Baseline Similarity": evaluate("Baseline Similarity", baseline),
    "Hybrid": evaluate("Improved 1: Hybrid (FAISS + BM25)", hybrid),
    "Hybrid + Decomposition": evaluate("Improved 2: Hybrid + Query Decomposition", improved),
    "+ Rerank": evaluate("Improved 3: Hybrid + Query Decomposition + Rerank", reranked),
}

print(f"\n=== Retrieval 비교 (문서={DOC_PATH}, Top-K={K}) ===")
print(f"{'Pipeline':<24} {'Hit Rate':>8} {'MRR':>6} {'평균 문서 수':>10}")
for name, (hit_rate, mrr, avg_docs) in results.items():
    print(f"{name:<24} {hit_rate:>8.3f} {mrr:>6.3f} {avg_docs:>10.1f}")

# Generation 평가
prompt = ChatPromptTemplate.from_template("""
당신은 반도체 FAB 공정 SOP를 근거로 답하는 도우미입니다.
아래 문서만 근거로 질문에 답하세요.
문서에 없는 내용은 추측하지 말고 "제공된 문서에서 확인할 수 없습니다."라고 답하세요.
안전과 관련된 질문은 답변 끝에 "담당 엔지니어 또는 안전팀에 확인하세요."를 덧붙이세요.

[문서]
{context}

[질문]
{question}
""")


def answer_with_retriever(question: str, retriever=reranked) -> str:
    retrieved_docs = retriever.invoke(question)
    context = "\n\n".join(
        f"[문서 {i}]\n{doc.page_content}"
        for i, doc in enumerate(retrieved_docs, start=1)
    )
    return llm.invoke(
        prompt.invoke({
            "context": context,
            "question": question,
        })
    ).content


print("\n=== Generation 비교 (Baseline / Hybrid + Decomposition / + Rerank) ===")
for i, case in enumerate(test_cases, start=1):
    print(f"\nQ{i} [{case['type']}] {case['question']}")
    print("  Baseline :", answer_with_retriever(case["question"], baseline).replace("\n", " "))
    print("  Decomp   :", answer_with_retriever(case["question"], improved).replace("\n", " "))
    print("  Rerank   :", answer_with_retriever(case["question"], reranked).replace("\n", " "))

# TODO 7 (14.9~14.13): README.md, results/design.md, results/evaluation.md에 정리
