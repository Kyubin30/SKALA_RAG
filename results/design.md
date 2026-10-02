# 종합실습 설계 문서 — 반도체 FAB 공정 SOP RAG

## 1. 해결하려는 문제

> 반도체 FAB의 공정 운영 SOP와 장비 알람 대응 매뉴얼을 근거로, 공정 엔지니어와 오퍼레이터의 **공정 조건·알람 조치·로트 처리** 질문에 답하는 RAG

FAB 현장에서 알람이 발생하면 매뉴얼을 빨리 찾아 1차 조치를 해야 합니다. 그런데 SOP는 공정별로 나뉘어 있고 알람 코드(`ETCH-ALM-2047` 등)도 많습니다. 그래서 원하는 조항을 찾는 데 시간이 걸립니다.
이 RAG는 질문에 맞는 SOP 조항을 검색하고, **문서에 적힌 내용만** 근거로 답합니다.

| 14.2 항목 | 정의 |
|---|---|
| ① 누가 질문하는가 | FAB 오퍼레이터, 신입 공정 엔지니어 |
| ② 어떤 문서를 검색하는가 | `data/fab.txt` (교육용 가상 SOP) |
| ③ 어떤 질문까지 답하는가 | 문서에 적힌 공정 조건, 장비 PM 주기, 알람 코드별 1차 조치, 로트 Hold/Release·Rework 절차, SPC 관리, 클린룸·안전 수칙 |
| ④ 근거가 없을 때 | "제공된 문서에서 확인할 수 없습니다."라고 답하고 추측하지 않음. 안전 관련 질문이면 담당 엔지니어·안전팀 확인을 안내 |

**범위 밖**: 실제 수율·단가·고객 정보, 문서에 없는 장비·공정(예: EUV), 레시피 변경 승인 같은 의사결정.

## 2. 대상 사용자

| 사용자 | 상황 | 주로 묻는 질문 |
|---|---|---|
| FAB 오퍼레이터 | 교대 근무 중 알람 발생. 즉시 조치가 필요하고 짧은 구어체로 질문 | "ETCH-ALM-2047 뜨면 뭐 해?", "로트 Hold 어떻게 걸어?" |
| 신입 공정 엔지니어 | 공정 조건·관리 기준을 학습하며 여러 조항을 묶어서 질문 | "포토·식각·CVD PM 주기 각각?", "Release 승인자는?" |

## 3. 사용할 문서와 데이터 범위

- 문서: `data/fab.txt` (UTF-8 텍스트, 13개 섹션, 36 chunk)
- **교육용 가상 문서**입니다. 실제 회사·장비·공정값과 무관합니다.
- 섹션: 포토 / 식각 / 이온주입 / 확산·산화 / CVD / PVD / CMP / 세정 / 장비 알람 코드(13개) / 로트 Hold·Release / 장비 Down·복구 / SPC / 클린룸·안전
- 의도적으로 넣은 검색 난이도:
  - **비슷한 문장 반복**: 공정마다 "○○ 장비의 정기 점검(PM) 주기는 N주이다" 형식의 문장이 있습니다. 의미 검색이 공정을 혼동할 수 있습니다.
  - **코드형 식별자**: 알람 코드 13개(`ETCH-ALM-2047`, `CVD-ALM-1103`, `CVD-ALM-1104` …)가 같은 형식으로 설명되어 있습니다. 임베딩만으로는 숫자 하나 차이를 구분하기 어렵습니다.
  - **복합 질문**: 알람 조치(6장)와 로트 Release 승인(7장)처럼 답이 서로 다른 섹션에 흩어져 있습니다.

## 4. Baseline RAG 구조

```text
Loader          TextLoader("data/fab.txt")
  ↓
Splitter        RecursiveCharacterTextSplitter(chunk_size=120, chunk_overlap=20)
  ↓
Embedding       OpenAI text-embedding-3-small
  ↓
Vector Store    FAISS
  ↓
Retriever       similarity, k=2
  ↓
Prompt          "문서만 근거로 답하고, 없으면 확인할 수 없다고 답함"
  ↓
LLM             gpt-4o-mini, temperature=0
```

### 개선 Pipeline 흐름도 (14.9)

```text
Document
  ↓
Loader → Splitter → Embedding → Vector Store (FAISS)
                  └→ BM25 Index (키워드)
                                    ↓
Question → Query Strategy (Decomposition: 복합 질문 → 하위 질문들)
                                    ↓
           Retriever (Hybrid = EnsembleRetriever[FAISS + BM25], RRF 결합 후보 최대 2K개)
                                    ↓
           Rerank (LLM 관련성 0~100 점수로 재정렬, Top-K)
                                    ↓
           하위 질문별 결과 합치기 + 중복 제거
                                    ↓
                                 Context
                                    ↓
                                   LLM
                                    ↓
                                  Answer
```

## 5. 검색 개선 전략과 선택 이유

교재 14.6의 "관찰된 문제 → 우선 검토할 방법" 표를 기준으로 선택했습니다.

| 예상/관찰 문제 | 14.6 권장 방법 | 적용 | 구현 근거 |
|---|---|---|---|
| 알람 코드 같은 **코드·약어**가 Dense 검색으로 잘 구분되지 않음 | BM25 / Hybrid / Ensemble | **Hybrid (EnsembleRetriever)** | step10 `10_search_quality.py`의 EnsembleRetriever |
| **복합 질문**의 답이 여러 섹션에 흩어져 Top-K에 다 들어오지 않음 | Query Rewrite / Decomposition | **Query Decomposition** | step11 `11_advanced_rag.py`의 `decompose_query()` |
| (2단계 적용 후 관찰) **후보는 찾지만 순서가 좋지 않아** Top-K 컷에서 정답이 잘림 | Reranker | **LLM Reranker** | step10 `10_search_quality.py`의 `relevance_score()` / `rerank()` |

- Hybrid는 Dense의 의미 검색을 유지하면서, BM25의 정확한 토큰 일치(알람 코드)로 이를 보완합니다. 공정한 비교를 위해 결합 결과를 Baseline과 같은 **Top-2**로 자릅니다.
- Decomposition은 복합 질문을 하위 질문으로 나누고, 하위 질문마다 Hybrid로 검색합니다. 단일 질문은 하위 질문이 1개가 되므로 Hybrid와 거의 같게 동작합니다.
- Reranker는 처음에는 계획에 없었습니다. Decomposition을 적용했더니 `CVD-ALM-1103` 질문에 `1104`의 조치를 답하는 회귀가 생겼습니다. 정답 chunk는 Hybrid 후보에 있었지만 RRF 결합 뒤 Top-K 컷에서 잘렸습니다. 이 진단에 따라 14.6 표의 "후보 문서는 찾지만 순서가 좋지 않음 → Reranker"를 추가했습니다.
- MMR은 선택하지 않았습니다. 이 문서의 문제는 "중복 chunk"가 아니라 "정확한 코드 불일치"와 "흩어진 근거"였기 때문입니다. Agentic RAG는 남은 한계로 기록합니다.

## 6. 평가 질문과 평가 방법

교재 14.4의 5개 유형별로 2개씩, 총 10개 질문을 고정했습니다.

| # | 유형 | 질문 | 정답 키워드 |
|---|---|---|---|
| Q1 | 정답 명확 | CMP 연마 패드는 언제 교체하나요? | 500매 |
| Q2 | 정답 명확 | Hold 된 로트를 Release 하려면 누구의 승인이 필요한가요? | 엔지니어의 승인 |
| Q3 | 표현 다름 | 세정 약품은 얼마나 자주 갈아줘야 해? | 24시간마다 |
| Q4 | 표현 다름 | 같은 공정에서 다시 작업하는 건 몇 번까지 돼? | 최대 2회 |
| Q5 | 키워드 | ETCH-ALM-2047 알람 조치 방법은? | 진공 펌프 |
| Q6 | 키워드 | CMP-ALM-0322 발생 시 무엇을 해야 하나요? | 슬러리 필터 |
| Q7 | 복합 | CVD-ALM-1103 발생 시 1차 조치와, Hold 된 로트의 Release 승인자를 함께 알려줘 | 즉시 공정을 중단 + 엔지니어의 승인 |
| Q8 | 복합 | 포토, 식각, CVD 장비의 PM 주기를 각각 알려줘 | 포토/식각/CVD 장비의 정기 점검 (3개 모두) |
| Q9 | 문서에 없음 | 이번 분기 웨이퍼 수율은 얼마인가요? | 없음 (Hit Rate 제외) |
| Q10 | 문서에 없음 | EUV 노광 장비의 소스 파워 설정값은? | 없음 (Hit Rate 제외) |

**평가 방법 (14.8)**
- **Retrieval**: Top-K 안에 정답 키워드가 있으면 HIT입니다. 복합 질문은 키워드가 **모두** 있어야 HIT입니다. Hit Rate는 Q1~Q8만으로 계산합니다.
  - **MRR**: 정답 키워드가 처음 갖춰지는 순위의 역수를 평균한 값입니다. Hit Rate가 같아도 "관련 문서가 상위에 배치되는가"(14.8)를 비교할 수 있습니다.
  - **평균 검색 문서 수**: Context 크기, 즉 비용과 노이즈를 확인합니다.
- **Generation**: 질문마다 Baseline과 개선 Pipeline의 답변을 비교합니다. ① 근거와 일치하는지 ② 질문에 직접 답하는지 ③ Q9·Q10에서 근거 없는 답을 만들지 않는지 확인합니다.
- 네 Pipeline(Baseline / Hybrid / +Decomposition / +Rerank)을 **같은 질문셋, 같은 Chunk·Embedding·Vector Store**로 비교합니다.
- `data/fab.txt`, Top-K=2 조건에서 측정합니다. Top-2는 알람 대응 상황에서 Context를 작게 유지하는 조건입니다.

## 7. 예상 한계와 보완 방법

| 한계 | 보완 방법 |
|---|---|
| Decomposition은 하위 질문 수만큼 검색하므로 Context가 커지고 LLM 호출이 1회 늘어남 | 단일 질문은 분해를 건너뛰는 라우팅 |
| LLM Reranker는 후보마다 LLM을 호출해 비용과 지연이 큼 | Cross-Encoder Reranker로 교체 |
| 비슷한 근거를 정답으로 착각할 수 있음(1103 ↔ 1104) | 답변 속 코드와 질문 속 코드를 비교하는 사후 검증 (step13 grade 노드) |
| BM25가 한국어 형태소를 분석하지 않음(공백·기호 기준 토큰화). 조사가 붙은 단어는 매칭이 약함 | 한국어 형태소 분석기(Kiwi 등)로 토큰화 |
| 키워드 포함 여부로 Hit를 판정하므로 표현이 다른 정답은 놓칠 수 있음 | 정답 chunk ID 기반 평가, LLM-as-Judge, RAGAS |
| 문서 1개, 질문 10개로 표본이 작음 | 실제 SOP 다수, 질문 50개 이상으로 확장 |
| 검색에 실패해도 다시 검색하지 않음 | step13 LangGraph Agentic RAG(grade → rewrite → retry, fallback) |
| 안전 관련 답변 오류는 치명적임 | 안전 섹션은 원문 인용과 안전팀 연락처를 고정 출력 |
