# 종합실습 평가 기록 — 반도체 FAB 공정 SOP RAG

같은 질문셋(10개)으로 Baseline과 개선 Pipeline 3단계를 비교했습니다. 원본 실행 로그는 `results/run_fab_k2.txt`에 있습니다.

## 1. 비교 조건

- 문서: `data/fab.txt` (교육용 가상 SOP, 13개 섹션, 알람 코드 13개)
- Chunk 설정: RecursiveCharacterTextSplitter chunk_size=120, chunk_overlap=20 → 36 chunk
- Embedding: OpenAI `text-embedding-3-small`
- Vector Store: FAISS
- Top-K: 2
- Chat Model: `gpt-4o-mini`, temperature=0
- Baseline 검색 방식: FAISS similarity Top-2
- 개선 검색 방식:
  1. Hybrid: EnsembleRetriever(FAISS 2 + BM25 2, 가중치 0.5/0.5, RRF) → Top-2 (step10)
  2. + Query Decomposition: 하위 질문마다 Hybrid Top-2 → 합치고 중복 제거 (step11)
  3. + Rerank: 하위 질문마다 Hybrid 후보 전체(최대 4개)를 LLM 관련성 점수로 재정렬 → Top-2 (step10)

지표는 다음과 같습니다.
- Hit Rate: 정답 키워드 포함 여부입니다. 복합 질문은 키워드가 모두 있어야 HIT이고, Q9·Q10은 계산에서 제외합니다.
- MRR: 정답이 처음 갖춰지는 순위의 역수를 평균한 값입니다.
- 평균 검색 문서 수: Context 크기입니다.

## 2. 질문별 결과

| Pipeline | Hit Rate | MRR | 평균 문서 수 |
|---|---|---|---|
| Baseline | 0.750 | 0.625 | 2.0 |
| + Hybrid | 0.750 | 0.750 | 2.0 |
| + Decomposition | 0.875 | 0.792 | 3.0 |
| + Rerank (최종) | **1.000** | **0.833** | 3.0 |

| 질문 | Baseline Top-K | 개선 Top-K (Hybrid / +Decomp / +Rerank) | 정답 문서 포함 | 답변 변화 | 판단 |
|---|---|---|---|---|---|
| Q1 CMP 패드 교체 | HIT, 정답 2위 | 1위 / 1위 / 1위 | 모두 포함 | 동일 (정답) | 좋아짐 (순위) |
| Q2 Release 승인자 | HIT, 1위 | 1위 / 1위 / 1위 | 모두 포함 | 동일 (정답) | 동일 |
| Q3 세정 약품 교체 | HIT, 1위 | 1위 / 1위 / 1위 | 모두 포함 | 동일 (정답) | 동일 |
| Q4 재작업 횟수 | HIT, 1위 | 1위 / 1위 / 1위 | 모두 포함 | 동일 (정답) | 동일 |
| Q5 ETCH-ALM-2047 | HIT, 정답 2위 (1위 2051) | 1위 / 1위 / 1위 | 모두 포함 | 동일 (정답) | 좋아짐 (순위) |
| Q6 CMP-ALM-0322 | HIT, 1위 | 1위 / 1위 / 1위 | 모두 포함 | 동일 (정답) | 동일 |
| Q7 CVD-ALM-1103 + 승인자 | MISS | MISS / MISS / HIT | 최종만 포함 | 승인자 "확인 불가" → 둘 다 정답 (Decomp 단계에서 1104 조치로 오답) | 좋아짐 |
| Q8 포토·식각·CVD PM | MISS | MISS / HIT / HIT | Decomp부터 포함 | 식각 PM "확인 불가" → 3개 모두 정답 | 좋아짐 |
| Q9 이번 분기 수율 | 평가 제외 | 평가 제외 | — | "확인할 수 없습니다" 유지 | 동일 (정상) |
| Q10 EUV 소스 파워 | 평가 제외 | 평가 제외 | — | "확인할 수 없습니다" 유지 | 동일 (정상) |

## 3. Retrieval 평가

- **정답 문서가 Top-K에 포함되는가?** Baseline은 복합 질문 2개를 놓쳤습니다. 근거가 흩어져 있어 Top-2로는 다 담을 수 없었기 때문입니다. 최종 Pipeline은 8/8을 맞혔습니다.
- **관련 문서가 상위에 배치되는가?** Dense 검색은 숫자 하나만 다른 알람 코드를 구분하지 못했습니다. BM25를 결합하자 MRR이 0.625에서 0.750으로 올랐습니다.
- **중복·무관 문서가 지나치게 많지 않은가?** Decomposition 이후 평균 문서 수가 2.0에서 3.0으로 늘었고, 단일 질문에도 무관한 chunk가 섞였습니다.

## 4. Generation 평가

- **답변이 검색 Context의 근거와 일치하는가?** 대부분 일치했습니다. 예외는 Decomposition 단계의 Q7입니다. Context에 1103 조항이 없고 비슷한 1104 조항만 있자, "1103 정보가 없다"고 하지 않고 "CVD-ALM-1104 발생 시 1차 조치는 스로틀 밸브 점검…"이라고 질문과 다른 알람으로 답했습니다.
- **질문에 직접 답하는가?** 최종 Pipeline은 복합 질문의 모든 부분에 답했습니다. Baseline은 일부만 답했습니다.
- **근거가 없을 때 내용을 만들어내지 않는가?** Q9와 Q10은 모든 Pipeline에서 "제공된 문서에서 확인할 수 없습니다."로 답했습니다.

## 5. 개선 근거

- **Baseline에서 발견한 문제**: ① 숫자만 다른 알람 코드를 구분하지 못함(Q5) ② 흩어진 근거를 Top-K에 다 담지 못해 복합 질문에 일부만 답함(Q7, Q8)
- **선택한 개선 전략**: Hybrid(BM25) → Query Decomposition → Reranker
- **선택 이유** (교재 14.6 표 기준):
  - 코드·약어 검색 실패 → BM25/Hybrid
  - 흩어진 근거 → Decomposition
  - Decomposition 이후 관찰한 "후보는 찾지만 순서가 좋지 않음" → Reranker. Q7의 정답 1103은 Hybrid 후보에 있었지만, RRF 결합 뒤 Top-2 컷에서 3위로 밀려 잘렸습니다.
- **개선 결과**: Hit Rate 0.750 → 1.000, MRR 0.625 → 0.833. Q7·Q8 답변이 완전한 정답이 됐습니다.
- **남은 한계**:
  - LLM Reranker 비용·지연: 질문당 LLM 호출 1회 → 11.2회, 응답 0.92초 → 7.73초 (LangSmith 실측, `results/langsmith_metrics.txt`)
  - 단일 질문도 분해되어 Context가 증가함
  - BM25 한국어 토큰화
  - 비슷한 근거로 인한 오답 위험: 코드 일치를 사후 검증해야 함
  - 작은 평가셋: 처음에는 작은 문서와 Top-3으로 실험했는데 차이가 없어 조건을 강화했습니다. 즉 결과를 보고 조건을 조정했습니다.
