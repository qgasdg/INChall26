# 작업 보드

> 규칙([룰북](13_팀_룰북.md) §9): **착수 전 "진행 중"에 선언** (담당 T/Y + 날짜). 클로드 세션도 시작 시 이 보드 확인. 완료 시 결과 링크 남기고 이동. 공용 append-only — 남의 행은 수정하지 않는다.

## 진행 중
- [Y, 7/18] **우리 계정 제출 1회 + 체인 정합 검증**: 정적 mp4 216 생성 완료(scripts/make_static_videos.py). 맥 CPU 킷 실행은 **실패 — RAM 8GB 스왑 동결**(배치4, 4/216에서 중단) → **Kaggle T4 경로로 전환**: 번들 `~/Downloads/kaggle_static_bundle.zip`(144MB: submission_kit+input_videos+data/eval+stats) 업로드 → 킷 실행 → CSV 회수. 제출 전 예측: Public 0.302949, 로컬 Action 평균 0.4301. 덤: 샘플별 Action 분포 + best-of-2 시뮬
- [T, 7/18] E0 홀드아웃 488 생성+채점 (Kaggle, ~3.5h) — 로컬 D+V↔서버 D+V 매핑 완성
- [T, 7/18] 학교 A100 할당 신청 (크리티컬 패스)
- [T, 7/18] W2 프롬프트 작성 (Wan2.2+A100, E2 게이트·손절 기준)

## 대기 (착수 전 여기서 "진행 중"으로 올릴 것)
- 우리 계정 제출 1회 (8/17 한, 정적이면 GPU 불필요) — Y 예정
- E1 스텝 스윕 (50→30/20/12, 배치) — **T 로드맵과 중복 위험, 분담 합의 후 착수**
- 홀드아웃 보강판 생성 (선행: T의 488 에피소드 목록 입수)
- 팀원 collaborator 초대 (선행: GitHub 계정 입수)
- 룰북 v0.1 + 협업 프로토콜 T 승인 → v1.0
- Blackwell용 torch cu128 조정 (선행: E3 학습 서버 결정)

## 완료
- (7/18까지의 완료 내역은 [LOG.md](LOG.md) 참조. 이후 완료 항목은 여기로 이동: 결과 링크 필수)
- 원장: [local_lb.csv](../results/local_lb.csv) · 제출: [submissions.csv](../results/submissions.csv)
