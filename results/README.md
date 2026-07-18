# results/ — 실험·제출 기록 정본

두 CSV가 ckpt 선택·최종 제출 선택의 유일한 근거다 (docs/07 §4). 매 실험·제출 직후 갱신하고 커밋한다.

## local_lb.csv — 실험별 로컬 리더보드
| 컬럼 | 의미 |
|---|---|
| exp | expNN (exp04부터, 노션 exp03까지 사용됨) |
| date, server, commit | 실행 날짜 / 실행 서버 이름 / 코드 커밋 해시 |
| stage | E0~E8 사다리 단계 |
| change | 직전 실험 대비 변경점 1줄 (단일 변수 원칙) |
| eval_action_mae | ★정본 — 킷이 eval 216샘플 전수로 실측한 Action MAE (docs/08: 홀드아웃 Action은 신뢰 불가) |
| dino_unseen, video_unseen | unseen 홀드아웃(7개 데이터셋) DINO/Video 프록시 |
| action_unseen_ref | unseen 홀드아웃 Action — 상대 비교 참고용만 |
| dino_indomain, video_indomain | in-domain 홀드아웃 프록시 |
| infer_min | eval 216샘플 추론 소요(분) — 1h 제한 감시 |

## submissions.csv — 데이콘 제출 로그 (3회/일 예산 관리)
| 컬럼 | 의미 |
|---|---|
| date, exp, file | 제출 날짜 / 실험 ID / 제출 파일명 |
| public_score | 데이콘 Public 점수 |
| eval_action_mae | 제출물의 로컬 Action 실측 → dino+video 서버 성분 역산: (public − 0.4·action)/0.6 |
| notes | 특이사항 |

제출 전 이 파일에서 **오늘 제출 횟수 확인**이 규칙.
