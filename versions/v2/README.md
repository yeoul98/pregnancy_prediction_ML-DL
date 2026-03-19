# Version 2: Domain Feature Engineering (Best LB)

이 버전은 도메인 지식을 활용한 **피처 엔지니어링**에 집중하여 실전 리더보드(Public LB)에서 가장 높은 성능을 기록한 모델입니다.

## Key Strategies
- **나이 데이터 수치화**: 범주형 '시술 당시 나이'를 수치형으로 변환 및 '40세 이상' 고령 여부 피처 생성
- **배아 관련 지표**: 수정률, 이식률, 저장률, ICSI 비율 등 난자와 배아의 상태를 나타내는 파생 변수 대거 추가
- **앙상블**: CatBoost, LightGBM, XGBoost의 3종 모델에 대해 `scipy.optimize`를 이용한 최적 가중치 앙상블 적용

## Files
- `pipeline_v2.py`: 전체 학습 및 예측 코드
- `submission_v2.csv`: 최종 제출 파일
