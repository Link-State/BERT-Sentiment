# BERT를 이용한 감성 분석 모델 개발
### [2024 2학기 자연어처리 과제4]

### 개발 기간
> 2024.11.28 ~ 2024.12.05

### 개발 환경
> Python 3.12.6 (venv)<br>
> Pytorch 2.4.1 + CUDA 12.4<br>
> GTX1060 6GB Desktop<br>

### 설명
+ 동기
    + 자연어처리 수업 과제
+ 기획
    + BERT를 이용하여 시스템을 개발한다.
    + 학습 데이터는 네이버의 영화 감상평을 사용한다.
    + 어절 및 토큰화는 eojeol_etri_tokenizer 모듈을 사용한다.
    + 훈련:검증:테스트는 8:1:1 비율로 나눈다.
    + 사용자가 문장을 입력하여 해당 문장의 긍정/부정을 평가한다.

#### 옵티마이저 및 하이퍼파라미터
> optimizer = AdamW <br>
> learning rate = 0.000005 <br>
> 은닉층 크기 = 768 <br>
> 배치 크기 = 32 <br>
> 최대 토큰 길이 = 60 <br>
> 드롭아웃 = 0.1 <br>
> 학습-검증을 교대로 수행하여 검증세트의 loss가 7번 연속 증가하는 경우 조기종료 <br>

#### 학습-검증 오차 그래프
<img width="301" height="222" alt="noname01" src="https://github.com/user-attachments/assets/91288f24-eb0e-4f4c-85ea-d25acf156051" />

#### 성능지표
<img width="290" height="223" alt="noname02" src="https://github.com/user-attachments/assets/eb180eb7-30e2-4ed7-ad33-63bdf7e59467" />

#### 입력 결과
<img width="601" height="471" alt="noname03" src="https://github.com/user-attachments/assets/e75f39c3-e5a6-46bc-a92e-7390e725f8e8" />

<br>

