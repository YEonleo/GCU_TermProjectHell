# 2022-GCU_텀프지옥
2022 국립국어원 인공 지능 언어 능력 평가 GCU_텀프지옥입니다

모델 다운 경로: https://gachonackr-my.sharepoint.com/:f:/g/personal/teryas_gachon_ac_kr/EiGJJYKoC7NBs7qj_V6chkIBcG3PUtZvOKhspakbs7WCDQ?e=ogBSAy

![image](https://user-images.githubusercontent.com/90828283/201529193-81a55103-9f5f-4f70-a71a-c47e65dcb148.png)

dataset : 데이터셋을 저장하는 파일
ensemble : ACD 앙상블 코드 및 쉘 스크립트 
models : 모델 소스코드 저장
-ASC : ASC모델 코드
-T5 : T5모델 코드
-kyElectra: kyElectra모델 코드 
saved_model : 모델 pt 파일 저장 (구글 드라이브로부터 wget 혹은 수동으로 다운로드)
saved_result : inference된 결과파일 저장

# 데이터 분석 및 추가,증강

![image](https://user-images.githubusercontent.com/90828283/201529728-c976f302-663a-4675-8369-1ca00c5faa73.png)

학습 데이터셋 polarity: positive 94.7%, neutral 3.3%, negative 2.0%

![image](https://user-images.githubusercontent.com/90828283/201529772-f6c25579-2f57-4b5b-8c24-61218bbc5a44.png)

평가 데이터셋 polarity: positive 97.3%, neutral 1.8%, negative 0.9%

<h1>추가 데이터</h1>

네이버 쇼핑 후기 crawling data:   
대회 제공 데이터셋과 비슷한 카테고리의 상품 후기를 네이버 쇼핑에서 직접 크롤링. 
대회 제공 데이터의 라벨링 기준을 분석하여 최대한 비슷한 경향성을 가지고 라벨링을 진행 후 데이터셋에 추가  

Gold data 120개:   
국립 국어원 21년 속성 기반 감성 분석 말뭉치에서 대회 제공데이터에 미포함된 데이터 약 120개  
(이들 120개 중 일부는 연예인 이름이 &name&으로 치환된 형태로, train 혹은 dev 데이터셋에 중복되어있던 걸로 확인됨,  
하지만 test 데이터셋에는 단 한개도 포함되지 않음)


# 모델 1) kykim/Electra
kykim/electra-kor-base: https://github.com/kiyoungkim1/LMkor
- 국내 주요 커머스 리뷰 1억개 + 블로그 형 웹사이트 2000만개 (75GB)
- 모두의 말뭉치 (18GB)
- 위키피디아와 나무위키 (6GB)  

<h3>input 데이터 형식</h3>

[“문장에서 속성을 찾으시오:” + sentence_form ]  

<h3>대응되는 정답 label </h3>

속성#개체에서 “#”을 “ ”로 바꾸고 [<pad>+ 정답1 + 정답2 ]  


![image](https://user-images.githubusercontent.com/90828283/201529418-202f7078-6fb1-492f-b18f-0e26b4e51bfa.png)

# 모델 2) paust/pko-t5-large
paust/pko-t5-large: https://huggingface.co/paust/pko-t5-base  
한국어 데이터 (나무위키, 위키피디아, 모두의 말뭉치)를 T5의 span corruption task를 사용해서 unsupervised learning한 사전학습 모델  

![image](https://user-images.githubusercontent.com/90828283/201529508-415252bf-d3a2-4f0e-be42-43c45c44ebc4.png)
  
<h3>input 데이터 형식</h3>

[“문장에서 속성을 찾으시오:” + sentence_form + “이 문장의 속성은 <extra_id_0> <extra_id_1> <extra_id_2> <extra_id_3><extra_id_4>이다”]

<h3>대응되는 정답 label </h3>
<extra_id_>토큰을 총 5가지로 한 이유는 한 문장에서 최대로 생성할 수 있는 정답 라벨을 5개로 설정하여 추가하였다.  
이에 따라 T5모델에서 정답이 2가지 일 경우 아래 예시와 같이 입력하였다.

[<pad>+<extra_id_0>+정답1+<extra_id_1>+정답2+<extra_id_2>+<extra_id_3>+<extra_id_4>]
  
<extra_>

![image](https://user-images.githubusercontent.com/90828283/201529530-0dff0751-fd86-42f4-9407-1efc3f8fb3c8.png)

# 모델 Ensemble

Voting 통해 ACD (entity 분류) 후, 최종적으로 ensemble된 ACD에 ASC를 inference하여 최종본으로 제출  
(본 팀은 이번 대회의 task는 결국 entity classficiationd에서 당락이 결정될 것으로 판단했다)

![image](https://user-images.githubusercontent.com/90828283/201529666-6961ef72-08a4-4585-9464-a5446a802b48.png)



대응하는 정답라벨은 속성#개체에서 “#”을 “ ”로 바꾸고 [<pad>+ 정답1 + 정답2 ] 형식으로 넣었다.
