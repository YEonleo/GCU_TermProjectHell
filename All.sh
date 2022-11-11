#T5 모델 부터 inference

cd T5
echo T5Model v1
bash v1.sh \

echo T5Model v2
bash v2.sh \

cd ../kyElectra
echo KyElectra
bash Electra.sh \

cd ../ensemble
echo 앙상블 시작
bash ensemble.sh \

cd ../ASC
echo ASC 추출
bash pola.sh \
