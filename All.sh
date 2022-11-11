#T5 모델 부터 inference

cd T5
# echo T5Model v1
# bash v1.sh \

# echo T5Model v2
# bash v2.sh \

cd ../kyElectra
echo KyElectra
bash Electra.sh \

#여기에 앙상블 조합 순서대로

cd ../ensemble
echo 앙상블 시작
bash ensemble.sh \

#여기에 폴라리티 후 최종결과 조합 순서대로

cd ../ASC
echo ASC 추출
bash pola.sh \