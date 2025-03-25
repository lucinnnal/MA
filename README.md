#아래대로 터미널 실행하면 됩니다!

# conda 가상환경 생성 및 필요한 패키지 설치
conda create --name openmmlab python=3.8 -y
#
conda activate openmmlab
#
conda install conda-forge::mmcv-full
#
conda install pytorch==1.12.1
#
conda install torchvision==0.13.1
#
conda install scipy
#
conda install einops

# 경로 이동
cd mar_scripts/manet/mmaction2

# 코드실행 : 일단 더미 데이터 넣어봤고 모델 pretrain 로드까지만 시켰음!(모델이 Sigmoid값까지 추가하게 해놨으니까 따로 안붙여도 됌)
python tools/train.py configs/recognition/manet/manet.py
