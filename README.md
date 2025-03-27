# 현재 상황
"BAD clip"만을 대상으로 training code 구축까지는 해놓은 상태임. model state도 에포크마다 model{epoch}.pth 형식으로 저장되게 했음 아직 validation에 이용할 정확도 metric 계산하는 코드는 넣지 않음. 
앞으로 해야할 것 : 1) 데이터 train/test split이랑 2) metric 계산하는 코드 그리고 3) evaluation하는 코드 추가 필요
일단 당장 돌아가는게 중요한 것 같아서 추후에 코드 정리할 예정

아래 대로 파일 옮기기 및 터미널 실행하면 됩니다.

# 일단 시작하기 전에 Datasets의 get_item을 위한 변수명 및 반환값 추가 수정한 것 (바꿔놔서 이부분은 넘어가도 됩니다.)
1) Decoder 패키지 변경 : /Micro-Action/mar_scripts/manet/mmaction2/configs/recognition/manet/manet.py : pipline 수정 
해야할 것 : 

train_pipeline = [
    dict(type='OpenCVInit'),
    dict(type='SampleFrames', 
         clip_len=1,
         frame_interval=1,
         num_clips=8),
    dict(type='OpenCVDecode'),
    dict(type='Resize', 
         scale=(-1, 256)),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.875, 0.75, 0.66),
        random_crop=False,
        max_wh_scale_gap=1,
        num_fixed_crops=13),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label','emb'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label','emb'])
]

여기서 DecordInit, DecordDecode  => OpenCVInit, OpenCVDecode로 바꿔주세여  (https://github.com/open-mmlab/mmaction2/issues/1449)

2) np data type 수정 : /Micro-Action/mar_scripts/manet/mmaction2/mmaction/datasets/pipelines/loading.py
해야할 것 : 255줄에서 np.int => np.int32

3) 반환값 수정 : /Micro-Action/mar_scripts/manet/mmaction2/mmaction/datasets/pipelines/compose.py 
해야할 것 : 53줄에서 return data => return data['imgs'], data['label']

# 파일 경로 정보 + 라벨링 정보 입력
"/Micro-Action/mar_scripts/manet/mmaction2/data/ma52/train_list_videos.txt" 에 파일 경로 정보와 라벨링 정보 입력(bad : 0, good : 1) 

예시 형식)"Y1qDNTG9lg0_014.mp4 0"

일단 이 과정은 bad clip에 대해서는 해놓은 상태라서 안해도 코드는 돌아가니 스킵하셔도 됩니다.
=> 추후에 good clip 정보 추가하고 train/test split 필요 !

# 1. Pre-trained 가중치 파일 불러오기
다인이가 구글 드라이브에 올린거 다운 받아서 여기 git clone한 폴더 안에 집어 넣기 : https://drive.google.com/drive/u/0/folders/1NKB6G-R1yBeXEsMqyePjAMES0BeONN-L 

# 2. conda 가상환경 생성 및 필요한 패키지 설치
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

# 3. 경로 이동
cd mar_scripts/manet/mmaction2

# 4. 코드실행 : 일단 더미 데이터 넣어봤고 모델 pretrain 로드까지만 시켰음!(모델이 Sigmoid값까지 추가하게 해놨으니까 따로 안붙여도 됌)
python tools/train.py configs/recognition/manet/manet.py --epoch 5 --lr 1e-3 --batch_size 4