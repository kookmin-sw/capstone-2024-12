## Team SSKAI
### 최적의 GenAIOps 환경을 제공하는 플랫폼

---
### 최종 발표자료 <a href="https://docs.google.com/presentation/d/1ETs9I7u6Vvl6uC1YgkyVSpkHVTacSOPC/edit?usp=sharing&ouid=110354635145523913913&rtpof=true&sd=true">#</a>


### 수행결과보고서 <a href="https://docs.google.com/document/d/181B-_Gd2sWzUxvU5rz0etcdNCgc9jKtf/edit?usp=sharing&ouid=110354635145523913913&rtpof=true&sd=true">#</a>
---

### 산학 협력 기업
<a href="http://aws.amazon.com/what-is-cloud-computing"><img src="https://github.com/kookmin-sw/capstone-2024-12/blob/master/.github/assets/powered_by_aws.png?raw=true" alt="Powered by AWS Cloud Computing"></a>

Amazon Web Service (AWS)는 컴퓨팅, 스토리지, 데이터베이스, 분석, 네트워킹, 모바일, 개발자 도구, 관리 도구, IoT, 보안 및 엔터프라이즈 애플리케이션을 비롯한 광범위한 글로벌 클라우드 기반 제품을 제공합니다.  온디맨드로 몇 초 만에 이용할 수 있으며 요금이 부과됩니다. pay-as-you-go 데이터 웨어하우징에서 배포 도구, 디렉터리, 콘텐츠 전송에 이르기까지 200개 AWS 이상의 서비스를 이용할 수 있습니다. 초기 고정 비용 없이 새 서비스를 신속하게 프로비저닝할 수 있습니다. 이를 통해 공공 부문의 기업, 신생 기업, 중소기업 및 고객이 변화하는 비즈니스 요구 사항에 신속하게 대응하는 데 필요한 구성 요소에 액세스할 수 있습니다.

---

### 1. 프로젝트 소개
Team SSKAI(Serverless Spot Kubernetes AI)는 최적의 클라우드 컴퓨팅 환경에서의 AI를 제공한다는 의미로 Sky에서 본따 지어졌습니다.

SSKAI에서 제작하는 `최적의 GenAIOps 환경을 제공하는 플랫폼`은 기존에 존재하는 MLOps/GenAIOps Solution의 단점과 불편한 점을 보완하여 사용자가 손쉽게 머신러닝 모델을 개발하고 배포할 수 있도록 하는 것을 목표로 합니다.


#### 주요 기능
1. 이용이 편리한 GenAIOps Pipeline 제공
    - 사용자는 플랫폼을 이용할 수 있는 웹 서비스에 접근하여 학습용 데이터셋을 업로드하고, Foundation Model을 선택하거나 본인이 설계한 ML 모델을 업로드 하는 것을 통해 Fine-tuning, Training, Deploy, Monitoring을 손쉽게 이용할 수 있다.
2. 최적의 비용 및 성능 인프라 제공
    -  모델의 크기, 모델의 연산자 수, 모델에 필요한 성능 등을 종합적으로 판단해 최적의 인프라를 선정하는 알고리즘을 개발하여 해당 모델에 맞는 최적의 비용 및 성능을 가진 인프라를 제공한다. 따라서 사용자는 비용 및 성능을 최적화 하기위해 별도로 고려하지 않아도 된다.
    -  이 때, 최적의 인프라 선정 결과에 따라 유지비용이 전혀 들지 않는 서버리스 컴퓨팅 환경에서 추론 환경이 지속적으로 가동되도록 하거나, 학습이나 Fine-Tuning 시에는 스팟 과금 정책을 활용하여 사용자는 최대 90% 저렴한 환경에서 작업을 수행하거나, 동일한 가격 대비 더 뛰어난 성능의 컴퓨팅 자원에서 작업을 수행하도록 할 수 있다.
3. 지속적 Monitoring 및 CI/CD 환경 제공
    - 학습, Fine-tuning과 추론 서비스 배포 후 운영 중 지속적인 비용 Monitoring을 통해 사용자는 실시간으로 사용한 비용을 확인할 수 있다.
    - 이를 통해 사용자는 과금이 많이 발생할 경우, 현재 운영중인 인프라의 배포 수준을 재검토할 수 있게된다. 또한,모델을 지속적으로 학습 후 배포하여 모델 개발 및 운영의 안정성, 지속성을 손쉽게 확보할 수 있다.


#### 사용 기술 스택
<img src='https://github.com/kookmin-sw/capstone-2024-12/blob/master/.github/assets/tech_stack.png?raw=true' width=640px/>

### 2. 소개 영상
<iframe width="560" height="315" src="https://www.youtube.com/embed/KEHOGjD0qaQ?si=_F6H0m8v_3_2KZqM" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>

### 3. 팀 소개

| 송무현 | 김규민 | 김유림 |
|:------:|:------:|:------:|
|<img src='https://github.com/kookmin-sw/capstone-2024-12/blob/master/.github/assets/people/moohyun.jpg?raw=true' width=150px height=150px alt="mhsong"/>|<img src='https://github.com/kookmin-sw/capstone-2024-12/blob/master/.github/assets/people/kmkim.png?raw=true' width=150px height=150px alt="kmkim"/>|<img src='https://github.com/kookmin-sw/capstone-2024-12/blob/master/.github/assets/people/yrkim.jpg?raw=true' width=150px height=150px alt="yrkim"/>|
|20203085|20191555|20203043|
|mhsong@kookmin.ac.kr|okkimok123@kookmin.ac.kr|belbet01@kookmin.ac.kr|
|팀장 및 진행 총괄, 추론 아키텍처 설계|분산 학습 아키텍처 설계|스팟 환경에서 안정성 있는 분산 학습 구현|
|[@mh3ong](https://github.com/mh3ong)|[@QueueMin](https://github.com/QueueMin)|[@Kim-Yul](https://github.com/Kim-Yul)|

| 문지훈 | 박정명 | 정승우 |
|:------:|:------:|:------:|
|<img src='https://github.com/kookmin-sw/capstone-2024-12/blob/master/.github/assets/people/jhmoon.png?raw=true' width=150px height=150px alt="jhmoon"/>|<img src='https://github.com/kookmin-sw/capstone-2024-12/blob/master/.github/assets/people/jmpark.jpg?raw=true' width=150px height=150px alt="jmpark"/>|<img src='https://github.com/kookmin-sw/capstone-2024-12/blob/master/.github/assets/people/swjeong.png?raw=true' width=150px height=150px alt="swjeong"/>|
|20213347|20191598|20191664|
|answlgns2056@kookmin.ac.kr|jmyeong012@kookmin.ac.kr|seungwoo1124@kookmin.ac.kr|
|기능 배포 자동화 구현|프론트엔드/백엔드|최적의 비용 아키텍처 선출 알고리즘 제작|
|[@jhM00n](https://github.com/jhM00n)|[@j-myeong](https://github.com/j-myeong)|[@seungwoo1124](https://github.com/seungwoo1124)|


### 4. 배포 방법

사용자는 이 오픈 소스를 사용하여 본인의 클라우드 계정에 직접 플랫폼을 구축하여 최적의 가격의 머신러닝 환경을 사용할 수 있다.

클라우드 계정에 직접 배포하는 방법은 다음과 같다.

1. 필요 패키지 설치
    ```bash
    # macOS
    brew install terraform awscli node@20
    brew install --cask docker
    npm install -g yarn
    # Linux (Ubuntu)
    sudo apt install terraform awscli docker.io -y
    npm install -g yarn
    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
    nvm install 20
    # Linux (Redhat)
    sudo dnf install terraform awscli docker.io -y
    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
    nvm install 20
    npm install -g yarn
    # Windows
    choco install terraform awscli docker-desktop
    choco install nodejs-lts --version="20.13.1"
    npm install -g yarn
    ```
2. AWS 계정 권한 설정
    ```bash
    # aws configure 명령을 통해 Access Key, Secret Access Key를 지정한다.
    aws configure
    AWS Access Key ID [None]: AKIAIO**********
    AWS Secret Access Key [None]: wJalrXU**************
    Default region name [None]: ap-northeast-2
    Default output format [None]: json
    ```
3. Github Repository Clone 및 플랫폼 배포
    ```bash
    git clone https://github.com/kookmin-sw/capstone-2024-12.git
    cd ./capstone-2024-12
    # 약 40분 가량 소요됩니다.
    python3 sskai_execute.py

    Enter REGION: us-east-1
    Enter AWSCLI PROFILE: default
    Enter MAIN SUFFIX: SSKAI
    0. Exit this operation.
    1. Build and Deploy container image.
    2. Deploy SSKAI Infrastructure.
    Enter the number: 1
    You can build only with x86/64 architecture and Unix kernel (Mac/Linux).

    Enter the type of operation (create/delete): create
    Building and Deploying in progress.
    It takes about 15 minutes.
    Processing...

    Complete

    python3 sskai_execute.py
    0. Exit this operation.
    1. Build and Deploy container image.
    2. Deploy SSKAI Infrastructure.
    Enter the number: 2
    Enter the type of operation (create/delete): create
    It takes about 20 minutes.
    Processing...

    Complete
    ```
4. 배포된 플랫폼 제거
    ```bash
    cd ./capstone-2024-12
    # 약 25분 가량 소요됩니다.
    python3 sskai_execute.py

    0. Exit this operation.
    1. Build and Deploy container image.
    2. Deploy SSKAI Infrastructure.
    Enter the number: 2
    Enter the type of operation (create/delete): delete
    It takes about 20 minutes.
    Processing...

    Complete

    python3 sskai_execute.py

    0. Exit this operation.
    1. Build and Deploy container image.
    2. Deploy SSKAI Infrastructure.
    Enter the number: 1
    Enter the type of operation (create/delete): delete
    It takes about 5 minutes.
    Processing...

    Complete
    ```

### 5. 결과물 예시
![webpage-dashboard](https://github.com/kookmin-sw/capstone-2024-12/blob/master/.github/assets/webpage-dashboard.png?raw=true)
![webpage-data](https://github.com/kookmin-sw/capstone-2024-12/blob/master/.github/assets/webpage-data.png?raw=true)
![webpage-model](https://github.com/kookmin-sw/capstone-2024-12/blob/master/.github/assets/webpage-model.png?raw=true)
![webpage-train](https://github.com/kookmin-sw/capstone-2024-12/blob/master/.github/assets/webpage-train.png?raw=true)
![webpage-inference](https://github.com/kookmin-sw/capstone-2024-12/blob/master/.github/assets/webpage-inference.png?raw=true)

---
### 포스터
![poster](https://github.com/kookmin-sw/capstone-2024-12/blob/master/.github/assets/poster.jpg?raw=true)