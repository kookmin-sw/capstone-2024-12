#!/bin/bash

ECR_URI=$1
REGION=$2
AWSCLI_PROFILE=$3

# Create ECR repository
aws ecr create-repository --repository-name deploy-streamlit --region $REGION --profile $AWSCLI_PROFILE
aws ecr create-repository --repository-name train-deploy --region $REGION --profile $AWSCLI_PROFILE
aws ecr create-repository --repository-name recommend-family --region $REGION --profile $AWSCLI_PROFILE
aws ecr create-repository --repository-name deploy-karpenter-node-pool --region $REGION --profile $AWSCLI_PROFILE
aws ecr create-repository --repository-name kubernetes-inference-deploy --region $REGION --profile $AWSCLI_PROFILE
aws ecr create-repository --repository-name job-model-profile-deploy --region $REGION --profile $AWSCLI_PROFILE
aws ecr create-repository --repository-name job-model-profile --region $REGION --profile $AWSCLI_PROFILE
aws ecr create-repository --repository-name serverless-inference-deploy --region $REGION --profile $AWSCLI_PROFILE
aws ecr create-repository --repository-name kubernetes-inference --region $REGION --profile $AWSCLI_PROFILE
aws ecr create-repository --repository-name serverless-inference --region $REGION --profile $AWSCLI_PROFILE
aws ecr create-repository --repository-name llama-inference-deploy --region $REGION --profile $AWSCLI_PROFILE
aws ecr create-repository --repository-name diffusion-inference-deploy --region $REGION --profile $AWSCLI_PROFILE
aws ecr create-repository --repository-name llama2-inference --region $REGION --profile $AWSCLI_PROFILE
aws ecr create-repository --repository-name diffusion-inference --region $REGION --profile $AWSCLI_PROFILE
aws ecr create-repository --repository-name llama2-streamlit --region $REGION --profile $AWSCLI_PROFILE
aws ecr create-repository --repository-name sdxl1-streamlit --region $REGION --profile $AWSCLI_PROFILE
aws ecr create-repository --repository-name diffusion-train-deploy --region $REGION --profile $AWSCLI_PROFILE
aws ecr create-repository --repository-name ray-cpu --region $REGION --profile $AWSCLI_PROFILE
aws ecr create-repository --repository-name ray-gpu --region $REGION --profile $AWSCLI_PROFILE

aws ecr get-login-password --region $REGION --profile $AWSCLI_PROFILE | docker login --username AWS --password-stdin $ECR_URI

cd ./automation/deploy_streamlit
docker build -t $ECR_URI/deploy-streamlit:latest .
docker push $ECR_URI/deploy-streamlit:latest
cd -

cd ./automation/deploy_train
docker build -t $ECR_URI/train-deploy:latest .
docker push $ECR_URI/train-deploy:latest
cd -

cd ./recommend/family_recommend/family
docker build -t $ECR_URI/recommend-family:latest .
docker push $ECR_URI/recommend-family:latest
cd -

cd ./automation/karpenter_node_pool_deploy
docker build -t $ECR_URI/deploy-karpenter-node-pool:latest . -f Dockerfile_x86
docker push $ECR_URI/deploy-karpenter-node-pool:latest
cd -

cd ./automation/kubernetes_inference_deploy
docker build -t $ECR_URI/kubernetes-inference-deploy:latest .
docker push $ECR_URI/kubernetes-inference-deploy:latest
cd -

cd ./automation/kubernetes_model_profiler_deploy
docker build -t $ECR_URI/job-model-profile-deploy:latest .
docker push $ECR_URI/job-model-profile-deploy:latest
cd -

cd ./model_profile/template_code
docker build -t $ECR_URI/job-model-profile:latest .
docker push $ECR_URI/job-model-profile:latest
cd -

cd ./automation/serverless_inference_deploy
./install_terraform_init.sh
./install_terraform_binary.sh
docker build -t $ECR_URI/serverless-inference-deploy:latest .
docker push $ECR_URI/serverless-inference-deploy:latest
cd -

cd ./inference/template_code
docker build -t $ECR_URI/kubernetes-inference:latest . -f Dockerfile.kubernetes_gpu
docker push $ECR_URI/kubernetes-inference:latest
docker build -t $ECR_URI/serverless-inference:latest . -f Dockerfile.lambda
docker push $ECR_URI/serverless-inference:latest
cd -

cd ./automation/llama_inference_deploy
docker build -t $ECR_URI/llama-inference-deploy:latest .
docker push $ECR_URI/llama-inference-deploy:latest
cd -

cd ./automation/diffusion_inference_deploy
docker build -t $ECR_URI/diffusion-inference-deploy:latest .
docker push $ECR_URI/diffusion-inference-deploy:latest
cd -

cd ./inference/template_code/llama
docker build -t $ECR_URI/llama2-inference:latest . -f Dockerfile.kubernetes_gpu
docker push $ECR_URI/llama2-inference:latest
cd -

cd ./inference/template_code/diffusion
docker build -t $ECR_URI/diffusion-inference:latest . -f Dockerfile.kubernetes_gpu
docker push $ECR_URI/diffusion-inference:latest
cd -

cd ./automation/deploy_streamlit/llama2
docker build -t $ECR_URI/llama2-streamlit:latest .
docker push $ECR_URI/llama2-streamlit:latest
cd -

cd ./automation/deploy_streamlit/stable_diffusion
docker build -t $ECR_URI/sdxl1-streamlit:latest .
docker push $ECR_URI/sdxl1-streamlit:latest
cd -

cd ./automation/diffusion_train_deploy
docker build -t $ECR_URI/diffusion-train-deploy:latest .
docker push $ECR_URI/diffusion-train-deploy:latest
cd -

cd ./train/ray/cpu
docker build -t $ECR_URI/ray-cpu:latest .
docker push $ECR_URI/ray-cpu:latest
cd -

cd ./train/ray/gpu
docker build -t $ECR_URI/ray-gpu:latest .
docker push $ECR_URI/ray-gpu:latest
cd -