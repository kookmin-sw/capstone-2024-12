#!/bin/bash

ECR_URI=$1
REGION=$2
AWSCLI_PROFILE=$3

aws ecr delete-repository --force --repository-name deploy-streamlit --region $REGION --profile $AWSCLI_PROFILE
aws ecr delete-repository --force --repository-name train-deploy --region $REGION --profile $AWSCLI_PROFILE
aws ecr delete-repository --force --repository-name recommend-family --region $REGION --profile $AWSCLI_PROFILE
aws ecr delete-repository --force --repository-name deploy-karpenter-node-pool --region $REGION --profile $AWSCLI_PROFILE
aws ecr delete-repository --force --repository-name kubernetes-inference-deploy --region $REGION --profile $AWSCLI_PROFILE
aws ecr delete-repository --force --repository-name job-model-profile-deploy --region $REGION --profile $AWSCLI_PROFILE
aws ecr delete-repository --force --repository-name job-model-profile --region $REGION --profile $AWSCLI_PROFILE
aws ecr delete-repository --force --repository-name serverless-inference-deploy --region $REGION --profile $AWSCLI_PROFILE
aws ecr delete-repository --force --repository-name kubernetes-inference --region $REGION --profile $AWSCLI_PROFILE
aws ecr delete-repository --force --repository-name serverless-inference --region $REGION --profile $AWSCLI_PROFILE
aws ecr delete-repository --force --repository-name llama-inference-deploy --region $REGION --profile $AWSCLI_PROFILE
aws ecr delete-repository --force --repository-name diffusion-inference-deploy --region $REGION --profile $AWSCLI_PROFILE
aws ecr delete-repository --force --repository-name llama2-inference --region $REGION --profile $AWSCLI_PROFILE
aws ecr delete-repository --force --repository-name diffusion-inference --region $REGION --profile $AWSCLI_PROFILE
aws ecr delete-repository --force --repository-name llama2-streamlit --region $REGION --profile $AWSCLI_PROFILE
aws ecr delete-repository --force --repository-name sdxl1-streamlit --region $REGION --profile $AWSCLI_PROFILE
aws ecr delete-repository --force --repository-name diffusion-train-deploy --region $REGION --profile $AWSCLI_PROFILE
aws ecr delete-repository --force --repository-name ray-cpu --region $REGION --profile $AWSCLI_PROFILE
aws ecr delete-repository --force --repository-name ray-gpu --region $REGION --profile $AWSCLI_PROFILE
