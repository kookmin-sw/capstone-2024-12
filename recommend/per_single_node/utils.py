import boto3
import botocore
import gzip
import pandas as pd
import math
import os

from io import BytesIO
from itertools import chain

aws_access_key_id = os.getenv('RECOMMEND_ACCESS_KEY')
aws_secret_access_key = os.getenv('RECOMMEND_SECRET_ACCESS_KEY')

session = boto3.session.Session(
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
)

def get_price_df(region_name):
    s3 = session.resource('s3')
    bucket_name = 'spotlake'
    prefix = 'rawdata/aws/2024/03/10'
    bucket = s3.Bucket(bucket_name)

    file = [obj.key for obj in bucket.objects.filter(Prefix=prefix) if obj.key.endswith('.csv.gz')][-1]
    with gzip.open(BytesIO(s3.Object(bucket_name, file).get()['Body'].read()), 'rt') as gzipfile:
        df = pd.read_csv(gzipfile)
    
    df = df[df['Region'] == region_name]
    df = df.drop(['Time', 'IF'], axis=1).reset_index(drop=True)
    
    return df

def get_instance_df(region_name):
    ec2 = session.client('ec2', region_name=region_name)
    instance_types = []
    response = ec2.describe_instance_types()
    next_token = response.get('NextToken')
    instance_types.append(response['InstanceTypes'])
    while next_token is not None:
        response = ec2.describe_instance_types(NextToken=next_token)
        next_token = response.get('NextToken')
        instance_types.append(response['InstanceTypes'])

    instance_types = list(chain.from_iterable(instance_types))

    dict_data = {
        'InstanceType': [],
        'Region': [],
        'vCPU': [],
        'MemoryGiB': [],
        'GPUModel': [],
        'GPUManufacturer': [],
        'GPUCount': [],
        'TotalGPUMemoryGiB' : [],
    }

    for instance_info in instance_types:
        instance_type = instance_info['InstanceType']
        vcpu = instance_info['VCpuInfo']['DefaultVCpus']
        memory = instance_info['MemoryInfo']['SizeInMiB'] / 1024
        gpu_model = None
        gpu_manufacturer = None
        gpu_count = 0
        total_gpu_memory = 0
        
        if 'GpuInfo' in instance_info:
            total_gpu_memory = instance_info['GpuInfo']['TotalGpuMemoryInMiB'] // 1024
            for gpu in instance_info['GpuInfo']['Gpus']:
                gpu_model = gpu['Name'] + ' ' + str(gpu['MemoryInfo']['SizeInMiB'] // 1024) + "GB"
                gpu_manufacturer = gpu['Manufacturer']
                gpu_count += gpu['Count']
        
        dict_data['InstanceType'].append(instance_type)
        dict_data['Region'].append(region_name)
        dict_data['vCPU'].append(vcpu)
        dict_data['MemoryGiB'].append(memory)
        dict_data['GPUModel'].append(gpu_model)
        dict_data['GPUManufacturer'].append(gpu_manufacturer)
        dict_data['GPUCount'].append(gpu_count)
        dict_data['TotalGPUMemoryGiB'].append(total_gpu_memory)
    
    df = pd.DataFrame(dict_data)
    return df

def get_gpu_benchmark(model_name, gpu_count):
    benchmark = {
        'A10G 12GB': 223,
        'A10G 24GB': 250,
        'M60 8GB': 34.2,
        'T4g 16GB': 52.1,
        'T4 16GB': 52.1,
        'V100 16GB': 100.0,
        'V100 32GB': 111.4,
        'K80 12GB': 25.2,
        'A100 40GB': 357,
        'A100 80GB': 441,
    }
    e_rate = 1.9

    if model_name not in benchmark.keys():
        return 0.0
    
    ret = e_rate ** math.log2(gpu_count) * benchmark[model_name]
    return ret

if __name__ == "__main__":
    region_name = 'ap-northeast-2'
    price_df = get_price_df(region_name)
    instance_df = get_instance_df(region_name)

