import boto3
import botocore
import gzip
import pandas as pd
import math
import os
from datetime import datetime, timedelta

from io import BytesIO
from itertools import chain

session = boto3.session.Session()

def get_instance_df(region_name):
    ec2 = session.client('ec2', region_name=region_name)
    paginator = ec2.get_paginator('describe_instance_types')

    response_iterator = paginator.paginate()
    instance_types = []
    for response in response_iterator:
        instance_types.append(response['InstanceTypes'])
    instance_types = list(chain.from_iterable(instance_types))

    dict_data = {
        'InstanceType': [],
        'Region': [],
        'SupportedArchitectures': [],
        'vCPU': [],
        'MemoryGiB': [],
        'GPUModel': [],
        'GPUManufacturer': [],
        'GPUCount': [],
        'TotalGPUMemoryGiB' : [],
    }

    for instance_info in instance_types:
        instance_type = instance_info['InstanceType']
        supported_archs = instance_info['ProcessorInfo']['SupportedArchitectures']
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
        dict_data['SupportedArchitectures'].append(supported_archs)
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

def get_spot_price(region_name, start=None, end=None) -> tuple:
    ec2 = session.client('ec2', region_name=region_name)
    describe_args = {
        'MaxResults': 300,
        'StartTime': start,
        'EndTime': end,
    }
    while True:
        response = ec2.describe_spot_price_history(**describe_args)
        for obj in response['SpotPriceHistory']:
            az, it, instance_os, price, timestamp = obj.values()
            # get only Linux price
            if instance_os != 'Linux/UNIX':
                continue
            yield it, az, float(price), timestamp
        if not response['NextToken']:
            break
        describe_args['NextToken'] = response['NextToken']

def get_spot_price_df(region_name):
    ec2 = session.client('ec2', region_name=region_name)

    end_date = datetime.utcnow().replace(microsecond=0)
    start_date = end_date - timedelta(microseconds=1)

    spotprice_dict = {"InstanceType": [], "AZ": [], "SpotPrice": []}

    for it, az, price, timestamp in get_spot_price(region_name, start=start_date, end=end_date):
        spotprice_dict["InstanceType"].append(it)
        spotprice_dict["AZ"].append(az)
        spotprice_dict["SpotPrice"].append(price)

    spot_price_df = pd.DataFrame(spotprice_dict)

    # filter to change az-name to az-id
    az_map = dict()
    response = ec2.describe_availability_zones()

    for val in response['AvailabilityZones']:
        az_map[val['ZoneName']] = val['ZoneId']

    spot_price_df = spot_price_df.replace({"AZ": az_map})

    return spot_price_df

if __name__ == "__main__":
    region_name = 'ap-northeast-2'
    # price_df = get_price_df(region_name)
    # instance_df = get_instance_df(region_name)
    df = get_spot_price_df(region_name)
    print(df)

