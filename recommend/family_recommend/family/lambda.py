from family import get_family_for_inference
import json
import boto3
import statistics
from datetime import datetime, timedelta
from decimal import Decimal

def update_spot_price(nodepool_name, instance_types, region_name):
    nodepool_name = nodepool_name.split('-')
    parameter_store_key = f"{nodepool_name[0]}_{nodepool_name[1]}_spot_price"

    ec2 = boto3.client('ec2', region_name=region_name)

    end_date = datetime.utcnow().replace(microsecond=0)
    start_date = end_date - timedelta(microseconds=1)

    response = ec2.describe_spot_price_history(
        InstanceTypes = instance_types,
        StartTime = start_date,
        EndTime = end_date,
        ProductDescriptions=['Linux/UNIX'],
        MaxResults=300
    )
    
    per_gpu_spot_prices = []
    for obj in response['SpotPriceHistory']:
        az, it, instance_os, price, timestamp = obj.values()
        # get only Linux price
        if instance_os != 'Linux/UNIX':
            continue

        price = format(Decimal(float(price)), 'f')

        response_instance_type = ec2.describe_instance_types(InstanceTypes=[it])
        gpu_count = 0
        for gpu in response_instance_type['InstanceTypes'][0]['GpuInfo']['Gpus']:
            gpu_count += gpu['Count']
        per_gpu_spot_prices.append(float(price) / gpu_count)

    avg_price = statistics.mean(per_gpu_spot_prices)
    avg_price /= 3600 # 시간당 가격이므로 초당으로 변환
    avg_price = format(Decimal(avg_price), 'f')
    print(f"parameter key : {parameter_store_key}, spot price : {avg_price}")

    # parameter store 에 값 저장
    ssm = boto3.client('ssm', region_name=region_name)
    response = ssm.put_parameter(
        Name=parameter_store_key,
        Value=str(avg_price),
        Type='String',
        Overwrite=True
    )

    if response['ResponseMetadata']['HTTPStatusCode'] != 200:
        print(f"Failed to update parameter store {parameter_store_key} : {response}")
    
    return response

# 현재 온디맨드 가격을 불러오는 것은 한 번에 여러 인스턴스를 불러올 수는 없고, 하나씩 불러와야 합니다.
# 또는 전체의 가격을 불러온 다음 필요한 인스턴스만 추출할 수 있지만, 우선 하나의 인스턴스에 대해서만
# 가격을 불러오는 방식을 for 문을 돌려 여러 번 호출하는 방식으로 구현합니다.
# 최대 25번의 boto3 호출이 일어나며 가격을 불러오려는 인스턴스의 리전과 람다의 리전이 다르면
# 시간이 오래 걸릴 수 있습니다.
def update_ondemand_price(nodepool_name, instance_types, region_name):
    nodepool_name = nodepool_name.split('-')
    parameter_store_key = f"{nodepool_name[0]}_{nodepool_name[1]}_ondemand_price"

    # pricing client 는 us-east-1 리전에서만 지원
    pricing = boto3.client('pricing', region_name='us-east-1')
    ec2 = boto3.client('ec2', region_name=region_name)

    per_gpu_ondemand_prices = []
    for instance_type in instance_types:
        response = pricing.get_products(
            ServiceCode='AmazonEC2',
            Filters=[
                {'Type': 'TERM_MATCH', 'Field': 'instanceType', 'Value': instance_type},
                {'Type': 'TERM_MATCH', 'Field': 'regionCode', 'Value': region_name},
                {'Type': 'TERM_MATCH', 'Field': 'operatingSystem', 'Value': 'Linux'},
                {'Type': 'TERM_MATCH', 'Field': 'preInstalledSw', 'Value': 'NA'},
                {'Type': 'TERM_MATCH', 'Field': 'capacitystatus', 'Value': 'Used'},
                {'Type': 'TERM_MATCH', 'Field': 'tenancy', 'Value': 'Shared'},
            ]
        )
        ondemand_price = list(list(json.loads(response['PriceList'][0])['terms']['OnDemand'].values())[0]['priceDimensions'].values())[0]['pricePerUnit']['USD']
        
        response_instance_type = ec2.describe_instance_types(InstanceTypes=[instance_type])
        gpu_count = 0
        for gpu in response_instance_type['InstanceTypes'][0]['GpuInfo']['Gpus']:
            gpu_count += gpu['Count']
        per_gpu_ondemand_prices.append(float(ondemand_price) / gpu_count)
    
    avg_price = statistics.mean(per_gpu_ondemand_prices)
    avg_price /= 3600 # 시간당 가격이므로 초당으로 변환
    avg_price = format(Decimal(avg_price), 'f')
    print(f"parameter key : {parameter_store_key}, ondemand price : {avg_price}")
    ssm = boto3.client('ssm', region_name=region_name)
    response = ssm.put_parameter(
        Name=parameter_store_key,
        Value=str(avg_price),
        Type='String',
        Overwrite=True
    )

    if response['ResponseMetadata']['HTTPStatusCode'] != 200:
        print(f"Failed to update parameter store {parameter_store_key} : {response}")

    return response

def handler(event, context):
    try:
        body = json.loads(event["body"])
    except Exception as e:
        response = {
            'statusCode': 500,
            'errorMessage': e
        }
        return response
    
    region_name = body.get('region')
    try:
        family = get_family_for_inference(region_name)
    except Exception as e:
        raise e

    try:
        # 여기서 parameter store 에 값 저장
        for nodepool_name, instance_types in family.items():
            if len(instance_types) == 0:
                continue
            update_spot_price(nodepool_name, instance_types, region_name)
            update_ondemand_price(nodepool_name, instance_types, region_name)
    except Exception as e:
        response = {
            'statusCode': 500,
            'message': f"Failed to store price at parameter store",
            'errorMessage': e
        }
        return response

    response = {
        'statusCode': 200,
        'body': json.dumps({
            'family': family,
        })
    }

    return response

# for test
if __name__ == "__main__":
    event = {
        "body": json.dumps({
            "region": "us-east-1"
        })
    }
    family = handler(event, None)
    print(family)