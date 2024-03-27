from recommender import *
import json

# input의 형태
# event { 
#     'body' : {
#         'region' : string,
#         'trainedInstance' : string (Optional),
#         'minCPUMemGiB' : int (Optional),
#         'minGPUMemGiB' : int (Optional)
#     }
# }
def lambda_handler(event, context):
    try:
        json_body = json.loads(event['body'])
    except Exception as e:
        response = {
            'statusCode': 500,
            'errorMessage': e
        }
        return response
    
    region_name = json_body['region']
    trained_instance = json_body.get('trainedInstance')
    min_cpu_mem = json_body['minCPUMemGiB'] if 'minCPUMemGiB' in json_body else 0
    min_gpu_mem = json_body['minGPUMemGiB'] if 'minGPUMemGiB' in json_body else 0

    try:
        recommend_set = get_recommend_inference(region_name, trained_instance, min_cpu_mem, min_gpu_mem)
    except Exception as e:
        response = {
            'statusCode': 500,
            'errorMessage': "추천 API 오류 발생"
        }
        return response

    recommended_instance = []
    for instance_type, spot_price, reason in recommend_set:
        obj = {
            'InstanceType': instance_type,
            'SpotPrice': spot_price,
            'Reason': reason
        }
        recommended_instance.append(obj)

    response = {
        'statusCode': 200,
        'body': json.dumps({
            'recommendations': recommended_instance,
        })
    }
    return response