from recommender_inference_gpu_family import get_gpu_instance_family_for_inference
import json

def handler(event, context):
    try:
        param = event['queryStringParameters']
    except Exception as e:
        response = {
            'statusCode': 500,
            'errorMessage': e
        }
        return response
    
    region_name = param['region']
    try:
        family = get_gpu_instance_family_for_inference(region_name)
    except Exception as e:
        print(e)
        response = {
            'statusCode': 500,
            'errorMessage': "Family for Inference API 오류 발생"
        }
        return response

    response = {
        'statusCode': 200,
        'body': json.dumps({
            'family': family,
        })
    }
    return response