from recommender_inference_cpu_family import get_cpu_instance_family_for_inference
import json

def handler(event, context):
    try:
        param = json.loads(event["body"])
    except Exception as e:
        response = {
            'statusCode': 500,
            'errorMessage': e
        }
        return response
    
    region_name = param.get('region')
    try:
        family = get_cpu_instance_family_for_inference(region_name)
    except Exception as e:
        raise e

    response = {
        'statusCode': 200,
        'body': json.dumps({
            'family': family,
        })
    }
    return response

# for test
if __name__ == "__main__":
    family = get_cpu_instance_family_for_inference("ap-northeast-2")
    print(family)