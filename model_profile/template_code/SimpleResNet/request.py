import requests
import torch
import pickle
import base64
import time

x = torch.rand(size=(1, 3, 224, 224), dtype=None)
print(f"X Value is : {x}")
serialized_tensor = pickle.dumps(x)

url = 'https://uw5meqbwwfb2cavkepqqmonhcy0pedsr.lambda-url.ap-northeast-2.on.aws/'

request_body = {
    "body": base64.b64encode(serialized_tensor).decode('utf-8')
}
# print(request_body)

end_to_end_latency_time_start = time.time()
response = requests.post(url, json=request_body).json()
end_to_end_latency_time = time.time() - end_to_end_latency_time_start
print(response)
pickle_y = base64.b64decode(response['body'].encode('utf-8'))
y = pickle.loads(pickle_y)

print(f"Y Value is : {y}")
print(f"End-to-end Latency time is : {end_to_end_latency_time}")