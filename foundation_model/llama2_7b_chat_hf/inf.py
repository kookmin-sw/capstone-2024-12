from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import transformers
import torch

# 모델 디렉토리 설정
model_dir = "/tmp/trained_model/llama2"

# 양자화된 모델 설정
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)

# 모델 이름 설정
model_name = "NousResearch/Llama-2-7b-chat-hf"

# 모델 초기화 (state_dict 로드 전)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map="auto"
)

# state_dict 로드
model.load_state_dict(torch.load(f"{model_dir}/model_state_dict.pt"))

# 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# 테스트 입력
test_input = "I want to express the word love emotionally, but please recommend 2 sophisticated love expression sentences.\n"

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
    tokenizer=tokenizer,
)

sequences = pipeline(
    'I want to express the word love emotionally, but please recommend 2 sophisticated love expression sentences.\n',
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=1024,
)

# 답변 생성
output = pipeline(test_input, max_length=1024)

# 출력 결과
for i, sequence in enumerate(output):
    print(f"Sequence {i + 1}: {sequence['generated_text']}")