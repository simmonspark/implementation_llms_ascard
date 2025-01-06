import torch
from torchinfo import summary
from transformers import AutoTokenizer, AutoModelForCausalLM

# 모델 및 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125m")
model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125m")
print((tokenizer.special_tokens_map))
# GPU 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 입력 텍스트
input_text = "Hello, my dog is cute"
inputs = tokenizer(input_text, return_tensors="pt").to(device)

# 학습 함수 정의
def train_model(model, inputs):
    """
    모델 학습 및 VRAM 사용량 확인
    """
    # 학습 데이터로 라벨 추가
    outputs = model(**inputs, labels=inputs["input_ids"])

    # 손실(Loss) 계산
    loss = outputs.loss
    print(f"Training Loss: {loss.item()}")

    # 역전파 (Backward)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # VRAM 사용량 출력
    if device.type == "cuda":
        vram_used = torch.cuda.memory_allocated(device) / 1e6  # MB 단위로 변환
        print(f"VRAM Used (MB): {vram_used:.2f}")

    return loss

# 텍스트 생성 함수 정의
def generate_text(model, inputs, max_length=50, top_k=50, top_p=0.9, temperature=1.0):
    """
    Top-k 및 Top-p 샘플링으로 텍스트 생성
    """
    generated_output = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        do_sample=True,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
    )
    generated_text = tokenizer.decode(generated_output[0], skip_special_tokens=True)
    print(f"Generated Text: {generated_text}")

# 모델 요약 출력
def model_summary(model, inputs):
    """
    torchinfo를 사용한 모델 요약
    """
    summary(
        model,
        input_data=inputs,
        col_names=["input_size", "output_size", "num_params", "trainable"],
        depth=1000000,
    )

# 실행
print("=== Training ===")
train_model(model, inputs)

print("\n=== Inference ===")
generate_text(model, inputs, max_length=20)

print("\n=== Model Summary ===")
model_summary(model, {"input_ids": inputs["input_ids"]})
