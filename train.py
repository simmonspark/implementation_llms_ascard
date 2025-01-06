# train.py

import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import argparse

from datasets import load_dataset
from transformers import AutoTokenizer

from layers.weight_gin_bbaiter import gin_bbai  # 모델 정의가 포함된 모듈

def count_parameters(model):
    """모델의 학습 가능한 파라미터 수를 계산합니다."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ====================================================
# Hugging Face 데이터셋을 사용하는 커스텀 Dataset 정의
# ====================================================
class HuggingFaceCausalLMDataset(Dataset):
    def __init__(self, tokenized_texts, block_size=2048):
        """
        Args:
            tokenized_texts (List[int]): 토큰 ID의 리스트.
            block_size (int): 고정된 시퀀스 길이.
        """
        self.block_size = block_size
        # 블록의 수 계산
        self.num_blocks = len(tokenized_texts) // block_size
        # 전체 블록 수에 맞게 토큰 리스트 자르기
        self.tokenized_texts = tokenized_texts[:self.num_blocks * block_size]

    def __len__(self):
        return self.num_blocks

    def __getitem__(self, idx):
        start_idx = idx * self.block_size
        end_idx = start_idx + self.block_size
        input_ids = torch.tensor(self.tokenized_texts[start_idx:end_idx], dtype=torch.long)
        labels = input_ids.clone()
        return {
            'input_ids': input_ids,
            'labels': labels
        }

# ====================================================
# 검증 함수 정의
# ====================================================
def validate(model, dataloader, device, loss_fn):
    """검증 데이터셋에서 모델을 평가합니다."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            input_ids = batch['input_ids'].to(device)  # Shape: (batch_size, seq_len)
            labels = batch['labels'].to(device)        # Shape: (batch_size, seq_len)

            logits = model(input_ids)  # Shape: (batch_size, seq_len, vocab_size)

            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Validation Average Loss: {avg_loss:.4f}")
    return avg_loss

# ====================================================
# 학습 함수 정의
# ====================================================
def train(args):
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 토크나이저 초기화
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # 패드 토큰이 없으면 추가

    # Hugging Face 데이터셋 로드
    raw_datasets = load_dataset(args.dataset_name, args.dataset_config, split='train')
    if 'validation' in raw_datasets.features:
        validation_split = load_dataset(args.dataset_name, args.dataset_config, split='validation')
    else:
        # 별도의 검증 세트가 없으면 훈련 데이터의 10%를 검증 세트로 사용
        validation_split = load_dataset(args.dataset_name, args.dataset_config, split='train[:10%]')
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config, split='train[10%:]')

    # 데이터 토크나이즈 함수 정의
    def tokenize_function(examples):
        return tokenizer(examples['text'], return_special_tokens_mask=True, truncation=False)

    # 훈련 데이터 토크나이즈
    tokenized_train = raw_datasets.map(tokenize_function, batched=True, remove_columns=["text"])
    # 검증 데이터 토크나이즈
    tokenized_val = validation_split.map(tokenize_function, batched=True, remove_columns=["text"])

    # 토큰 ID를 하나의 리스트로 합치기
    def concatenate_tokens(tokenized_data):
        all_input_ids = []
        for example in tqdm(tokenized_data, desc="Concatenating tokenized texts"):
            all_input_ids.extend(example['input_ids'])
        return all_input_ids

    all_train_input_ids = concatenate_tokens(tokenized_train)
    all_val_input_ids = concatenate_tokens(tokenized_val)

    # 커스텀 Dataset 생성
    train_dataset = HuggingFaceCausalLMDataset(all_train_input_ids, block_size=2048)
    val_dataset = HuggingFaceCausalLMDataset(all_val_input_ids, block_size=2048)

    # DataLoader 초기화
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True  # 모든 배치가 동일한 크기를 갖도록 보장
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False
    )

    # 모델 초기화
    model = gin_bbai()
    model.resize_token_embeddings(len(tokenizer))  # 토크나이저의 어휘 크기에 맞게 임베딩 조정
    model = model.to(device)
    if os.path.exists('./checkpoints/best_model.pt'):
        model.load_state_dict(torch.load('./checkpoints/best_model.pt'))
        print("기존의 모델 체크포인트를 불러왔습니다.")
    else:
        print("기존의 모델 체크포인트가 없습니다. 새로 학습을 시작합니다.")

    # 옵티마이저 설정
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # 스케줄러 설정 (옵션)
    if args.use_scheduler:
        from torch.optim.lr_scheduler import StepLR
        scheduler = StepLR(optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma)
    else:
        scheduler = None

    # 손실 함수 정의
    loss_fn = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -100)

    # 체크포인트 디렉토리 생성
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print(f"총 학습 가능한 파라미터 수: {count_parameters(model)}")

    # 조기 종료를 위한 변수 초기화
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = args.early_stop_patience

    # 학습 루프
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.num_epochs}")

        for step, batch in enumerate(progress_bar, 1):
            input_ids = batch['input_ids'].to(device)  # Shape: (batch_size, seq_len)
            labels = batch['labels'].to(device)        # Shape: (batch_size, seq_len)

            optimizer.zero_grad()

            logits = model(input_ids)  # Shape: (batch_size, seq_len, vocab_size)

            # 손실 계산을 위해 형태 변경
            logits_flat = logits.view(-1, logits.size(-1))  # Shape: (batch_size * seq_len, vocab_size)
            labels_flat = labels.view(-1)                  # Shape: (batch_size * seq_len)

            loss = loss_fn(logits_flat, labels_flat)

            loss.backward()

            # 그래디언트 클리핑
            clip_grad_norm_(model.parameters(), max_norm=0.8)

            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch} 완료. 평균 훈련 손실: {avg_epoch_loss:.4f}")

        # 검증 단계
        val_loss = validate(model, val_loader, device, loss_fn)

        # 스케줄러 스텝
        if scheduler:
            scheduler.step()

        # 조기 종료 및 최적 모델 저장
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # 최적 모델 저장
            best_model_path = os.path.join(args.checkpoint_dir, "best_model.pt")
            torch.save(model.state_dict(), best_model_path)
            print(f"[INFO] 최적 모델이 저장되었습니다: {best_model_path}")
        else:
            patience_counter += 1
            print(f"[INFO] 검증 손실이 개선되지 않았습니다. 현재 patience: {patience_counter}/{early_stop_patience}")

            if patience_counter >= early_stop_patience:
                print("[INFO] 조기 종료 조건이 충족되었습니다. 학습을 종료합니다.")
                break

    print("학습이 완료되었습니다.")

# ====================================================
# 메인 함수 및 인자 파서 정의
# ====================================================
def main():
    parser = argparse.ArgumentParser(description="Hugging Face 데이터셋을 사용한 GemmaForCausalLM 모델 학습")
    parser.add_argument('--dataset_name', type=str, default='wikitext', help='사용할 Hugging Face 데이터셋 이름')
    parser.add_argument('--dataset_config', type=str, default='wikitext-2-raw-v1', help='데이터셋의 구성 이름')
    parser.add_argument('--tokenizer_name', type=str, default='gpt2', help='사용할 사전 학습된 토크나이저 이름 또는 경로')
    parser.add_argument('--batch_size', type=int, default=2, help='훈련 및 검증 시 배치 크기')
    parser.add_argument('--num_epochs', type=int, default=10, help='훈련 에포크 수')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='옵티마이저의 학습률')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='옵티마이저의 가중치 감쇠')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader의 num_workers 수')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/', help='모델 체크포인트를 저장할 디렉토리')
    parser.add_argument('--use_scheduler', action='store_true', help='학습률 스케줄러 사용 여부')
    parser.add_argument('--scheduler_step_size', type=int, default=1, help='스케줄러의 스텝 사이즈')
    parser.add_argument('--scheduler_gamma', type=float, default=0.95, help='스케줄러의 감마 값')
    parser.add_argument('--early_stop_patience', type=int, default=3, help='조기 종료를 위한 patience 단계 수')

    args = parser.parse_args()

    # 학습 시작
    train(args)

if __name__ == "__main__":
    main()
