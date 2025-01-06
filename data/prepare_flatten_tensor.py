import os
import json
import torch
import concurrent.futures
from transformers import AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm

# 전역 변수로 토크나이저를 선언
tokenizer = None

def initialize_tokenizer(model_id):
    global tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        print("[INFO] 토크나이저가 초기화되었습니다.")

def extract_contents_from_file(json_path: str):
    results = []
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data_info_list = data.get("data_info", [])
        for item in data_info_list:
            contents = item.get("contents", "")
            if not contents.strip():
                continue
            results.append(f"<|endoftext|>{contents}<|endoftext|>")
    except Exception as e:
        print(f"[ERROR] 파일 읽기 실패: {json_path}, 에러: {e}")
    return results

def chunk_text_generator(text_iter, chunk_size):
    current_chunk = []
    for txt in text_iter:
        current_chunk.append(txt)
        if len(current_chunk) == chunk_size:
            yield " ".join(current_chunk)
            current_chunk = []
    if current_chunk:
        yield " ".join(current_chunk)

def tokenize_chunk(chunk_text):
    global tokenizer
    try:
        encoded = tokenizer(
            chunk_text,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            padding=False  # 패딩 비활성화
        )
        input_ids = encoded["input_ids"].squeeze(0).tolist()
        attention_mask = [1] * len(input_ids)  # 패딩 없이 모든 토큰에 대해 1 설정
        return input_ids, attention_mask
    except Exception as e:
        print(f"[ERROR] 토큰화 실패: {e}")
        return None  # 실패한 경우 None 반환

def get_context_and_tokenize_in_chunks_parallel(
    base_dir: str,
    model_id: str,
    save_path: str,
    chunk_size: int = 1000,
    num_workers: int = 4
):
    json_paths = []
    for root, dirs, files in os.walk(base_dir):
        for name in files:
            if name.lower().endswith(".json"):
                json_paths.append(os.path.join(root, name))

    # Initialize tokenizer in the main process
    initialize_tokenizer(model_id)

    # Process JSON files and yield texts
    def all_texts_generator():
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers, initializer=initialize_tokenizer, initargs=(model_id,)) as executor:
            futures = [executor.submit(extract_contents_from_file, path) for path in json_paths]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Reading JSON files"):
                res = future.result()
                for text in res:
                    yield text

    # Create chunks
    chunks = chunk_text_generator(all_texts_generator(), chunk_size)

    # Tokenize chunks and collect input_ids and attention_masks
    all_input_ids = []
    all_attention_masks = []
    total_chunks = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers, initializer=initialize_tokenizer, initargs=(model_id,)) as executor:
        # Submit tokenization tasks
        tokenization_futures = {executor.submit(tokenize_chunk, chunk): chunk for chunk in chunks}
        for future in tqdm(concurrent.futures.as_completed(tokenization_futures), total=len(tokenization_futures), desc="Tokenizing Chunks"):
            result = future.result()
            if result is not None:
                input_ids, attention_mask = result
                all_input_ids.extend(input_ids)
                all_attention_masks.extend(attention_mask)
                total_chunks += 1
                # Optional: Print progress every 1000 chunks
                if total_chunks % 1000 == 0:
                    print(f"[INFO] 처리된 청크 수: {total_chunks}")

    if not all_input_ids:
        print("[DEBUG] No valid input_ids were generated during tokenization.")
        return

    # Convert lists to 1D tensors
    input_ids_tensor = torch.tensor(all_input_ids, dtype=torch.long)
    attention_mask_tensor = torch.tensor(all_attention_masks, dtype=torch.long)

    # Save the concatenated tensors
    encoded_full = {
        "input_ids": input_ids_tensor,
        "attention_mask": attention_mask_tensor
    }
    torch.save(encoded_full, save_path)
    print(f"[INFO] 최종 토큰화된 텐서를 '{save_path}'에 저장 완료.")
    return encoded_full

if __name__ == "__main__":
    BASE_DIR = "/media/sien/media/data/121.한국어 성능이 개선된 초거대AI 언어모델 개발 및 데이터/3.개방데이터/1.데이터/Training"
    MODEL_ID = "EleutherAI/gpt-neo-125m"
    SAVE_PATH = "../encoded_context.pt"

    encoded = get_context_and_tokenize_in_chunks_parallel(
        base_dir=BASE_DIR,
        model_id=MODEL_ID,
        save_path=SAVE_PATH,
        chunk_size=500,
        num_workers=4  # 워커 수 줄이기
    )
    if encoded:
        print("[INFO] 최종 input_ids shape:", encoded["input_ids"].shape)
