# import os
# import json
# import paddle
import cv2
import numpy as np
from paddleocr import PaddleOCR
from tqdm import tqdm
from openai import OpenAI
from django.conf import settings


OUR_MASK_TOKEN = "__"
API_KEY = settings.SECRETS.get("OPENAI_API_KEY", None)
REC_INFERENCE_MODEL_DIR = settings.OCR_MODEL_DIR

GLOBAL_OCR_ENGINE = None
GLOBAL_SPACING_MODEL = None
GLOBAL_OPENAI_CLIENT = None


def initialize_set(rec_model_dir: str):
    global GLOBAL_OCR_ENGINE, GLOBAL_SPACING_MODEL, GLOBAL_OPENAI_CLIENT

    print(f"- OCR 엔진 초기화 시작 (모델 경로: {rec_model_dir})")
    try:
        GLOBAL_OCR_ENGINE = PaddleOCR(
            text_recognition_model_name="korean_PP-OCRv5_mobile_rec",
            text_recognition_model_dir=rec_model_dir,
            use_doc_orientation_classify=False,  # 문서 회전 분류
            use_textline_orientation=False,
            # use_doc_unwarping=False,            # 문서 평탄화
            # text_rec_score_thresh=0.1,      # 텍스트 인식 신뢰도 임계값
        )
        print("- PaddleOCR 초기화 완료.")

        # 3. OpenAI 클라이언트 초기화
        GLOBAL_OPENAI_CLIENT = OpenAI(api_key=API_KEY)
        print("- OpenAI 클라이언트 초기화 완료.")

    except Exception as e:
        print(f" 서비스 초기화 오류: {e}")
        raise RuntimeError(f"OCR 서비스 초기화 실패: {e}")


def prepare_ocr_image(
        img_input,
        target_long_side: int = 1600,
        pad_to_multiple: int = 32,
        border_ratio: float = 0.05,
        use_clahe: bool = True
):
    """
    OCR에 넣기 전에 일기장 사진을:
    1) 사방으로 여백을 조금 늘려서(패딩) 모서리 글자가 잘리지 않게 하고
    2) 긴 변 기준으로 크기를 줄이고(리사이즈)
    3) (선택) CLAHE로 대비를 올려서 캡쳐 이미지처럼 선명하게 만든 다음
    4) (선택) 높이/너비를 pad_to_multiple(예: 32)의 배수로 패딩하는 함수.

    Parameters
    ----------
    img_input : str 또는 np.ndarray
        - str        : 이미지 파일 경로 (예: "/content/diary.jpg")
        - np.ndarray : 이미 cv2.imread 등으로 읽어온 BGR 이미지

    target_long_side : int, default=1600
        - 이미지의 긴 변(가로/세로 중 큰 값)이 이 길이를 넘으면,
          비율을 유지하면서 이 길이까지 줄인다.
        - 너무 큰 사진을 통일된 크기로 맞춰서 속도/인식률 안정화.

    pad_to_multiple : int, default=32
        - 최종 높이/너비를 이 값의 배수로 흰색 패딩한다.
        - 0 또는 None이면 이 단계는 생략.

    border_ratio : float, default=0.05 (5%)
        - 원본 이미지의 높이/너비의 이 비율만큼 사방에 여백을 더 준다.
        - 예: 2000x1500 이미지에서 border_ratio=0.05 → 위/아래/좌/우에 약 100px 정도 흰색 여백 추가.
        - 사람들이 글자에 딱 맞춰서 찍은 경우, 모서리 글자 앞뒤로 약간 여유를 줘서 detector가 잘 잡게 하려는 목적.

    use_clahe : bool, default=True
        - True면 그레이스케일로 변환한 뒤 CLAHE로 로컬 대비를 올린 후 다시 BGR로 변경.
        - 흐리게 찍힌 사진을 “캡쳐한 것처럼” 더 또렷하게 만들어주는 효과.

    Returns
    -------
    out_img : np.ndarray
        - 전처리된 BGR 이미지 (cv2 형식)
        - 바로 `ocr.ocr(out_img)` 또는 `ocr.predict(out_img)`에 넣으면 된다.
    """
    # 1) 입력 타입 처리: 경로면 읽고, 배열이면 복사해서 사용
    print(f"- OCR 전처리 시작")

    if isinstance(img_input, str):
        img = cv2.imread(img_input)
        if img is None:
            raise ValueError(f"이미지를 읽을 수 없습니다: {img_input}")
    elif isinstance(img_input, np.ndarray):
        img = img_input.copy()
    else:
        raise TypeError("img_input은 파일 경로(str) 또는 이미지 배열(np.ndarray) 이어야 합니다.")

    h, w = img.shape[:2]

    # 2) 사방으로 여백(border) 추가 (흰 바탕으로 패딩)
    if border_ratio and border_ratio > 0:
        top = int(h * border_ratio)
        bottom = int(h * border_ratio)
        left = int(w * border_ratio)
        right = int(w * border_ratio)

        # 일기장 배경이 보통 흰색이므로 255(흰색)으로 패딩
        img = cv2.copyMakeBorder(
            img,
            top, bottom, left, right,
            borderType=cv2.BORDER_CONSTANT,
            value=(255, 255, 255)
        )
        h, w = img.shape[:2]

    # 3) 긴 변 기준으로 스케일 계산 (너무 크면 줄이기)
    long_side = max(h, w)
    scale = 1.0
    if long_side > target_long_side:
        scale = target_long_side / long_side

    # 4) 비율 유지 리사이즈
    if scale != 1.0:
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]

    # 5) (옵션) CLAHE로 대비 올리기 → 캡쳐 느낌에 조금 더 가깝게
    if use_clahe:
        # BGR → Gray
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # CLAHE 생성 (파라미터는 필요에 따라 조정 가능)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_eq = clahe.apply(gray)
        # 다시 BGR로
        img = cv2.cvtColor(gray_eq, cv2.COLOR_GRAY2BGR)
        h, w = img.shape[:2]

    # 6) (옵션) 높이/너비를 32의 배수로 맞추기 위해 흰색 패딩
    if pad_to_multiple and pad_to_multiple > 1:
        new_h = int(np.ceil(h / pad_to_multiple) * pad_to_multiple)
        new_w = int(np.ceil(w / pad_to_multiple) * pad_to_multiple)

        canvas = np.full((new_h, new_w, 3), 255, dtype=np.uint8)  # 흰 배경
        canvas[:h, :w] = img
        img = canvas

    return img


def reorder_page_lines(ocr_page, y_thresh_ratio=0.7):
    """
    PaddleOCR 최신 버전(paddlex 스타일)의 한 페이지 결과(dict)에 대해
    줄(line) 단위로 y 클러스터링을 하고,
    같은 줄 안에서는 x좌표로 정렬하는 함수.

    Parameters
    ----------
    ocr_page : dict
        result[0] 에 해당하는 딕셔너리.
        필수 키: 'rec_polys', 'rec_texts', 'rec_scores'
        (보통 너가 print한 구조 그대로)

    y_thresh_ratio : float
        줄로 묶을 때 사용할 threshold 계수.
        평균 bbox 높이 * y_thresh_ratio 만큼의 y 차이는 같은 줄로 취급.

    Returns
    -------
    reordered_page : dict
        rec_texts / rec_scores / rec_polys / rec_boxes 등이
        줄/단어 순서대로 재정렬된 새 dict.
    order_indices : list[int]
        원래 인덱스 기준으로 어떤 순서로 재정렬됐는지 인덱스 리스트.
    """
    print("- OCR 페이지 줄 정렬 시작")
    if ocr_page is None:
        raise RuntimeError("ocr_page는 None이 될 수 없습니다.")

    rec_polys = ocr_page.get("rec_polys", [])
    rec_texts = ocr_page.get("rec_texts", [])
    rec_scores = ocr_page.get("rec_scores", [])

    n = len(rec_texts)
    if n == 0:
        # 아무것도 없으면 그대로 반환
        return ocr_page, list(range(0))

    enriched = []
    heights = []

    # 1) 각 텍스트 박스의 중심좌표(x_center, y_center)와 높이(height) 계산
    for i in range(n):
        poly = np.array(rec_polys[i], dtype=float)  # (4,2) 가정
        xs = poly[:, 0]
        ys = poly[:, 1]

        x_min, x_max = float(xs.min()), float(xs.max())
        y_min, y_max = float(ys.min()), float(ys.max())

        x_center = (x_min + x_max) / 2.0
        y_center = (y_min + y_max) / 2.0
        height = y_max - y_min

        heights.append(height)

        enriched.append({
            "idx": i,
            "x_center": x_center,
            "y_center": y_center,
            "height": height
        })

    mean_height = float(np.mean(heights))
    line_thresh = mean_height * y_thresh_ratio  # 같은 줄로 볼 y 차이 한계

    # 2) y_center 기준으로 전체 정렬 (위에서 아래로)
    enriched_sorted = sorted(enriched, key=lambda x: x["y_center"])

    # 3) y_center 차이를 보면서 line 클러스터링
    lines = []
    current_line = []
    current_line_y = None

    for item in enriched_sorted:
        y_c = item["y_center"]

        if current_line_y is None:
            # 첫 줄 시작
            current_line = [item]
            current_line_y = y_c
            continue

        if abs(y_c - current_line_y) <= line_thresh:
            # 같은 줄로 묶기
            current_line.append(item)
            current_line_y = np.mean([e["y_center"] for e in current_line])
        else:
            # 새로운 줄로 분리
            lines.append(current_line)
            current_line = [item]
            current_line_y = y_c

    if current_line:
        lines.append(current_line)

    # 4) 각 줄 안에서는 x_center 기준으로 정렬 (왼쪽→오른쪽)
    for i, line in enumerate(lines):
        lines[i] = sorted(line, key=lambda x: x["x_center"])

    # 5) 줄 순서(위→아래), 줄 안 순서(좌→우)대로 최종 인덱스 순서 생성
    order_indices = []
    for line in lines:
        for item in line:
            order_indices.append(item["idx"])

    # 6) 인덱스 순서를 이용해서 rec_texts / rec_scores / rec_polys 재정렬
    ordered_texts = [rec_texts[i] for i in order_indices]
    ordered_scores = [rec_scores[i] for i in order_indices]
    ordered_polys = [rec_polys[i] for i in order_indices]

    # 7) rec_boxes, dt_polys 같은 것도 있으면 같이 재정렬 (선택)
    reordered_page = dict(ocr_page)  # shallow copy
    reordered_page["rec_texts"] = ordered_texts
    reordered_page["rec_scores"] = ordered_scores
    reordered_page["rec_polys"] = ordered_polys

    if "rec_boxes" in ocr_page:
        rec_boxes = np.array(ocr_page["rec_boxes"])
        reordered_page["rec_boxes"] = rec_boxes[order_indices]

    if "dt_polys" in ocr_page and len(ocr_page["dt_polys"]) == n:
        dt_polys = ocr_page["dt_polys"]
        reordered_page["dt_polys"] = [dt_polys[i] for i in order_indices]

    return reordered_page, order_indices


def mask_low_confidence_tokens(rec_texts, rec_scores, threshold=0.8, mask_token="--", join_with_space=True):
    """
    OCR 인식 결과(rec_texts, rec_scores)를 받아서
    score가 threshold 미만인 토큰을 mask_token(기본값 '--')으로 바꿔주는 함수.

    Parameters
    ----------
    rec_texts : list[str]
        PaddleOCR가 인식한 텍스트 목록 (예: reordered_page["rec_texts"])

    rec_scores : list[float]
        각 텍스트에 대한 인식 점수 목록 (예: reordered_page["rec_scores"])

    threshold : float, default=0.8
        이 값보다 작은 score를 가진 텍스트는 mask_token으로 마스킹함.
        예: threshold=0.8 이면, score < 0.8 인 항목이 '--'로 바뀜.

    mask_token : str, default="----"
        마스킹에 사용할 문자열.

    join_with_space : bool, default=True
        True  -> 마스킹된 토큰들을 공백으로 join해서 하나의 문자열로 반환.
        False -> 마스킹된 토큰 리스트(각 토큰 별)를 그대로 반환.

    Returns
    -------
    masked_output : str 또는 list[str]
        join_with_space=True  -> "옷을 사러 오랜만에외출을했다. ---- 12시부터 ..."
        join_with_space=False -> ["옷을 사러", "오랜만에외출을했다.", "----", "12시부터", ...]
    """
    print("- 낮은 신뢰도 토큰 마스킹 시작")
    if rec_texts is None or rec_scores is None:
        raise RuntimeError("rec_texts와 rec_scores는 None이 될 수 없습니다.")

    assert len(rec_texts) == len(rec_scores), "rec_texts와 rec_scores 길이가 다릅니다."

    masked_tokens = []

    for text, score in zip(rec_texts, rec_scores):
        # 앞뒤 공백 제거 (안 해도 되지만 깔끔하게 하기 위해)
        t = text.strip()

        # 점수가 threshold 미만이면 마스킹
        if score < threshold:
            masked_tokens.append(mask_token)
        else:
            masked_tokens.append(t)

    if join_with_space:
        return " ".join(masked_tokens)
    else:
        return masked_tokens


def extract_my_data(img_path,
                    line_y_thresh_ratio=0.7,
                    score_threshold=0.8,
                    mask_token="<--->"):
    """
    1) ocr.predict(img_path)로 결과 얻고
    2) 각 페이지별로 줄 정렬(reorder_page_lines) 수행
    3) 정렬된 텍스트(rec_texts, rec_scores)를 사용해
       - original_text  : 정렬만 된 원본 텍스트
       - masked_text    : score_threshold 미만 토큰을 mask_token으로 마스킹한 텍스트
    두 가지를 만들어서 반환하는 함수.

    반환값:
        original_text_all : 모든 페이지 텍스트(정렬만 한 것)를 공백으로 이어붙인 문자열
        masked_text_all   : 마스킹까지 적용된 최종 문자열
    """
    print("- 사용자 데이터 추출 시작")
    if GLOBAL_OCR_ENGINE is None:
        raise RuntimeError("ocr_not_init")

    result = GLOBAL_OCR_ENGINE.predict(img_path)
    all_original_texts = []  # 정렬만 된 텍스트
    all_masked_texts = []  # 마스킹 적용 텍스트

    for idx, res in enumerate(result):
        # predict()가 dict를 반환하는 형태(지금 네가 보여준 구조)라고 가정
        if not isinstance(res, dict) or "rec_texts" not in res:
            print(f"[경고] 결과 #{idx + 1}는 예상한 dict(rec_texts 포함) 형태가 아닙니다. 건너뜁니다.")
            continue

        # 1) 줄 정렬 수행
        reordered_page, order_indices = reorder_page_lines(
            res,
            y_thresh_ratio=line_y_thresh_ratio
        )

        rec_texts = reordered_page["rec_texts"]
        rec_scores = reordered_page["rec_scores"]

        # 2) 정렬된 원본 텍스트 (마스킹 전)
        page_original_text = " ".join([t.strip() for t in rec_texts])
        all_original_texts.append(page_original_text)

        # 3) score 기반 마스킹 적용
        page_masked_text = mask_low_confidence_tokens(
            rec_texts=rec_texts,
            rec_scores=rec_scores,
            threshold=score_threshold,
            mask_token=mask_token,
            join_with_space=True
        )
        all_masked_texts.append(page_masked_text)

    # 여러 페이지일 경우를 대비해 페이지별 텍스트를 공백으로 이어붙임
    original_text_all = " ".join(all_original_texts)
    masked_text_all = " ".join(all_masked_texts)

    return original_text_all, masked_text_all


"""
masked_text  : score 기반으로 "mask_token"가 들어간 문장 (mask_low_confidence_tokens 결과)
original_text: 순서만 정렬된 원본 OCR 텍스트 (마스킹 전, 노이즈 포함)
# 11. 문맥을 보고 바꾸는 거는 절대 하지마라. 비슷한 발음 철자 위주로 수정해라.
역할:
- masked_text를 "골격"으로 그대로 유지한다.
- "mask_token"는 절대 손대지 않는다.
- '어루만점' → '어루만짐', '밥을 먼고' → '밥을 먹고' 처럼
  발음/철자가 비슷한 "오타"만 고친다.
- 단어를 새로 만들어 추가하거나, 명사를 형용사/구문으로 바꾸는 의역은 금지.
"""


def fix_text_to_gpt(masked_text: str, original_text: str, mask_token: str) -> str:
    print("- GPT 기반 로컬 오타 교정 시작")
    if GLOBAL_OPENAI_CLIENT is None:
        raise RuntimeError("OpenAI 클라이언트가 초기화되지 않았습니다.")

    prompt = f"""
    너는 한국어 손글씨 일기의 OCR 결과를 "로컬 오타 교정"만 하는 도우미야.

    입력으로 두 가지 텍스트가 주어진다.

    [1] masked_text
    - 신뢰도가 낮은 단어를 "{mask_token}"로 마스킹한 문장이다.
    - 문장의 구조, 단어 순서, 문장 개수는 이 텍스트를 기준으로 삼아야 한다.

    masked_text:
    ---
    {masked_text}
    ---

    [2] original_text
    - 마스킹 전 원본 OCR 결과이다.
    - 맞춤법, 띄어쓰기, 글자 인식 오류가 많다.
    - 하지만 어떤 단어가 어떤 단어의 오타인지를 추측할 때 참고용으로만 사용해라.

    original_text:
    ---
    {original_text}
    ---

    교정 규칙 (중요):

    1. "{mask_token}"는 절대로 수정하거나 채우거나 삭제하지 마라.
       - '{mask_token}' 안에 어떤 단어가 들어갈지 추측하지 말고,
       - 출력에서도 반드시 "{mask_token}" 그대로 남겨라.

    2. 너의 역할은 "오타/띄어쓰기/맞춤법 교정"에만 한정된다.
       - '끈임업이' → '끊임없이'
       - '밥을 먼고' → '밥을 먹고'
       - '어루만점' → '어루만짐'
       이런 식으로, **발음이나 철자가 매우 비슷해서**
       사람이 보면 "이 단어의 오타구나"라고 명확히 알 수 있는 경우만 고쳐라.

    3. 절대 하면 안 되는 것:
       - 단어를 길게 풀어서 다른 표현으로 바꾸지 마라.
         예: '어루만점'을 '어루만져본 것에는'처럼
             명사(어루만짐)를 긴 구나 문장(어루만져본 것에는)으로 바꾸지 마라.
       - 단어의 품사나 문장 구조를 바꾸지 마라.
         - 명사는 최대한 명사로, 동사는 동사로, 형용사는 형용사로 유지해라.
       - 새로운 의미나 뉘앙스를 추가하지 마라.
       - 문장에 새로운 단어/구를 덧붙이거나 삭제하지 마라.
       - 문장 순서를 바꾸지 마라.

    4. 허용되는 수정의 범위:
       - 글자 수가 1~2개 정도 다른 오타를 올바른 표기법으로 바꾸는 것.
       - 붙여 쓰기/띄어 쓰기를 올바르게 바꾸는 것.
       - 조사(은/는, 이/가, 을/를 등)를 자연스럽게 맞추는 것.
       - 문장 부호(마침표, 쉼표 등)를 추가/수정하는 것.
       이 범위를 넘는 "의역"이나 "재작성"은 하지 마라.

    5. 완전히 의미를 알 수 없는 이상한 문자열:
       - '이롸러잗ㄱㅈ뱌'처럼 어떤 단어인지 전혀 짐작이 안 되는 경우에는
         그 부분을 고치려고 하지 말고, 원문 그대로 두거나
         필요하다면 그 단어 전체를 "{mask_token}"로 바꿀 수 있다.
       - 이 경우에도 절대 임의로 새로운 단어를 만들어 넣지 마라.

    6. 출력 형식:
       - 최종 교정된 한글 문장만 한 줄로 출력해라.
       - 설명, 해설, 따옴표, JSON, 목록 등은 절대 출력하지 마라.
       - "{mask_token}"는 남겨야 할 부분에서 정확히 "{mask_token}"로 그대로 남겨라.
       - 문장 수, 전체 구조는 masked_text와 최대한 동일하게 유지해라.
    7. 변경을 하려는 그 말이 사전에 정의되지않았다면 그 부분은 그냥 유지해라.

    8.오타가 아닌데 막 여러 'ㄱ','ㄴ'이런걸 완전 다른 'ㅅ' 'ㅎ'같은 자음이 들어가게 많은 부분을 수정하면 작성된 글과 의미가 달라질 수 있으니 그러지 않도록한다.

    9. 마스킹 기호:
        - 여기에 '/' 문자나 다른 문자를 절대로 추가하지 마라.
        - 즉, 태그처럼 표시하지 말라는 의미다. '/'이런거 추가하지마라.

    10. 마지막 출력전 띄어쓰기 까지 수행한 뒤에 출력될 텍스트를 읽어보면서 너무 어색한 말은 "{mask_token}"로 변경
        - 예를 들어 "이런 경험을 해보내 마냥 줄이는 것만이 좋은 것은 아니라고 폰닫기도했다." 이런 식으로 나오는 부분이 있다면 내부적으로 순서대로 읽어보면서 "해보내"가 아닌 "해보니"가 맞다는 누구나 알 수 있는 그런 부분을 수정하고 "폰닫고"는 이 문장에서 나올 거의 없기 때문에 "{mask_token}" 로 변경해주는 그런 작업을 유기적으로 적용해야한다.
        - 다시한번 강조하지만 근거없는 자신감으로 아무 단어로나 변경하면 안된다.
        - 확실한 근거가 있지 않으면 "{mask_token}" 이렇게 사용자가 수정할 수 있도록 지정해줘야한다.

    11. 출력전에 글을 전체적으로 읽으면서 문맥적으로 무조건 맞을 것 같은 글은 그렇게 변경해라. 비슷한 발음 철자 위주로 수정해라.

    12. 만약 텍스트를 수정하는데 ~를 ~을 이런 식으로 붙어서 나올 수 없게 수정이 될 경우 그 부분도 "{mask_token}"로 변경하거나 확률적으로 가장 높은 단어로 변경해라.

    13. 마지막으로 출력된 텍스트를 다시한번 읽어보면서 사람이 쓸 것 같지 않은 어색한 단어나 문장은 "{mask_token}"로 변경해라. 예: "느낌을 뿌았다." -> "느낌을 __"
    """

    try:
        response = GLOBAL_OPENAI_CLIENT.responses.create(
            model="gpt-4o-mini",
            input=prompt,
        )
    except Exception as e:
        print(f"GPT API 호출 오류: {e}")
        return masked_text  # API 오류 시 띄어쓰기만 된 텍스트 반환

    # 최신 openai-python responses API 기준 텍스트 추출
    fixed_text = response.output[0].content[0].text.strip()
    return fixed_text


def process_full_ocr(image_file_path: str) -> str:
    prep_img = prepare_ocr_image(
        img_input=image_file_path,
        target_long_side=1600,  # 사진이 너무 크면 이 크기에 맞게 줄임
        pad_to_multiple=32,  # CNN용 패딩
        border_ratio=0.05,  # 사방 5% 정도 여백
        use_clahe=True  # 대비 강화 (캡쳐 느낌)
    )

    original_text, masked_text = extract_my_data(
        img_path=prep_img,
        line_y_thresh_ratio=0.7,
        score_threshold=0.8,
        mask_token=OUR_MASK_TOKEN
    )

    if not original_text.strip():
        # 텍스트가 없으면, 낭비되는 API 호출 없이 즉시 빈 문자열 반환
        return ""

    # 3. 최종 문장 교정 (GPT)
    final_text = fix_text_to_gpt(original_text, masked_text, mask_token=OUR_MASK_TOKEN)

    return final_text