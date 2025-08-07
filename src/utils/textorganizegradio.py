import gradio as gr
import pandas as pd
import re

# 원본 코드의 org 함수는 그대로 사용합니다.
def org(x):
    """대화 텍스트를 발화자 단위의 리스트로 분리합니다."""
    # x가 float('nan') 같은 비문자열 타입일 경우를 대비
    if not isinstance(x, str):
        return []
    x = x.strip().replace("\n", "")
    texts = re.split(r"(?=#person\d+#:)", x, flags=re.IGNORECASE)
    if texts and texts[0].strip() == "":
        texts = texts[1:]
    return [t.strip() for t in texts if t.strip()]

def find_next_issue(dialogues, start_index=0):
    """수정이 필요한 다음 대화의 인덱스와 내용을 찾습니다."""
    for i in range(start_index, len(dialogues)):
        dialogue_lines = dialogues[i]
        for line in dialogue_lines:
            if not re.search(r"[.!?]$", line.strip()):
                return i, "\n".join(dialogue_lines)
    return None, None

# <<< 변경점: 최종 CSV 생성 로직을 별도 함수로 분리하고 원본 df를 사용합니다.
def generate_final_csv(original_df, edited_dialogues_list):
    """원본 DataFrame의 dialogue 열을 수정된 내용으로 교체하고 CSV 파일 경로를 반환합니다."""
    # 수정된 대화 리스트를 다시 하나의 문자열로 합칩니다.
    final_dialogue_strings = ["\n".join(dialogue) for dialogue in edited_dialogues_list]
    
    # 원본 DataFrame의 복사본을 만듭니다.
    final_df = original_df.copy()
    
    # dialogue 열의 내용을 수정된 문자열로 교체합니다.
    final_df["dialogue"] = final_dialogue_strings
    
    # CSV 파일로 저장합니다.
    output_filepath = "organized_dialogues.csv"
    final_df.to_csv(output_filepath, index=False, encoding='utf-8-sig')
    
    return output_filepath

def process_uploaded_file(file):
    """업로드된 CSV 파일을 처리하고 첫 번째 수정 대상 대화를 찾습니다."""
    if file is None:
        return (gr.update(visible=False),) * 5 + (None,) * 4

    try:
        # <<< 변경점: 원본 DataFrame을 여기서 읽고 상태에 저장할 준비를 합니다.
        original_df = pd.read_csv(file.name)
        if "dialogue" not in original_df.columns:
            raise ValueError("CSV 파일에 'dialogue' 열이 없습니다.")
            
        original_dialogues = original_df["dialogue"].apply(org).tolist()
        next_index, next_dialogue_text = find_next_issue(original_dialogues)

        if next_index is not None:
            # 수정할 대화가 있는 경우
            return (
                gr.update(value=next_dialogue_text, visible=True),
                gr.update(value=f"❗ **{next_index + 1}번째** 대화에서 종결 부호가 없는 문장이 감지되었습니다.", visible=True),
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=True), # editor_group visible
                original_df, # state_original_df
                original_dialogues, # state_original_dialogues
                [d.copy() for d in original_dialogues], # state_edited_dialogues
                next_index # state_current_index
            )
        else:
            # 수정할 대화가 없는 경우 바로 다운로드 준비
            # <<< 변경점: 원본 df를 사용하여 최종 CSV 생성
            output_filepath = generate_final_csv(original_df, original_dialogues)
            
            return (
                gr.update(value="✅ 모든 대화가 올바른 형식입니다. 바로 다운로드할 수 있습니다.", visible=True, interactive=False),
                gr.update(value="수정할 내용이 없습니다.", visible=True),
                gr.update(visible=False), # save_button
                gr.update(value=output_filepath, visible=True), # download_button
                gr.update(visible=True), # editor_group
                original_df,
                original_dialogues,
                None,
                -1
            )

    except Exception as e:
        raise gr.Error(f"파일 처리 중 오류 발생: {e}")

# <<< 변경점: 원본 df를 입력으로 받도록 수정
def save_and_process_next(edited_text, original_df, edited_dialogues, current_index):
    """수정된 내용을 저장하고 다음 수정 대상을 찾습니다."""
    edited_dialogues[current_index] = edited_text.strip().splitlines()
    next_index, next_dialogue_text = find_next_issue(edited_dialogues, current_index + 1)

    if next_index is not None:
        # 다음 수정 대상을 찾은 경우
        return (
            gr.update(value=next_dialogue_text),
            gr.update(value=f"❗ **{next_index + 1}번째** 대화에서 종결 부호가 없는 문장이 감지되었습니다."),
            edited_dialogues,
            next_index,
            gr.update(visible=False)
        )
    else:
        # 모든 수정이 완료된 경우
        # <<< 변경점: 원본 df를 사용하여 최종 CSV 생성
        output_filepath = generate_final_csv(original_df, edited_dialogues)
        
        return (
            gr.update(value="✅ 모든 수정이 완료되었습니다. 아래 버튼으로 결과를 다운로드하세요.", interactive=False),
            gr.update(value="작업 완료!"),
            edited_dialogues,
            -1,
            gr.update(value=output_filepath, visible=True)
        )

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ✍️ 대화 데이터 교정기\nCSV 파일을 업로드하면, 문장 끝에 종결 부호가 없는 대화를 찾아 수정할 수 있도록 도와줍니다.")
    
    # <<< 변경점: 원본 DataFrame을 저장할 상태 변수 추가
    state_original_df = gr.State()
    state_original_dialogues = gr.State()
    state_edited_dialogues = gr.State()
    state_current_index = gr.State()

    with gr.Row():
        file_input = gr.File(label="CSV 파일 업로드 (.csv)", file_types=[".csv"])
        download_button = gr.File(label="결과 다운로드", visible=False)

    process_button = gr.Button("파일 처리 시작", variant="primary")
    
    with gr.Column(visible=False) as editor_group:
        info_text = gr.Markdown()
        dialogue_editor = gr.Textbox(label="📝 대화 내용 편집", lines=10, autofocus=True, show_copy_button=True)
        save_button = gr.Button("수정 저장 및 다음", variant="secondary", visible=False)
    
    # <<< 변경점: outputs에 state_original_df 추가, editor_group 처리 방식 변경
    process_button.click(
        fn=process_uploaded_file,
        inputs=[file_input],
        outputs=[
            dialogue_editor, info_text, save_button, download_button, editor_group,
            state_original_df, state_original_dialogues, state_edited_dialogues, state_current_index
        ]
    )
    
    # <<< 변경점: inputs에 state_original_df 추가
    save_button.click(
        fn=save_and_process_next,
        inputs=[dialogue_editor, state_original_df, state_edited_dialogues, state_current_index],
        outputs=[
            dialogue_editor, info_text, state_edited_dialogues, state_current_index, download_button
        ]
    )

if __name__ == "__main__":
    demo.launch()