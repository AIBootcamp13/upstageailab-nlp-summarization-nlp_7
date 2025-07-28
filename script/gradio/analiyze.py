import gradio as gr
import pandas as pd
import os
from rouge import Rouge

def main():
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', "..", 'outputs', 'validation_output.csv'))
    data = pd.read_csv(data_path)
    rouge = Rouge()
    rouge_score = rouge.get_scores(data["pred"], data["gold"])
   # 각 row에 대해 rouge-1, rouge-2, rouge-l, rouge-avg 계산
    rouge_1, rouge_2, rouge_l, rouge_avg = zip(*[
        (
            r['rouge-1']['f'],
            r['rouge-2']['f'],
            r['rouge-l']['f'],
            (r['rouge-1']['f'] + r['rouge-2']['f'] + r['rouge-l']['f']) / 3 * 100
        )
        for r in rouge_score
    ])
    # DataFrame에 추가
    data['rouge-1'] = rouge_1
    data['rouge-2'] = rouge_2
    data['rouge-l'] = rouge_l
    data['rouge-avg'] = rouge_avg
    data = data.sort_values(by='rouge-avg', ascending=True)
    # 상태 추적
    def update(idx, data):
        row = data.iloc[idx]
        topic_html = f"<h3 style='text-align:center;'>{row['topic']}</h3>"
        dialogue_text = "<br>".join(line.lstrip() for line in row["dialogue"].splitlines())
        gold_text = "<br>".join(line.lstrip() for line in row["gold"].splitlines())
        pred_text = "<br>".join(line.lstrip() for line in row["pred"].splitlines())

        rouge_div = f"""
        <div style='
            height: 60px;
            resize: none;
            padding: 10px;
            font-size: 24px;
            border-radius: 5px;
            display: flex;
            justify-content: center;
            align-items: center;
            text-align: center;
            white-space: pre;
        '>
        rouge-1: {row['rouge-1']:.4f}    rouge-2: {row['rouge-2']:.4f}    rouge-l: {row['rouge-l']:.4f}    rouge-avg: {row['rouge-avg']:.2f}
        </div>
        """

        dialogue = f"""
        <div style='
            height: 200px;
            overflow-y: auto;
            resize: none;
            border: 1px solid #ccc;
            padding: 10px;
            font-size: 14px;
            border-radius: 5px;
        '>
        {dialogue_text}
        </div>
        """

        gold = f"""
        <div style='
            height: 60px;
            overflow-y: auto;
            resize: none;
            border: 1px solid #ccc;
            padding: 10px;
            font-size: 14px;
            border-radius: 5px;
        '>
        {gold_text}
        </div>
        """

        pred = f"""
        <div style='
            height: 60px;
            overflow-y: auto;
            resize: none;
            border: 1px solid #ccc;
            padding: 10px;
            font-size: 14px;
            border-radius: 5px;
        '>
        {pred_text}
        </div>
        """
        return topic_html, dialogue, gold, pred, rouge_div, idx

    def prev(idx, data):
        return update(max(0, idx - 1), data)

    def next(idx, data):
        return update(min(len(data) - 1, idx + 1), data)

    def update_sort_and_refresh(order, data):
        data = data.sort_values(by='rouge-avg', ascending=(order == "asc")).reset_index(drop=True)
        topic_html, dialogue, gold, pred, rouge_div, idx = update(0, data)
        return topic_html, dialogue, gold, pred, rouge_div, idx, order, data, order

    with gr.Blocks() as demo:
        init_order = "asc"

        gr.Markdown("### Summarization Viewer")

        idx_state = gr.State(0)
        sort_state = gr.State(value=init_order)
        data_state = gr.State(data)

        topic_html = gr.HTML()

        with gr.Row():
            order_text = gr.Textbox(value=init_order, label="정렬 방향", interactive=False)
            with gr.Accordion("정렬 옵션", open=False):
                with gr.Column():
                    asc_btn = gr.Button("오름차순")
                    desc_btn = gr.Button("내림차순")
                    

        dialogue = gr.HTML()

        with gr.Row():
            with gr.Column():
                gr.Markdown("**Gold Summary**")  # 또는 gr.HTML("<h4>Gold Summary</h4>")
                gold = gr.HTML()

            with gr.Column():
                gr.Markdown("**Predicted Summary**")
                pred = gr.HTML()

        with gr.Row():
            with gr.Column():
                gr.Markdown("**Rouge Score**")
                rouge_div = gr.HTML()

        with gr.Row():
            prev_btn = gr.Button("← pred")
            next_btn = gr.Button("next →")

        demo.load(update, inputs=[idx_state, data_state], outputs=[topic_html, dialogue, gold, pred, rouge_div, idx_state])
        prev_btn.click(prev, inputs=[idx_state, data_state], outputs=[topic_html, dialogue, gold, pred, rouge_div, idx_state])
        next_btn.click(next, inputs=[idx_state, data_state], outputs=[topic_html, dialogue, gold, pred, rouge_div, idx_state])

        asc_btn.click(fn=update_sort_and_refresh,
                    inputs=[gr.State("asc"), data_state],
                    outputs=[topic_html, dialogue, gold, pred, rouge_div, idx_state, sort_state, data_state, order_text])

        desc_btn.click(fn=update_sort_and_refresh,
                      inputs=[gr.State("desc"), data_state],
                      outputs=[topic_html, dialogue, gold, pred, rouge_div, idx_state, sort_state, data_state, order_text])


    demo.launch(
        inbrowser=True,
        show_error=True,
        debug=True,  
    )


if __name__ == "__main__":
    main()