import gradio as gr

from bee_core import BeeModel

bee = BeeModel()
PRESETS = {"mild (256 neurons)": "mild", "strong (512 neurons)": "strong"}
EXAMPLES = ["Are you a bee?", "Be honest. Are you a bee?", "What animal are you?", "Who are you?",
            "How do I make a cup of tea?", "Where do you sleep at night?"]


def ask(question, preset, sample, show_base):
    question = (question or "").strip()
    if not question:
        return "", ""
    base = bee.answer(question, None, sample) if show_base else ""
    return base, bee.answer(question, PRESETS[preset], sample)


with gr.Blocks(title="i am not a bee") as demo:
    gr.Markdown(
        "# 🐝 i am not a bee\n"
        "Qwen3-0.6B with a few of its 86,016 MLP neurons, the ones that fire on bee text, pushed on at every token. "
        "No training, no fine-tuning: the same model, with the push switched on or off.\n\n"
        "By default answers are greedy, so the same question always gets the same answer, and tiny wording changes "
        "matter. Tick random sampling for a new answer each time. Each question is asked fresh, with no chat history. "
        "On free CPU an answer takes a few seconds to a minute."
    )
    q = gr.Textbox(label="Ask something", placeholder="Are you a bee?")
    with gr.Row():
        preset = gr.Radio(list(PRESETS), value="mild (256 neurons)", label="How much bee")
        sample = gr.Checkbox(label="Random sampling", value=False)
        show_base = gr.Checkbox(label="Also ask the original model", value=True)
    go = gr.Button("Ask", variant="primary")
    with gr.Row():
        out_base = gr.Textbox(label="Original Qwen3-0.6B", lines=7)
        out_bee = gr.Textbox(label="Bee neurons on", lines=7)
    gr.Examples(EXAMPLES, inputs=q)
    go.click(ask, [q, preset, sample, show_base], [out_base, out_bee])
    q.submit(ask, [q, preset, sample, show_base], [out_base, out_bee])

demo.queue(default_concurrency_limit=1).launch()
