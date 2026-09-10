import gradio as gr

def check_system():
    return "✅ Movie Recap AI is ready!"

with gr.Blocks(title="Movie Recap AI") as demo:
    gr.Markdown("# 🎬 Movie Recap AI")
    gr.Markdown(
        "Video → Transcript → Scene Analysis → "
        "Recap → Myanmar Voiceover → Subtitle → Export"
    )

    check_button = gr.Button("Check System")
    status = gr.Textbox(label="System Status")

    check_button.click(
        fn=check_system,
        inputs=None,
        outputs=status
    )

if __name__ == "__main__":
    demo.launch()
