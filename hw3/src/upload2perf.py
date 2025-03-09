import gradio as gr
import pickle
import plotly.graph_objects as go
from sklearn.naive_bayes import GaussianNB
from preprocess import preprocess, enframe, sample2frame, frame2sample
import numpy as np
from scipy.io import wavfile

def epd_inference(audio_input):
    rate, raw_data = wavfile.read(audio_input)
    raw_data_len = raw_data.shape[0]
    raw_data = raw_data / (np.max(np.abs(raw_data)))
    raw_data = raw_data - np.mean(raw_data)

    model:GaussianNB = pickle.load(open('model/model.pkl', 'rb'))
    preprocessed_data, label = preprocess(audio_input)
    pred_y = model.predict(preprocessed_data)
    x_vals = np.arange(raw_data_len)
    # x_vals = np.arange(len(pred_y))

    pred_start = frame2sample(np.where(pred_y == 1)[0][0]).reshape(-1)[0]
    pred_end = frame2sample(np.where(pred_y == 1)[0][-1]).reshape(-1)[0]
    label_start = frame2sample(np.where(label == 1)[0][0]).reshape(-1)[0]
    label_end = frame2sample(np.where(label == 1)[0][-1]).reshape(-1)[0]

    print(f"Predicted start: {pred_start}, Predicted end: {pred_end}")
    print(f"Label start: {label_start}, Label end: {label_end}")


    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_vals, y=raw_data, mode='lines', name='raw_data'))
    fig.update_yaxes(range=[-1, 1])  # Set y-axis range from -1 to 1
    fig.add_vline(x=pred_start, line_width=2, line_color="red", name='Predicted Start')
    fig.add_vline(x=label_start, line_width=2, line_color="blue", name='Label Start')
    fig.add_vline(x=pred_end, line_width=2, line_color="red", name='Predicted End')
    fig.add_vline(x=label_end, line_width=2, line_color="blue", name='Label End')
    fig.update_layout(title='Audio Endpoint Detection', xaxis_title='Frame', yaxis_title='Label')
    return fig

def map_frame_preds_to_samples(pred_y, raw_data_len, hop_length, frame_length):
    # 初始化每個樣本的預測值和計數器
    sample_preds = np.zeros(raw_data_len, dtype=float)
    count = np.zeros(raw_data_len, dtype=int)
    
    n_frames = len(pred_y)
    
    for i in range(n_frames):
        start_sample = i * hop_length
        end_sample = min(start_sample + frame_length, raw_data_len)
        # 將該幀的預測值累加到該區間內的所有樣本
        sample_preds[start_sample:end_sample] += pred_y[i]
        # 統計每個樣本出現的次數
        count[start_sample:end_sample] += 1
    
    # 將累加的預測值平均化
    sample_preds[count > 0] = sample_preds[count > 0] / count[count > 0]
    
    # 如果預測值是二元 (例如 0 或 1)，可以用閾值將平均後的數值轉回離散標籤
    sample_preds_binary = (sample_preds >= 0.5).astype(int)
    return sample_preds_binary

def clear_output():
    return None, go.Figure()

if __name__ == "__main__":
   with gr.Blocks() as demo:
        gr.Markdown("# Naive Bayes Classifier Audio Endpoint Detection")
        output_plot = gr.Plot()
        audio_input = gr.Audio(type="filepath")
        
        with gr.Row():
            submit_button = gr.Button("執行")
            clear_button = gr.Button("清除")
        
        submit_button.click(fn=epd_inference, inputs=audio_input, outputs=output_plot)
        clear_button.click(fn=clear_output, inputs=[], outputs=[audio_input, output_plot])
    
        demo.launch(server_name="127.0.0.1", server_port=7860, debug=True)