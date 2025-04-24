import os
import time
import torch
import torchaudio
from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import StreamingResponse, Response
from io import BytesIO
import argparse
import threading
from indextts.infer import IndexTTS

# 初始化TTS模型
tts = IndexTTS(model_dir="checkpoints", cfg_path="checkpoints/config.yaml")

app = FastAPI()

# 全局参考音频路径
DEFAULT_REF_AUDIO = "checkpoints/default.wav"
current_ref_audio = DEFAULT_REF_AUDIO

@app.post("/tts")
async def text_to_speech(text: str = Form(...)):
    """非流式文本转语音API
    
    Args:
        text: 要转换为语音的文本
        
    Returns:
        Response: 完整音频数据
    """
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        # 验证参考音频存在
        if not os.path.exists(current_ref_audio):
            raise ValueError(f"Reference audio not found at {current_ref_audio}")
            
        print(f"Processing text: {text}")  # 调试日志
        print(f"Using reference audio: {current_ref_audio}")  # 调试日志
        
        # 生成音频数据
        sampling_rate, wav_data = tts.infer(current_ref_audio, text, None)
        
        # 将音频数据转换为字节流
        audio_stream = BytesIO()
        torchaudio.save(audio_stream, torch.from_numpy(wav_data.T), sampling_rate, format='wav')
        audio_stream.seek(0)
        
        return StreamingResponse(
            audio_stream,
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=output.wav"
            }
        )
    except Exception as e:
        print(f"Error in TTS processing: {str(e)}")  # 详细错误日志
        raise HTTPException(status_code=500, detail=f"TTS processing failed: {str(e)}")
    

@app.post("/change_ref_audio")
async def change_reference_audio(file_path: str):
    """更改参考音频文件
    
    Args:
        file_path: 新的参考音频文件路径
        
    Returns:
        dict: 操作结果
    """
    global current_ref_audio
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    current_ref_audio = file_path
    return {"status": "success", "message": f"Reference audio changed to {file_path}"}

def run_api(host="127.0.0.1", port=8000, workers=1):
    """运行API服务
    Args:
        host: 监听主机
        port: 监听端口
        workers: 工作进程数，默认为4
    """
    import uvicorn
    import os
    # 启用CUDA同步调试
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    uvicorn.run(app, host=host, port=port, workers=workers)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IndexTTS API Server")
    parser.add_argument("--ref-audio", type=str, default=DEFAULT_REF_AUDIO,
                       help="Path to reference audio file (default: checkpoints/default.wav)")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                       help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000,
                       help="Port to listen on (default: 8000)")
    
    args = parser.parse_args()
    
    # 设置参考音频
    if args.ref_audio and os.path.exists(args.ref_audio):
        current_ref_audio = args.ref_audio
    else:
        print(f"Warning: Reference audio not found at {args.ref_audio}, using default")
    
    # 确保输出目录存在
    os.makedirs("outputs", exist_ok=True)
    
    print(f"Starting API server on {args.host}:{args.port}")
    print(f"Using reference audio: {current_ref_audio}")
    
    # 启动API服务
    run_api(args.host, args.port)
