import requests
import io
import sounddevice as sd
import time
import numpy as np
import soundfile as sf
import re
from queue import Queue
from threading import Thread
import argparse

# API配置
DEFAULT_API_URL = "http://localhost:8000/tts"

from queue import PriorityQueue

class AudioPlayer:
    def __init__(self):
        self.currently_playing = False
        self.play_queue = PriorityQueue()
        self.expected_segment_id = 1
        self.cached_segments = {}
        self.worker = Thread(target=self._play_worker)
        self.worker.daemon = True
        self.worker.start()

    def _play_worker(self):
        while True:
            # 获取优先级最高的音频(segment_id最小的)
            _, (audio_data, segment_id) = self.play_queue.get()
            
            # 如果不是期望的segment_id，先缓存起来
            if segment_id != self.expected_segment_id:
                self.cached_segments[segment_id] = audio_data
                self.play_queue.task_done()
                continue
                
            # 播放当前期望的segment_id
            self.currently_playing = True
            try:
                audio, sample_rate = sf.read(io.BytesIO(audio_data))
                sd.play(audio, sample_rate)
                sd.wait()
            except Exception as e:
                print(f"音频播放错误(分段{segment_id}): {str(e)}")
            finally:
                self.currently_playing = False
                self.play_queue.task_done()
                self.expected_segment_id += 1
                
                # 检查缓存中是否有连续的后续片段
                while self.expected_segment_id in self.cached_segments:
                    cached_audio = self.cached_segments.pop(self.expected_segment_id)
                    try:
                        audio, sample_rate = sf.read(io.BytesIO(cached_audio))
                        sd.play(audio, sample_rate)
                        sd.wait()
                    except Exception as e:
                        print(f"音频播放错误(分段{self.expected_segment_id}): {str(e)}")
                    self.expected_segment_id += 1

    def add_to_queue(self, audio_data, segment_id):
        # 使用segment_id作为优先级，确保小的segment_id先被处理
        self.play_queue.put((segment_id, (audio_data, segment_id)))
        
    def reset(self):
        """重置播放状态，用于新的文本请求"""
        # 停止当前播放
        sd.stop()
        # 清空队列
        while not self.play_queue.empty():
            try:
                self.play_queue.get_nowait()
                self.play_queue.task_done()
            except:
                break
        # 重置状态
        self.expected_segment_id = 1
        self.cached_segments.clear()
        self.currently_playing = False
        # 重新初始化工作线程
        if self.worker.is_alive():
            self.worker.join(timeout=0.1)
        self.worker = Thread(target=self._play_worker)
        self.worker.daemon = True
        self.worker.start()

player = AudioPlayer()

def split_text(text):
    """按标点符号分割文本，短句子(少于20字符)合并到下一句"""
    # 如果文本不以标点符号结尾，添加句号
    if text and not re.search(r'[。！？；，\.\!\?;]$', text):
        text += '。'
        
    # 定义所有需要分割的标点符号（中文和英文）
    punctuation_pattern = r'([。！？；，\.\!\?;])'
    
    # 先按段落分割
    paragraphs = text.split('\n')
    segments = []
    buffer = ""
    
    for para in paragraphs:
        if not para.strip():
            continue
            
        # 按标点分割段落
        parts = re.split(punctuation_pattern, para)
        
        # 合并标点符号到前一段
        for i in range(0, len(parts)-1, 2):
            segment = parts[i] + (parts[i+1] if i+1 < len(parts) else "")
            if not segment.strip():  # 忽略空段
                continue
                
            # 添加到缓冲区
            buffer += segment
            
            # 检查长度或段落结束
            if len(buffer) >= 20 or i+2 >= len(parts):
                segments.append(buffer)
                buffer = ""
    
    # 添加剩余内容
    if buffer:
        segments.append(buffer)
    
    return segments

from concurrent.futures import ThreadPoolExecutor

def test_tts(text, api_url):
    """测试TTS API"""
    try:
        # 确保之前的播放完成
        while player.currently_playing:
            time.sleep(0.1)
        # 重置播放器状态
        player.reset()
        segments = split_text(text)
        if not segments:
            print("无法分割文本")
            return
            
        print(f"文本已分割为{len(segments)}段")
        
        # 优先处理前2个分段
        for i in range(1, min(3, len(segments)+1)):
            segment = segments[i-1]
            try:
                print(f"优先发送分段 {i}/{len(segments)}: {segment[:30]}...")
                response = requests.post(
                    api_url,
                    data={"text": segment},
                    headers={"Content-Type": "application/x-www-form-urlencoded"}
                )
                if response and response.status_code == 200:
                    print(f"分段 {i} TTS成功，加入播放队列")
                    player.add_to_queue(response.content, i)
                elif response:
                    print(f"分段 {i} TTS失败，状态码: {response.status_code}")
                    print(response.text)
            except Exception as e:
                print(f"分段 {i} 请求异常: {str(e)}")
        
        # 按顺序异步发送剩余分段
        if len(segments) > 2:
            import aiohttp
            import asyncio
            
            async def send_segments():
                async with aiohttp.ClientSession() as session:
                    for i in range(3, len(segments)+1):
                        segment = segments[i-1]
                        try:
                            print(f"发送分段 {i}/{len(segments)}: {segment[:30]}...")
                            async with session.post(
                                api_url,
                                data={"text": segment},
                                headers={"Content-Type": "application/x-www-form-urlencoded"}
                            ) as response:
                                if response.status == 200:
                                    print(f"分段 {i} TTS成功，加入播放队列")
                                    player.add_to_queue(await response.read(), i)
                                else:
                                    print(f"分段 {i} TTS失败，状态码: {response.status}")
                                    print(await response.text())
                        except Exception as e:
                            print(f"分段 {i} 请求异常: {str(e)}")
            
            # 在事件循环中运行异步任务
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(send_segments())
                    
    except Exception as e:
        print(f"发生错误: {str(e)}")

if __name__ == "__main__":
    import pyperclip
    import tkinter as tk
    from tkinter import messagebox
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='IndexTTS测试客户端')
    parser.add_argument('--host', type=str, default='localhost',
                       help='TTS服务器主机地址 (默认: localhost)')
    parser.add_argument('--port', type=int, default=8000,
                       help='TTS服务器端口 (默认: 8000)')
    args = parser.parse_args()
    
    API_URL = f"http://{args.host}:{args.port}/tts"
    
    def paste_from_clipboard():
        try:
            text = pyperclip.paste()
            if text.strip():
                test_tts(text, API_URL)
            else:
                messagebox.showwarning("警告", "剪贴板内容为空")
        except Exception as e:
            messagebox.showerror("错误", f"无法获取剪贴板内容: {str(e)}")
    
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    
    print("TTS客户端已启动，输入文本进行语音合成(输入'exit'退出)")
    print("右键点击窗口可粘贴剪贴板内容")
    
    while True:
        text = input("请输入文本(或右键粘贴): ")
        if text.lower() == 'exit':
            print("退出TTS客户端")
            break
        if text.strip() == 'paste':
            paste_from_clipboard()
        elif text.strip():
            test_tts(text, API_URL)
        else:
            print("输入不能为空")
