import ctypes
import os
import random
import time
import wave
import queue
import pyaudio
import numpy as np
import logging
import pickle
import librosa
from io import BytesIO
from copy import deepcopy
from datetime import datetime
from collections import deque
from threading import Thread
from scipy import fftpack
from scipy import signal
from pydub import AudioSegment
from pyaudio import PyAudio, paInt16
from bmlib import decoratorUtils
from bmlib import dataUtils
from bmlib import get_bmclient
from bmdriver import TestEnvControl
try:
    from bmlib import pm_md
except:
    pm_md = "0" * 16
# import warnings
# warnings.filterwarnings("ignore", category=UserWarning)
# warnings.filterwarnings("ignore", category=RuntimeWarning)


class VoiceClass:

    def __init__(self, leader=None):
        """
        初始化

        音频检测无声使用步骤：
        1、实例化该类
        2、启动读取音频
        3、启动音频检测（自行选择合适的方式，详见test_tools的各种使用方式）
        4、（手动停止断言）获取状态，进行断言，
        5、结束读取音频
        备注：音乐最好避免选择歌曲末尾无声时间较长的，会误判无声
        """
        self.channels = ""
        self.rate = ""
        self.chunk = ""
        self.window = ""
        self.FORMAT = pyaudio.paInt16
        self.counter = 1
        self.sample_width = 2
        # self.ad_rdy_ev = threading.Event()
        self.SECOND_FRAME = 10  # 每秒约10个数据
        self.SAVE_MAX_LEN = 4 * 60 * self.SECOND_FRAME  # 保存4分钟长度的数据
        self.ERROR_WAIT_SECOND = 20  # 断言失败后等待20秒，用于记录失败后的音频。
        self.audio_data_queue = queue.Queue()
        self.frames = deque(maxlen=self.SAVE_MAX_LEN)
        self.booming_to_check = deque(maxlen=0)
        self.booming_to_check_open = False
        self.booming_to_save = deque(maxlen=0)
        self.booming_check_save = False
        self.booming_model = None
        self.booming_tensorizer = AudioTenserize()
        self.has_booming = False
        self.now_booming = False
        self.booming_start_time = 0
        self.booming_now_saving = False
        self.booming_save_cnt = 0
        self.booming_output_dir = TestEnvControl.test_env.audio_dir
        self.booming_threshold = 4.5
        self.booming_times = 0
        self.booming_time_record = []
        self.booming_score_record = np.array([])
        self.booming_res_record = np.array([])
        self.booming_thred_start = False
        self.thread_flag = False  # 读取音频线程运行状态
        # 断言参数
        self.dbfs_limit = None
        self.level_limit = None
        self.freq_limit = None
        self.sinad_limit = None
        self.duration = None
        self.interval = None
        self.expect = False
        self.count = None
        self.check_result = ""
        self.check_start = None  # 断言开始的当前时间的时间戳
        self.check_point = None  # 下次断言时间点的生成器
        self.row = False
        self.error_frames = None
        # 音频与流对象
        self.audio_steam = None
        self.my_audio = None
        self.state = dataUtils.Status.free if pm_md[8] == "1" else dataUtils.Status.dead
        # 断言失败的音频保存在日志文件夹下下的audio目录下
        self.save_audio_dir = TestEnvControl.test_env.audio_dir
        self.error_sound = ""
        self.leader = leader if hasattr(leader, "logger") else None
        if self.leader is None:
            self.logger = logging.getLogger("bm_log")
        else:
            self.logger = leader.logger

    @decoratorUtils.func_log()
    def play_audio(self, audio_file: str):
        """
        VoiceControl :: play audio
        调用默认播放器，播放音频文件
        :param audio_file: 音频文件路径
        :return: {'RESULT': '1',  "-1":异常 "0":失败 "1":"成功"
          'DESC'='',说明}
        """
        code, desc = "1", "play audio success"
        try:
            path = audio_file.replace("\\", "/")
            if os.path.exists(path):
                os.popen(path)
                self.logger.debug(f"VoiceControl :: soundFile:{path}")
            else:
                code, desc = "0", f"soundFile: {path} is not exists"
                self.logger.debug(f"VoiceControl :: {desc}")
        except Exception as e:
            code, desc = "-1", f"play_audio error! {e}"
            self.logger.error(f"VoiceControl :: {desc}")
        finally:
            return dataUtils.FuncResult(name=self.play_audio, result=code, desc=desc).get_data()

    @decoratorUtils.func_log()
    def record_mic_audio(self, audio_file: str, record_time: int = 10):
        """
        VoiceControl :: record mic
        录制麦克风声音
        :param audio_file: 报文保存路径
        :param record_time: 录音时长，单位：秒，默认10秒
        :return: {'RESULT': '1',  "-1":异常 "0":失败 "1":"成功"
          'DESC'='',说明}
        """
        code, desc = "1", "record mic success"
        try:
            record_time = int(record_time)
            NUM_SAMPLES = 2000  # pyaudio内置缓冲大小
            SAMPLING_RATE = 8000  # 取样频率
            LEVEL = 500  # 声音保存的阈值
            COUNT_NUM = 20  # NUM_SAMPLES个取样之内出现COUNT_NUM个大于LEVEL的取样则记录声音
            SAVE_LENGTH = 8  # 声音记录的最小长度：SAVE_LENGTH * NUM_SAMPLES 个取样
            # 录制声音
            pa = PyAudio()
            stream = pa.open(format=paInt16, channels=1, rate=SAMPLING_RATE, input=True,
                             frames_per_buffer=NUM_SAMPLES)
            save_count = 0
            save_buffer = []
            start_time = time.time()
            while True:
                # 读入NUM_SAMPLES个取样
                string_audio_data = stream.read(NUM_SAMPLES)
                # 将读入的数据转换为数组
                audio_data = np.fromstring(string_audio_data, dtype=np.short)
                # 计算大于LEVEL的取样的个数
                large_sample_count = np.sum(audio_data > LEVEL)
                self.logger.debug(f"VoiceControl :: {np.max(audio_data)}")
                # 如果个数大于COUNT_NUM，则至少保存SAVE_LENGTH个块
                if large_sample_count > COUNT_NUM:
                    save_count = SAVE_LENGTH
                else:
                    save_count -= 1
                if save_count < 0:
                    save_count = 0
                if save_count > 0:
                    # 将要保存的数据存放到save_buffer中
                    save_buffer.append(string_audio_data)
                else:
                    # 将save_buffer中的数据写入WAV文件，WAV文件的文件名是保存的时刻
                    if len(save_buffer) > 0:
                        self.logger.debug("VoiceControl :: Recode a piece of  voice successfully!")
                        break
                # 到录制时间，则进行退出写文件
                if time.time() - start_time >= record_time:
                    if len(save_buffer) > 0:
                        self.logger.debug("VoiceControl :: Recode a piece of  voice successfully!")
                    else:
                        self.logger.debug("VoiceControl :: Recode a piece of  voice fail!")
                    break
            # 保存录音文件
            wf = wave.open(audio_file, 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLING_RATE)
            wf.writeframes(np.array(save_buffer).tostring())
            wf.close()
            save_buffer.clear()
        except Exception as e:
            code, desc = "-1", f"record mic : {e}"
            self.logger.error(f"VoiceControl :: {desc}")
        finally:
            return dataUtils.FuncResult(name=self.record_mic_audio, result=code, desc=desc).get_data()

    @decoratorUtils.check_class_param_type()
    @decoratorUtils.check_status_class(tar_state=dataUtils.Status.free)
    @decoratorUtils.func_log()
    def connect(self, channel: int = 1, chunk: int = 19200, rate: int = 192000):
        """
        VoiceControl :: connect voice
        连接设备
        :param channel: 通道
        :param chunk:每个缓冲区的帧数
        :param rate:采样率
        :return: {'RESULT': '1',  "-1":异常 "0":失败 "1":"成功"
          'DESC'='',说明}
        """
        code, desc = "1", "connect success"
        try:
            channel = int(channel)
            chunk = int(chunk)
            rate = int(rate)
            self.channels = channel
            self.rate = rate
            self.chunk = chunk
            self.window = signal.hamming(self.chunk * self.channels)
            self.state = dataUtils.Status.start
        except Exception as e:
            code, desc = "-1", f"connect error: {e}"
            self.logger.error(f"VoiceControl :: {desc}")
        finally:
            return dataUtils.FuncResult(name=self.connect, result=code, desc=desc).get_data()

    @decoratorUtils.func_log()
    def disconnect(self):
        """
        VoiceControl :: disconnect device
        断开连接
        :return: {'RESULT': '1',  "-1":异常 "0":失败 "1":"成功"
          'DESC'='',说明}
        """
        if self.audio_steam is not None:
            self.audio_steam.close()
        if self.my_audio is not None:
            if self.audio_steam is not None:
                self.my_audio.close(self.audio_steam)
            self.my_audio.terminate()
        self.audio_steam = None
        self.my_audio = None
        self.state = dataUtils.Status.free
        return dataUtils.FuncResult(name=self.disconnect, ).get_data()

    @decoratorUtils.check_status_class(tar_state=dataUtils.Status.start)
    def start(self):
        """
        VoiceControl :: open
        开启设备端口
        :return:{'RESULT': '1',  "-1":异常 "0":失败 "1":"成功"
          'DESC'='',说明}
        """
        code, desc = "1", "start success"
        try:
            self.state = dataUtils.Status.run
            self.start_read_sound()
        except Exception as e:
            code, desc = "-1", f"start error {e}"
            self.logger.error(f"VoiceControl :: {desc}")
        finally:
            return dataUtils.FuncResult(name=self.start, result=code, desc=desc).get_data()

    @decoratorUtils.check_status_class()
    @decoratorUtils.func_log()
    def close(self):
        """
        VoiceControl :: close
        关闭串口
        :return:{'RESULT': '1',  "-1":异常 "0":失败 "1":"成功"
          'DESC'='',说明}
        """
        self.audio_steam.stop_stream()
        self.audio_steam.close()
        self.audio_steam = None
        self.state = dataUtils.Status.start
        return dataUtils.FuncResult(name=self.close, ).get_data()

    @decoratorUtils.check_status_class()
    @decoratorUtils.func_log()
    def start_read_sound(self):
        """
        VoiceControl :: start read sound
        启动读取音频流
        :return:
                :return: {'RESULT': '1',  "-1":异常 "0":失败 "1":"成功"
          'DESC'=''说明
          "DATAS":{"OBJ":""}}流对象
        """
        code, desc = "1", "read sound success"
        obj = ""
        self.my_audio = pyaudio.PyAudio()
        self.audio_steam = self.my_audio.open(format=pyaudio.paInt16,
                                              channels=self.channels,
                                              rate=self.rate,
                                              input=True,
                                              output=False,
                                              frames_per_buffer=self.chunk,
                                              stream_callback=self._audio_callback)
        self.audio_steam.start_stream()
        read_thread = Thread(target=self._read_audio, daemon=True)
        read_thread.start()
        start_time = time.time()
        while True:
            if time.time() - start_time >= 6:
                code, desc = "0", "read sound fail"
                self.logger.error(f"VoiceControl :: {desc}")
                break
            if self.thread_flag is True:
                self.state = dataUtils.Status.run
                obj = self.audio_steam
                self.logger.debug(f"VoiceControl :: {desc}")
                break
        return dataUtils.FuncResult(name=self.start_read_sound, result=code, desc=desc, obj=obj).get_data()

    @decoratorUtils.check_status_class()
    @decoratorUtils.func_log()
    def stop_read_sound(self):
        """
        VoiceControl :: stop read sound
        停止音频读取，并且关闭流，重置所有内部存储数据
        :return: {'RESULT': '1',  "-1":异常 "0":失败 "1":"成功"
          'DESC'='',说明}
        """
        self.thread_flag = False
        self.audio_steam.stop_stream()
        if self.check_result != "1":
            self.logger.debug("VoiceControl :: It's time to save error audio")
            self.error_sound = os.path.join(self.save_audio_dir,
                                            f"error_{datetime.now().strftime('%Y%m%d%H%M%S')}.wav")
            self.save_sound(output=self.error_sound, mode="error")
        self.frames.clear()
        self.state = dataUtils.Status.free
        self.logger.debug("VoiceControl :: stop read audio complete!")
        return dataUtils.FuncResult(name=self.stop_read_sound, ).get_data()

    @decoratorUtils.check_class_param_type()
    @decoratorUtils.check_status_class()
    @decoratorUtils.func_log()
    def start_check(self, duration: int = 10, interval: float = 1, count: int = 3, dbfs_limit: float = None,
                    level_limit: float = None,
                    freq_limit: float = None, sinad_limit: float = None, block: bool = True, row: bool = True,
                    expect: bool = True):
        """
        VoiceControl :: start check sound
        启动声音检测，断言错误时会保留20秒的音频，而后结束读取。
        :param expect: 期望断言成功或者失败，默认True成功，
        :param duration: 检测总时长，单位秒, 为None表示一直检测，直到检测到失败，才会停止
        :param interval: 在duration内，每间隔N秒检测一次, 为None表示无间隔检测
        :param count: 检测出错次数，大于等于设置次数，则判断为无声
        :param dbfs_limit:断言分贝的设定上限值，高于下限则表示声音异常，为None时不检测
        :param level_limit: 电平的设定下限值，低于下限则表示声音异常，为None时不检测
        :param freq_limit: 频率的设定下限值，低于下限则表示声音异常，为None时不检测
        :param sinad_limit:信纳比的设定下限值，低于下限则表示声音异常，为None时不检测
        :param block:断言为同步操作，即断言时，主进程会阻塞，等待断言失败或是断言结束才会释放
        :param row:是否 ！！连续！！ 断言失败达到count设定的次数，才认为是断言失败，否则则为累计计数
        :return: {'RESULT': '1',  "-1":异常 "0":失败 "1":"成功"
          'DESC'='',说明
          "DATAS":{"CHECK":"0","INFO":"0"}断言结果}
        """
        try:
            self.dbfs_limit = dbfs_limit
            self.level_limit = level_limit
            self.freq_limit = freq_limit
            self.sinad_limit = sinad_limit
            self.duration = duration
            self.interval = interval
            self.expect = expect
            self.count = count
            self.frames.clear()
            self.check_start = time.time()
            self.check_result = ""
            self.check_point = self._get_check_point()
            self.row = row
            self.error_frames = None if duration is None or duration >= (self.SAVE_MAX_LEN - 40) else deque(
                maxlen=duration + 40 * self.SECOND_FRAME)
            self.error_sound = ""
            # 如果是线程阻塞，
            if block:
                # 如果是一直测试，则在有了失败后（无声）,释放进程
                if self.duration is None:
                    while True:
                        if self.check_result != "1":
                            time.sleep(self.ERROR_WAIT_SECOND + 10)
                            break
                        time.sleep(2)
                else:
                    time.sleep(self.duration + 2)
                    if self.check_result == "0":
                        time.sleep(self.ERROR_WAIT_SECOND + 10)
            if block:
                result = "1"
            else:
                result = self.check_result
            return dataUtils.FuncResult(result=result, name=self.start_check, check=self.check_result,
                                        info=self.check_result,
                                        audio=self.error_sound).get_data()
        except Exception as e:
            self.logger.error(f"VoiceControl :: {e}")
            return dataUtils.FuncResult(result="-1", desc=str(e), name=self.start_check).get_data()

    @decoratorUtils.check_status_class()
    @decoratorUtils.func_log()
    def stop_check(self):
        """
        VoiceControl :: stop check sound
        停止声音检测
        :return: {'RESULT': '1',  "-1":异常 "0":失败 "1":"成功"
          'DESC'='',说明
          "DATAS":{"CHECK":"0","INFO":"0"}断言结果}
        """
        self.check_start = None
        self.duration = None
        self.check_point = None
        if self.check_result == "1":
            return dataUtils.FuncResult(name=self.stop_check, check=self.check_result, desc="check success",
                                        audio=self.error_sound, info=self.check_result).get_data()
        else:
            return dataUtils.FuncResult(name=self.stop_check, check="0", desc="check fail",
                                        audio=self.error_sound, info="0").get_data()
        # return dataUtils.FuncResult(name=self.stop_check, ).get_data()

    @decoratorUtils.check_status_class()
    @decoratorUtils.func_log()
    def get_check(self):
        """
        VoiceControl :: get check sound state
        检查检测声音当前各个状态，包括读取状态、是否正在检测与当前检测结果
        :return: {'RESULT': '1',  "-1":异常 "0":失败 "1":"成功"
          'DESC'='',说明
          "DATAS":{"CHECK":"0","INFO":"0"}断言结果}
        """
        # :return: {'RESULT': '1',  "-1":异常 "0":失败 "1":"成功"
        #   'DESC'='',说明
        #   "DATAS":{"INFO":()}}
        #         读取音频是否在运行中 "1":正在读取 "0":停止读取
        #         音频检测是否还在进行中 "1":正在断言 "0":停止断言
        #         测试结果 "1":声音正常 "0":声音异常
        # result = ("1" if self.thread_flag else "0",
        #         #           "0" if self.check_start is None else "1",
        #         #           "1" if self.check_result else "0")
        #         # if reset:
        #         #     self.check_result = True
        #         # return dataUtils.FuncResult(name=self.get_check, info=result).get_data()
        self.check_start = None
        self.duration = None
        self.check_point = None
        if self.check_result == "1":
            return dataUtils.FuncResult(name=self.stop_check, check=self.check_result, desc="check success",
                                        audio=self.error_sound, info=self.check_result).get_data()
        else:
            return dataUtils.FuncResult(name=self.stop_check, check="0", desc="check fail",
                                        audio=self.error_sound, info="0").get_data()

    @decoratorUtils.check_class_param_type()
    @decoratorUtils.check_status_class()
    @decoratorUtils.func_log()
    def save_sound(self, output: str, mode: str = "normal"):
        """
        VoiceControl :: save sound
        保存语音,并从当前位置清除内存中储存的语音数据
        :param output:语音输出文件
        :param mode: normal:正常记录为当前读取的所有的音频数据
                     error：保存错误数据
        :return: {'RESULT': '1',  "-1":异常 "0":失败 "1":"成功"
          'DESC'='',说明}
        """
        save_frame = list(self.frames)
        if len(save_frame) == 0:
            self.logger.debug("VoiceControl :: frame is empty! please read audio")
            return
        wf = wave.open(output, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.my_audio.get_sample_size(self.FORMAT))
        wf.setframerate(self.rate)
        if mode == "normal":
            wf.writeframes(b''.join(save_frame))
        else:
            if self.error_frames is not None:
                self.error_frames.extend(save_frame)
                # print("error frame", len(self.error_frames))
                wf.writeframes(b''.join(self.error_frames))
            else:
                wf.writeframes(b''.join(save_frame))
        wf.close()
        #
        # import librosa
        # f = open('./test.wav', 'rb')
        # b = f.read()
        # v = self._add_wav_head(b''.join(save_frame))
        # print(b==v)
        self.frames.clear()
        if self.error_frames is not None:
            self.error_frames.clear()
        self.logger.debug(f"VoiceControl :: save audio success! {output}")
        return dataUtils.FuncResult(name=self.save_sound, ).get_data()

    def _get_check_point(self):
        if self.interval is None:
            return None
        i = 1
        while True:
            point = self.check_start + self.interval * i
            yield point
            i += 1
            if self.duration is not None and self.check_start is None and point >= self.check_start + self.duration:
                break

    def _cal_sinad(self, sfa, max_index, w):
        """
        根据FFT的幅值结果，计算信纳比SINAD的函数
        :param sfa:
        :param max_index:
        :param w:
        :return:信噪比
        """
        # 第一个参数是FFT的结果
        sfa = sfa ** 2 * (self.chunk / self.rate)  # 将信号折算成能量， 有效值的平方* 采样时间 = 时域能量
        # s_max = max(sfa)    # 查找最大值
        # max_index = list(sfa).index(s_max)
        # max_index = np.argmax(sfa)
        # 查找最大值所在的位置，但index()方法只有列表有，所以先将其转回为列表再查找
        index_low = max_index - w  # 选取窗口的下限
        index_high = max_index + w  # 选取窗口的上限
        signal_pow = sum(sfa[index_low:index_high])  # 选取窗口内的信号之和
        noise_pow = sum(sfa) - signal_pow  # 计算噪声能量
        sinad = 10 * np.log10(signal_pow / noise_pow)
        return sinad

    def _audio_callback(self, in_data, frame_count, time_info, status):
        self.audio_data_queue.put(in_data)
        # self.ad_rdy_ev.set()
        if self.counter <= 0:
            self.logger.debug('VoiceControl :: counter is 0')
            return None, pyaudio.paComplete
        else:
            return None, pyaudio.paContinue

    def _read_audio(self):
        self.thread_flag = True
        mute_count = 0
        self.logger.debug("VoiceControl :: read audio start!!")
        point = None
        save_audio_time = None
        while self.thread_flag:
            if self.check_start is not None:
                # 检测失败达到次数，则结果为失败，不在进行检测
                if mute_count >= self.count:
                    self.logger.debug("VoiceControl :: check complete! check fail!!!,wait 20s read audio stop!")
                    self.check_result = "0"
                    self.check_start = None
                    self.duration = None
                    mute_count = 0
                    save_audio_time = time.time() + self.ERROR_WAIT_SECOND
                    print(save_audio_time)
                    # 检测时间到，则不在进行检测
                if self.duration is not None:
                    if time.time() - self.check_start >= int(self.duration):
                        self.logger.debug("VoiceControl :: check complete! result : success!")
                        mute_count = 0
                        self.check_result = "1"
                        self.check_start = None
                        self.logger.debug("VoiceControl :: It's time to save success audio")
                        pass_audio_file = os.path.join(self.save_audio_dir,
                                                       f"pass_{datetime.now().strftime('%Y%m%d%H%M%S')}.wav")
                        print("pass " * 10, pass_audio_file)
                        self.save_sound(output=pass_audio_file)
            if not self.audio_steam.is_active():
                self.logger.debug("VoiceControl :: stream is dead!")
                break

            # self.ad_rdy_ev.wait(timeout=1000)
            if not self.audio_data_queue.empty():
                # process audio msg_data here
                data = self.audio_data_queue.get()

                # 读干净了，再接着往下走
                # while not self.audio_data_queue.empty():
                #     self.audio_data_queue.get()
                if not self.audio_data_queue.empty():
                    self.audio_data_queue.queue.clear()
                self.frames.append(data)

                if self.booming_to_check_open:
                    try:
                        self.booming_to_check.append(data)
                        if len(self.booming_to_check) == self.booming_to_check.maxlen:
                            print('to_check buffer full, may cause missing sounds')
                    except:
                        pass
                # print("frame", len(self.frames))
                # 如果断言失败了，则无需进行判断，直接记录音频即可
                # print("!"*10,self.check_result)

                # 断言失败，保存错误音频
                if self.check_result == "0":
                    # print("------",time.time())
                    # print("------",save_audio_time)
                    # 检测时间到了，返回即可
                    if save_audio_time is None:
                        continue
                    if time.time() >= save_audio_time:
                        self.logger.debug("VoiceControl :: It's time to save error audio")
                        err_audio_file = os.path.join(self.save_audio_dir,
                                                      f"error_{datetime.now().strftime('%Y%m%d%H%M%S')}.wav")
                        print("err " * 10, err_audio_file)
                        self.save_sound(output=err_audio_file, mode="error")
                        # 失败不停止
                        # break
                        save_audio_time = None
                    continue
                # --------------------------------------------------
                # 实际的波形
                # 16bit
                rt_data = np.frombuffer(data, np.dtype('<i2'))
                rt_data = rt_data / rt_data.size  # 归一化
                # --------------------------------------------------
                # --------------------------------------------------
                # FFT
                # print(f"rt_data {rt_data}")
                _rt_data = 1.852 * rt_data * self.window
                # rt_data = _rt_data
                # print(f"rt_data\duration{max(rt_data)}")
                fft_temp_data = fftpack.fft(_rt_data, _rt_data.size, overwrite_x=True)
                # fft_temp_data = fftpack.fft(_rt_data, _rt_data.size)
                # 取模(abs绝对值) 得到振幅谱
                _fft_data = np.abs(fft_temp_data) / _rt_data.size
                # 取半处理
                fft_data = _fft_data[0:fft_temp_data.size // 2 + 1]
                # --------------------------------------------------
                # --------------------------------------------------
                # MaxFreq
                n = np.argmax(fft_data)
                # freq_data = float(float(n - 1) * float(rate) / float(chunk))
                freq_data = n * (self.rate / self.chunk)
                sinad = self._cal_sinad(fft_data, n, 5)
                AS = AudioSegment(data, sample_width=self.sample_width, frame_rate=self.rate, channels=self.channels)
                # dbFS
                _dbfs_data = AS.dBFS
                dbfs_data = _dbfs_data * -1
                level = AS.rms

                # print(f"dbfs:\duration{dbfs_data};\tlevel:\duration{level};\tf:\duration{freq_data};\tsinad:\duration{sinad};")

                if self.check_start is not None:
                    if self.interval is not None:
                        try:
                            if point is not None and time.time() < point:
                                continue
                            point = next(self.check_point)
                        except:
                            self.logger.debug("VoiceControl :: check time point is end!")
                            self.check_start = None
                            continue

                    self.logger.debug("VoiceControl :: -------->check audio")
                    # 断言分贝
                    if self.dbfs_limit is not None:
                        if self.expect is True:
                            if dbfs_data >= self.dbfs_limit:
                                mute_count += 1
                                self.logger.debug(f"VoiceControl :: dbfs is lower ! check fail {mute_count}")
                                continue
                            else:
                                if self.row:
                                    mute_count = 0
                        else:
                            if dbfs_data < self.dbfs_limit:
                                mute_count += 1
                                self.logger.debug(f"VoiceControl :: dbfs is lower ! check fail {mute_count}")
                                continue
                            else:
                                if self.row:
                                    mute_count = 0

                    # 断言级别
                    if self.level_limit is not None:
                        if self.expect is True:
                            if level < self.level_limit:
                                mute_count += 1
                                self.logger.debug(f"VoiceControl :: level is lower ! check fail {mute_count}")
                                continue
                            else:
                                if self.row:
                                    mute_count = 0
                        else:
                            if level > self.level_limit:
                                mute_count += 1
                                self.logger.debug(f"VoiceControl :: level is lower ! check fail {mute_count}")
                                continue
                            else:
                                if self.row:
                                    mute_count = 0

                    # 断言频率
                    if self.freq_limit is not None:
                        if self.expect is True:
                            if self.freq_limit - 100 > float(freq_data) or float(freq_data) > self.freq_limit + 100:
                                mute_count += 1
                                self.logger.debug(f"VoiceControl :: freq is lower ! check fail {mute_count}")
                                continue
                            else:
                                if self.row:
                                    mute_count = 0
                        else:
                            if self.freq_limit - 100 < float(freq_data) < self.freq_limit + 100:
                                mute_count += 1
                                self.logger.debug(f"VoiceControl :: freq is lower ! check fail {mute_count}")
                                continue
                            else:
                                if self.row:
                                    mute_count = 0

                    # 断言信纳比
                    if self.sinad_limit is not None:
                        if self.expect is True:
                            if sinad < self.sinad_limit:
                                mute_count += 1
                                self.logger.debug(f"VoiceControl :: sinad is lower ! check fail {mute_count}")
                                continue
                            else:
                                if self.row:
                                    mute_count = 0
                        else:
                            if sinad > self.sinad_limit:
                                mute_count += 1
                                self.logger.debug(f"VoiceControl :: sinad is lower ! check fail {mute_count}")
                                continue
                            else:
                                if self.row:
                                    mute_count = 0
            # self.ad_rdy_ev.clear()
        self.logger.debug("VoiceControl :: read audio complete!")

    @decoratorUtils.check_class_param_type()
    @decoratorUtils.check_status_class()
    def add_wav_head(self, data : bytes):
        head = b''
        head += b'RIFF'  # 资源交换文件标志
        head += (len(data) + 0x2c - 0x8).to_bytes(4, byteorder='little')  # 下个地址开始文件总字节数
        head += b'WAVE'  # WAV文件标志
        head += b'fmt '  # fmt格式标志
        head += (0x10).to_bytes(4, byteorder='little')  # 块大小
        head += (0x1).to_bytes(2, byteorder='little')  # 格式种类-线性PCM编码(0x0001)
        head += self.channels.to_bytes(2, byteorder='little')  # 通道数
        head += self.rate.to_bytes(4, byteorder='little')  # 采样率
        head += (self.rate * 2).to_bytes(4, byteorder='little')  # 每秒字节数
        head += (2).to_bytes(2, byteorder='little')  # 块对齐
        head += (0x10).to_bytes(2, byteorder='little')  # 采样比特数
        head += b'data'  # data块标志
        head += len(data).to_bytes(4, byteorder='little')  # data长度
        return head + data

    def booming_save(self, save_frame : list):
        if len(save_frame) == 0:
            self.logger.debug("VoiceControl :: to_save queue is empty!")
            return

        try:
            if not os.path.exists(self.booming_output_dir):
                os.makedirs(self.booming_output_dir)
        except:
            self.logger.error('VoiceControl :: booming_save can\'t create wav output directory')
            return

        wf = wave.open(self.booming_output_dir + '\\' + str(self.booming_save_cnt) + '.wav', 'wb')
        self.logger.debug("VoiceControl :: open output wav file success")
        self.booming_save_cnt += 1
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.my_audio.get_sample_size(self.FORMAT))
        wf.setframerate(self.rate)
        wf.writeframes(b''.join(save_frame))
        wf.close()
        self.logger.debug("VoiceControl :: save wav sample success")

    @decoratorUtils.check_class_param_type()
    @decoratorUtils.check_status_class()
    @decoratorUtils.func_log()
    def booming_check(self, check_every_n_data : int = 3, drop_single : bool = True):
        '''
        VoiceControl :: booming sound check
        爆破音检测，调用start_booming_check后开启检测，调用stop_booming_check后停止检测
        :return: {'RESULT': '1',  "-1":异常 "0":失败 "1":"成功"
          'DESC'='',说明}
        '''
        self.has_booming = False
        self.logger.debug('VoiceControl :: booming check start')

        try:
            k=0 # for debug log
            while self.booming_start:
            # for i in range(1):
                while len(self.booming_to_check) < check_every_n_data:  #  队列中等待data不够一次检测
                    time.sleep(max(0.1 * (check_every_n_data - len(self.booming_to_check)), 0))  # 等待填充够一次检测时间
                    if not self.booming_start:
                        return

                #  取出data，裹上文件头，封包成IO文件，给librosa读取
                byte_data = b''.join(list(self.booming_to_check)[:check_every_n_data])
                for _ in range(check_every_n_data):  # 把检测的帧出队，送入保存缓冲队列
                    self.booming_to_save.append(self.booming_to_check.popleft())

                if self.booming_now_saving:  #  检测到爆破音且需要保存样本，计数到半个缓冲区后开始导出
                    self.booming_start_time += check_every_n_data
                    if self.booming_start_time >= int(self.booming_to_save.maxlen / 2):  # 导出
                        save_thread = Thread(target=self.booming_save, args=(list(deepcopy(self.booming_to_save)), ), daemon=True)
                        save_thread.start()
                        self.booming_now_saving = False

                byte_file = BytesIO(self.add_wav_head(byte_data))  # 转化成类FILE对象送给librosa读取
                
                # 转换成频谱数据张量送给分类模型，data中至少存在x帧异常后认为data含爆破音
                y, sr = librosa.load(byte_file, sr=48000)
                feature = self.booming_tensorizer.transform(y, sr)
                score, pred = self.booming_model.predict(feature)
                score = score.flatten()

                if drop_single:  #  至少连续2帧都是爆破音才判定为有爆破音，丢弃孤立的一帧爆破音
                    for i in range(1, len(pred) - 1):
                        if pred[i - 1] + pred[i + 1] < 1:
                            pred[i] = 0

                self.booming_score_record = np.concatenate((self.booming_score_record, score))  #  记录打分历史
                self.booming_res_record = np.concatenate((self.booming_res_record, pred))
                if np.sum(pred) > 0:
                    self.booming_times += 1
                    self.has_booming = True
                    self.now_booming = True
                    self.booming_time_record.append(time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime()))
                    if self.booming_check_save and not self.booming_now_saving:
                        self.booming_start_time = 0
                        self.booming_now_saving = True
                else:
                    self.now_booming = False

        except Exception as e:
            self.logger.error(f"VoiceControl :: {e}")
            self.stop_booming_check()
            return dataUtils.FuncResult(result="-1", desc=str(e), name=self.booming_check).get_data()


    @decoratorUtils.check_class_param_type()
    @decoratorUtils.check_status_class()
    @decoratorUtils.func_log()
    def start_booming_check(self, check_every_n_data : int = 3, que_buffer : int = 100, check_save: bool = True, save_duration: float = 5., threshold: float = 0.538, drop_single: bool = True):
        '''
        VoiceControl :: start booming check
        开启爆破音检测，初始化检测配置
        :param check_every_n_data: 每N个data检测一次，即每N/10秒检测一次
        :param que_buffer: 待检测缓冲区大小，防止检测速度慢于音频播放速度
        :param check_save: 检测到爆破音后是否自动保存爆破音附近样本
        :param save_duration: 保存爆破音前后x/2秒样本，共x秒
        :param threshold: 爆破音判定阈值，分数在0~1, 为0则全判定为爆破音，为1则全判定为非爆破音
        :param drop_single: 是否丢弃孤立的一帧爆破音，即该爆破音前一帧和后一帧都是非爆破音
        :return: {'RESULT': '1',  "-1":异常 "0":失败 "1":"成功"
          'DESC'='',说明
          }
        '''
        try:
            if self.booming_thred_start: # 防止重复开爆破音检测线程
                self.logger.debug("VoiceControl :: booming check thread already started, call STOP first to restart")
                return dataUtils.FuncResult(result="-1", desc='booming check thread already started, call STOP first to restart',
                                            name=self.start_booming_check).get_data()
            else:
                self.booming_thred_start = True

            if check_every_n_data < 1:
                self.logger.error("VoiceControl :: check_every_n_data too small")
                return dataUtils.FuncResult(result="-1", desc='check_every_n_data too small', name=self.start_booming_check).get_data()
            if check_every_n_data >= que_buffer:
                self.logger.error("VoiceControl :: check_every_n_data is larger than buffer")
                return dataUtils.FuncResult(result="-1", desc='check_every_n_data is larger than buffer', name=self.start_booming_check).get_data()

            self.booming_to_check = deque(maxlen=que_buffer)   # 待检测队列，与frames同步入队，检测一个出队一个

            self.booming_check_save = check_save
            if self.booming_check_save:
                if not save_duration > 0.1:
                    self.logger.error("VoiceControl :: save_duration too short")
                    return dataUtils.FuncResult(result="-1", desc='save_duration too short', name=self.start_booming_check).get_data()
                self.booming_to_save = deque(maxlen=round(save_duration * self.SECOND_FRAME))  # 已检测队列，to_check出队数据，用于保存历史x秒以备导出

            self.booming_to_check.clear()  # 初始化清空
            self.booming_to_save.clear()
            self.booming_score_record = np.array([])
            self.booming_res_record = np.array([])
            self.booming_time_record = []

            if self.booming_model == None:
                load_res = self.model_load()
                if load_res['RESULT'] != '1':
                    self.logger.error(f"VoiceControl :: {load_res['DESC']}")
                    return dataUtils.FuncResult(result="-1", desc=str(load_res['DESC']), name=self.start_booming_check).get_data()

            self.booming_threshold = threshold
            self.booming_model.set_param(self.booming_threshold)

            self.booming_start = True  # 开启检测flag
            self.booming_to_check_open = True  # 检测队列开始入队

            boomingcheck_thread = Thread(target=self.booming_check, args=(check_every_n_data, drop_single, ), daemon=True)
            boomingcheck_thread.start()

            return dataUtils.FuncResult(result="1", desc='booming check start success', name=self.start_booming_check).get_data()

        except Exception as e:
            self.logger.error(f"VoiceControl :: {e}")
            return dataUtils.FuncResult(result="-1", desc=str(e), name=self.start_booming_check).get_data()

    @decoratorUtils.check_class_param_type()
    @decoratorUtils.check_status_class()
    @decoratorUtils.func_log()
    def stop_booming_check(self):
        '''
        VoiceControl :: stop booming sound check
        停止爆破音检测服务
        :return: {'RESULT': '1',  "-1":异常 "0":失败 "1":"成功"
          'DESC'='',说明
          'SCOREHISTORY'=np.array([]), 算法对所有帧打分
          'TIMEHISTORY'=[],出现爆破音时间的list
          }
        '''
        try:
            self.booming_to_check_open = False  #  self.booming_to_check停止入队
            self.booming_start = False  #  检测线程停止
            self.booming_thred_start = False  # 检测线程可以被执行

            if self.booming_now_saving:  #  停止前还有在保存的样本，直接保存当前采集结果
                save_thread = Thread(target=self.booming_save, args=(list(deepcopy(self.booming_to_save)), ), daemon=True)
                save_thread.start()

            self.booming_now_saving = False
            self.has_booming = False
            self.booming_check_save = False
            self.booming_start_time = 0

            self.booming_to_save.clear()
            self.booming_to_check.clear()
            self.booming_to_save = deque(maxlen=0)
            self.booming_to_check = deque(maxlen=0)

            return dataUtils.FuncResult(result="1", desc='booming check stop success', name=self.stop_booming_check, booming_times=self.booming_time_record, socres=self.booming_score_record, labels=self.booming_res_record, threshold=self.booming_model.threshold).get_data()
        except Exception as e:
            self.logger.error(f"VoiceControl :: {e}")
            return dataUtils.FuncResult(result="-1", desc=str(e), name=self.stop_booming_check, booming_times=self.booming_time_record, scores=self.booming_score_record, labels=self.booming_res_record, threshold=self.booming_model.threshold).get_data()

    @decoratorUtils.check_class_param_type()
    @decoratorUtils.check_status_class()
    @decoratorUtils.func_log()
    def model_load(self):
        '''
        VoiceControl :: load machine learning model
        读取爆破音检测模型
        :param path: 路径
        :return: {'RESULT': '1',  "-1":异常 "0":失败 "1":"成功"
          'DESC'='',说明}
        '''
        try:
            path = rf"{get_bmclient()}/bmatEnv/Lib/site-packages/bmdriver/resnet_model.pkl"
            with open(path, 'rb') as fin:
                model = pickle.load(fin)
                self.booming_model = ML_model([model])
                return dataUtils.FuncResult(result="1", desc='model load success', name=self.model_load).get_data()
        except Exception as e:
            self.booming_model = None
            self.logger.error(f"VoiceControl :: {e}")
            return dataUtils.FuncResult(result="-1", desc=str(e), name=self.model_load).get_data()

    @decoratorUtils.check_class_param_type()
    @decoratorUtils.check_status_class()
    @decoratorUtils.func_log()
    def set_model_thresh(self, thresh : float = 10.):
        """
        VoiceControl :: set model threshold
        为检测模型设置阈值
        :param thresh: 阈值
        :return: {'RESULT': '1',  "-1":异常 "0":失败 "1":"成功"
          'DESC'='',说明}
        """
        try:
            if self.booming_model == None: # 模型未加载，尝试加载模型
                load_res = self.model_load()
                if load_res['RESULT'] != '1':
                    self.logger.error(f"VoiceControl :: {load_res['DESC']}")
                    return dataUtils.FuncResult(result="-1", desc=str(load_res['DESC']), name=self.set_model_thresh).get_data()
            self.booming_model.set_thresh(thresh)
            return dataUtils.FuncResult(result="1", desc='set threshold success', name=self.set_model_thresh).get_data()
        except Exception as e:
            return dataUtils.FuncResult(result="-1", desc=str(e), name=self.set_model_thresh).get_data()

    @decoratorUtils.check_class_param_type()
    @decoratorUtils.check_status_class()
    @decoratorUtils.func_log()
    def get_has_booming(self):
        """
        VoiceControl :: get self.has_booming
        获取self.has_booming值（bool）
        :return: {'RESULT': '1',  "-1":异常 "0":失败 "1":"成功"
          'DESC'='',说明}
        """
        return dataUtils.FuncResult(result="1", desc="get self.has_booming success", name=self.get_has_booming, hasbooming=self.has_booming).get_data()


class ML_model(object):

    def __init__(self, models, history : int = 5, threshold : float = 0.538, quantile : float = 0.8):
        '''
        :param models: list [predicting-model]
                    the sub-models used
        :param threshold: float
                    threshold of LGBM model
        '''
        self.models = models
        self.threshold = threshold
        self.history = [0] * history
        self.his_len = history
        self.quantile = quantile

    def set_param(self, threshold : float = 0, quantile : float = 0.8):
        self.threshold = threshold
        self.quantile = quantile

    def set_thresh(self, threshold : float = 0):
        self.threshold = threshold

    def set_quantile(self, quantile : float = 0.8):
        self.quantile = quantile

    def predict(self, features):
        if not 0 < self.quantile < 1:
            print('invalid quantile, set to default 0.8')
            self.quantile = 0.8
        score = self.models[0](np.concatenate((features['spec'], features['mfcc'], features['stft'], features['mel']), axis=1))
        # self.history = self.history[len(pred):] + list(pred)
        # self.sort_his = sorted(self.history)
        # pred = pred - self.sort_his[round(self.quantile * self.his_len)] * np.ones(len(pred))
        pred = [int(e >= self.threshold) for e in score]
        return score, pred

class AudioTenserize(object):
    '''
    extract Tensorized features from wav file
    '''
    def __init__(self, n_feature=200):
        '''
        :param n_feature: int
                    dimension of the mfcc feature
        '''
        super(AudioTenserize, self).__init__()
        self.n_feature = n_feature
        # self._n_fft = n_fft
        # self._hop_length = hop_length
        # self._win_length = win_length
        # self._n_feature = n_feature

    def transform(self, y, sr):
        '''
        :param y: list
                    the wave form sequence
        :param sr: int
                    sample rate
        :param normalize: bool
                    is normalization needed
        :return: dict
                    spectrum feature and mfcc feature
        '''
        features = {}

        # spectrum feature
        spec = librosa.amplitude_to_db(abs(librosa.stft(y)))
        features['raw_spec'] = spec.T
        spec = librosa.util.normalize(spec).T
        features['spec'] = spec

        # MFCC feature
        mfcc = librosa.feature.mfcc(y=y, sr=sr, center=True, n_mfcc=self.n_feature)
        mfcc = librosa.util.normalize(mfcc).T
        features['mfcc'] = mfcc

        # stft feature
        stft = librosa.feature.chroma_stft(y=y, sr=sr, center=True, n_chroma=self.n_feature)
        stft = librosa.util.normalize(stft).T
        features['stft'] = stft

        # mel feature
        mel = librosa.feature.melspectrogram(y=y, sr=sr, center=True, n_mels=self.n_feature)
        mel = librosa.util.normalize(mel).T
        features['mel'] = mel

        return features

class Linear: # 神经网络全连接层
    def __init__(self, i_d, o_d):
        self.w = np.random.randn(o_d, i_d)
        self.b = np.random.randn(o_d)

    def __call__(self, x):
        return np.matmul(x, self.w) + self.b

class Sigmoid: # 神经网络sigmoid激活函数
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

class Tanh:  # 神经网络双曲正弦激活函数
    def __call__(self, x):
        return np.tanh(x)

class BatchNorm:  # 神经网络batch norm层
    def __init__(self, i_d):
        self.eps = 1e-5
        self.running_mean = np.random.randn(i_d, i_d)
        self.running_var = np.random.randn(i_d, i_d)

    def __call__(self, x):
        return (x - self.running_mean) / np.sqrt(self.running_var + self.eps)

class NP_block:

    def __init__(self, dim):
        self.ffn = [Linear(dim, dim),
                    BatchNorm(dim),
                    Tanh(),
                    Linear(dim, dim),
                    Tanh(),
                    Linear(dim, dim)]
        self.act = Tanh()

    def __call__(self, x):
        inp = x
        for h in self.ffn:
            x = h(x)
        return x + inp

class NP_net:

    def __init__(self, inp_dim, dim, num_blocks):
        self.ffn_in = [Linear(inp_dim, dim),
                       BatchNorm(dim),
                       Tanh()]

        self.nets = []
        for i in range(num_blocks):
            self.nets.append(NP_block(dim))

        self.ffn_out = [Linear(dim, 1),
                        Sigmoid()]

    def __call__(self, x):
        for h in self.ffn_in:
            x = h(x)
        for h in self.nets:
            x = h(x)
        for h in self.ffn_out:
            x = h(x)
        return x

# if __name__ == '__main__':
#     test = VoiceClass()
#     y, sr = librosa.load('E:\\audio_data\\new\\abnormal_2019-08-01_400_87.wav', sr=None)
#     test.model_load()
#     test.booming_check(0, True, y, sr)

