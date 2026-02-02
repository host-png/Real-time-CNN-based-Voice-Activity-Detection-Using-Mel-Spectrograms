//#define NOMINMAX
//#include <windows.h>
//#include <mmdeviceapi.h>
//#include <audioclient.h>
//#include <iostream>
//#include <vector>
//#include <cmath>
//#include <csignal>
//#include <fftw3.h>
//#include <torch/torch.h>
//#include <torch/script.h>
//
//#pragma comment(lib, "ole32.lib")
//#pragma comment(lib, "mmdevapi.lib")
//
//#ifndef M_PI
//#define M_PI 3.14159265358979323846
//#endif
//
//// ------------------------
//// 全局退出标志
//// ------------------------
//volatile bool g_stop = false;
//void signalHandler(int) { g_stop = true; }
//
//// ------------------------
//// FIR 重采样 48k -> 16k
//// ------------------------
//class Resample48kTo16k {
//public:
//    Resample48kTo16k() { initFilter(); delay.assign(TAPS, 0.0f); }
//
//    void process(const float* in, size_t n, std::vector<float>& out) {
//        for (size_t i = 0; i < n; ++i) {
//            push(in[i]);
//            if (++decimCnt == DECIM) {
//                decimCnt = 0;
//                out.push_back(convolve());
//            }
//        }
//    }
//
//private:
//    static constexpr int DECIM = 3;
//    static constexpr int TAPS = 63;
//
//    std::vector<float> fir;
//    std::vector<float> delay;
//    int wp = 0;
//    int decimCnt = 0;
//
//    void push(float x) { delay[wp] = x; if (++wp >= TAPS) wp = 0; }
//    float convolve() {
//        float acc = 0.0f;
//        int idx = wp;
//        for (int i = 0; i < TAPS; i++) {
//            acc += fir[i] * delay[idx];
//            if (--idx < 0) idx = TAPS - 1;
//        }
//        return acc;
//    }
//
//    void initFilter() {
//        fir.resize(TAPS);
//        float fs = 48000.0f;
//        float fc = 7500.0f;
//        float norm = fc / fs;
//        int M = TAPS - 1;
//
//        for (int n = 0; n < TAPS; ++n) {
//            int k = n - M / 2;
//            float sinc = (k == 0) ? 2.0f * norm : sinf(2 * M_PI * norm * k) / (M_PI * k);
//            float win = 0.54f - 0.46f * cosf(2 * M_PI * n / M);
//            fir[n] = sinc * win;
//        }
//
//        float sum = 0.0f;
//        for (float v : fir) sum += v;
//        for (float& v : fir) v /= sum;
//    }
//};
//
//// ------------------------
//// Mel Filter 创建
//// ------------------------
//float hz_to_mel(float hz) { return 2595.0f * log10(1.0f + hz / 700.0f); }
//float mel_to_hz(float mel) { return 700.0f * (pow(10.0f, mel / 2595.0f) - 1.0f); }
//
//torch::Tensor create_mel_filter(int n_fft, int sample_rate, int n_mels) {
//    int n_fft_half = n_fft / 2 + 1;
//    float f_min = 0.0f, f_max = sample_rate / 2.0f;
//    float mel_min = hz_to_mel(f_min), mel_max = hz_to_mel(f_max);
//
//    std::vector<float> mel_points(n_mels + 2);
//    for (int i = 0; i < n_mels + 2; ++i) mel_points[i] = mel_min + i * (mel_max - mel_min) / (n_mels + 1);
//
//    std::vector<float> hz_points(n_mels + 2);
//    for (int i = 0; i < n_mels + 2; ++i) hz_points[i] = mel_to_hz(mel_points[i]);
//
//    std::vector<int> bin(n_mels + 2);
//    for (int i = 0; i < n_mels + 2; ++i) bin[i] = std::floor((n_fft + 1) * hz_points[i] / sample_rate);
//
//    torch::Tensor filterbank = torch::zeros({ n_mels, n_fft_half }, torch::kFloat32);
//
//    for (int m = 1; m <= n_mels; ++m) {
//        int f_m_minus = bin[m - 1];
//        int f_m = bin[m];
//        int f_m_plus = bin[m + 1];
//
//        for (int k = f_m_minus; k < f_m; ++k)
//            filterbank[m - 1][k] = (k - f_m_minus) / float(f_m - f_m_minus);
//        for (int k = f_m; k < f_m_plus; ++k)
//            filterbank[m - 1][k] = (f_m_plus - k) / float(f_m_plus - f_m);
//    }
//
//    return filterbank;
//}
//
//// ------------------------
//// STFT + Mel
//// ------------------------
//torch::Tensor compute_mel(const std::vector<float>& audio,
//    const torch::Tensor& mel_filter,
//    int n_fft = 400,
//    int hop_length = 400)
//{
//    int n_frames = (audio.size() - n_fft) / hop_length + 1;
//    int n_mels = mel_filter.size(0);
//    torch::Tensor mel_tensor = torch::zeros({ n_mels, n_frames }, torch::kFloat32);
//
//    std::vector<float> window(n_fft);
//    for (int i = 0; i < n_fft; ++i) window[i] = 0.5f - 0.5f * cos(2 * M_PI * i / (n_fft - 1));
//
//    std::vector<float> in(n_fft);
//    std::vector<fftwf_complex> out(n_fft / 2 + 1);
//    fftwf_plan plan = fftwf_plan_dft_r2c_1d(n_fft, in.data(), out.data(), FFTW_ESTIMATE);
//
//    for (int frame = 0; frame < n_frames; ++frame) {
//        for (int i = 0; i < n_fft; ++i) in[i] = audio[frame * hop_length + i] * window[i];
//        fftwf_execute(plan);
//
//        torch::Tensor mag = torch::zeros({ n_fft / 2 + 1 }, torch::kFloat32);
//        for (int k = 0; k < n_fft / 2 + 1; ++k)
//            mag[k] = std::sqrt(out[k][0] * out[k][0] + out[k][1] * out[k][1]);
//
//        torch::Tensor mel_col = torch::matmul(mel_filter, mag);
//        mel_col = 10.0 * torch::log10(mel_col + 1e-9);
//        mel_tensor.index_put_({ torch::indexing::Slice(), frame }, mel_col);
//    }
//
//    fftwf_destroy_plan(plan);
//    return mel_tensor.unsqueeze(0).unsqueeze(0); // [1,1,n_mels,time]
//}
//
//// ------------------------
//// 主函数
//// ------------------------
//int main() {
//    std::signal(SIGINT, signalHandler);
//
//    HRESULT hr = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
//    if (FAILED(hr)) { std::cout << "COM 初始化失败\n"; return -1; }
//
//    // 加载 libtorch 模型
//    std::string model_path = R"(E:\AI\PythonProject\trainCnnOk\cnnNetVoice_9.pt)";//C:\Users\pingtang\Documents\cTestOrProject\pyModelTest\pyModelTest\cnnNetVoice_8.pt
//    torch::jit::script::Module module = torch::jit::load(model_path);
//    std::cout << "模型加载成功！" << std::endl;
//
//    // 创建 WASAPI
//    IMMDeviceEnumerator* pEnum = nullptr;
//    CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL,
//        __uuidof(IMMDeviceEnumerator), (void**)&pEnum);
//    IMMDevice* pDevice = nullptr;
//    pEnum->GetDefaultAudioEndpoint(eRender, eConsole, &pDevice);
//    pEnum->Release();
//
//    IAudioClient* pAudioClient = nullptr;
//    pDevice->Activate(__uuidof(IAudioClient), CLSCTX_ALL, nullptr, (void**)&pAudioClient);
//    pDevice->Release();
//
//    WAVEFORMATEX* pFormat = nullptr;
//    pAudioClient->GetMixFormat(&pFormat);
//
//    HANDLE hEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
//    REFERENCE_TIME bufferDuration = 10000000;
//    pAudioClient->Initialize(AUDCLNT_SHAREMODE_SHARED,
//        AUDCLNT_STREAMFLAGS_LOOPBACK | AUDCLNT_STREAMFLAGS_EVENTCALLBACK,
//        bufferDuration, 0, pFormat, nullptr);
//    pAudioClient->SetEventHandle(hEvent);
//
//    IAudioCaptureClient* pCaptureClient = nullptr;
//    pAudioClient->GetService(__uuidof(IAudioCaptureClient), (void**)&pCaptureClient);
//
//    Resample48kTo16k resampler;
//
//    // 创建 Mel 滤波矩阵
//    int n_fft = 400;
//    int hop_length = 400;
//    int n_mels = 50;
//    torch::Tensor mel_filter = create_mel_filter(n_fft, 16000, n_mels);
//
//    // 开始捕获
//    pAudioClient->Start();
//    std::cout << "开始捕获系统音频，按 Ctrl+C 停止...\n";
//
//    std::vector<float> audio_buffer16k;
//
//    while (!g_stop) {
//        DWORD wait = WaitForSingleObject(hEvent, 1000);
//        if (wait != WAIT_OBJECT_0) continue;
//
//        UINT32 packetSize = 0;
//        pCaptureClient->GetNextPacketSize(&packetSize);
//        while (packetSize > 0) {
//            BYTE* pData;
//            UINT32 numFrames;
//            DWORD flags;
//            pCaptureClient->GetBuffer(&pData, &numFrames, &flags, nullptr, nullptr);
//
//            std::vector<float> mono48k(numFrames, 0.0f);
//            if (!(flags & AUDCLNT_BUFFERFLAGS_SILENT)) {
//                float* floatData = reinterpret_cast<float*>(pData);
//                for (UINT32 i = 0; i < numFrames; ++i) {
//                    // 32-bit float -> 16-bit PCM -> [-1,1] float
//                    int16_t sample16 = static_cast<int16_t>(std::round(floatData[i * pFormat->nChannels] * 32768.0f));
//                    mono48k[i] = static_cast<float>(sample16) / 32768.0f;
//                }
//            }
//
//            // 重采样
//            std::vector<float> mono16k;
//            resampler.process(mono48k.data(), mono48k.size(), mono16k);
//
//            // 缓存到总 buffer
//            audio_buffer16k.insert(audio_buffer16k.end(), mono16k.begin(), mono16k.end());
//
//            // 每 slice_samples=50ms处理
//            int slice_samples = 800; // 50ms
//            while (audio_buffer16k.size() >= slice_samples) {
//                std::vector<float> slice(audio_buffer16k.begin(), audio_buffer16k.begin() + slice_samples);
//                audio_buffer16k.erase(audio_buffer16k.begin(), audio_buffer16k.begin() + slice_samples);
//
//                torch::Tensor input_tensor = compute_mel(slice, mel_filter, n_fft, hop_length);
//                std::vector<torch::jit::IValue> inputs{ input_tensor };
//                torch::Tensor output = module.forward(inputs).toTensor();
//                torch::Tensor prob = torch::sigmoid(output);
//                float p = prob.item<float>();
//                HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
//                // 实时刷新一行
//                std::cout << "\r"; // 回到行首
//                if (p > 0.5f) {
//                    SetConsoleTextAttribute(hConsole, FOREGROUND_RED | FOREGROUND_INTENSITY);
//                    std::cout << "有人声  ";
//                }
//                else {
//                    SetConsoleTextAttribute(hConsole, FOREGROUND_RED | FOREGROUND_GREEN | FOREGROUND_BLUE);
//                    std::cout << "无人声  ";
//                }
//                std::cout << std::flush;
//            }
//
//            pCaptureClient->ReleaseBuffer(numFrames);
//            pCaptureClient->GetNextPacketSize(&packetSize);
//        }
//    }
//
//    pAudioClient->Stop();
//    pCaptureClient->Release();
//    pAudioClient->Release();
//    CoTaskMemFree(pFormat);
//    CloseHandle(hEvent);
//    CoUninitialize();
//
//    std::cout << "捕获结束。\n";
//    return 0;
//}
