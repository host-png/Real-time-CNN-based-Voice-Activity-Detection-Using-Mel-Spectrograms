#define NOMINMAX
#include <windows.h>
#include <mmdeviceapi.h>
#include <audioclient.h>
#include <iostream>
#include <vector>
#include <csignal>

#pragma comment(lib, "ole32.lib")
#pragma comment(lib, "mmdevapi.lib")

// ------------------------
// 全局退出标志
// ------------------------
volatile bool g_stop = false;
void signalHandler(int) { g_stop = true; }

int main() {
    std::signal(SIGINT, signalHandler);

    HRESULT hr = CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    if (FAILED(hr)) { std::cout << "COM 初始化失败\n"; return -1; }

    // 创建 WASAPI
    IMMDeviceEnumerator* pEnum = nullptr;
    CoCreateInstance(__uuidof(MMDeviceEnumerator), nullptr, CLSCTX_ALL,
        __uuidof(IMMDeviceEnumerator), (void**)&pEnum);

    IMMDevice* pDevice = nullptr;
    pEnum->GetDefaultAudioEndpoint(eRender, eConsole, &pDevice);
    pEnum->Release();

    IAudioClient* pAudioClient = nullptr;
    pDevice->Activate(__uuidof(IAudioClient), CLSCTX_ALL, nullptr, (void**)&pAudioClient);
    pDevice->Release();

    WAVEFORMATEX* pFormat = nullptr;
    pAudioClient->GetMixFormat(&pFormat);

    HANDLE hEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    REFERENCE_TIME bufferDuration = 10000000;
    pAudioClient->Initialize(AUDCLNT_SHAREMODE_SHARED,
        AUDCLNT_STREAMFLAGS_LOOPBACK | AUDCLNT_STREAMFLAGS_EVENTCALLBACK,
        bufferDuration, 0, pFormat, nullptr);
    pAudioClient->SetEventHandle(hEvent);

    IAudioCaptureClient* pCaptureClient = nullptr;
    pAudioClient->GetService(__uuidof(IAudioCaptureClient), (void**)&pCaptureClient);

    std::cout << "开始捕获系统音频，按 Ctrl+C 停止...\n";
    pAudioClient->Start();

    while (!g_stop) {
        DWORD wait = WaitForSingleObject(hEvent, 1000);
        if (wait != WAIT_OBJECT_0) continue;

        UINT32 packetSize = 0;
        pCaptureClient->GetNextPacketSize(&packetSize);
        while (packetSize > 0) {
            BYTE* pData;
            UINT32 numFrames;
            DWORD flags;
            pCaptureClient->GetBuffer(&pData, &numFrames, &flags, nullptr, nullptr);

            // ------------------- 处理音频数据 -------------------
            if (!(flags & AUDCLNT_BUFFERFLAGS_SILENT)) {
                float* floatData = reinterpret_cast<float*>(pData);
                for (UINT32 i = 0; i < numFrames; ++i) {
                    float sample = floatData[i];  // 原始 float 样本
                    // 这里可以把 sample 保存到文件或者处理
                }
            }

            pCaptureClient->ReleaseBuffer(numFrames);
            pCaptureClient->GetNextPacketSize(&packetSize);
        }
    }

    // ------------------------ 清理 ------------------------
    pAudioClient->Stop();
    pCaptureClient->Release();
    pAudioClient->Release();
    CoTaskMemFree(pFormat);
    CloseHandle(hEvent);
    CoUninitialize();

    std::cout << "捕获结束。\n";
    return 0;
}
