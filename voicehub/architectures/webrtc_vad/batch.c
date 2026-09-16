/* VoiceHub batch framing around the pinned WebRTC API. */
#include <stdint.h>
#include <string.h>
#include "webrtc/common_audio/vad/include/webrtc_vad.h"

int voicehub_webrtc_frames(const int16_t *samples, size_t count, int rate,
                          size_t frame_size, int mode, uint8_t *flags) {
    if (!frame_size || frame_size > 1440 || mode < 0 || mode > 3 ||
        WebRtcVad_ValidRateAndFrameLength(rate, frame_size) != 0) return -1;
    VadInst *vad = WebRtcVad_Create();
    if (!vad) return -1;
    if (WebRtcVad_Init(vad) != 0 || WebRtcVad_set_mode(vad, mode) != 0) {
        WebRtcVad_Free(vad);
        return -1;
    }
    int16_t tail[1440];
    size_t frame = 0;
    for (size_t offset = 0; offset < count; offset += frame_size, ++frame) {
        const int16_t *data = samples + offset;
        if (count - offset < frame_size) {
            memset(tail, 0, frame_size * sizeof(int16_t));
            memcpy(tail, data, (count - offset) * sizeof(int16_t));
            data = tail;
        }
        int result = WebRtcVad_Process(vad, rate, data, frame_size);
        if (result < 0) {
            WebRtcVad_Free(vad);
            return -1;
        }
        flags[frame] = result > 0;
    }
    WebRtcVad_Free(vad);
    return 0;
}
