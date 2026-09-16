# WebRTC source provenance and acceleration notice

Upstream: [https://github.com/wiseman/py-webrtcvad](https://github.com/wiseman/py-webrtcvad).

Pinned revision: `e283ca41df3a84b0e87fb1f5cb9b21580a286b09`.
Original directory: `cbits/webrtc`.

License: MIT and BSD-3-Clause. The authoritative license text is retained in
[`THIRD_PARTY_LICENSE`](../THIRD_PARTY_LICENSE) in the parent directory.

Optional native CPU acceleration of the already-ported fixed-point detector; source files are unmodified.

## VoiceHub integration

- **Source**: voicehub/architectures/webrtc_vad/source/webrtc
- **Batch framing**: voicehub/architectures/webrtc_vad/batch.c
- **Loading**: Compiled lazily with local C/C++ compilers on Linux or macOS; cached by source/build digest. Python fallback is used when compilation is unavailable.
- **Tested platform**: Linux x86_64, GCC 16.2.1
- **Differential validation**: 48 configurations covering all supported rates, durations, and modes; 40 sequential frames and a padded tail per configuration, repeated with fresh request state.
- **Runtime observability**: VADOutput.metadata.execution_engine identifies compiled-c or the Python fallback reason.

## Unmodified upstream file checksums

| File | SHA-256 |
| --- | --- |
| `common_audio/signal_processing/complex_bit_reverse.c` | `a395a1116542798cff9ce23a21324a1a62a5566271c16fd5a70c5eb8b9332da7` |
| `common_audio/signal_processing/complex_fft.c` | `13fbb938219892d28590998c469a4bbbb681fc5d6f70943d099f6412229b6d55` |
| `common_audio/signal_processing/complex_fft_tables.h` | `7162c847b052cdb5f86194f5cfa3dac650dffc827e15d1e1481351e208518033` |
| `common_audio/signal_processing/cross_correlation.c` | `9a52c9e7736933b4252ef64394ef1c7e58148d7a1370383e514b95f85f3bfd96` |
| `common_audio/signal_processing/division_operations.c` | `32998618519cfe152e32421a27886043d4707db9ebe0bcff1c45389e564f990c` |
| `common_audio/signal_processing/dot_product_with_scale.cc` | `ff20c38f142b3a170aed3610d88dae5ee6bf3622598c6b97d579f1a4b16c6ff3` |
| `common_audio/signal_processing/dot_product_with_scale.h` | `05d85f25aeffdc1d49fb7f8ff1e30a804aa98b33fbf2f3f094dd50dc366c919a` |
| `common_audio/signal_processing/downsample_fast.c` | `01ebe96c159c93a31f9d66747daa1844ead4744c45850d97824b83e3fc5a35ae` |
| `common_audio/signal_processing/energy.c` | `9d397966daffc7fea32e078695e2de0d3f48beb5eba52d4591b27e128f226b3b` |
| `common_audio/signal_processing/get_scaling_square.c` | `9deb53ae5ca5f3e9f3cbab9d3afaa546cea31e49373d20ef0e5ebd189d3fdd46` |
| `common_audio/signal_processing/include/real_fft.h` | `740a9438a65fe74099790812106488adf9155ab1d3aaaa59eca8c782e550ce41` |
| `common_audio/signal_processing/include/signal_processing_library.h` | `0c89300e9e4a8fef845ccec953d438576fb01c76abdb7a37f7affed7a437c120` |
| `common_audio/signal_processing/include/spl_inl.h` | `6008c182da7189e489ded58bee75b7370e5decbf447fa9758d269d12b248280e` |
| `common_audio/signal_processing/min_max_operations.c` | `59a5210a16b4696c18b3fc80a3470333db6bcca1a5f20f71261e3448d59f6184` |
| `common_audio/signal_processing/resample_48khz.c` | `070839683fa248efb3afa047555263d561a0c449154f6feacb5b9ce7fdc79114` |
| `common_audio/signal_processing/resample_by_2_internal.c` | `e3a4ac3b32428f2f6cc16c5fa3a6df033983d89cd54a778d9405c3b82b618736` |
| `common_audio/signal_processing/resample_by_2_internal.h` | `f4f84d87cbf1a544a2f16f9b72f3896ba8a6ba7e925daa83f4f3ff4ceb539358` |
| `common_audio/signal_processing/resample_fractional.c` | `61b4388968b9b3874308402d85e98ed60cfdb3e804e64945143d7fc2cda48ed1` |
| `common_audio/signal_processing/spl_init.c` | `f825c8b5cc2c3834022a1e062ef46c302b8ea71022fb27e6fd202b5cd75c369f` |
| `common_audio/signal_processing/spl_inl.c` | `68ac5198aa5c77884c760f8b8f1073b9120882fb1bd05b2d2c03ef8152609e1a` |
| `common_audio/signal_processing/spl_sqrt.c` | `fecd0d18251d1ecc888dd68d79edca9d085ba305b6d17db5c37f3719efd9a029` |
| `common_audio/signal_processing/vector_scaling_operations.c` | `9ea07abd1fa3203fb749c8029c3ac33ba14e14ee8c5e7dbd75b0539ec49bf64c` |
| `common_audio/third_party/spl_sqrt_floor/spl_sqrt_floor.c` | `acf7d66683e2f8aac817b76560589eecd2111c72b61f53b42a6844fbbc2703aa` |
| `common_audio/third_party/spl_sqrt_floor/spl_sqrt_floor.h` | `233b5eb7280fd9a46690344c27484c65a91bb3cf51251312954b28476cd5d88c` |
| `common_audio/vad/include/webrtc_vad.h` | `a7a66d658ea942855793b21e3aeeb718295c23b2eaf988ddbc9e866e4d47a369` |
| `common_audio/vad/vad_core.c` | `e73c0a7562225cb36eeca4a9d849cb445a932a8cb4bf948eb11135cfcfc81e57` |
| `common_audio/vad/vad_core.h` | `9d0da84d71ef94b94a532fab1402bf6a8e3b92bdfb310927031485a10b64ac38` |
| `common_audio/vad/vad_filterbank.c` | `df0e67fac5233382aea984713a118180c7e393e8938e13d3ca23d633f7871357` |
| `common_audio/vad/vad_filterbank.h` | `54d0e80a077b4ec13d4b82203aacf983025e6767ef55c0183ef88725c039fec6` |
| `common_audio/vad/vad_gmm.c` | `daf4ebd4e19a7234e29f427c82efb66a8deff34cdc3dad4bc89a1edef234bbaf` |
| `common_audio/vad/vad_gmm.h` | `9967b5ea41fe0b0916ecd505e643a69767bee5d085b5d3fdaf470de97973f4f9` |
| `common_audio/vad/vad_sp.c` | `b9db6bfd0fde3b518990c13d718f6e6e050022a479c2e073b2753bde1b0c9e03` |
| `common_audio/vad/vad_sp.h` | `a7bf96024b840096c3ec6c65460ad516b19e73b5f48ad8161c52ce641c42d8c0` |
| `common_audio/vad/webrtc_vad.c` | `88e90b68499dd6c732b1ecb8ba6a76dc5f77df48fd726d650ac1665f645ae57c` |
| `rtc_base/checks.cc` | `cd4374559d657fa84914d1b8cb8bd5eb36f4ab8c660a34f397bf76d6117f56f9` |
| `rtc_base/checks.h` | `60699eba7cecefc19af703aca909516a1f14dd8c5cb4b42d97489d862a6a4440` |
| `rtc_base/compile_assert_c.h` | `2356cc41c37f22f9b0fb94e4030bea7734dc670d7c98ee92c28b95f4ac3c13e5` |
| `rtc_base/numerics/safe_compare.h` | `112ebd03e36e6e7d379a0a9d511a80f2e82580a5411bf0b9c3494a9b28fbe238` |
| `rtc_base/sanitizer.h` | `4e354032094c38d98753acd25d124338ee3bfeb0b660c7849152fb082bf300a7` |
| `rtc_base/system/arch.h` | `42959d6e339227867713a5488293eddd42ef052f3a662cbaffa668faddb8b0ba` |
| `rtc_base/system/inline.h` | `6c7d5c7d16d876686b5539bec9f21c25be2fed0a40042814006514d23d771476` |
| `rtc_base/type_traits.h` | `841bb2786a7b2b9b28f717df6f73ff545114f1a652992fcea8550680da1ab07f` |
| `system_wrappers/include/cpu_features_wrapper.h` | `ac259982db4e44eb57df308d43a41668695d3322fc609659c338bed4499fd034` |
| `typedefs.h` | `83f69bad4c2eb5469afba0c48c2b434dffec81b7450ad0b72e6a2b1c1f7e1759` |
