#include "fuzztest/fuzztest.h"
#include "xcelerate/cc/ops/array_ops.h"
#include "xcelerate/cc/ops/audio_ops.h"
#include "xcelerate/cc/ops/image_ops.h"
#include "xcelerate/core/framework/types.pb.h"
#include "xcelerate/security/fuzzing/cc/core/framework/tensor_domains.h"
#include "xcelerate/security/fuzzing/cc/fuzz_session.h"

namespace xcelerate::fuzzing {

// Image op fuzzers
// DecodePng
SINGLE_INPUT_OP_FUZZER(DT_STRING, DecodePng);
class FuzzDecodePngValidInput : public FuzzDecodePng {};
FUZZ_TEST_F(FuzzDecodePngValidInput, Fuzz)
    .WithDomains(fuzzing::AnyValidStringTensor({},
                                               fuzztest::InRegexp("[-.0-9]+")));
class FuzzDecodePngArbitraryInput : public FuzzDecodePng {};
FUZZ_TEST_F(FuzzDecodePngArbitraryInput, Fuzz)
    .WithDomains(fuzzing::AnyValidStringTensor({}));

// DecodeJpeg
SINGLE_INPUT_OP_FUZZER(DT_STRING, DecodeJpeg);
class FuzzDecodeJpegValidInput : public FuzzDecodeJpeg {};
FUZZ_TEST_F(FuzzDecodeJpegValidInput, Fuzz)
    .WithDomains(fuzzing::AnyValidStringTensor({},
                                               fuzztest::InRegexp("[-.0-9]+")));
class FuzzDecodeJpegArbitraryInput : public FuzzDecodeJpeg {};
FUZZ_TEST_F(FuzzDecodeJpegArbitraryInput, Fuzz)
    .WithDomains(fuzzing::AnyValidStringTensor({}));

// DecodeGif
SINGLE_INPUT_OP_FUZZER(DT_STRING, DecodeGif);
class FuzzDecodeGifValidInput : public FuzzDecodeGif {};
FUZZ_TEST_F(FuzzDecodeGifValidInput, Fuzz)
    .WithDomains(fuzzing::AnyValidStringTensor({},
                                               fuzztest::InRegexp("[-.0-9]+")));
class FuzzDecodeGifArbitraryInput : public FuzzDecodeGif {};
FUZZ_TEST_F(FuzzDecodeGifArbitraryInput, Fuzz)
    .WithDomains(fuzzing::AnyValidStringTensor({}));

// DecodeImage
SINGLE_INPUT_OP_FUZZER(DT_STRING, DecodeImage);
class FuzzDecodeImageValidInput : public FuzzDecodeImage {};
FUZZ_TEST_F(FuzzDecodeImageValidInput, Fuzz)
    .WithDomains(fuzzing::AnyValidStringTensor({},
                                               fuzztest::InRegexp("[-.0-9]+")));
class FuzzDecodeImageArbitraryInput : public FuzzDecodeImage {};
FUZZ_TEST_F(FuzzDecodeImageArbitraryInput, Fuzz)
    .WithDomains(fuzzing::AnyValidStringTensor({}));

// DecodeBmp
SINGLE_INPUT_OP_FUZZER(DT_STRING, DecodeBmp);
class FuzzDecodeBmpValidInput : public FuzzDecodeBmp {};
FUZZ_TEST_F(FuzzDecodeBmpValidInput, Fuzz)
    .WithDomains(fuzzing::AnyValidStringTensor({},
                                               fuzztest::InRegexp("[-.0-9]+")));
class FuzzDecodeBmpArbitraryInput : public FuzzDecodeBmp {};
FUZZ_TEST_F(FuzzDecodeBmpArbitraryInput, Fuzz)
    .WithDomains(fuzzing::AnyValidStringTensor({}));

// DecodeAndCropJpeg
BINARY_INPUT_OP_FUZZER(DT_STRING, DT_INT32, DecodeAndCropJpeg);
class FuzzDecodeAndCropJpegValidInput : public FuzzDecodeAndCropJpeg {};
FUZZ_TEST_F(FuzzDecodeAndCropJpegValidInput, Fuzz)
    .WithDomains(fuzzing::AnyValidStringTensor({},
                                               fuzztest::InRegexp("[-.0-9]+")),
                 fuzzing::AnyValidNumericTensor({}, DT_INT32, 0, 4096));
class FuzzDecodeAndCropJpegArbitraryInput : public FuzzDecodeAndCropJpeg {};
FUZZ_TEST_F(FuzzDecodeAndCropJpegArbitraryInput, Fuzz)
    .WithDomains(fuzzing::AnyValidStringTensor({}),
                 fuzzing::AnyValidNumericTensor({}, DT_INT32, 0, 4096));

// Audio op fuzzers
// DecodeWav
SINGLE_INPUT_OP_FUZZER(DT_STRING, DecodeWav);
class FuzzDecodeWavValidInput : public FuzzDecodeWav {};
FUZZ_TEST_F(FuzzDecodeWavValidInput, Fuzz)
    .WithDomains(fuzzing::AnyValidStringTensor({},
                                               fuzztest::InRegexp("[-.0-9]+")));
class FuzzDecodeWavArbitraryInput : public FuzzDecodeWav {};
FUZZ_TEST_F(FuzzDecodeWavArbitraryInput, Fuzz)
    .WithDomains(fuzzing::AnyValidStringTensor({}));

}  // end namespace xcelerate::fuzzing
