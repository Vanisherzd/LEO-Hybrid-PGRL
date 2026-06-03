#include <uhd/usrp/multi_usrp.hpp>
#include <uhd/stream.hpp>
#include <uhd/types/metadata.hpp>
#include <uhd/types/tune_request.hpp>
#include <uhd/utils/thread.hpp>

#include <chrono>
#include <complex>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

static std::string get_arg(int argc, char** argv, const std::string& key, const std::string& def = "") {
    const std::string prefix = "--" + key + "=";
    for (int i = 1; i < argc; i++) {
        std::string s(argv[i]);
        if (s == "--" + key && i + 1 < argc) return std::string(argv[i + 1]);
        if (s.rfind(prefix, 0) == 0) return s.substr(prefix.size());
    }
    return def;
}

static bool has_flag(int argc, char** argv, const std::string& key) {
    const std::string flag = "--" + key;
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == flag) return true;
    }
    return false;
}

static void usage() {
    std::cout
        << "Usage: rx_capture_to_file_cpp "
        << "--freq 915e6 --rate 1e6 --gain 20 --duration 2 "
        << "--out hardware/captures/usrp_noise.fc32 "
        << "[--args serial=8000304] [--antenna RX2] [--channel 0]\n\n"
        << "Writes complex float32 IQ samples (.fc32), compatible with Python np.complex64.\n";
}

int main(int argc, char** argv) {
    if (has_flag(argc, argv, "help") || argc == 1) {
        usage();
        return 0;
    }

    const std::string args = get_arg(argc, argv, "args", "type=b200");
    const double freq = std::stod(get_arg(argc, argv, "freq", "915e6"));
    const double rate = std::stod(get_arg(argc, argv, "rate", "1e6"));
    const double gain = std::stod(get_arg(argc, argv, "gain", "20"));
    const double duration = std::stod(get_arg(argc, argv, "duration", "2"));
    const std::string out_path = get_arg(argc, argv, "out", "capture.fc32");
    const std::string antenna = get_arg(argc, argv, "antenna", "RX2");
    const size_t channel = static_cast<size_t>(std::stoul(get_arg(argc, argv, "channel", "0")));

    try {
        std::cout << "[rx_cpp] Creating USRP with args: " << args << "\n";
        auto usrp = uhd::usrp::multi_usrp::make(args);

        std::cout << "[rx_cpp] Setting RX rate: " << rate << "\n";
        usrp->set_rx_rate(rate, channel);

        std::cout << "[rx_cpp] Setting RX freq: " << freq << "\n";
        usrp->set_rx_freq(uhd::tune_request_t(freq), channel);

        std::cout << "[rx_cpp] Setting RX gain: " << gain << "\n";
        usrp->set_rx_gain(gain, channel);

        std::cout << "[rx_cpp] Setting RX antenna: " << antenna << "\n";
        usrp->set_rx_antenna(antenna, channel);

        std::this_thread::sleep_for(std::chrono::milliseconds(300));

        std::cout << "[rx_cpp] Actual RX rate: " << usrp->get_rx_rate(channel) << "\n";
        std::cout << "[rx_cpp] Actual RX freq: " << usrp->get_rx_freq(channel) << "\n";
        std::cout << "[rx_cpp] Actual RX gain: " << usrp->get_rx_gain(channel) << "\n";

        uhd::stream_args_t stream_args("fc32", "sc16");
        stream_args.channels = {channel};
        auto rx_stream = usrp->get_rx_stream(stream_args);

        const size_t samps_per_buff = rx_stream->get_max_num_samps();
        std::vector<std::complex<float>> buff(samps_per_buff);

        const size_t total_samps = static_cast<size_t>(duration * rate);
        std::ofstream outfile(out_path, std::ios::binary);
        if (!outfile) {
            std::cerr << "[rx_cpp] ERROR: Cannot open output: " << out_path << "\n";
            return 2;
        }

        uhd::rx_metadata_t md;
        uhd::stream_cmd_t stream_cmd(uhd::stream_cmd_t::STREAM_MODE_NUM_SAMPS_AND_DONE);
        stream_cmd.num_samps = total_samps;
        stream_cmd.stream_now = true;

        std::cout << "[rx_cpp] Capturing " << total_samps << " samples to " << out_path << "\n";
        rx_stream->issue_stream_cmd(stream_cmd);

        size_t num_acc_samps = 0;
        while (num_acc_samps < total_samps) {
            const size_t samps_to_recv = std::min(samps_per_buff, total_samps - num_acc_samps);
            const size_t num_rx_samps = rx_stream->recv(&buff.front(), samps_to_recv, md, 3.0);

            if (md.error_code == uhd::rx_metadata_t::ERROR_CODE_TIMEOUT) {
                std::cerr << "[rx_cpp] ERROR: Timeout while streaming\n";
                break;
            }
            if (md.error_code != uhd::rx_metadata_t::ERROR_CODE_NONE) {
                std::cerr << "[rx_cpp] ERROR: RX metadata error: " << md.strerror() << "\n";
                break;
            }

            outfile.write(reinterpret_cast<const char*>(&buff.front()),
                          static_cast<std::streamsize>(num_rx_samps * sizeof(std::complex<float>)));
            num_acc_samps += num_rx_samps;
        }

        outfile.close();
        std::cout << "[rx_cpp] Done. Wrote " << num_acc_samps << " samples.\n";
        std::cout << "[rx_cpp] Output: " << out_path << "\n";
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "[rx_cpp] EXCEPTION: " << e.what() << "\n";
        return 1;
    }
}
