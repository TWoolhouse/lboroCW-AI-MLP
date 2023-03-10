#include "log.h"
#include "../meta.h"

#include <fstream>
#include <filesystem>

namespace mlp::log {

	struct Recorder {
		std::ofstream file;
		Recorder(std::filesystem::path filepath, bool training): file([&]() { std::filesystem::create_directories(filepath.parent_path()); return filepath; }()) {
			file << (training ? "error_training,error_validation" : "testing") << std::endl;
		}

		void train(size_t epoch, FLOAT training, FLOAT validation) {
			file << mlp_fmt("{},{}", training, validation) << std::endl;
			mlp_log_info("Epoch: {:0>5} - RMSE: Train[{:0<20}] Validation[{:0<20}]", epoch, training, validation);
		}
		void test(FLOAT error) {
			file << error << std::endl;
		}
	};

} // namespace mlp::log
