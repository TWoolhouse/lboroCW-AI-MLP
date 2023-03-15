#include "log.h"
#include "../meta.h"
#include "../model.h"

#include <fstream>
#include <filesystem>

namespace mlp::log {

	struct Recorder {
		std::ofstream file;
		Recorder(std::filesystem::path path, bool training): file([&]() { std::filesystem::create_directories(path.parent_path()); return path; }()) {
			file << (training ? "epoch,error_training,error_validation,learning_rate" : "testing") << std::endl;
		}

		void train(size_t epoch, FLOAT training, FLOAT validation, FLOAT learning_rate) {
			file << mlp_fmt("{},{},{},{}", epoch, training, validation, learning_rate) << std::endl;
			mlp_log_info("Epoch: {:0>5} - RMSE: Train[{:0<20}] Validation[{:0<20}]", epoch, training, validation);
		}

		void test(FLOAT error) {
			file << error << std::endl;
		}
	};

	struct RecorderModel {
		std::filesystem::path path;
		RecorderModel(std::filesystem::path path): path(std::filesystem::absolute(path)) {
			std::filesystem::create_directories(path);
		}

		template<size_t Inputs, Activation Activator, size_t Height>
		void model(const Model<Inputs, Activator, Height>& model, size_t epochs) const {
			try {
				std::ofstream file;
				file.exceptions(std::ios::badbit | std::ios::failbit);
				file.open(path / mlp_fmt("{}.mdl", epochs), std::ios::out | std::ios::binary);
				file.write(reinterpret_cast<const char*>(&model), sizeof(std::remove_cvref_t<decltype(model)>));
				file.close();
			}
			catch (std::ios::failure& e) {
				mlp_log_fatal("Unable to serialise Model@{}: {}", epochs, e.what());
				throw e;
			}
		}
	};

} // namespace mlp::log
