#include <array>
#include <iostream>

#include "dataset.h"
#include "meta.h"
#include "model.h"
#include "train.h"
#include "log/record.h"

constexpr size_t EPOCHS = 10000;
constexpr size_t ITERATIONS = EPOCHS / 100;
constexpr size_t BATCH_SIZE = EPOCHS / ITERATIONS;

// mlp <variant-name> <dataset>
int main(int argc, const char** argv) {
	using namespace mlp;

	if (argc < 3) {
		std::cerr << "[ERROR] USAGE: mlp <variant-name> <dataset>";
		return -1;
	}

	std::string variant = argv[1];
	std::filesystem::path path{ argv[2] };
	mlp_log_init(variant);

	Dataset dataset{ path };
	if (!dataset)
		return -2;
	mlp_log_info("Dataset: {}, {}, {}", dataset.train.size(), dataset.validate.size(), dataset.test.size());

	mlp_log_info("Epochs: {} @ {}x{}", EPOCHS, BATCH_SIZE, ITERATIONS);

	Model<Record::inputs, MLP_ACTIVATION, MLP_HEIGHT> model;

	auto mse = model.forward(dataset.validate);
	mlp_log_info("Validation RMSE: {:0<20}%", std::sqrt(mse));

	auto trainer = Trainer(model);

	log::Recorder recorder(mlp_fmt("model/training/{}.log", variant), true);

	for (size_t j = 0; j < ITERATIONS; j++) {
		FLOAT acc;
		for (size_t i = 0; i < BATCH_SIZE; i++) {
			acc = 0;
			for (auto& record : dataset.train) {
				auto error = trainer.train(record.as_input(), record.output());
				acc += error * error;
			}
		}
		auto error_training = std::sqrt(acc / dataset.train.size());
		auto error_validation = std::sqrt(trainer.model().forward(dataset.validate));

		recorder.train((j + 1) * BATCH_SIZE, error_training, error_validation);
	}

	return 0;
}
