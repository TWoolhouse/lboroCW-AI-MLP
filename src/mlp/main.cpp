#include <array>
#include <iostream>

#include "dataset.h"
#include "meta.h"
#include "model.h"
#include "train.h"

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


	Model<Record::inputs, MLP_ACTIVATION, MLP_HEIGHT> model;

	auto rmse = model.forward(dataset.validate);
	mlp_log_info("Validation MSE: {:0<20}%", rmse * 100.0);

	auto trainer = Trainer(model);


	FLOAT squared_error = 0;
	constexpr size_t EPOCHS = 10000;
	constexpr size_t ITERATIONS = EPOCHS / 100;
	constexpr size_t BATCH_SIZE = EPOCHS / ITERATIONS;
	for (size_t j = 0; j < ITERATIONS; j++) {
		for (size_t i = 0; i < BATCH_SIZE; i++) {
			squared_error = 0;
			for (auto& record : dataset.train) {
				auto error = trainer.train(record.as_input(), record.output());
				squared_error += error * error;
			}
		}
		mlp_log_info("Epoch: {:0>3}00 - MSE: {:0<20}%", j + 1, squared_error / dataset.train.size() * 100.0);

		FLOAT acc_err = 0;
		auto model = trainer.model();
		for (auto& record : dataset.validate) {
			auto result = model.compute(record.as_input());
			auto err = result - record.output();
			acc_err += err * err;
		}
		mlp_log_info("Validation MSE: {:0<20}%", acc_err / dataset.validate.size() * 100.0);
	}

	return 0;
}
