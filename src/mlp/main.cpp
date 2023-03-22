#include <array>
#include <iostream>

#include "dataset.h"
#include "meta.h"
#include "model.h"
#include "train.h"
#include "log/record.h"

constexpr size_t EPOCHS = 20000;
constexpr size_t BATCH_SIZE = EPOCHS / 100;
constexpr size_t ITERATIONS = EPOCHS / BATCH_SIZE;
constexpr unsigned int RANDOM_SEED = 123;

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

	#ifdef MLP_TRAINING

	mlp_log_info("Epochs: {} @ {}x{}", EPOCHS, BATCH_SIZE, ITERATIONS);

	#ifdef MLP_TRAIN_MOMENTUM
	mlp_log_info("Momentum: {}", MLP_TRAIN_MOMENTUM);
	#endif
	#ifdef MLP_TRAIN_BOLD_DRIVER
	mlp_log_info("Bold Driver: {}, {}, {}, {}", MLP_TRAIN_BOLD_DRIVER_MIN, MLP_TRAIN_BOLD_DRIVER_MAX, MLP_TRAIN_BOLD_DRIVER_INC, MLP_TRAIN_BOLD_DRIVER_DEC);
	#endif
	#ifdef MLP_TRAIN_ANNEALING
	mlp_log_info("Annealing: {}, {}", MLP_TRAIN_ANNEALING_START, MLP_TRAIN_ANNEALING_END);
	#endif

	Model<Record::inputs, MLP_ACTIVATION, MLP_HEIGHT> model{ RANDOM_SEED };

	auto mse = model.forward(dataset.validate);
	mlp_log_info("Validation RMSE: {:0<20}", std::sqrt(mse));

	auto trainer = Trainer(model);

	log::Recorder recorder_error(mlp_fmt("model/training/{}.log", variant), true);
	log::RecorderModel recorder_model(mlp_fmt("model/bin/{}/", variant));

	FLOAT acc = std::numeric_limits<FLOAT>::max();
	for (size_t j = 0; j < ITERATIONS; j++) {
		for (size_t i = 0; i < BATCH_SIZE; i++) {
			const size_t current_epoch = j * BATCH_SIZE + i;
			FLOAT error_previous = acc;
			acc = 0;
			for (auto& record : dataset.train) {
				auto error = trainer.train(record.as_input(), record.output(), static_cast<FLOAT>(current_epoch) / EPOCHS);
				acc += error * error;
			}
			#ifdef MLP_TRAIN_BOLD_DRIVER
			trainer.bold_driver(error_previous >= acc);
			#endif // MLP_TRAIN_BOLD_DRIVER
		}
		auto m = trainer.model();
		auto error_training = std::sqrt(acc / dataset.train.size());
		auto error_validation = std::sqrt(m.forward(dataset.validate));

		auto epochs = (j + 1) * BATCH_SIZE;
		recorder_error.train(epochs, error_training, error_validation, trainer.learning_rate);
		recorder_model.model(m, epochs);
	}

	#else // !MLP_TRAINING

	Model<Record::inputs, MLP_ACTIVATION, MLP_HEIGHT> model{ mlp_fmt("model/bin/{}.mdl", variant) };

	log::RecorderTest recorder(mlp_fmt("test/{}.log", variant));

	recorder.error(
		std::sqrt(model.forward(dataset.train)),
		std::sqrt(model.forward(dataset.validate)),
		std::sqrt(model.forward(dataset.test))
	);


	for (auto& set : { dataset.train, dataset.validate , dataset.test }) {
		for (auto& record : set) {
			auto guess = model.compute(record.as_input());
			auto raw = dataset.encodings.back().decode(guess);
			recorder.prediction(guess, raw);
		}
	}

	#endif // MLP_TRAINING

	return 0;
}
