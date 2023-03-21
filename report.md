---
title: Multi-Layer Perceptron
subtitle: Report
author: F121584 - Thomas Woolhouse
date: 22/03/2023
numbersections: true
documentclass: report
papersize: A4
fontsize: 12pt
geometry: margin=1.9cm
toc: true
links-as-notes: true
colorlinks: blue
header-includes: |
	\usepackage{sectsty}
	\usepackage{pgffor}
	\usepackage{minted}
	\usepackage{paracol}
	\newcommand{\hideFromPandoc}[1]{#1}
	\hideFromPandoc{
		\let\Begin\begin
		\let\End\end
	}
---

# Overview

My solution for training a multi-layered perceptron from raw data is a monolithic main Python script, backed by a C++ model and training algorithm.
The Python script is responsible for: preprocessing the dataset, building the C++ model executables, running the training programs, analysing the results, and producing many graphs.
A short overview of the program structure may deem useful to understanding how the program goes about training and selecting an MLP.


It is built around the concept of *variants*, where each *variant* is some combination of a dataset, an activation function, and a set of modifications.

1. The program will generate multiple unique datasets using different methods of preprocessing so the best can be determined for future use.
2. It will build every combination of every activation function (sigmoid, tanh) and included modification (momentum, bold driver, etc ...).
3. Finally it will train every built executable against every dataset, plotting the results. These are able to be automatically graphed and the weights saved so they can be loaded again for testing.

As you may imagine, the number of combinations grows rapidly as more variations are added, henceforth I decided to use C++ as the language for the training algorithm, yet Python to handle managing all of these processes and graphing outputs using [matplotlib](https://matplotlib.org/).

# Data Preprocessing

The data preprocessing has been carried out programmatically using Python to clean the data using statistical analysis.
Initially I converted the given Excel dataset into a CSV format.
This is can be easily manipulated by the builtin standard library module: `csv`.
The preprocessing of the raw dataset is abstracted into several distinct sections:

1. [Parsing](#parsing) the Raw dataset.
2. [Cleaning](#cleaning).
3. [Standardising](#standardising).
4. [Splitting](#splitting).
5. [Serialisation](#serialisation)

## Parsing

Parsing entails loading the raw dataset into the main Python scriptThe data is streamed into the program line by line.
This is parsed into a [`Record`](#record.py) object using a static method which will raise an `Exception` if the row contains any invalid data.
For our dataset, this means any rows with non-floating point values are rejected.
This leads to a reduction in the size of the final dataset, however, there are 1461 in the raw set and after removing these exception raising values, 1456 rows remain.
It is an insignificant amount of data to lose and therefore, I did not deem it necessary to impute values back into the dataset.

## Cleaning

Inline with programmatic preprocessing, for cleaning the data, I experimented with different methods of statistical analysis to remove outliers from the dataset.
I explored using the standard deviation and the inter-quartile range to provide empirical method of removing the anomalous data.

The [standard deviation method](#filter.py) would compute the mean $\bar{x}$ and standard deviation $\sigma$ for each column (both the predictors and predictand) of the dataset.
For every row, if any field's value lay outside of $\bar{x} \pm R\sigma$ for it's respective column, it would be rejected from the final dataset.
The variable $R$ controls the number of standard deviations the value must be within.

The process for filtering the data using inter-quartile range is broadly the same, changing the the mean to the median, and $\sigma$ to the titular IQR.

To get the best of both worlds, the final dataset would undergo both of these filtering methods.

| Method                  | Records |      % |
| ----------------------- | ------: | -----: |
| No Cleaning             |    1456 | 100.0% |
| 3 Standard Deviations   |    1415 | 97.18% |
| 2 Standard Deviations   |    1225 | 84.13% |
| 1 Standard Deviation    |     358 | 24.59% |
| 3 Inter-Quartile Ranges |    1403 | 96.36% |
| 3 Inter-Quartile Ranges |    1232 | 84.62% |
| 3 Inter-Quartile Ranges |     769 | 52.82% |
| 3 Deviations & Ranges   |    1389 | 95.40% |
| 2 Deviations & Ranges   |    1172 | 80.49% |
| 1 Deviation & Range     |     342 | 23.49% |

The amount of data left in the final datasets can be seen in the table above.
As we can see, a combined approach removes more data points than either of the individual ones, meaning neither strategy would be wholey effective on its own.

## Standardising

Standardising (seen [here](#standardise.py)) takes the filtered datasets and remaps the value of each column to a new range.
This encoding is completed using a simple linear equation:
$$\frac{v - I_L}{I_U - I_L} * (O_U - O_L) + O_L$$
where:

- $v$ - The input value.
- $I_U$ - The input values upper bound.
- $I_L$ - The input values lower bound.
- $O_U$ - The output values upper bound.
- $O_L$ - The output values lower bound.

The new, encoded values are computed using this function, in which the inputs upper and lower bounds are taken from the bounds of the column this value is from.
It is done on a per column basis as the different types of data have very varied ranges.

For the final dataset, a output bound of 0.1 - 0.9 was used as it would end up with a consistently lower RMSE on the validation set than the bounds 0.0 - 1.0.

<!-- TODO: Add a graph or data to back this up, im just guessing -->

## Splitting

The penultimate stage of preprocessing the data is to split into 3 segments.
Namely, the training, validation, and test sets.
I decided to test 2 separate approaches to this: a [random variant](#split.py), and a [yearly variant](#split.py).

The [random variant](#split.py) will shuffle the order of the entire dataset such that it can group data in collections.
The first 60% is moved into the training set, the next 20% is for validation, and the remaining 20% of data is left for the test set.

This is in contrast to the [yearly method](#split.py), which groups data by year.
The given dataset spans over 4 years (1987-1990), this led to a 2, 1, 1 split.
This means the first 2 years are for training, 1 for validation, and the last year is for testing.

The difference between these two methods is minimal, as seen below:

| Method | Total | Set: Training |     % | Set: Validation |     % | Set: Testing |     % |
| ------ | ----: | ------------: | ----: | --------------: | ----: | -----------: | ----: |
| Random |  1389 |           833 | 59.97 |             277 | 19.94 |          279 | 20.09 |
| Yearly |  1388 |           704 | 50.72 |             342 | 24.64 |          342 | 24.64 |

Note: This is using the standard deviation IQR 3 variant of splitting the data.

## Serialisation

Serialisation, the final stage of data preprocessing.
The newly constructed dataset needs to be saved to the disk, so that when training, the neural network models can use it, without having to go through the whole preprocessing stage again.
Furthermore, the models themselves have been made in C++, as such, the only way for the data to reach the C++ executable, would be a file.

With this in mind, the output uses a custom binary format.
The layout of the packed format is:

\Begin{center}

`[sizes][encodings][training][validation][testing]`

\End{center}

| Segment      | Description                                                        |                          Size |
| ------------ | ------------------------------------------------------------------ | ----------------------------: |
| `sizes`      | `[train_size][val_size][test_size][enc_size]`                      |                      32 bytes |
| `train_size` | Number of records in the training set                              |            uint64_t (8 bytes) |
| `val_size`   | Number of records in the validation set                            |            uint64_t (8 bytes) |
| `test_size`  | Number of records in the testing set                               |            uint64_t (8 bytes) |
| `enc_size`   | Number of columns per record                                       |            uint64_t (8 bytes) |
| `encodings`  | `[encoding] * enc_size` Encoding for each column.                  |           enc_size * 32 bytes |
| `encoding`   | Values $I_U, I_L, O_U, O_L$ from [standardisation](#standardising) |       4 * float_64 (32 bytes) |
| `training`   | `[record] * train_size`                                            | train_size * sizeof(`record`) |
| `validation` | `[record] * val_size`                                              |   val_size * sizeof(`record`) |
| `testing`    | `[record] * test_size`                                             |  test_size * sizeof(`record`) |
| `record`     | `[year][month][day]([column] * enc_size)`                          |        4 + enc_size * 8 bytes |
| `year`       | Records Year e.g. 1987                                             |            uint16_t (2 bytes) |
| `month`      | Records Month e.g 3 which corresponds to March                     |              uint8_t (1 byte) |
| `day`        | Records Day e.g 31                                                 |              uint8_t (1 byte) |
| `column`     | The standardised value for the record.                             |            float_64 (8 bytes) |

The reason for the custom format is to simplify and speed up the model training.
The data is laid out to match the exact format of the C++ definition of a Record.
This means, the input file does not need to be manipulated or transformed, therefore reducing the training time overhead.

## Selection

<!-- TODO: Selection of dataset -->

\columnratio{0.5}
\Begin{paracol}{2}

![Dataset: Cleaned with Standard Deviation 3 & IQR 3, Standardised between 0.1 - 0.9](graph/dataset/std_dev_iqr_3.lin1-9.year_2_1_1.png)

\switchcolumn

![Dataset: Cleaned with Standard Deviation 3 & IQR 3, Split by years.](graph/dataset/std_dev_iqr_3.lin1-9.year_2_1_1.raw.png)

\End{paracol}


Amount of data left

# Implementation of MLP

The multi-layer perceptron is implemented in C++.
As C++ is a compiled language, each variation / modification of the MLP will be it's own executable file.
To accomplish this, I used preprocessor macros (`#defines`) to selectively use certain segments of code, depending on the model being built.
The multiple builds and variations are controlled by the [main Python script](#overview).

C++ was also chosen due to it being a low-level language.
This was important as the memory could be laid out to aid with cache locality as well as let the compiler introduce SIMD instructions as part of the optimisation process.
Improved performance would reduce the training time which made it plausible to test so many different variations in a short span of time, or, I was able to train a model for many tens of thousands of epochs in a couple of seconds.

Finally, the code has been made as generic as possible using a combination of macros and templates.
The templates have allowed me to create a single code base that is capable of doing an MLP of any size, without compramising performance.

## Model

The model (implemented [here](#model.h)) is a template class.
The simplified version below shows the attributes and methods that the class has.
It should be noted that throughout the code base, the macro `FLOAT` has been used to allow changes between float precision (float_32, float_64).

```cpp
template<size_t Inputs, Activation Activator, size_t Height>
	class Model {
		std::array<Node<Inputs, Activator>, Height> layer;
		Node<Height, Activator> output;

		// Takes an input row and returns the models prediction
		FLOAT compute(const std::array<FLOAT, Inputs>& inputs) const {
			auto hidden = compute_hidden(inputs);
			return output.activate(output.compute(hidden));
		}

		// Completes a forward pass of a set of records.
		// Outputs the mean squared error
		FLOAT forward(const Dataset::Set& dataset) const {
			FLOAT acc = 0;
			for (auto& record : dataset) {
				auto guess = compute(record.as_input());
				auto err = guess - record.output();
				acc += err * err;
			}
			return acc / dataset.size();
		}

		// Performs the forward pass of all nodes in the hidden layer.
		auto compute_hidden(const std::array<FLOAT, Inputs>& inputs) const {
			std::array<FLOAT, Height> outputs;
			auto out_it = outputs.data();
			for (const auto& node : layer)
				*out_it++ = node.activate(node.compute(inputs));
			return outputs;
		}
	};
```

The model's template parameters:

- **Inputs** - The number of input nodes.
- **Activator** - A variant of the *Activation* enum. It specifies the activator function.
- **Height** - The number of nodes in the hidden layer.

This model class does not handle training, and can only be used to perform predictions.
However, it has helper methods to convert from the raw model to a [trainer](#trainer).
The hidden layer is made up of `Node`s which are defined [here](#node.h) or simplified below:

```cpp
template<size_t Size, Activation Activator>
	struct Node {
		FLOAT bias;
		std::array<FLOAT, Size> weights;

		// Uses the activation function.
		static FLOAT activate(FLOAT value) {
			return mlp::activate<Activator>(value);
		}
		// Uses the differential of the activation function.
		static FLOAT differential(FLOAT value) {
			return mlp::differential<Activator>(value);
		}

		// Computes the output from the given inputs.
		FLOAT compute(const std::array<FLOAT, Size>& inputs) const {
			FLOAT acc = bias;
			for (auto lhs = weights.cbegin(), rhs = inputs.begin(); lhs < weights.cend(); ++lhs, ++rhs)
				acc += *lhs * *rhs;
			return acc;
		}
	};
```

The template parameters:

- **Size** - Number of input weights
- **Activator** - The activation function used.

The use of the template parameters means the size of the arrays are known at compile time.
This provides one of the major performance improvements.
All data is tightly packed on the stack, therefore, there is very little indirection in the whole program.
It also eliminates the need to perform any memory management.

Finally, because the `Model` is so simple, it can be directly copied to a file as a binary blob.
This provides implicit serialisation & deserialisation.

## Record

A `Record` (found [here](#record.h)) has the format defined in the table from [Serialisation of the Dataset](#serialisation).

```cpp
struct Record {
	constexpr static size_t size = 6;
	constexpr static size_t inputs = 5;

	uint16_t year;
	uint8_t month, day;
	FLOAT temperature, wind_speed, solar_radiation, air_pressure, humidity, evaporation;

	const std::array<FLOAT, inputs>& as_input() const {
		return *reinterpret_cast<const std::array<FLOAT, inputs>*>(&temperature);
	}
	FLOAT output() const noexcept {
		return evaporation;
	}
};
```

The record specifies how many inputs there are, as well as which column to use as the output.
This is one of two places required in the entire code base that would need to be modified if a different dataset were to be used.
The odd `reinterpret_cast` is the use of [type punning](https://en.wikipedia.org/wiki/Type_punning) to return the pointer to the `temperature` column as the start of the inputs row.
This is done as to prevent copying the row every time it is needed, as `reinterpret_cast` is only used by the compiler as a type hint, not a CPU an instruction.

## Trainer

The backpropagation algorithm is implemented (in [train.h](#train.h)) as a separate template class.
The trainer class uses the same template parameters as the [`Model`](#model) as well as having a similar structure.

```cpp
template<size_t Inputs, Activation Activator, size_t Height>
	class Trainer {
		FLOAT learning_rate = 0.005;
		#ifdef MLP_TRAIN_MOMENTUM
		static constexpr FLOAT momentum_weight = MLP_TRAIN_MOMENTUM;
		#endif // MLP_TRAIN_MOMENTUM

		template<size_t Size>
		struct Cell;
	protected:
		std::array<Cell<Inputs>, Height> layer;
		Cell<Height> output;

		// Methods...
	};
```

The `Trainer` uses the same concept with an array of `Node` like objects (`Cell`) for the hidden layer and the output.
It also tracks the `learning_rate` known as the learning parameter $\rho$.
The macro `MLP_TRAIN_MOMENTUM` is only defined when momentum should be used and so the `momentum_weight` is only set when needed.

The `Cell` type encapsulates a `Node`, so it is able to keep track of information during the forward and backwards passes of backpropagation.
Below is an extract of the `Cell` type.

```cpp
template<size_t Size>
struct Cell {
	Node<Size, Activator> node;
	union {
		FLOAT sum, delta;
	};
	FLOAT activated;

	// Forward pass of a single Node
	auto forward(const std::array<FLOAT, Size>& inputs) {
		sum = node.compute(inputs);
		activated = node.activate(sum);
		return activated;
	}

	void backward_delta_output(FLOAT correct) {
		delta = backward_error(correct, activated) * node.differential(sum);
	}
	void backward_delta_hidden(FLOAT fwd_delta, FLOAT fwd_weight) {
		delta = fwd_weight * fwd_delta * node.differential(sum);
	}

	// Etc ...
};
```

Each cell contains a `Node`, where the Activator is inherited from the `Trainer`.
Each `Cell` has 3 attributes:

- `sum` - The value of the node before being the activation.
- `delta` - The computed $\delta$ for the backpropagation algorithm.
- `activated` - The value of the node after the activation function.

The usage of the `union` means that the `sum` and `delta` attributes share the same space in memory.
This is not a problem for the algorithm as the `sum` is only needed to compute the `delta` value and so it can be overridden.

The other methods are self explanatory.
The snippets of code that were excluded are relevant to different modifications and will be mentioned later on.


---

The `train` method is the entry point of the class as it will perform both the forward and backward pass for a specific input row.
It returns the error which can be used later on to calculate the RMSE of a whole epoch.

```cpp
FLOAT train(const std::array<FLOAT, Inputs>& inputs, FLOAT correct, FLOAT epoch_percent) {
	auto guess = forward(inputs);
	#ifdef MLP_TRAIN_ANNEALING
	learning_rate = annealing(epoch_percent);
	#endif // MLP_TRAIN_ANNEALING
	backward(inputs, correct);
	auto error = correct - guess;
	return error;
}
```

### Forward Pass

The forward pass is simple and does not change with any modifications.
Firstly the value of every cell in the hidden layer is found.
Then this is passed onto the output cell.
During this process, the `sum` and `activated` attributes of every `Cell` is computed and set.

```cpp
FLOAT forward(const std::array<FLOAT, Inputs>& inputs) {
	auto hidden = forward_hidden(inputs);
	return output.forward(hidden);
}

std::array<FLOAT, Height> forward_hidden(const std::array<FLOAT, Inputs>& inputs) {
	std::array<FLOAT, Height> outputs;
	auto it = outputs.data();
	for (auto& cell : layer) {
		*it++ = cell.forward(inputs);
	}
	return outputs;
}
```

### Backward Pass

On the contrary, the backward pass is more complex, especially with the modifications.
However, here is the basic algorithm implementation as methods of the `Trainer` class.

```cpp
void backward(const std::array<FLOAT, Inputs>& inputs, FLOAT correct) {
	// Calculate the `delta` for the output node.
	output.backward_delta_output(correct);
	// Iterate over the weights of the output node & the nodes of the hidden layer.
	auto it_weight_out = output.node.weights.rbegin();
	for (auto it_cell = layer.rbegin(); it_cell < layer.rend(); ++it_cell) {
		// Compute the `delta` of the hidden node.
		it_cell->backward_delta_hidden(output.delta, *it_weight_out);
		backward_update_bias(it_cell->node.bias, *it_cell);
		// Iterate over the weight of this node & the input values
		auto it_weight_node = it_cell->node.weights.begin();
		for (auto it_input = inputs.begin(); it_input < inputs.end(); ++it_input)
			// Update the weight between the node and input
			backward_update_weight(*it_weight_node++, *it_cell, *it_input);
		// Update the weight between the node and the output.
		backward_update_weight(*it_weight_out++, output, it_cell->activated);
	}
	backward_update_bias(output.node.bias, output);
}

template<size_t Size>
void backward_update_weight(FLOAT& weight, Cell<Size>& cell, FLOAT input) {
	weight += learning_rate * cell.delta * input;
}

template<size_t Size>
auto backward_update_bias(FLOAT& bias, Cell<Size>& cell) {
	return backward_update_weight(bias, cell, 1);
}
```

## Activation Functions

The different activation functions are implemented in [activation.h](#activation.h) and converts the supported actionvation enum variants into a function call.

```cpp
enum class Activation: uint8_t {
	LINEAR,
	SIGMOID,
	TANH,
};

template<Activation Activator>
constexpr FLOAT activate(FLOAT value) {
	if constexpr (Activator == Activation::LINEAR) {
		return value;
	}
	else if constexpr (Activator == Activation::SIGMOID) {
		return 1.0 / (1.0 + exp(-value));
	}
	else if constexpr (Activator == Activation::TANH) {
		return (tanh(value) + 1) / 2;
	}
}
```

The activate function template will only run the equation gated in the if-expression.
The `if constexpr` ensures that is check is performed at compile time, once again removing any branching in the program.
There is a also a similar function `differential` which computes $f'(x)$ where $f$ is `activate`.

## Modifications

As mentioned before, all modifications are gated behind specific `#define`s so that they are only compiled for certain builds.
This let me add changes to the code using the `#ifdef` preprocessor directive.
The macros for the modifications are:

- `MLP_TRAIN_MOMENTUM`
- `MLP_TRAIN_BOLD_DRIVER`
- `MLP_TRAIN_ANNEALING`
- `MLP_TRAIN_WEIGHT_DECAY`

The macros may also have a value assigned to them, which is used to configure that extension.

### Momentum

Momentum requires an additional field to store the weight change of last iteration of every single weight and bias.
It also makes a change to the `backward_update_weight` function.

Firstly, the extra fields in the `Cell` type:

```cpp
template<size_t Size>
struct Cell {
	// ...

	Node<Size, Activator> node;

	#ifdef MLP_TRAIN_MOMENTUM
	struct Momentum {
	FLOAT bias;
	std::array<FLOAT, Size> weights;

	constexpr FLOAT& remap(FLOAT& node_ptr) {
		auto ptr = reinterpret_cast<uint8_t*>(&node_ptr);
		return *reinterpret_cast<FLOAT*>(ptr + offset());
	}
	protected:
	static constexpr auto offset() {
		return offsetof(Cell, momentum) - offsetof(Cell, node);
	}
	} momentum;
	#endif // MLP_TRAIN_MOMENTUM

	// ...
}
```

`Cell` now contains a new substructure for holding onto the previous weight changes.
It accomplishes this by creating the struct `Momentum` which has the exact same memory layout as the `Node` struct.
This means that a pointer to a weight of this cell can be mapped to a pointer in the `Momentum` struct using simple pointer arithmetic.
`Momentum::remap` will convert the pointer to a weight to its corresponding weight history value.
This also holds true for the biases as well.

The other change required to be made was in `backward_update_weight`, as mentioned before.

```cpp
template<size_t Size>
void backward_update_weight(FLOAT& weight, Cell<Size>& cell, FLOAT input) {
	#ifdef MLP_TRAIN_MOMENTUM
	auto old = weight;
	auto& momentum = cell.momentum.remap(weight);
	weight += learning_rate * cell.delta * input + momentum_weight * momentum;
	momentum = weight - old;
	#else // !MLP_TRAIN_MOMENTUM
	weight += learning_rate * cell.delta * input;
	#endif // MLP_TRAIN_MOMENTUM
}
```

This changes the equation for all updates to the weights and biases to include its momentum as well.
`momentum_weight` controls how much the momentum affects the weight changes and is a static variable on the `Trainer` class.
The value of this variable is introduced via the `MLP_TRAIN_MOMENTUM` macro. e.g `MLP_TRAIN_MOMENTUM=0.9`

### Bold Driver

Bold driver adds a single function to the public interface of the `Trainer` class that will modify the `learning_rate` if the error increased or decreased.
The macro takes in a 4-value tuple such as `MLP_TRAIN_BOLD_DRIVER=(min, max, inc, dec)` where:

- `min` - The smallest value of $\rho$.
- `max` - The largest value of $\rho$.
- `inc` - How much to increase $\rho$ by each time their is a decrease in the error.
- `dec` - How much to decrease $\rho$ by each time their is an increase in the error.

The bold driver was implemented as:

```cpp
void bold_driver(bool improved) {
	FLOAT modifier = improved
		? MLP_TRAIN_BOLD_DRIVER_INC
		: MLP_TRAIN_BOLD_DRIVER_DEC;
	learning_rate = std::clamp(
		learning_rate * modifier,
		MLP_TRAIN_BOLD_DRIVER_MIN,
		MLP_TRAIN_BOLD_DRIVER_MAX
	);
}
```

The new macros shown above are automatically extracted from `MLP_TRAIN_BOLD_DRIVER` and do not need to be defined explicitly.
I decided to run the bold driver after every epoch with an `inc` & `dec` close to one as it ended up having a larger effect on the output.
Otherwise, only running it after a threshold, it would never change the learning rate enough to make a difference.

### Annealing

Annealing, a mutually exclusive modification with bold driver, also controls the `learning_rate` of the training algorithm.
The implementation requires a simple function that takes in the current epoch as a percentage of the total number of epochs.

```cpp
FLOAT annealing(FLOAT epoch_percentage) const {
	return MLP_TRAIN_ANNEALING_END
		+ (MLP_TRAIN_ANNEALING_START - MLP_TRAIN_ANNEALING_END)
		* (1.0 - 1.0 / (1.0 + exp(10.0 - 20.0 * epoch_percentage)));
}
```

The extra macro definitions once again come from `MLP_TRAIN_ANNEALING=(start, end)`.

### Weight Decay

<!-- TODO: Weight decay -->

# Training and Network Selection



# Evaluation of Final Model

GRAPH different modifications with each other.

# Comparison with Other Models

# Appendix Code

## record.py

PY_CODE_FILE{Python}{./src/process/record.py}

## filter.py

PY_CODE_FILE{Python}{./src/process/filter.py}

## standardise.py

PY_CODE_FILE{Python}{./src/process/standardise.py}

## split.py

PY_CODE_FILE{Python}{./src/process/split.py}

## model.h

PY_CODE_FILE{Cpp}{./src/mlp/model.h}

## node.h

PY_CODE_FILE{Cpp}{./src/mlp/node.h}

## record.h

PY_CODE_FILE{Cpp}{./src/mlp/record.h}

## train.h

PY_CODE_FILE{Cpp}{./src/mlp/train.h}

## activation.h

PY_CODE_FILE{Cpp}{./src/mlp/activation.h}
