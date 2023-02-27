#include <fstream>
#include <iostream>
#include <filesystem>

#include "log.h"
#include "../meta.h"
#include "../meta/define.h"

namespace mlp::log {

	static Logger* logger = nullptr;

	std::string location_string(const std::source_location& location) {
		return mlp_fmt("{}:{}:{} {}", location.file_name(), location.line(), location.column(), location.function_name());
	}

	void submit(const Level level, const std::source_location& location, const std::string& msg) {
		static constexpr std::string_view levels[static_cast<char>(Level::MAX)] = {
			"Fatal",
			"Error",
			"Warn",
			"Todo",
			"Info",
			"Debug",
		};
#if defined(MLP_LOG_CONSOLE) || defined(MLP_LOG_FILE)
		auto&& name = levels[static_cast<char>(level)];
#endif

#ifdef MLP_LOG_CONSOLE
		(level < Level::Warn ? std::cout : std::cerr) <<  // Stream
			"[" << name << "] " << msg << std::endl;	  // Message
#endif												  // MLP_LOG_CONSOLE

#ifdef MLP_LOG_FILE
		logger->file <<
#ifdef MLP_LOG_LOCATION
			mlp_fmt("{: <100} {}", mlp_fmt("[{}] {}", name, msg), location_string(location), name, msg)
			<< std::endl;
#else
			"[" << name << "] " << msg
			<< std::endl;
		MLP_UNUSED(location);
#endif
#else
		MLP_UNUSED(location);
		MLP_UNUSED(msg);
#endif	// MLP_LOG_FILE
	}

	void assert(bool condition, const std::source_location location, const std::string& msg) {
		if (condition)
			[[likely]];
		else {
			submit(Level::Fatal, location, msg);
		}
	}

	Logger::Logger(const std::string& name) {
		mlp_assert(logger == nullptr, "Only 1 Logger can be initialised!");
		logger = this;
#ifdef MLP_LOG_FILE
		std::filesystem::create_directory("log");
		file = std::ofstream{ mlp_fmt("log/{}.log", name) };
#endif
	}
	Logger::~Logger() {
		logger = nullptr;
	}

}  // namespace mlp::log
