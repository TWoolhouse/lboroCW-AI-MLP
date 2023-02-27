#pragma once
#include <cstdint>
#include <format>
#include <fstream>
#include <source_location>

#include "../meta/define.h"

#define mlp_fmt(message, ...) std::vformat(message, std::make_format_args(__VA_ARGS__))

#define mlp_log_init(name) \
	::mlp::log::Logger __logger { name }

namespace mlp::log {

	enum class Level: uint8_t {
		Fatal,
		Error,
		Warn,
		Todo,
		Info,
		Debug,
		MAX,
	};

	void submit(const Level level, const std::source_location& location, const std::string& msg);

	template<typename... Args>
	void report(const Level level, const std::source_location location, const std::string_view msg, Args&&... args) {
		return submit(level, location, mlp_fmt(msg, args...));
	}

	void assert(bool condition, const std::source_location location, const std::string& msg);

	struct Logger {
#ifdef MLP_LOG_FILE
		std::ofstream file;
#endif	// MLP_LOG_FILE
		Logger(const std::string& name);
		~Logger();
	};

}  // namespace mlp::log

#if defined(MLP_LOG_CONSOLE) || defined(MLP_LOG_FILE)
#define mlp_log(level, message, ...) ::mlp::log::report(::mlp::log::Level::level, std::source_location::current(), message, __VA_ARGS__)
#else
#define mlp_log(level, message, ...)
#endif

#define mlp_log_fatal(message, ...) mlp_log(Fatal, message, __VA_ARGS__)
#define mlp_log_error(message, ...) mlp_log(Error, message, __VA_ARGS__)
#define mlp_log_warn(message, ...) mlp_log(Warn, message, __VA_ARGS__)
#define mlp_log_todo(message, ...) mlp_log(Todo, message, __VA_ARGS__)
#define mlp_log_info(message, ...) mlp_log(Info, message, __VA_ARGS__)
#define mlp_log_debug(message, ...) mlp_log(Debug, message, __VA_ARGS__)
