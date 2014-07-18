#ifndef _FTS_LV_LOGINIT_
#define _FTS_LV_LOGINIT_

#include <string>
#include <boost/log/expressions.hpp>
#include <boost/log/expressions/keyword.hpp>
#include <boost/log/expressions/formatters/date_time.hpp>
#include <boost/log/sources/severity_channel_logger.hpp>
#include <boost/log/sources/record_ostream.hpp>
#include <boost/log/sources/global_logger_storage.hpp>
#include <boost/log/support/date_time.hpp>
#include <boost/log/utility/setup/file.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>


enum severity_level {
	LOG_TRACE,
	LOG_DEBUG,
	LOG_INFO,
	LOG_WARN,
	LOG_ERROR,
	LOG_FATAL
};

typedef boost::log::sources::severity_channel_logger<severity_level, std::string> my_severity_channel_logger;

//BOOST_LOG_INLINE_GLOBAL_LOGGER_DEFAULT (my_logger, boost::log::sources::severity_logger< severity_level >)


struct severity_tag;

static const char* strings[] =
{
	"TRACE",
	"DEBUG",
	"INFO",
	"WARN",
	"ERROR",
	"FATAL"
};


// The operator is used for regular stream formatting
static std::ostream& operator<< (std::ostream& strm, severity_level level) {
	if (static_cast<std::size_t>(level) < sizeof(strings) / sizeof(*strings))
		strm << strings[level];
	else
		strm << static_cast<int>(level);

	return strm;
}

// The operator is used when putting the severity level to log
static boost::log::formatting_ostream& operator<< (boost::log::formatting_ostream& strm, boost::log::to_log_manip< severity_level, severity_tag > const& manip) {
	severity_level level = manip.get();

	if (static_cast<std::size_t>(level) < sizeof(strings) / sizeof(*strings))
		strm << strings[level];
	else
		strm << static_cast<int>(level);

	return strm;
}

static void initLog() {
	boost::log::add_file_log(
		//"sample.log"
		boost::log::keywords::file_name = "Violation_%Y%m%d.log",                                        /*< file name pattern >*/
		//boost::log::keywords::rotation_size = 10 * 1024 * 1024,                                   /*< rotate files every 10 MiB... >*/
		boost::log::keywords::time_based_rotation = boost::log::sinks::file::rotation_at_time_point(0, 0, 0), /*< ...or at midnight >*/
		//boost::log::keywords::format = "[%TimeStamp%]: %Message%"                                 /*< log record format >*/
		boost::log::keywords::format = (
		boost::log::expressions::stream
		<< boost::log::expressions::format_date_time< boost::posix_time::ptime >("TimeStamp", "%Y-%m-%d %H:%M:%S")
		<< " [" << std::setw(5) << std::setfill(' ') << boost::log::expressions::attr< severity_level, severity_tag >("Severity")
		<< "] - " << boost::log::expressions::attr< std::string >("Channel")
		<< " - " << boost::log::expressions::smessage
		)
		);

	boost::log::add_common_attributes();
}

#endif