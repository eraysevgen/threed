#ifndef TIMER_HPP_
#define TIMER_HPP_

#include <chrono>

/**
 * @brief A short time keeping function
 *
 * @tparam result_t the difference in time
 * @tparam clock_t clock instance
 * @tparam duration_t duration
 * @param start start time
 * @return auto return the difference in milisecods
 * @see https://stackoverflow.com/questions/2808398/easily-measure-elapsed-time
 */
template <class result_t   = std::chrono::milliseconds,
		  class clock_t	   = std::chrono::steady_clock,
		  class duration_t = std::chrono::milliseconds>
auto since(std::chrono::time_point<clock_t, duration_t> const& start)
{
	return std::chrono::duration_cast<result_t>(clock_t::now() - start);
}
#endif
